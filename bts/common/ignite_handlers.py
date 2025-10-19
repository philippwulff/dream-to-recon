import os
import time
import threading

import torch
import numpy as np
import torch.nn as nn
from typing import Union, Optional, Any, Callable
from logging import Logger
from ignite.engine import Engine, EventEnum, Events
from ignite.contrib.handlers import TensorboardLogger
from ignite.contrib.handlers.base_logger import BaseHandler
from ignite.handlers import global_step_from_engine, DiskSaver
from configs.structured_configs.main_config import MainConfig
import ignite.distributed as idist
import torch.distributed as dist
from typing import Optional, Callable, Any
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger

def get_save_handler(config: MainConfig):
    return DiskSaver(config.RECON_EXP_DIR, create_dir=False, require_empty=False)


class VisualizationHandler(BaseHandler):
    def __init__(self, tag, visualizer, global_step_transform=None):
        self.tag = tag
        self.visualizer = visualizer
        self.gst = global_step_transform
        super().__init__()

    def __call__(self, engine: Engine, logger: TensorboardLogger, event_name: Union[str, EventEnum]) -> None:

        if not isinstance(logger, TensorboardLogger):
            raise RuntimeError("Handler 'VisualizationHandler' works only with TensorboardLogger")

        if self.gst is None:
            gst = global_step_from_engine(engine)
        else:
            gst = self.gst
        global_step = gst(engine, event_name)  # type: ignore[misc]

        if not isinstance(global_step, int):
            raise TypeError(
                f"global_step must be int, got {type(global_step)}."
                " Please check the output of global_step_transform."
            )

        self.visualizer(engine, logger, global_step, self.tag)
        

class GradientLoggingHandler:
    def __init__(self, model: nn.Module, tb_logger: TensorboardLogger, logger: Optional[Logger] = None, tag: Optional[str] = None):
        self.model = model
        self.tb_logger = tb_logger
        self.logger = logger
        self.tag = f"-{tag}" if tag else ""

    def __call__(self, engine: Engine) -> None:
        # Per-parameter gradient norms
        all_grad_norms = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if param.grad is not None:      # If this param is part of the modules-to-be-trained
                    # PARAM NORMS
                    p_norm = param.detach().data.norm(p=2)
                    if not torch.isnan(p_norm) and not torch.isinf(p_norm):
                        self.tb_logger.add_scalar(f"param_norm{self.tag}/{name}", p_norm.item(), engine.state.iteration)                
                        self.tb_logger.add_histogram(f"param_hist{self.tag}/{name}", param.detach().data.cpu().numpy(), engine.state.iteration)
                    elif self.logger:
                        self.logger.warn(f"Found NaN or Inf in param {name}")
                    
                    # GRAD NORMS
                    grad_norm = param.grad.detach().data.norm(p=2)
                    all_grad_norms.append(grad_norm)
                    if not torch.isnan(grad_norm) and not torch.isinf(grad_norm):
                        self.tb_logger.add_scalar(f"grad_norm{self.tag}/{name}", grad_norm.item(), engine.state.iteration)                
                        self.tb_logger.add_histogram(f"grad_hist{self.tag}/{name}", param.grad.detach().data.cpu().numpy(), engine.state.iteration)
                    elif self.logger:
                        self.logger.warn(f"Found NaN or Inf Grad in param {name} with requires_grad=True")
                elif self.logger:
                    self.logger.warn(f"Found None in param {name} with requires_grad=True.")
        
        # Compute the overall gradient norm following `torch.nn.utils.clip_grad_norm_`.
        all_grad_norms = torch.stack([p.grad.detach().norm(p=2) for p in self.model.parameters() if p.grad is not None])
        grads_norm = all_grad_norms.norm(p=2)
        if not torch.isnan(grads_norm) and not torch.isinf(grads_norm):
            self.tb_logger.add_scalar(f"grad_norm{self.tag}/total", grads_norm.item(), engine.state.iteration)                
        elif self.logger:
            self.logger.warn(f"Found NaN or Inf in total grad norm.")
            
            
class MetricLoggingHandler(BaseHandler):
    def __init__(self, tag, optimizer=None, log_loss=True, log_metrics=True, log_timings=True, log_weights=False, log_vram=False, global_step_transform=None):
        self.tag = tag
        self.optimizer = optimizer
        self.log_loss = log_loss
        self.log_metrics = log_metrics
        self.log_timings = log_timings
        self.gst = global_step_transform
        self.log_weights = log_weights
        self.log_vram = log_vram
        super(MetricLoggingHandler, self).__init__()

    def __call__(self, engine: Engine, logger: TensorboardLogger, event_name: Union[str, EventEnum]):
        if not isinstance(logger, TensorboardLogger):
            raise RuntimeError("Handler 'MetricLoggingHandler' works only with TensorboardLogger")

        if self.gst is None:
            gst = global_step_from_engine(engine)
        else:
            gst = self.gst
        global_step = gst(engine, event_name)  # type: ignore[misc]

        if not isinstance(global_step, int):
            raise TypeError(
                f"global_step must be int, got {type(global_step)}."
                " Please check the output of global_step_transform."
            )

        writer = logger.writer

        # Optimizer parameters
        if self.optimizer is not None:
            params = {
                k: float(param_group["lr"]) for k, param_group in enumerate(self.optimizer.param_groups)
            }

            for k, param in params.items():
                writer.add_scalar(f"lr-{self.tag}/{k}", param, global_step)

        epoch = engine.state.epoch
        writer.add_scalar(f"epoch/{self.tag}-engine", epoch, global_step)

        if self.log_loss:
            # Plot losses
            loss_dict = engine.state.output["loss_dict"]
            for k, v in loss_dict.items():
                writer.add_scalar(f"loss-{self.tag}/{k}", v, global_step)

        if self.log_metrics:
            # Plot metrics
            metrics_dict = engine.state.metrics
            metrics_dict_custom = engine.state.output["metrics_dict"]

            for k, v in metrics_dict.items():
                writer.add_scalar(f"metrics-{self.tag}/{k}", v, global_step)
            for k, v in metrics_dict_custom.items():
                writer.add_scalar(f"metrics-{self.tag}/{k}", v, global_step)

        if self.log_timings:
            # Plot timings
            timings_dict = engine.state.times
            timings_dict_custom = engine.state.output["timings_dict"]
            for k, v in timings_dict.items():
                if k == "COMPLETED":
                    continue
                writer.add_scalar(f"timing-{self.tag}/{k}", v, global_step)
            for k, v in timings_dict_custom.items():
                writer.add_scalar(f"timing-{self.tag}/{k}", v, global_step)
        
        if self.log_weights:
            grads_norm = engine.state.output.get("grads_norm", None)
            grad_norm_is_nan = engine.state.output.get("grad_norm_is_nan", None)
            if grads_norm is not None:
                writer.add_scalar(f"weights-{self.tag}/grads_norm", grads_norm, global_step)
                writer.add_scalar(f"weights-{self.tag}/grad_norm_is_nan", grad_norm_is_nan, global_step)
            writer.add_scalar(f"weights-{self.tag}/GradScaler.scale", engine.state.output["GradScaler.scale"], global_step)
            writer.add_scalar(f"weights-{self.tag}/weights_norm", engine.state.output["weights_norm"], global_step)
            writer.add_scalar(f"weights-{self.tag}/weights_abs_max", engine.state.output["weights_abs_max"], global_step)
        
        if self.log_vram:
            writer.add_scalar(f"Performance-{self.tag}/VRAM_max_memory_allocated", engine.state.output["max_memory_allocated"], global_step)
            writer.add_scalar(f"Performance-{self.tag}/VRAM_memory_allocated", engine.state.output["memory_allocated"], global_step)
            writer.add_scalar(f"Performance-{self.tag}/VRAM_max_memory_reserved", engine.state.output["max_memory_reserved"], global_step)
            writer.add_scalar(f"Performance-{self.tag}/RAM_available", engine.state.output["ram_available"], global_step)
            writer.add_scalar(f"Performance-{self.tag}/RAM_used", engine.state.output["ram_used"], global_step)
            writer.add_scalar(f"Performance-{self.tag}/batch_size", engine.state.output["batch_size"], global_step)

class IterationTimeHandler:
    def __init__(self):
        self._start_time = None

    def start_iteration(self, engine):
        self._start_time = time.time()

    def end_iteration(self, engine):
        if self._start_time is None:
            t_diff = 0
            iters_per_sec = 0
        else:
            t_diff = max(time.time() - self._start_time, 1e-6)
            iters_per_sec = 1 / t_diff
        if not hasattr(engine.state, "times"):
            engine.state.times = {}
        else:
            engine.state.times["secs_per_iter"] = t_diff
            engine.state.times["iters_per_sec"] = iters_per_sec


class DataloaderTimeHandler:
    def __init__(self):
        self._start_time = None

    def start_get_batch(self, engine):
        self._start_time = time.time()

    def end_get_batch(self, engine):
        if self._start_time is None:
            t_diff = 0
            iters_per_sec = 0
        else:
            t_diff = max(time.time() - self._start_time, 1e-6)
            iters_per_sec = 1 / t_diff
        if not hasattr(engine.state, "times"):
            engine.state.times = {}
        else:
            engine.state.times["get_batch_secs"] = t_diff


class CacheHandler:
    """
    Handler for caching and loading intermediate computation results during training.
    
    7.800.000 * 4 + 640 * 192 + 192 * 288 * 4 (in float16) = 31,544,064 * 2 bytes = 63 MB
    """
    
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        keys_to_cache: Optional[set[str]] = None,
        cache_prefix: str = "train",
        trigger_cleanup_at: int | None = None,
        cleanup_fn: Callable | None = None,
        persistent: bool = False,
        tensorboard_logger: TensorboardLogger = None,
        log_every: int = 100,
        # shared_memory: Any = None
    ):
        """
        Args:
            cache_dir: Directory to store persistent cache if enabled
            keys_to_cache: Set of keys to cache from state dict. If None, will be determined from first batch
            cache_prefix: Prefix for cache keys (e.g. 'train' or 'val')
            persistent: Whether to save cache to disk between runs
        """
        # self.cache_dir = Path(cache_dir) if cache_dir else None
        self.keys_to_cache = keys_to_cache
        self.cache_prefix = cache_prefix
        self.persistent = persistent
        self.cache_dir = cache_dir
        self.cache: dict[str, Any] = {}
        self.trigger_cleanup_at = trigger_cleanup_at
        self.cleanup_fn = cleanup_fn
        self.tensorboard_logger = tensorboard_logger
        self.log_every = log_every
        if self.persistent:
            assert self.cache_dir is not None, "Persistent cache requires a cache directory"
            # rank = idist.get_rank()
            # if rank == 0:
            os.makedirs(self.cache_dir, exist_ok=True)
            # wait for rank 0 to create the directory
            # dist.barrier()
        self._writes = 0
        self._reads = 0
        # self.shared_memory = shared_memory
        
        # if self.persistent and self.cache_dir:
        #     os.makedirs(self.cache_dir, exist_ok=True)
            
    def _get_cache_key(self, idx: int) -> str:
        """Generate cache key for a specific sample index."""
        return f"{self.cache_prefix}_{idx}"
    
    def _should_cache_key(self, key: str) -> bool:
        """Determine if a state dict key should be cached."""
        if self.keys_to_cache is None:
            # Initialize keys_to_cache from first batch
            # Add keys that contain tensor data we want to cache
            return any(substr in key for substr in [
                "poses_gt",
                "projs_gt",
                "gt_invalid",
                "depths_in",
                "depths_gt",
                "sampled_densities_gt",
                "sampled_densities_invalid",
                "sampled_densities_occluded_and_empty",
                "xyz"
            ])
        return key in self.keys_to_cache
    
    def _cache_batch(self, state_dict: dict[str, Any], indices: torch.Tensor) -> None:
        """Cache relevant data from a batch."""
        for i, idx in enumerate(indices):
            key = self._get_cache_key(idx.item())
            # if not self.shared_memory.has(key):
            #     self.shared_memory.add(key, {
            #         k: state_dict[k][i].clone().cpu()
            #         for k in state_dict
            #         if self._should_cache_key(k)
            #     })
            cache_data = {
                k: state_dict[k][i].clone().cpu()
                for k in state_dict
                if self._should_cache_key(k)
            }
            if self.persistent:
                file_path = os.path.join(self.cache_dir, f"{key}.pt")
                if not os.path.exists(file_path):
                    torch.save(cache_data, file_path)
                    self._writes += 1
            elif key not in self.cache:
                self.cache[key] = cache_data
                self._writes += 1
    
    def _load_cached_batch(self, state_dict: dict[str, Any], indices: torch.Tensor) -> None:
        """Load cached data into state dict for a batch."""
        device = indices.device
        batch_size = len(indices)
        
        # Collect all available cached data for this batch
        cached_data = {}
        for idx in indices:
            key = self._get_cache_key(idx.item())
            
            # Try loading from memory cache
            if self.persistent:
                file_path = os.path.join(self.cache_dir, f"{key}.pt")
                if os.path.exists(file_path):
                    cached_data[idx.item()] = torch.load(file_path)
                    self._reads += 1
            elif key in self.cache:
                cached_data[idx.item()] = self.cache[key]
                self._reads += 1
            # if self.shared_memory.has(key):
            #     cached_data[idx.item()] = self.shared_memory.get(key)
            # Try loading from disk if persistent
        
        # If we have cached data for the full batch, update state dict
        if len(cached_data) == batch_size:
            for tensor_key in cached_data[indices[0].item()]:
                if tensor_key not in state_dict:
                    state_dict[tensor_key] = torch.stack([
                        cached_data[idx.item()][tensor_key]
                        for idx in indices
                    ]).to(device)
    
    def attach(self, engine: Engine) -> None:
        """Attach handler to engine events."""
        engine.add_event_handler(Events.ITERATION_STARTED, self._pre_forward)
        engine.add_event_handler(Events.ITERATION_COMPLETED, self._post_forward)
        if self.tensorboard_logger:
            engine.add_event_handler(Events.ITERATION_COMPLETED(every=self.log_every), self.log_cache_size)
    
    def _pre_forward(self, engine: Engine) -> None:
        """Load cached data before forward pass if available."""
        state_dict = engine.state.batch
        if isinstance(state_dict, dict) and "idxs" in state_dict:
            self._load_cached_batch(state_dict, state_dict["idxs"][:, 0])
    
    def _post_forward(self, engine: Engine) -> None:
        """Cache data after forward pass."""
        # The output from my training iteration function
        state_dict = engine.state.output["output"]
        if isinstance(state_dict, dict) and "idxs" in state_dict:
            self._cache_batch(state_dict, state_dict["idxs"][:, 0])
        
        cache_size = len([_ for _ in os.listdir(self.cache_dir) if _.startswith(self.cache_prefix)]) if self.persistent else len(self.cache)
        if self.cleanup_fn is not None and self.trigger_cleanup_at is not None and cache_size >= self.trigger_cleanup_at:
            self.cleanup_fn()

    def log_cache_size(self, engine: Engine) -> None:
        """Log cache size."""
        if self.tensorboard_logger:
            # rank = engine.state.distributed.get_rank() if hasattr(engine.state, "distributed") else 0
            rank = idist.get_rank()
            self.tensorboard_logger.writer.add_scalar(f"CacheHandler/size-rank{rank}-{self.cache_prefix}", len(self.cache), engine.state.iteration)
            self.tensorboard_logger.writer.add_scalar(f"CacheHandler/writes-rank{rank}-{self.cache_prefix}", self._writes, engine.state.iteration)
            self.tensorboard_logger.writer.add_scalar(f"CacheHandler/reads-rank{rank}-{self.cache_prefix}", self._reads, engine.state.iteration)


class SharedMemoryCacheHandler:
    """
    Handler for caching and loading intermediate computation results during distributed training.
    Uses shared memory to share cache across GPU ranks on the same node.
    """
    
    def __init__(
        self,
        keys_to_cache: Optional[set[str]] = None,
        cache_prefix: str = "train",
        trigger_cleanup_at: int | None = None,
        cleanup_fn: Callable | None = None,
        tensorboard_logger: TensorboardLogger = None,
        log_every: int = 100,
        use_shared_memory: bool = True
        # shared
    ):
        self.keys_to_cache = keys_to_cache
        self.cache_prefix = cache_prefix
        self.trigger_cleanup_at = trigger_cleanup_at
        self.cleanup_fn = cleanup_fn
        self.tensorboard_logger = tensorboard_logger
        self.log_every = log_every
        self.use_shared_memory = use_shared_memory
        
        self.rank = idist.get_rank()
        self.world_size = idist.get_world_size()
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        # Shared memory cache structure:
        # - shared_tensors: dict mapping cache keys to shared memory tensors
        self.shared_tensors: dict[str, dict[str, torch.Tensor]] = {}
        
        # Lock for thread-safe access to shared memory
        self._lock = threading.Lock()
            
    def _get_cache_key(self, idx: int) -> str:
        return f"{self.cache_prefix}_{idx}"
    
    def _should_cache_key(self, key: str) -> bool:
        if self.keys_to_cache is None:
            return any(substr in key for substr in [
                "poses_gt", "projs_gt", "gt_invalid", "depths_in", "depths_gt",
                "sampled_densities_gt", "sampled_densities_invalid",
                "sampled_densities_occluded_and_empty", "xyz"
            ])
        return key in self.keys_to_cache
    
    def _cache_batch(self, state_dict: dict[str, Any], indices: torch.Tensor) -> None:
        """Cache relevant data from a batch in shared memory if enabled."""
        for i, idx in enumerate(indices):
            cache_key = self._get_cache_key(idx.item())
            
            if cache_key not in self.shared_tensors:
                with self._lock:
                    self.shared_tensors[cache_key] = {}

                    # Add a test value
                    if self.rank == 0:
                        test_tensor = torch.tensor([self.rank], dtype=torch.int).share_memory_()
                        self.shared_tensors["_test"] = {"_test": test_tensor}
                        print(f"Rank 0: Set test value for {cache_key}")
                        dist.broadcast(test_tensor, src=0)
                    elif self.rank == 1: 
                        if '_test' in self.shared_tensors:
                            dist.broadcast(self.shared_tensors["_test"]['_test'], src=0)
                            test_val = self.shared_tensors["_test"]['_test'].item()
                            print(f"Rank 1: Found test value {test_val} for {cache_key}")
                    
                    for tensor_key in state_dict:
                        if self._should_cache_key(tensor_key):
                            tensor: torch.Tensor = state_dict[tensor_key][i].clone().cpu()
                            
                            if self.use_shared_memory:
                                # Create a shared memory tensor and copy data into it.
                                shared_tensor = tensor.share_memory_()
                                self.shared_tensors[cache_key][tensor_key] = shared_tensor
                            else:
                                self.shared_tensors[cache_key][tensor_key] = tensor
                    
                    # Synchronize across ranks on the same node
                    if self.use_shared_memory:
                        dist.barrier(device_ids=[self.local_rank])
    
    def _load_cached_batch(self, state_dict: dict[str, Any], indices: torch.Tensor) -> None:
        """Load cached data from shared memory if available."""
        device = indices.device
        batch_size = len(indices)
        
        cached_data = {}
        for idx in indices:
            cache_key = self._get_cache_key(idx.item())
            
            if cache_key in self.shared_tensors:
                cached_data[idx.item()] = {
                    tensor_key: self.shared_tensors[cache_key][tensor_key]
                    for tensor_key in self.shared_tensors[cache_key]
                }
        
        if len(cached_data) == batch_size:
            for tensor_key in cached_data[indices[0].item()]:
                if tensor_key not in state_dict:
                    state_dict[tensor_key] = torch.stack([
                        cached_data[idx.item()][tensor_key]
                        for idx in indices
                    ]).to(device)
    
    def attach(self, engine: Engine) -> None:
        """Attach handler to engine events."""
        engine.add_event_handler(Events.ITERATION_STARTED, self._pre_forward)
        engine.add_event_handler(Events.ITERATION_COMPLETED, self._post_forward)
        if self.tensorboard_logger:
            engine.add_event_handler(Events.ITERATION_COMPLETED(every=self.log_every), self.log_cache_size)
    
    def _pre_forward(self, engine: Engine) -> None:
        state_dict = engine.state.batch
        if isinstance(state_dict, dict) and "idxs" in state_dict:
            self._load_cached_batch(state_dict, state_dict["idxs"][:, 0])
    
    def _post_forward(self, engine: Engine) -> None:
        state_dict = engine.state.output["output"]
        if isinstance(state_dict, dict) and "idxs" in state_dict:
            self._cache_batch(state_dict, state_dict["idxs"][:, 0])
        
        if self.cleanup_fn is not None and self.trigger_cleanup_at is not None:
            with self._lock:
                if len(self.shared_tensors) >= self.trigger_cleanup_at:
                    self.cleanup_fn()
                    # if self.use_shared_memory:
                    #     dist.barrier(device_ids=[self.local_rank])

    def log_cache_size(self, engine: Engine) -> None:
        """Log cache size to tensorboard."""
        if self.tensorboard_logger:
            self.tensorboard_logger.writer.add_scalar(
                f"CacheHandler/size-rank{self.rank}-{self.cache_prefix}", 
                len(self.shared_tensors), 
                engine.state.iteration
            )


class ModelCleaner:
    """
    Callback class to delete the GT synthesizer and cascade wrapper from the model after all engines give their OK.
    """
    def __init__(
            self, 
            config: MainConfig,
            trainer: Engine,
            model: nn.Module,
            get_dataflow_fn: Callable,
            new_batch_size: int,
            logger: Logger
        ):
        self.bts_config = config.BTS
        self.bts_config.BATCH_SIZE = new_batch_size
        self.n_proc_per_node = config.NPROC_PER_NODE

        self.train_ok = False
        self.val_ok = False
        self.vis_ok = False
        self.is_cleaned = False

        self.trainer = trainer
        self.model = model
        self.get_dataflow_fn = get_dataflow_fn
        self.new_batch_size = new_batch_size
        self.logger = logger

    def set_train_ok(self):
        self.train_ok = True
    
    def set_val_ok(self):
        self.val_ok = True

    def set_vis_ok(self):
        self.vis_ok = True

    def __call__(self, engine):
        if self.is_cleaned:
            return

        if not all([self.train_ok, self.val_ok, self.vis_ok]):
            return
        
        # Delete the GT synthesizer and cascade wrapper from the model
        if hasattr(self.model, "gt_synthesizer") and self.model.gt_synthesizer is not None:
            self.logger.info("Cleaning model. Deleting 'gt_synthesizer'...")
            self.model.gt_synthesizer.to("cpu")
            self.model.gt_synthesizer = None
        if hasattr(self.model, "cascade_wrapper") and self.model.cascade_wrapper is not None:
            self.logger.info("Cleaning model. Deleting 'cascade_wrapper'...")
            self.model.cascade_wrapper.to("cpu")
            self.model.cascade_wrapper = None
        torch.cuda.empty_cache()

        self.logger.info(f"Setting new dataloader with all-process batch size {self.new_batch_size}")
        # Switch the dataloader
        new_loader, _, _ = self.get_dataflow_fn(self.bts_config, self.n_proc_per_node, self.logger)
        self.trainer.set_data(new_loader)
        if len(new_loader) == 0:
            self.logger.error("New dataloader is empty! New batch size must be too large.")
        # set_data() does not update epoch length, so we need to do it manually
        if self.trainer.state.epoch_length is not None:
            self.trainer.state.epoch_length = len(new_loader)

        # We also need to account for the new batch size in the iteration count
        # self.trainer.state.iteration = ...

        self.is_cleaned = True