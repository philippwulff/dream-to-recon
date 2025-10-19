import os
import json
import time
from dataclasses import asdict
import psutil

import ignite.distributed as idist
import torch
from typing import Dict, Callable, List
from ignite.contrib.engines import common
from ignite.contrib.handlers import TensorboardLogger
from ignite.engine import Engine, Events, EventEnum
from ignite.handlers import Checkpoint, global_step_from_engine
from ignite.utils import manual_seed, setup_logger
# from torch.cuda.amp import autocast, GradScaler
# from torch.amp import autocast


from utils.array_ops import to
from utils.metrics import MeanMetric
from utils.profiler import Profiler
from configs.structured_configs.main_config import MainConfig
from bts.common.train_eval_utils import log_basic_info, log_metrics, get_checkpoint_path

from logging import Logger
from torch.nn.utils import clip_grad_norm_
from bts.common.ignite_handlers import VisualizationHandler, GradientLoggingHandler, MetricLoggingHandler, IterationTimeHandler, DataloaderTimeHandler, CacheHandler, SharedMemoryCacheHandler, get_save_handler, ModelCleaner
from bts.losses.base_loss import BaseLoss


class BackpropEvents(EventEnum):
    """Following: https://pytorch-ignite.ai/how-to-guides/08-custom-events/"""
    BACKWARD_STARTED = 'backward_started'
    BACKWARD_COMPLETED = 'backward_completed'
    OPTIM_STEP_COMPLETED = 'optim_step_completed'


def base_training(local_rank: int, config: MainConfig, get_dataflow: Callable, initialize: Callable, get_metrics: Callable, visualize: Callable):
    rank = idist.get_rank()
    manual_seed(config.BTS.SEED + rank)
    device = idist.device()

    logger = setup_logger(name=config.NAME)
    log_basic_info(logger, config)

    if rank == 0:
        os.makedirs(config.RECON_EXP_DIR, exist_ok=True)
        with open(os.path.join(config.RECON_EXP_DIR, "config.json"), "w") as f:
            json.dump(asdict(config), f, indent=4)

    # Setup dataflow, model, optimizer, criterion
    loaders = get_dataflow(config.BTS, config.NPROC_PER_NODE, logger)
    if len(loaders) == 2:
        train_loader, val_loader = loaders
        vis_loader = None
    else:
        train_loader, val_loader, vis_loader = loaders

    train_dataset_len = len(train_loader.dataset)
    val_dataset_len = len(val_loader.dataset)
    vis_dataset_len = len(vis_loader.dataset)

    model, optimizer, criterions, lr_scheduler = initialize(config, logger)

    logger.info(f"# of model parameters: {sum(p.numel() for p in model.parameters())}")

    # Let's now setup evaluator engine to perform model's validation and compute metrics
    metrics = get_metrics(config.BTS, device)
    metrics_loss = {k: MeanMetric((lambda y: lambda x: x["loss_dict"][y])(k)) for crit in criterions for k in crit.get_loss_metric_names()}

    # if config.BTS.LOSS_DURING_VALIDATION:
    #     eval_metrics = {**metrics, **metrics_loss}       # TODO this may be wrong
    # else:
    #     eval_metrics = metrics
    
    profiler = None
    if config.PROFILER.ENABLED:
        profiler = Profiler(os.path.join(config.RECON_EXP_DIR, config.PROFILER.TRACE_SUBDIR_NAME))
        logger.info(f"Running with profiler for {config.PROFILER.get_total_profiling_steps()} steps.")

    # Create trainer for current task
    trainer = create_trainer(model, optimizer, criterions, lr_scheduler, train_loader.sampler if hasattr(train_loader, "sampler") else None, config, logger, profiler, metrics={})

    # We define two evaluators as they wont have exactly similar roles:
    # - `evaluator` will save the best model based on validation score
    criterions_or_none = criterions if config.BTS.LOSS_DURING_VALIDATION else None
    evaluator = create_evaluator(model, metrics=metrics, criterions=criterions_or_none, config=config, tag="val")

    if vis_loader is not None:
        visualizer = create_evaluator(model, metrics=metrics, criterions=criterions_or_none, config=config, tag="vis")
    else:
        visualizer = None

    def run_validation(engine):
        epoch = trainer.state.epoch
        state = evaluator.run(val_loader)
        log_metrics(logger, epoch, state.times["COMPLETED"], "Test", state.metrics)

    def run_visualization(engine):
        epoch = trainer.state.epoch
        state = visualizer.run(vis_loader)
        log_metrics(logger, epoch, state.times["COMPLETED"], "Vis", state.metrics)

    if not config.BTS.VAL_USE_ITERS:
        trainer.add_event_handler(Events.STARTED | Events.EPOCH_COMPLETED(every=config.BTS.VALIDATE_EVERY) | Events.COMPLETED, run_validation)
    else:
        trainer.add_event_handler(Events.STARTED | Events.ITERATION_COMPLETED(every=config.BTS.VALIDATE_EVERY) | Events.COMPLETED, run_validation)

    if visualizer:
        if not config.BTS.VIS_USE_ITERS:
            trainer.add_event_handler(Events.STARTED | Events.EPOCH_COMPLETED(every=config.BTS.VISUALIZE_EVERY) | Events.COMPLETED, run_visualization)
        else:
            trainer.add_event_handler(Events.STARTED | Events.ITERATION_COMPLETED(every=config.BTS.VISUALIZE_EVERY) | Events.COMPLETED, run_visualization)

    tb_logger = None
    if rank == 0:
        # Setup TensorBoard logging on trainer and evaluators. Logged values are:
        #  - Training metrics, e.g. running average loss values
        #  - Learning rate
        #  - Evaluation train/test metrics
        tb_logger = TensorboardLogger(log_dir=config.RECON_EXP_DIR)

        trainer_timer = IterationTimeHandler()
        trainer_timer_data = DataloaderTimeHandler()
        trainer.add_event_handler(Events.ITERATION_STARTED, trainer_timer.start_iteration)
        trainer.add_event_handler(Events.ITERATION_COMPLETED, trainer_timer.end_iteration)
        trainer.add_event_handler(Events.GET_BATCH_STARTED, trainer_timer_data.start_get_batch)
        trainer.add_event_handler(Events.GET_BATCH_COMPLETED, trainer_timer_data.end_get_batch)

        evaluator_timer = IterationTimeHandler()
        evaluator_timer_data = DataloaderTimeHandler()
        evaluator.add_event_handler(Events.ITERATION_STARTED, evaluator_timer.start_iteration)
        evaluator.add_event_handler(Events.ITERATION_COMPLETED, evaluator_timer.end_iteration)
        evaluator.add_event_handler(Events.GET_BATCH_STARTED, evaluator_timer_data.start_get_batch)
        evaluator.add_event_handler(Events.GET_BATCH_COMPLETED, evaluator_timer_data.end_get_batch)

        if visualizer:
            visualizer_timer = IterationTimeHandler()
            visualizer_timer_data = DataloaderTimeHandler()
            visualizer.add_event_handler(Events.ITERATION_STARTED, visualizer_timer.start_iteration)
            visualizer.add_event_handler(Events.ITERATION_COMPLETED, visualizer_timer.end_iteration)
            visualizer.add_event_handler(Events.GET_BATCH_STARTED, visualizer_timer_data.start_get_batch)
            visualizer.add_event_handler(Events.GET_BATCH_COMPLETED, visualizer_timer_data.end_get_batch)

        gst = lambda engine, event_name: trainer.state.epoch
        gst_it_epoch = lambda engine, event_name: (trainer.state.epoch - 1) * engine.state.epoch_length + engine.state.iteration - 1
        eval_gst_it_iters = lambda engine, event_name: (((trainer.state.epoch - 1) * trainer.state.epoch_length + trainer.state.iteration) // config.BTS.VALIDATE_EVERY) * engine.state.epoch_length + engine.state.iteration - 1
        vis_gst_it_iters =  lambda engine, event_name: (((trainer.state.epoch - 1) * trainer.state.epoch_length + trainer.state.iteration) // config.BTS.VISUALIZE_EVERY) * engine.state.epoch_length + engine.state.iteration - 1

        eval_gst_ep_iters = lambda engine, event_name: (((trainer.state.epoch - 1) * trainer.state.epoch_length + trainer.state.iteration) // config.BTS.VALIDATE_EVERY)
        vis_gst_ep_iters = lambda engine, event_name: (((trainer.state.epoch - 1) * trainer.state.epoch_length + trainer.state.iteration) // config.BTS.VISUALIZE_EVERY)

        eval_gst_it = eval_gst_it_iters if config.BTS.VAL_USE_ITERS else gst_it_epoch
        vis_gst_it = vis_gst_it_iters if config.BTS.VIS_USE_ITERS else gst_it_epoch

        eval_gst_ep = eval_gst_ep_iters if config.BTS.VAL_USE_ITERS else gst
        vis_gst_ep = vis_gst_ep_iters if config.BTS.VIS_USE_ITERS else gst

        tb_logger.attach(trainer, MetricLoggingHandler("train", optimizer, log_weights=True, log_vram=True), Events.ITERATION_COMPLETED(every=config.BTS.LOG_EVERY_ITERS))
        if config.LOG_GRADIENTS:
            # Use trainer.add_event_handler for custom events.
            trainer.add_event_handler(BackpropEvents.BACKWARD_COMPLETED(every=config.BTS.LOG_EVERY_ITERS), GradientLoggingHandler(model, tb_logger, logger, tag="train"))

        tb_logger.attach(evaluator, MetricLoggingHandler("val", log_loss=config.BTS.LOSS_DURING_VALIDATION, global_step_transform=eval_gst_ep), Events.EPOCH_COMPLETED)
        if visualizer:
            tb_logger.attach(visualizer, MetricLoggingHandler("vis", log_loss=True, global_step_transform=vis_gst_ep), Events.EPOCH_COMPLETED)

        # Plot config to tensorboard
        config_json = json.dumps(asdict(config), indent=2)
        config_json = "".join("\t" + line for line in config_json.splitlines(True))
        tb_logger.writer.add_text("config", text_string=config_json, global_step=0)

        if visualize is not None:
            train_log_interval = asdict(config.BTS).get("log_tb_train_every_iters", -1)
            val_log_interval = asdict(config.BTS).get("log_tb_val_every_iters", train_log_interval)
            vis_log_interval = asdict(config.BTS).get("log_tb_vis_every_iters", 1)

            if train_log_interval > 0:
                tb_logger.attach(
                    trainer,
                    VisualizationHandler(tag="training", visualizer=visualize),
                    Events.ITERATION_COMPLETED(every=train_log_interval))
            if val_log_interval > 0:
                tb_logger.attach(
                    evaluator,
                    VisualizationHandler(tag="val", visualizer=visualize, global_step_transform=eval_gst_it),
                    Events.ITERATION_COMPLETED(every=val_log_interval))
            if visualizer and config.BTS.log_tb_vis_every_iters > 0:
                tb_logger.attach(
                    visualizer,
                    VisualizationHandler(tag="vis", visualizer=visualize, global_step_transform=vis_gst_it),
                    Events.ITERATION_COMPLETED(every=config.BTS.log_tb_vis_every_iters))

    if config.BTS.CACHE_SYNTHETIC_GT:
        # world_size = idist.get_world_size()
        model_cleaner = ModelCleaner(
            config=config,
            trainer=trainer,
            model=model,
            get_dataflow_fn=get_dataflow,
            new_batch_size=int(
                config.BTS.BATCH_SIZE * config.BTS.BATCH_SIZE_MULTIPLE_AFTER_CLEANUP if config.BTS.BATCH_SIZE_MULTIPLE_AFTER_CLEANUP else config.BTS.BATCH_SIZE
                ),
            logger=logger
        )
        
        cache_dir = config.CUSTOM_RECON_CACHE_DIR if config.CUSTOM_RECON_CACHE_DIR else config.RECON_CACHE_DIR
        train_cache_handler = CacheHandler(cache_prefix="train", trigger_cleanup_at=train_dataset_len, cleanup_fn=model_cleaner.set_train_ok, tensorboard_logger=tb_logger, log_every=100, cache_dir=cache_dir, persistent=True)
        train_cache_handler.attach(trainer)
        val_cache_handler = CacheHandler(cache_prefix="val", trigger_cleanup_at=val_dataset_len, cleanup_fn=model_cleaner.set_val_ok, tensorboard_logger=tb_logger, log_every=10, cache_dir=cache_dir, persistent=True)
        val_cache_handler.attach(evaluator)
        if visualizer is not None:
            vis_cache_handler = CacheHandler(cache_prefix="vis", trigger_cleanup_at=vis_dataset_len, cleanup_fn=model_cleaner.set_vis_ok, tensorboard_logger=tb_logger, log_every=10, cache_dir=cache_dir, persistent=True)
            vis_cache_handler.attach(visualizer)

        trainer.add_event_handler(Events.EPOCH_COMPLETED, model_cleaner)

    if True:#"save_best" in asdict(config):       # TODO enable this
        # Store 2 best models by validation accuracy starting from num_epochs / 2:
        best_model_handler = Checkpoint(
            {"model": model},
            get_save_handler(config),
            filename_prefix="best",
            n_saved=3,
            global_step_transform=global_step_from_engine(trainer),
            score_name="val_loss",
            # score_function=Checkpoint.get_default_score_fn(config.BTS.SAVE_BEST.METRIC, score_sign=config.BTS.SAVE_BEST.SIGN),
            score_function=lambda engine: -engine.state.output["loss_dict"]["loss_total"],
        )
        evaluator.add_event_handler(
            # Events.COMPLETED(lambda *_: trainer.state.epoch > config.BTS.NUM_EPOCHS // 2), best_model_handler
            Events.COMPLETED(lambda *_: True), best_model_handler
        )
        best_density_model_handler = Checkpoint(
            {"model": model},
            get_save_handler(config),
            filename_prefix="best",
            n_saved=3,
            global_step_transform=global_step_from_engine(trainer),
            score_name="density_loss",
            score_function=lambda engine: -engine.state.output["loss_dict"].get("density_loss_total", 0.0),
        )
        evaluator.add_event_handler(
            Events.COMPLETED(lambda *_: True), best_density_model_handler
        )

    # In order to check training resuming we can stop training on a given iteration
    if config.BTS.STOP_ITERATION is not None:

        @trainer.on(Events.ITERATION_STARTED(once=config.BTS.STOP_ITERATION))
        def _():
            logger.info(f"Stop training on {trainer.state.iteration} iteration")
            trainer.terminate()

    try:
        trainer.run(
            train_loader, 
            # Override epochs and epoch length during profiling
            max_epochs=1 if profiler else config.BTS.NUM_EPOCHS, 
            epoch_length=config.PROFILER.get_total_profiling_steps() if profiler else None,
        )
    except Exception as e:
        logger.exception("")
        raise e

    if rank == 0:
        tb_logger.close()
        
    if profiler:
        profiler.cleanup()


def create_trainer(
    model: torch.nn.Module, 
    optimizer: torch.optim.Optimizer, 
    criterions: List[BaseLoss], 
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler, 
    train_sampler, 
    config: MainConfig, 
    logger: Logger, 
    profiler=None, 
    metrics: Dict = {},
):

    device = idist.device()

    assert len(criterions) > 0, "At least one criterion should be provided."

    # Setup Ignite trainer:
    # - let's define training step
    # - add other common handlers:
    #    - TerminateOnNan,
    #    - handler to setup learning rate scheduling,
    #    - ModelCheckpoint
    #    - RunningAverage` on `train_step` output
    #    - Two progress bars on epochs and optionally on iterations

    # The scale factor often causes infs/NaNs to appear in gradients for the first few iterations as its value calibrates.
    # https://pytorch.org/docs/stable/amp.html#gradient-scaling
    scaler = torch.cuda.amp.GradScaler(enabled=config.AMP.ENABLED)
    
    # if config.BTS.BACKPROP_GRAD_CLIP_VAL > 0.:
    #     # https://stackoverflow.com/questions/54716377/how-to-do-gradient-clipping-in-pytorch
    #     def grad_clamp(grad): 
    #         # grad.norm(p=2)
    #         # return 
    #         return grad.clamp(-config.BTS.BACKPROP_GRAD_CLIP_VAL, config.BTS.BACKPROP_GRAD_CLIP_VAL)
    #     for group in optimizer.param_groups:
    #         for p in group['params']:
    #             # The hook is called with the gradient: https://pytorch.org/docs/stable/generated/torch.Tensor.register_hook.html#torch-tensor-register-hook
    #             p.register_hook(grad_clamp)
    
    parameters = [p for group in optimizer.param_groups for p in group['params']]

    def train_step(engine: Engine, data: dict):
        if "t__get_item__" in data:
            timing = {"t__get_item__": torch.mean(data["t__get_item__"]).item()}
        else:
            timing = {}

        _start_time = time.time()

        data = to(data, device)

        timing["t_to_gpu"] = time.time() - _start_time
        
        model.train()

        # FORWARD
        _start_time = time.time()
        with torch.cuda.amp.autocast(enabled=config.AMP.ENABLED):
            if config.PROFILER.ENABLED:
                profiler.step()
            data = model(data)
        timing["t_forward"] = time.time() - _start_time

        # LOSS
        _start_time = time.time()
        overall_loss = torch.tensor(0.0, device=device)
        loss_metrics = {}
        for criterion in criterions:
            loss, loss_dict = criterion(data, parameters=parameters)
            overall_loss = overall_loss + loss
            loss_metrics.update(loss_dict)
        timing["t_loss"] = time.time() - _start_time
        
        # Add the weight decay loss of the adam optimizer
        if config.BTS.WEIGHT_DECAY > 0.:
            loss_metrics["loss_weight_decay"] = .5 * config.BTS.WEIGHT_DECAY * torch.sum(torch.stack([torch.sum(p**2) for p in parameters if p is not None and p.requires_grad]))
        
        loss_metrics["loss_total"] = overall_loss.item() + loss_metrics.get("loss_weight_decay", 0.0)
        
        # BACKWARD
        _start_time = time.time()
        optimizer.zero_grad()
        engine.fire_event(BackpropEvents.BACKWARD_STARTED)
        # Scale float16 losses to prevent underflowing gradients when using AMP.
        # See: https://wandb.ai/wandb_fc/tips/reports/How-To-Use-GradScaler-in-PyTorch--VmlldzoyMTY5MDA5
        # https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.GradScaler
        scaler.scale(overall_loss).backward()
        
        # Optionally, clip the gradient using its norm
        grads_norm = None
        if config.OVERALL_GRAD_CLIP_NORM > 0:
            # Following: https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-clipping
            # Unscales the gradients of optimizer's assigned params in-place
            scaler.unscale_(optimizer)
            # Returns the total norms before clipping.
            grads_norm = clip_grad_norm_(
                parameters=parameters, 
                max_norm=config.OVERALL_GRAD_CLIP_NORM
            )
        engine.fire_event(BackpropEvents.BACKWARD_COMPLETED)
        
        # scaler.step() first unscales gradients of the optimizer's params (if not done during grad clipping).
        # If gradients don't contain infs/NaNs, optimizer.step() is then called,
        # otherwise, optimizer.step() is skipped.
        scaler.step(optimizer)
        engine.fire_event(BackpropEvents.OPTIM_STEP_COMPLETED)
        scaler.update()
        timing["t_backward"] = time.time() - _start_time
        
        # Enforce minimum gradient scale value
        if config.AMP.ENABLED and config.AMP.MIN_SCALE and scaler._scale < config.AMP.MIN_SCALE:
            scaler._scale = torch.tensor(config.AMP.MIN_SCALE).to(scaler._scale)

        # Parameter norm
        p_norms = torch.stack([p.norm(p=2) for p in parameters if p is not None and p.requires_grad])
        p_abs_max = torch.stack([p.abs().max() for p in parameters if p is not None and p.requires_grad])
                
        ret = {
            "output": data,
            "loss_dict": loss_metrics,
            "timings_dict": timing,
            "metrics_dict": {},
            "GradScaler.scale": scaler._scale if hasattr(scaler, "_scale") else 0.0,
            "max_memory_allocated": torch.cuda.max_memory_allocated() / 1024**3,
            "memory_allocated": torch.cuda.memory_allocated() / 1024**3,
            "max_memory_reserved": torch.cuda.max_memory_reserved() / 1024**3,
            "ram_available": psutil.virtual_memory().available / 1024**3,
            "ram_used": psutil.virtual_memory().used / 1024**3,
            "weights_norm": p_norms.norm(p=2),
            "weights_abs_max": p_abs_max.max(),
            "batch_size": data["imgs"][0].shape[0],
        }
        if grads_norm is not None and not any([torch.isnan(grads_norm), torch.isinf(grads_norm)]):
            ret["grad_norm_is_nan"] = 1
            ret["grads_norm"] = grads_norm
        else:
            ret["grad_norm_is_nan"] = 0
        
        return ret

    trainer = Engine(train_step)
    trainer.register_events(*BackpropEvents)
    trainer.logger = logger

    for name, metric in metrics.items():
        metric.attach(trainer, name)

    to_save = {"trainer": trainer, "model": model, "optimizer": optimizer, "lr_scheduler": lr_scheduler}

    common.setup_common_training_handlers(
        trainer=trainer,
        train_sampler=train_sampler,
        to_save=to_save,
        save_every_iters=config.BTS.CHECKPOINT_EVERY,
        save_handler=get_save_handler(config),
        lr_scheduler=lr_scheduler,
        output_names=None,
        with_pbars=False,
        clear_cuda_cache=False,
        log_every_iters=config.BTS.LOG_EVERY_ITERS,
        stop_on_nan=config.STOP_ON_NAN,
    )

    # NOTE: don't move to initialization, as to save it is also needed here
    match config.BTS.RESUME_FROM:
        case None:
            checkpoint_fp = None
        case "latest":
            checkpoint_fp = get_checkpoint_path("latest", config.RECON_EXP_DIR)
        case _:
            checkpoint_fp = get_checkpoint_path(config.BTS.RESUME_FROM)
            
    if checkpoint_fp is not None:
        logger.info(f"Resuming from a checkpoint: {checkpoint_fp.as_posix()}")
        checkpoint = torch.load(checkpoint_fp.as_posix(), map_location="cpu")
        Checkpoint.load_objects(to_load=to_save, checkpoint=checkpoint)

        # return [group['lr'] * self.gamma
        # TODO add other schedulers
        if isinstance(lr_scheduler, torch.optim.lr_scheduler.StepLR):
            for group in optimizer.param_groups:
                if config.BTS.LR != group["lr"]:
                    logger.info(f"Overriding optimizer's initial learning rate from checkpoint: {group['lr']} -> {config.BTS.LR}")
                    group["lr"] = config.BTS.LR

    return trainer



def create_evaluator(model, metrics: Dict[str, MeanMetric], criterions: List[BaseLoss], config: MainConfig, tag="val"):
    device = idist.device()

    @torch.no_grad()
    def evaluate_step(engine: Engine, data: Dict):
        model.eval()
        if "t__get_item__" in data:
            timing = {"t__get_item__": torch.mean(data["t__get_item__"]).item()}
        else:
            timing = {}

        data = to(data, device)

        with torch.cuda.amp.autocast(enabled=config.AMP.ENABLED):
            data = model(data)

        for name in metrics.keys():
            data[name] = data[name].mean()

        loss_overall = torch.tensor(0.0, device=device)
        if criterions is not None:
            loss_dict_full = {}
            for criterion in criterions:
                loss, loss_dict = criterion(data, parameters=[])
                loss_dict_full.update(loss_dict)
                loss_overall += loss
            loss_dict_full["loss_total"] = loss_overall.item()
        else:
            loss_dict_full = {}

        return {
            "output": data,
            "loss_dict": loss_dict_full,
            "timings_dict": timing,
            "metrics_dict": {}
        }

    evaluator = Engine(evaluate_step)

    for name, metric in metrics.items():
        metric.attach(evaluator, name)
    
    if idist.get_rank() == 0 and (not config.WITH_CLEARML):
        common.ProgressBar(desc=f"Evaluation ({tag})", persist=False).attach(evaluator)

    return evaluator


