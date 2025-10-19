from pathlib import Path
from typing import Optional
from dataclasses import asdict
from logging import Logger

import torch
import ignite
import ignite.distributed as idist

from configs.structured_configs.main_config import MainConfig


def log_basic_info(logger: Logger, config: MainConfig):
    logger.info(f"Run {config.NAME}")
    logger.info(f"- PyTorch version: {torch.__version__}")
    logger.info(f"- Ignite version: {ignite.__version__}")
    if torch.cuda.is_available():
        # explicitly import cudnn as
        # torch.backends.cudnn can not be pickled with hvd spawning procs
        from torch.backends import cudnn

        logger.info(f"- GPU Device: {torch.cuda.get_device_name(idist.get_local_rank())}")
        logger.info(f"- CUDA version: {torch.version.cuda}")
        logger.info(f"- CUDNN version: {cudnn.version()}")

    logger.info("\n")
    logger.info("Configuration:")
    for key, value in asdict(config.BTS).items():
        logger.info(f"\t{key}: {value}")
    logger.info("\n")

    if idist.get_world_size() > 1:
        logger.info("\nDistributed setting:")
        logger.info(f"\tbackend: {idist.backend()}")
        logger.info(f"\tworld size: {idist.get_world_size()}")
        logger.info("\n")
        
        
def log_metrics(logger: Logger, epoch, elapsed, tag, metrics):
    metrics_output = "\n".join([f"\t{k}: {v}" for k, v in metrics.items()])
    logger.info(f"\nEpoch {epoch} - Evaluation time: {elapsed:.2f}s - {tag} metrics:\n {metrics_output}")
    
    
def log_metrics_current(logger: Logger, metrics: dict):
    def f(engine):
        out_str = "\n" + "\t".join([f"{v.compute():.3f}".ljust(8) for v in metrics.values()])
        out_str += "\n" + "\t".join([f"{k}".ljust(8) for k in metrics.keys()])
        logger.info(out_str)
    return f


def get_checkpoint_path(filepath_or_latest: str, checkpoint_dir: Optional[str] = None):
    """
    Checkpoints should be stored as 
        `path/to/checkpoint<optional text with leading underscore>_<some number>.pt`
    """
    
    if filepath_or_latest == "latest":
        # Get the most recent checkpoint
        checkpoint_files = list(Path(checkpoint_dir).glob("training_checkpoint_*.pt"))
        # latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
        if checkpoint_files:
            path = max(checkpoint_files, key=lambda p: float(p.stem.split("_")[-1]))
        else:
            path = None

        # dirs = os.listdir(cfg.CONTROLNET_EXP_DIR)
        # dirs = [d for d in dirs if d.startswith("checkpoint")]
        # dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        # path = dirs[-1] if len(dirs) > 0 else None
        # path = os.path.join(cfg.CONTROLNET_EXP_DIR, path)
    else:
        # filepath = os.path.join(checkpoint_dir, filename_or_latest) if checkpoint_dir is not None else filename_or_latest
        path = Path(filepath_or_latest)
        assert path.exists(), f"Checkpoint '{path.as_posix()}' is not found."
        
    return path