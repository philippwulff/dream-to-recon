import os
import hydra
from omegaconf import DictConfig

import torch
import torch._dynamo
import ignite.distributed as idist
from torch.multiprocessing.spawn import ProcessRaisedException

from configs.structured_configs.config_utils import register_default_configs, check_and_post_init_config
from configs.structured_configs.main_config import MainConfig

register_default_configs()

@hydra.main(version_base=None, config_path="configs", config_name="base_main_config")
def main(cfg: DictConfig):
    
    cfg: MainConfig = check_and_post_init_config(cfg)

    os.environ["NCCL_DEBUG"] = "INFO"   # debug information that is displayed from NCCL
    
    torch.autograd.set_detect_anomaly(cfg.DETECT_ANOMALY, check_nan=cfg.DETECT_ANOMALY_CHECK_NAN)
    if any([cfg.SYNTHETIC_GT.COMPILE_REFINER, cfg.SYNTHETIC_GT.COMPILE_DEPTH_PREDICTOR]):
        torch._dynamo.config.suppress_errors = True

    backend = cfg.BACKEND
    master_port = cfg.MASTER_PORT if backend else None
    if backend == "xla-tpu" and cfg.AMP.ENABLED:
        raise RuntimeError("The value of with_amp should be False if backend is xla")
    
    match cfg.JOB_TYPE:
        case "bts":
            from bts.ignite_training.trainer import training
        case "bts_overfit":        
            from bts.ignite_training.trainer_overfit import training
        case _:
            raise ValueError(f"Invalid job type: {cfg.JOB_TYPE}")

    def run(mp):
        with idist.Parallel(backend=backend, nproc_per_node=cfg.NPROC_PER_NODE, master_port=mp) as parallel:
            parallel.run(training, cfg)
    
    NUM_RETRIES = 10
    for _ in range(NUM_RETRIES):
        try:
            run(master_port)    
            break
        except ProcessRaisedException as e:
            if "errno: 98 - Address already in use" in str(e.msg) and master_port:
                print(f"Caught: {e}")
                master_port += 1
                continue
            else:
                raise e

if __name__ == "__main__":
    main()
