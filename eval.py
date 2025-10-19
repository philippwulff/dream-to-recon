import os
import hydra
from omegaconf import DictConfig

import ignite.distributed as idist

from configs.structured_configs.main_config import MainConfig
from configs.structured_configs.config_utils import register_default_configs, check_and_post_init_config

register_default_configs()

@hydra.main(version_base=None, config_path="configs", config_name="base_main_config")
def main(cfg: DictConfig):

    cfg: MainConfig = check_and_post_init_config(cfg)

    os.environ["NCCL_DEBUG"] = "INFO"

    backend = cfg.BACKEND
    master_port = cfg.MASTER_PORT if backend else None
    if backend == "xla-tpu" and cfg.AMP.ENABLED:
        raise RuntimeError("The value of with_amp should be False if backend is xla")
    
    match cfg.JOB_TYPE:
        case "eval_recon_nvs":
            from bts.ignite_evaluation.evaluator import evaluation
        case "eval_controlnet_input_view":        
            from bts.ignite_evaluation.evaluator_controlnet_input_view import evaluation
        case "eval_controlnet_novel_view":        
            from bts.ignite_evaluation.evaluator_controlnet_novel_view import evaluation
        case "eval_recon_lidar_occ" | "eval_controlnet_lidar_occ": 
            from bts.ignite_evaluation.evaluator_lidar import evaluation
        case _:
            raise ValueError(f"Evaluation job type {cfg.JOB_TYPE} is not available.")

    with idist.Parallel(backend=backend, nproc_per_node=cfg.NPROC_PER_NODE, master_port=master_port) as parallel:
        parallel.run(evaluation, cfg)


if __name__ == "__main__":
    main()
