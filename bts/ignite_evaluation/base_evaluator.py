import yaml
import os
import json
from dataclasses import asdict
from pathlib import Path
import ignite.distributed as idist
import torch
from ignite.contrib.engines import common
from ignite.engine import Engine, Events
from ignite.utils import manual_seed, setup_logger
from torch.cuda.amp import autocast
from configs.structured_configs.main_config import MainConfig
from bts.common.train_eval_utils import log_basic_info, log_metrics, log_metrics_current
from utils.array_ops import to
import pandas as pd
from ignite.contrib.handlers import TensorboardLogger
from ignite.handlers import global_step_from_engine
from ignite.metrics import FID, InceptionScore


def base_evaluation(local_rank, config: MainConfig, get_dataflow, initialize, get_metrics, visualize):
    rank = idist.get_rank()
    manual_seed(config.BTS.SEED + rank)
    device = idist.device()

    logger = setup_logger(name=config.NAME)

    log_basic_info(logger, config)

    model = initialize(config, logger)
    RECONSTRUCTOR_EVAL = "recon" in config.JOB_TYPE     # TODO kind of hacky
    # RECONSTRUCTOR_EVAL = True
    
    if rank == 0:
        logger.info(f"Running eval job: {config.JOB_TYPE}")
        if RECONSTRUCTOR_EVAL:
            output_dir = Path(config.RECON_EVAL_DIR)
        else:
            output_dir = Path(config.CONTROLNET_EVAL_DIR)
        if not output_dir.exists():
            output_dir.mkdir(parents=True)
        logger.info(f"Output path: {output_dir}")
        if "cuda" in device.type:
            config.CUDA_DEVICE_NAME = torch.cuda.get_device_name(local_rank)

    data_config = config.BTS.DATA if RECONSTRUCTOR_EVAL else config.CONTROLNET.DATA
    test_loader = get_dataflow(data_config, num_workers=config.BTS.NUM_WORKERS)

    logger.info(f"Job type: {config.JOB_TYPE}. Loading data from {'BTS.DATA' if RECONSTRUCTOR_EVAL else 'CONTROLNET.DATA'}.")

    if hasattr(test_loader, "dataset"):
        logger.info(f"Dataset test length: {len(test_loader.dataset)}")
    
    model.to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    metrics = get_metrics(config, device)
    evaluator = create_evaluator(model, metrics=metrics, config=config)
    evaluator.add_event_handler(Events.ITERATION_COMPLETED(every=config.BTS.LOG_EVERY_ITERS), log_metrics_current(logger, metrics))
    
    tb_logger = TensorboardLogger(log_dir=output_dir)
    
    # Run a visualization function every few iters and also
    # log the figures to tensorboard and save them to files.
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    def visualize_(engine: Engine):
        step = global_step_from_engine(engine)(engine, Events.ITERATION_COMPLETED)
        visualize(engine, tb_logger, figures_dir, step)
        
    evaluator.add_event_handler(Events.ITERATION_COMPLETED(every=config.BTS.VISUALIZE_EVERY), visualize_)

    # Plot config to tensorboard
    config_json = json.dumps(asdict(config), indent=2)
    config_json = "".join("\t" + line for line in config_json.splitlines(True))
    tb_logger.writer.add_text("config", text_string=config_json, global_step=0)
    # Save config to output folder.
    with open(os.path.join(output_dir, "config.yaml"), 'w') as f:
        yaml.dump(asdict(config), f, default_flow_style=False)

    try:
        state = evaluator.run(test_loader, max_epochs=1)
        
        if rank == 0:
            # Log the final metrics after completion.
            log_metrics(logger, evaluator.state.epoch, state.times["COMPLETED"], "Test", state.metrics)
            for k, v in state.metrics.items():
                tb_logger.writer.add_scalar(f"eval_metrics/{k}", v)
            
            df = pd.DataFrame({
                "experiments_name": config.NAME, 
                "job_type": config.JOB_TYPE,
                "reconstructor_checkpoint": str(model.checkpoint_fp),
                "controlnet_checkpoint": config.CONTROLNET.MODEL.CONTROLNET_MODEL_NAME_OR_PATH,
                "num_test_samples": len(test_loader),
                **state.metrics,
            }, index=[0])
            logger.info("\n#### Results ####")
            logger.info(df.T[::-1])
            df.to_csv(os.path.join(output_dir, "results.csv")) 
            
    except Exception as e:
        logger.exception("")
        raise e


def create_evaluator(model, metrics: dict, config: MainConfig, tag="val"):
    device = idist.device()

    @torch.no_grad()
    def evaluate_step(engine: Engine, data):
        model.eval()
        if "t__get_item__" in data:
            timing = {"t__get_item__": torch.mean(data["t__get_item__"]).item()}
        else:
            timing = {}

        data = to(data, device)

        with autocast(enabled=config.AMP.ENABLED):
            data = model(data)

        loss_metrics = {}

        return {
            "output": data,
            "loss_dict": loss_metrics,
            "timings_dict": timing,
            "metrics_dict": {}
        }

    evaluator = Engine(evaluate_step)

    # Attach metrics to the engine
    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    if idist.get_rank() == 0 and not config.WITH_CLEARML:
        common.ProgressBar(desc=f"Evaluation ({tag})", persist=False).attach(evaluator)

    return evaluator
