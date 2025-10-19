from .bts import BTSNet
from .bts_old import BTSNetOld
from .bts_direct import BTSDirect
from configs.structured_configs.bts_config import ModelConfig
from dataclasses import asdict


def get_model_parameters(cfg, model):
    if hasattr(model, 'module'):
        # DDP wrapped
        return model.module.renderer.net.parameters()
    else:
        return model.renderer.net.parameters()
    

def make_model(cfg: ModelConfig):
    assert hasattr(cfg, "ARCH"), "ModelConfig must have an ARCH attribute."
    match cfg.ARCH:
        case "BTSNet":
            cls = BTSNet
        case "BTSNetOld":
            cls = BTSNetOld
        case "BTSDirect":
            cls = BTSDirect
        case _:
            raise ValueError(f"Model {cfg.ARCH} not available.")
    # model = cls.from_config(cfg)
    model = cls(asdict(cfg))
    return model