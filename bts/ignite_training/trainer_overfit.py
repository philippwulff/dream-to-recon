import math
from copy import copy
from typing import Optional, Union, Iterable, Sequence

import ignite.distributed as idist
import torch
from ignite.contrib.handlers import TensorboardLogger
from ignite.engine import Engine
from torch import optim, nn
from torch.utils.data import DataLoader, Dataset, Sampler, Subset
from torch.utils.data.dataloader import T_co, _collate_fn_t, _worker_init_fn_t
from torchvision.utils import make_grid

from datasets.data_util import make_datasets
from bts.common.scheduler import make_scheduler
from bts.renderer import NeRFRenderer
from bts.losses.depth_loss import DepthReconstructionLoss
from bts.ignite_training.trainer import BTSWrapper, BTSNet, BTSDirect, get_metrics, visualize
from utils.array_ops import map_fn, unsqueezer, to
from bts.ignite_training.base_trainer import base_training
from configs.structured_configs.main_config import MainConfig
from configs.structured_configs.bts_config import BTSConfig
from bts.gt_synthesis import GTSynthesisWrapper
from bts.losses import make_loss

from dataclasses import asdict
from torch.cuda.amp import autocast


class EncoderDummy(nn.Module):
    def __init__(self, size, feat_dim, num_views=1) -> None:
        super().__init__()

        self.feats = nn.Parameter(torch.randn(num_views, feat_dim, *size))
        self.latent_size = feat_dim

    def forward(self, x):
        n = x.shape[0]

        return [self.feats.expand(n, -1, -1, -1)]


# class DataloaderDummy(DataLoader):

#     def __init__(self, dataset: Dataset[T_co], batch_size: Optional[int] = 1, shuffle: Optional[bool] = None,
#                  sampler: Union[Sampler, Iterable, None] = None,
#                  batch_sampler: Union[Sampler[Sequence], Iterable[Sequence], None] = None, num_workers: int = 0,
#                  collate_fn: Optional[_collate_fn_t] = None, pin_memory: bool = False, drop_last: bool = False,
#                  timeout: float = 0, worker_init_fn: Optional[_worker_init_fn_t] = None, multiprocessing_context=None,
#                  generator=None, *, prefetch_factor: int = None, persistent_workers: bool = False,
#                  pin_memory_device: str = ""):
#         super().__init__(dataset, batch_size, shuffle, sampler, batch_sampler, num_workers, collate_fn, pin_memory,
#                          drop_last, timeout, worker_init_fn, multiprocessing_context, generator,
#                          prefetch_factor=prefetch_factor, persistent_workers=persistent_workers,
#                          pin_memory_device=pin_memory_device)

#         self.element = to(map_fn(map_fn(dataset.__getitem__(0), torch.tensor), unsqueezer), "cuda:0")

#     def _get_iterator(self):
#         return iter([self.element])

#     def __iter__(self):
#         return super().__iter__()

#     def __len__(self) -> int:
#         return 1


class BTSWrapperOverfit(BTSWrapper):
    def __init__(self, renderer: NeRFRenderer, gt_synthesizer: GTSynthesisWrapper, config: MainConfig, eval_nvs: bool = False, size=None) -> None:
        super().__init__(renderer, gt_synthesizer, config, eval_nvs)

        encoder_dummy = EncoderDummy(size, asdict(config.BTS.MODEL_CONF)["encoder"]["d_out"], num_views=1)
        # self.encoder_dummy = EncoderDummy(size, asdict(config.BTS.MODEL_CONF)["encoder"]["d_out"], num_views=config["num_multiviews"])

        self.renderer.net.encoder = encoder_dummy
        self.renderer.net.flip_augmentation = False


def training(local_rank, config):
    return base_training(local_rank, config, get_dataflow, initialize, get_metrics, visualize)


def get_dataflow(config: BTSConfig, logger=None):
    # - Get train/test datasets
    if idist.get_local_rank() > 0:
        # Ensure that only local rank 0 download the dataset
        # Thus each node will download a copy of the dataset
        idist.barrier()

    # train_dataset, _ = make_datasets(asdict(config.DATA))
    train_dataset = Subset(
        make_datasets(asdict(config.DATA))[0],
        # asdict(config).get("example", config.DATA.skip),
        [0]
    )

    # train_dataset.length = 1
    train_dataset.dataset._skip = config.DATA.skip

    vis_dataset = copy(train_dataset)
    test_dataset = copy(train_dataset)

    vis_dataset.dataset.return_depth = True
    test_dataset.dataset.return_depth = True

    if idist.get_local_rank() == 0:
        # Ensure that only local rank 0 download the dataset
        idist.barrier()

    # Setup data loader also adapted to distributed config: nccl, gloo, xla-tpu
    train_loader = DataLoader(train_dataset)
    test_loader = DataLoader(test_dataset)
    vis_loader = DataLoader(vis_dataset)
    # train_loader = DataloaderDummy(train_dataset)
    # test_loader = DataloaderDummy(test_dataset)
    # vis_loader = DataloaderDummy(vis_dataset)

    return train_loader, test_loader, vis_loader


def initialize(config: MainConfig, logger=None):
    
    net = globals()[config.BTS.MODEL_CONF.ARCH](asdict(config.BTS.MODEL_CONF))
    renderer = NeRFRenderer.from_conf(asdict(config.BTS.RENDERER))
    renderer = renderer.bind_parallel(net, gpus=None).eval()
    
    gt_synthesizer = GTSynthesisWrapper.from_conf(config, config.SYNTHETIC_GT, cam_incl_adjust=config.BTS.DATA.CAM_INCL_ADJUST, make_refiner=True)
    gt_synthesizer = gt_synthesizer.eval()
    gt_synthesizer.requires_grad_(False)

    model = BTSWrapperOverfit(
        renderer,
        gt_synthesizer,
        config,
        eval_nvs=config.BTS.MODE == "nvs",
        size=config.BTS.DATA.image_size,
    )

    model = idist.auto_model(model)
    
    def get_model_parameters(model):
        if hasattr(model, 'module'):
            # DDP wrapped
            return model.module.renderer.net.parameters()
        else:
            return model.renderer.net.parameters()

    optimizer = optim.Adam(get_model_parameters(model), lr=config.BTS.LR)
    # optimizer = optim.Adam(model.renderer.net.parameters(), lr=config.BTS.LR)
    optimizer = idist.auto_optim(optimizer)

    lr_scheduler = make_scheduler(config.BTS.SCHEDULER, optimizer)

    # criterion = DepthReconstructionLoss(asdict(config.BTS.LOSS), config.BTS.MODEL_CONF.USE_AUTOMASKING)
    criterion = make_loss(config.BTS.LOSSES, use_automasking=config.BTS.MODEL_CONF.USE_AUTOMASKING)

    return model, optimizer, criterion, lr_scheduler


