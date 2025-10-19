import os
import sys
from pathlib import Path
from typing import Tuple
from torch.utils.data import Dataset

import cv2
import hydra as hydra
from matplotlib import pyplot as plt
import numpy as np
import torch
from hydra import compose, initialize
from omegaconf import OmegaConf
from configs.structured_configs.config_utils import register_default_configs, check_and_post_init_config
from configs.structured_configs.main_config import MainConfig
from datasets.data_util import make_datasets
from torchvision.transforms import CenterCrop

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.append(os.path.abspath(os.getcwd()))

# bts_path = "~/storage/user/BTS"
# pythonpath = os.environ.get("PYTHONPATH", "")
# os.environ["PYTHONPATH"] = pythonpath + os.pathsep + bts_path if pythonpath else bts_path

from datasets.realestate10k.realestate10k_dataset import RealEstate10kDataset
from datasets.kitti_360.kitti_360_dataset import Kitti360Dataset
from datasets.kitti_raw.kitti_raw_dataset import KittiRawDataset

os.system("nvidia-smi")

gpu_id = 0

device = f'cuda:0'
if gpu_id is not None:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
if torch.cuda.is_available():
    print(f"Using GPU {torch.cuda.get_device_name()} at index {gpu_id}.")
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

r, c, = 0, 0
n_rows, n_cols = 3, 3


def plot(img, fig, axs, i=None):
    global r, c
    if r == 0 and c == 0:
        plt.show()
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 2))
    axs[r][c].imshow(img, interpolation="none")
    if i is not None:
        axs[r][c].title.set_text(f"{i}")
    c += 1
    r += c // n_cols
    c %= n_cols
    r %= n_rows
    return fig, axs


def load_and_setup_config(config_name: str, data_config_name: str | None = None, merge_config_name: str | None = None) -> MainConfig:
    register_default_configs()
    
    with initialize(version_base=None, config_path="../configs", job_name="scripts"):
        config = compose(config_name=config_name)
        if data_config_name is not None:
            # A little hacky, but we need to cache, delete and 
            # reassign fields that hydra cannot override.
            # data_config = compose(config_name=data_config_name, overrides=[f"+CONTROLNET/DATA={data_config_name}", f"+BTS/DATA={data_config_name}"])
            data_config = compose(config_name=data_config_name)
            bts_data = data_config.BTS.DATA
            ctrl_data = data_config.CONTROLNET.DATA
            sampler = data_config.SYNTHETIC_GT.NV_CAM_SAMPLER
            del data_config.BTS.DATA
            del data_config.CONTROLNET.DATA
            del data_config.SYNTHETIC_GT.NV_CAM_SAMPLER
            config = OmegaConf.merge(config, data_config)
            config.BTS.DATA = bts_data
            config.CONTROLNET.DATA = ctrl_data
            config.SYNTHETIC_GT.NV_CAM_SAMPLER = sampler
    
    config = check_and_post_init_config(config)
    
    return config

def setup_task(config: MainConfig, subdir_name: str, return_dataset="test") -> Tuple[Dataset, Path]:
    
    # cp_path = Path(f"out/kitti_360/pretrained")
    # cp_path = Path(config.RECON_EXP_DIR)
    # subsubdir_name = cp_path.name
    # cp_path = next(cp_path.glob("training*.pt"))
    
    # We agree to always use 'config.BTS.DATA' instead of 'config.CONTROLNET.DATA' in scripts.
    out_path = Path(os.path.join("media", subdir_name, config.BTS.DATA.type))
    
    config.BTS.DATA.MAX_TRAIN_DATASET_LEN = None
    config.BTS.DATA.MAX_VAL_DATASET_LEN = None
    config.BTS.DATA.MAX_TEST_DATASET_LEN = None
    train_dataset, val_dataset, test_dataset = make_datasets(config.BTS.DATA)
    dataset = {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset,
    }[return_dataset]

    print("Setup folders")
    out_path.mkdir(exist_ok=True, parents=True)
    
    return dataset, out_path


def save_plot(img, file_name=None, grey=False, mask=None, dry_run=False):
    if mask is not None:
        if mask.shape[-1] != img.shape[-1]:
            mask = np.broadcast_to(np.expand_dims(mask, -1), img.shape)
        img = np.array(img)
        img[~mask] = 0
    if dry_run:
        plt.imshow(img)
        plt.title(file_name)
        plt.show()
    else:
        cv2.imwrite(file_name, cv2.cvtColor((img * 255).clip(max=255).astype(np.uint8), cv2.COLOR_RGB2BGR) if not grey else (img * 255).clip(max=255).astype(np.uint8))


def flatten_pad_stack(imgs, flatten="row"):
    """
    In: list of [B, N, C, H, W] with varying H and W
    Out: if flatten="row" then [len, B, C, maxH, maxW*N] otherwise [len, B, C, maxH*N, maxW]
    """
    B, _, C, _, _ = imgs[0].shape
    if flatten == "row":
        imgs = [img.permute(0, 2, 3, 1, 4).reshape(B, C, img.size(3), img.size(1)*img.size(4)) for img in imgs]
    else:
        imgs = [img.permute(0, 2, 1, 3, 4).reshape(B, C, img.size(1)*img.size(3), img.size(4)) for img in imgs]
    # Pad all imgs to maximal size
    max_h = max([img.size(-2) for img in imgs])
    max_w = max([img.size(-1) for img in imgs])
    transform = CenterCrop([max_h, max_w])
    imgs = [transform(img) for img in imgs]
    return torch.stack(imgs)


def setup_kitti360(out_folder, split="test", split_name="seg"):
    resolution = (192, 640)

    dataset = Kitti360Dataset(
        data_path="data/KITTI-360",
        pose_path="data/KITTI-360/data_poses",
        split_path=f"datasets/kitti_360/splits/{split_name}/{split}_files.txt",
        return_fisheye=False,
        return_stereo=False,
        return_depth=False,
        frame_count=1,
        dilation=30,
        target_image_size=resolution,
        fisheye_rotation=(25, -25),
        color_aug=False)

    config_path = "exp_kitti_360"

    # cp_path = Path(f"out/kitti_360/pretrained")
    # cp_name = cp_path.name
    # cp_path = next(cp_path.glob("training*.pt"))
    cp_name = None
    cp_path = None

    out_path = Path(f"media/{out_folder}/kitti_360/{cp_name}")

    cam_incl_adjust = torch.tensor(
    [  [1.0000000,  0.0000000,  0.0000000, 0],
       [0.0000000,  0.9961947, -0.0871557, 0],
       [0.0000000,  0.0871557,  0.9961947, 0],
       [0.0000000,  000000000,  0.0000000, 1]
    ],
    dtype=torch.float32).view(1, 4, 4)

    return dataset, config_path, cp_path, out_path, resolution, cam_incl_adjust


def setup_kittiraw(out_folder, split="test"):
    resolution = (192, 640)

    dataset = KittiRawDataset(
        data_path="data/KITTI-Raw",
        pose_path="datasets/kitti_raw/out",
        split_path=f"datasets/kitti_raw/splits/eigen_zhou/{split}_files.txt",
        frame_count=1,
        target_image_size=resolution,
        return_stereo=True,
        return_depth=False,
        color_aug=False)

    config_path = "exp_kitti_raw"

    cp_path = Path(f"out/kitti_raw/pretrained")
    cp_name = cp_path.name
    cp_path = next(cp_path.glob("training*.pt"))

    out_path = Path(f"media/{out_folder}/kitti_raw/{cp_name}")

    cam_incl_adjust = None

    return dataset, config_path, cp_path, out_path, resolution, cam_incl_adjust


def setup_re10k(out_folder, split="test"):
    resolution = (256, 384)

    dataset = RealEstate10kDataset(
        data_path="data/RealEstate10K",
        split_path=f"datasets/realestate10k/splits/mine/{split}_files.txt" if split != "train" else None,
        frame_count=1,
        target_image_size=resolution)

    config_path = "exp_re10k"

    cp_path = Path(f"out/re10k/pretrained")
    cp_name = cp_path.name
    cp_path = next(cp_path.glob("training*.pt"))

    out_path = Path(f"media/{out_folder}/re10k/{cp_name}")

    cam_incl_adjust = None

    return dataset, config_path, cp_path, out_path, resolution, cam_incl_adjust


def render_poses(renderer, ray_sampler, poses, projs, black_invalid=False):
    """_summary_

    Args:
        renderer:
        ray_sampler:
        poses:
        projs:
        black_invalid:

    Returns:
        _type_: _description_
    """
    all_rays, _ = ray_sampler.sample(None, poses[:, :1], projs[:, :1])      # [n*nv, n_pts, 8]
    render_dict = renderer(all_rays, want_weights=True)

    render_dict["fine"] = dict(render_dict["coarse"])
    render_dict = ray_sampler.reconstruct(render_dict)

    depth = render_dict["coarse"]["depth"].squeeze(1)[0].cpu()
    frame = render_dict["coarse"]["rgb"][0].cpu()

    invalid = (render_dict["coarse"]["invalid"].squeeze(-1) * render_dict["coarse"]["weights"]).sum(-1).squeeze() > .8

    if black_invalid:
        depth[invalid] = depth.max()
        frame[invalid.unsqueeze(0).unsqueeze(-1), :] = 0

    return frame, depth


print("+++ Inference Setup Complete +++")
