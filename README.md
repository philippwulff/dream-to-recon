# Dream-to-Recon: Monocular 3D Reconstruction with Diffusion-Depth Distillation from Single Images

<div align="center" autoplay controls>
<p>
  <a href="https://arxiv.org/abs/2508.02323">
    <img src="https://badgen.net/static/arXiv/2508.02323/a42c25/" alt="arXiv: 2508.02323">
  </a>
  <a href="https://philippwulff.github.io/dream-to-recon/">
    <img src="https://badgen.net/static/Project/Website//" alt="Project Website">
  </a>
</p>
<h3>
  <a href="https://philippwulff.github.io">Philipp Wulff</a>,&nbsp&nbsp
  <a href="https://fwmb.github.io">Felix Wimbauer</a>,&nbsp&nbsp
  <a href="https://dominikmuhle.github.io/">Dominik Muhle</a>,&nbsp&nbsp
  <a href="https://vision.in.tum.de/members/cremers">Daniel Cremers</a>
</h3>
<video width="100%" loop>
  <source src="https://github.com/philippwulff/dream-to-recon/releases/download/v0.0.0/teaser_crop_3fps_3x.mp4" type="video/mp4">
</video>
<em>
We leverage a diffusion model and a depth predictor to generate high-quality scene geometry from a single image. Then, we distill a feed-forward scene reconstruction model, which performs on par with methods trained via multi-view supervision.</em>
</div>
<br>


This is the official implementation of the paper:

> **Dream-to-Recon: Monocular 3D Reconstruction with Diffusion-Depth Distillation from Single Images**
>
> [Philipp Wulff](https://philippwulff.github.io)<sup>1</sup>, [Felix Wimbauer](https://fwmb.github.io)<sup>1,2</sup>, [Dominik Muhle](https://dominikmuhle.github.io/)<sup>1,2,3</sup> and [Daniel Cremers](https://vision.in.tum.de/members/cremers)<sup>1,2</sup><br>
> <sup>1</sup>Technical University of Munich, <sup>2</sup>MCML, <sup>3</sup>SE3 Labs
> 
> *ICCV 2025*

If you find our work useful, please consider citing our paper:

```
@inproceedings{wulff2025dreamtorecon,
  title     = {Dream-to-Recon: Monocular 3D Reconstruction with Diffusion-Depth Distillation from Single Images},
  author    = {Wulff, Philipp and Wimbauer, Felix and Muhle, Dominik and Cremers, Daniel},
  booktitle = {IEEE International Conference on Computer Vision (ICCV)},
  year      = {2025}
}
```

# 🪧 Overview 
<div align="center">
<img src="https://philippwulff.github.io/dream-to-recon/static/images/method_v2.svg" alt="Method Overview"/>
<em>

Dream-to-Recon comprises three steps: a) We train a view completion model (VCM) that inpaints occluded areas and refines warped views. Training uses only a single view per scene and leverages forward-backward warping for data generation. b) The VCM is applied iteratively alongside a depth prediction network to synthesize virtual novel views, enabling progressive refinement of the 3D geometry. c) The synthesized scene geometries are then used to distill a feed-forward scene reconstruction model by supervising occupancy and virtual depth.
</em>
</div>
<br>

# 🏗️️ Setup

```shell
git clone https://github.com/philippwulff/dream-to-recon

# We use Conda to manage our Python 🐍 environment
conda env create -f environment.yml
conda activate dtr
```

## 💾 Datasets 

All data should be placed under the `data/` folder (or linked to there, e.g. with `ln -s /path/to/KITTI-360/* data/KITTI-360`) in order to match our config files for the 
different datasets. The folder structure should look like:
```bash
data
├── KITTI-360
│   ├── calibration
│   ├── data_2d_raw
│   ├── data_3d_raw
│   └── data_poses
└── waymo
    ├── testing
    │   ├── 39847154216997509_6440_000_6460_000
    │   │   ├── frames
    │   │   ├── lidar
    │   │   └── poses.npy
    │   └── ...
    ├── training
    └── validation
```

Run this to get the pre-processed evaluation ground truth data (LiDAR occupancy volumes):
```bash
bash fetch_assets.sh GT
```

All non-standard data (like precomputed poses and datasplits) comes with this repository and can be found in the `datasets/` folder.

**KITTI-360**

To download KITTI-360, go to https://www.cvlibs.net/datasets/kitti-360/index.php and create an account.
We require the perspective images, raw velodyne scans, calibrations, and vehicle poses.

**Waymo**

The Waymo dataset format differs between versions and we used a Waymo dataset that is shared among our chair, but I believe it is V2.0.1. To download this version, install `gsutil` and run `gsutil -m cp gs://waymo_open_dataset_v_2_0_1 data/waymo`.

**Other Dataset Implementations**

This repository contains dataloader implementations for other datasets, too. 
These are **not officially supported** and are **not guaranteed to work out of the box**.
However, they might be helpful when extending this codebase.

## 📸 Checkpoints

We provide download links for pretrained models for **KITTI-360** and **Waymo**.
Models will be stored under `out/<dataset>/pretrained/<checkpoint-name>.pth`.

```shell
# Run this from the root directory.
bash fetch_assets.sh checkpoints
```

# 🏃 Running the Pre-trained Models

We provide a script to run our pretrained models with custom data.
The script can be found under `scripts/images/gen_img_3d_vol_custom.py` and takes the following flags:

- `--img <path>` / `i <path>`: Path to input image. The image will be resized to match the model's default resolution. `media/example/` contains two example images. 
- `--model <model>` / `-m  <model>`: Which pretrained model to use (`KITTI-360` (default), `Waymo`).

Note that we use the default projection matrices for the respective datasets.

```bash
# Save outputs to disk
python scripts/images/gen_img_3d_vol_custom.py --img media/example/0000.png --model KITTI-360
```

## 📽️ Re-producing Our Image and Video Results

We provide scripts to generate images and videos from the outputs of our models.
Generally, you can adapt the model and configuration for the output by changing some constant in the scripts.
Generated files are stored under `media/`.

|  | Command |
| - | - |
| **Generate Videos** |  |
| 3D occupancy while driving | `python scripts/videos/gen_vid_3d_vol.py` |
| Novel views, occlusion maps, etc. for different trajectories and scenes | `python scripts/videos/gen_vid_nvs.py` |
| BEV density maps while driving | `python scripts/videos/gen_vid_nvs.py` |
| **Generate Images** |  |
| 3D occupancy | `python scripts/images/gen_img_3d_vol.py` |
| VCM inputs & outputs | `python scripts/images/gen_img_vcm.py` |
| Occlusion detection outputs | `python scripts/images/gen_img_occlusion_detection.py` |

# 🏋 Training

## View Completion Model (Controlnet)

We trained our view completion models on a single Nvidia A40 GPU with 48GB memory.
```bash
# On KITTI-360
accelerate launch train_controlnet.py -cn controlnet_full_512x768
# On Waymo
accelerate launch train_controlnet.py -cn controlnet_full_512x768_waymo
# To fine-tune the KITTI-360 checkpoint on Waymo
accelerate launch train_controlnet.py -cn controlnet_full_512x768_waymo CONTROLNET.TRAIN.RESUME_FROM_CHECKPOINT="path/to/kitti360_checkpoint.pt"

```

## Feed-forward Scene Reconstruction Model
Our scene reconstruction models were trained on 4 Nvidia A40 GPU with 48GB memory each.

```bash
# On KITTI-360
python train.py -cn exp_recon_full NAME="my_recon_kitti_360"
# On Waymo
python train.py -cn waymo NAME="my_recon_waymo"
```

# 📊 Evaluation

We further provide configurations to reproduce the evaluation results from the paper. These produce NVS metrics to evaluate the view completion model and LiDAR occupancy reconstruction results for the scene reconstruction model.

```bash
# Make sure you have downloaded our checkpoints before running this.
# Evaluate all view completion models.
bash eval_view_completion.sh
# Evaluate all reconstruction models.
bash eval_reconstruction.sh
```

# 🗣️ Acknowledgements

Acknowledgements: This work was funded by the ERC Advanced Grant ”SIMULACRON” (agreement #884679), the GNI Project ”AI4Twinning”, and the DFG project CR 250/26-1 ”4D YouTube”.

This repository is based on [BehindTheScenes](https://github.com/Brummi/BehindTheScenes), [PixelNeRF](https://github.com/sxyu/pixel-nerf) and [Monodepth2](https://github.com/nianticlabs/monodepth2).
