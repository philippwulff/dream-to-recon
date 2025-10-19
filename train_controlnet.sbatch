#!/bin/bash
#SBATCH --job-name="train controlnet"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1,VRAM:40G
#SBATCH --mem=32G
#SBATCH --time=240:00:00
#SBATCH --exclude=node1
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/user/wph/slurm/logs/slurm-%j-%x.out
#SBATCH --error=/storage/user/wph/slurm/logs/slurm-%j-%x.out
#SBATCH --partition=DEADLINEBIG
#SBATCH --comment="iccv"
#SBATCH --mail-user=slurmbot-SKirjMcCKeoiWavHiq@vision.in.tum.de

export NCCL_DEBUG=INFO
pwd; hostname; date
nvidia-smi
# Launching in the right conda environment automatically uses it in the job
echo $CONDA_DEFAULT_ENV


# accelerate launch controlnet/train_controlnet.py -cn exp_controlnet_base ++CONTROLNET.TRAIN.RESUME_FROM_CHECKPOINT="latest" ++CONTROLNET.TRAIN.TRAIN_BATCH_SIZE=20
# accelerate launch controlnet/train_controlnet.py -cn exp_controlnet_rgbm ++CONTROLNET.TRAIN.TRAIN_BATCH_SIZE=20
# accelerate launch controlnet/train_controlnet.py -cn exp_controlnet_rgbm_unmasked ++CONTROLNET.TRAIN.TRAIN_BATCH_SIZE=20
# accelerate launch controlnet/train_controlnet.py -cn exp_controlnet_rgbd ++CONTROLNET.TRAIN.TRAIN_BATCH_SIZE=20
# accelerate launch controlnet/train_controlnet.py -cn exp_controlnet_rgbdm ++CONTROLNET.TRAIN.TRAIN_BATCH_SIZE=20
# accelerate launch controlnet/train_controlnet.py -cn exp_controlnet_rgb_prompt ++CONTROLNET.TRAIN.TRAIN_BATCH_SIZE=20 ++CONTROLNET.TRAIN.RESUME_FROM_CHECKPOINT="latest"
# accelerate launch controlnet/train_controlnet.py -cn exp_controlnet_rgbdm_unmasked_prompt ++CONTROLNET.TRAIN.TRAIN_BATCH_SIZE=20

# accelerate launch controlnet/train_controlnet.py -cn exp_controlnet_crops_rgb_10_epochs ++CONTROLNET.TRAIN.TRAIN_BATCH_SIZE=24
# accelerate launch controlnet/train_controlnet.py -cn exp_controlnet_crops_rgb_384x384NV ++CONTROLNET.TRAIN.TRAIN_BATCH_SIZE=28
# accelerate launch controlnet/train_controlnet.py -cn exp_controlnet_crops_rgb_512x768 ++CONTROLNET.TRAIN.TRAIN_BATCH_SIZE=16
# accelerate launch controlnet/train_controlnet.py -cn exp_controlnet_crops_rgbm ++CONTROLNET.TRAIN.TRAIN_BATCH_SIZE=24
# accelerate launch controlnet/train_controlnet.py -cn exp_controlnet_crops_rgbdm_unmasked_prompt ++CONTROLNET.TRAIN.TRAIN_BATCH_SIZE=24 ++CONTROLNET.TRAIN.RESUME_FROM_CHECKPOINT="latest"
# accelerate launch controlnet/train_controlnet.py -cn exp_controlnet_crops_rgb_prompt ++CONTROLNET.TRAIN.TRAIN_BATCH_SIZE=24 ++CONTROLNET.TRAIN.RESUME_FROM_CHECKPOINT="latest"
# accelerate launch controlnet/train_controlnet.py -cn exp_controlnet_crops_rgbm_unmasked ++CONTROLNET.TRAIN.TRAIN_BATCH_SIZE=24 ++CONTROLNET.TRAIN.RESUME_FROM_CHECKPOINT="latest"
# accelerate launch controlnet/train_controlnet.py -cn exp_controlnet_crops_rgbd ++CONTROLNET.TRAIN.TRAIN_BATCH_SIZE=24 ++CONTROLNET.TRAIN.RESUME_FROM_CHECKPOINT="latest"

# accelerate launch controlnet/train_controlnet.py -cn exp_controlnet_crops_rgb_384x384NV_bfloat16 ++CONTROLNET.TRAIN.TRAIN_BATCH_SIZE=32
# accelerate launch train_controlnet.py -cn exp_controlnet_512x768_rgbdm_unmasked ++CONTROLNET.TRAIN.TRAIN_BATCH_SIZE=12

# accelerate launch train_controlnet.py -cn controlnet_rgb_unclip
# accelerate launch train_controlnet.py -cn controlnet_co3d_rgb ++CONTROLNET.TRAIN.RESUME_FROM_CHECKPOINT="latest"


# accelerate launch train_controlnet.py -cn controlnet_full
# accelerate launch train_controlnet.py -cn controlnet_full_512x768

# accelerate launch train_controlnet.py -cn controlnet_rgb_naive



# accelerate launch train_controlnet.py -cn controlnet_co3d_full CONTROLNET.DATA.CATEGORY_NAME=car NAME=controlnet_co3d_full_car SYNTHETIC_GT.Z_FAR=15 AMP.ENABLED=false
# accelerate launch train_controlnet.py -cn controlnet_co3d_full CONTROLNET.DATA.CATEGORY_NAME=cake NAME=controlnet_co3d_full_cake
# accelerate launch train_controlnet.py -cn controlnet_co3d_full CONTROLNET.DATA.CATEGORY_NAME=backpack NAME=controlnet_co3d_full_backpack CONTROLNET.TRAIN.RESUME_FROM_CHECKPOINT="latest"
# accelerate launch train_controlnet.py -cn controlnet_co3d_full CONTROLNET.DATA.CATEGORY_NAME=motorcycle NAME=controlnet_co3d_full_motorcycle CONTROLNET.TRAIN.RESUME_FROM_CHECKPOINT="latest"
# accelerate launch train_controlnet.py -cn controlnet_co3d_full CONTROLNET.DATA.CATEGORY_NAME=bench NAME=controlnet_co3d_full_bench SYNTHETIC_GT.Z_FAR=10
# accelerate launch train_controlnet.py -cn controlnet_co3d_full x NAME=controlnet_co3d_full_sandwich

accelerate launch train_controlnet.py -cn controlnet_full_512x768_waymo CONTROLNET.TRAIN.RESUME_FROM_CHECKPOINT="latest"


### VAL
# python controlnet/test.py -cn exp_controlnet_crops_rgb_10_epochs
# python controlnet/test.py -cn exp_controlnet_crops_rgb_384x384NV
# python controlnet/test.py -cn exp_controlnet_crops_rgb_512x768
# python controlnet/test.py -cn exp_controlnet_crops_rgbm
# python controlnet/test.py -cn exp_controlnet_crops_rgbm_unmasked
# python controlnet/test.py -cn exp_controlnet_crops_rgb_prompt
# python controlnet/test.py -cn exp_controlnet_crops_rgbd
# python controlnet/test.py -cn exp_controlnet_crops_rgbdm_unmasked_prompt
###
