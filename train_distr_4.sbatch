#!/bin/bash
#SBATCH --job-name="train distr 4"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4,VRAM:40G
#SBATCH --mem=128G
#SBATCH --time=120:00:00
#SBATCH --exclude=node1
#SBATCH --mail-type=ALL
#SBATCH --partition=DEADLINEBIG
#SBATCH --comment="iccv rebuttal"
#SBATCH --output=/storage/user/wph/slurm/logs/slurm-%j-%x.out
#SBATCH --error=/storage/user/wph/slurm/logs/slurm-%j-%x.out
#SBATCH --mail-user=slurmbot-SKirjMcCKeoiWavHiq@vision.in.tum.de

export NCCL_DEBUG=INFO
pwd; hostname; date
nvidia-smi
# Launching in the right conda environment automatically uses it in the job
echo $CONDA_DEFAULT_ENV

# SBATCH --constraint="GPU_MODEL:rtx_8000|GPU_MODEL:rtx_a6000"


# --- launch cmds ---

export SLURM_NTASKS=4

# SHARED="++BTS.NPROC_PER_NODE=4 ++BTS.BACKEND="nccl" ++BTS.BATCH_SIZE=8 ++BTS.DATA.MAX_DATASET_LEN=1000"

# python train.py -cn exp_bts_synthetic ++NAME="EXP70" ++SYNTHETIC_GT.Z_NEAR_GT=5 ++SYNTHETIC_GT.APPLY_CAM_INCL_ADJUST=true
# python train.py -cn exp_bts_synthetic ++NAME="EXP71" ++BTS.DATA.data_predicted_depth=false ++SYNTHETIC_GT.DEPTH_PREDICTOR_NAME="UniDepth"


# --- BEGIN BTS 10k imgs ----

# ++MASTER_PORT=12876 
SHARED="++NPROC_PER_NODE=4 ++BACKEND="gloo" ++BTS.BATCH_SIZE=8 ++BTS.DATA.MAX_TRAIN_DATASET_LEN=10000"

# python train.py -cn exp_bts_synthetic ++NAME="EXP80" ${SHARED} ++BTS.DATA.data_predicted_depth=false ++SYNTHETIC_GT.DEPTH_PREDICTOR_NAME="UniDepth" ++BTS.MASTER_PORT=12876 ++SYNTHETIC_GT.N_RETRIES_TO_SAMPLE_VALID_NOVEL_VIEW=2 ++SYNTHETIC_GT.MIN_MEAN_OCCLUDED_PIXELS=0.03 ++SYNTHETIC_GT.MIN_MEAN_VALID_PIXELS=0.5 ++BTS.RESUME_FROM="/storage/user/wph/BTS/out/kitti_360/EXP80_backend-nccl-4_20240425-093403/training_checkpoint_17000.pt"
# python train.py -cn exp_bts_synthetic ++NAME="EXP81" ${SHARED} ++BTS.DATA.data_predicted_depth=false ++SYNTHETIC_GT.DEPTH_PREDICTOR_NAME="UniDepth" ++BTS.MASTER_PORT=12876 ++SYNTHETIC_GT.N_RETRIES_TO_SAMPLE_VALID_NOVEL_VIEW=2 ++SYNTHETIC_GT.MIN_MEAN_OCCLUDED_PIXELS=0.03 ++SYNTHETIC_GT.MIN_MEAN_VALID_PIXELS=0.5 ++BTS.WEIGHT_DECAY=0.000001 ++BTS.LOSSES.DensityGridRegularizationLoss.LAMBDA_REG=1 ++BTS.LOSSES.DensityGridRegularizationLoss.THRESHOLD=1 ++BTS.MODEL_CONF.SAMPLED_DENSITY_LAMBDA=16
# python train.py -cn exp_bts_synthetic ++NAME="EXP82" ${SHARED} ++BTS.DATA.data_predicted_depth=false ++SYNTHETIC_GT.DEPTH_PREDICTOR_NAME="UniDepth" ++BTS.MASTER_PORT=12876 ++SYNTHETIC_GT.N_RETRIES_TO_SAMPLE_VALID_NOVEL_VIEW=2 ++SYNTHETIC_GT.MIN_MEAN_OCCLUDED_PIXELS=0.03 ++SYNTHETIC_GT.MIN_MEAN_VALID_PIXELS=0.5 ++BTS.LOSSES.DensityGridRegularizationLoss.LAMBDA_REG=1 ++BTS.LOSSES.DensityGridRegularizationLoss.THRESHOLD=1 ++BTS.MODEL_CONF.SAMPLED_DENSITY_LAMBDA=16
# python train.py -cn exp_bts_synthetic ++NAME="EXP83" ${SHARED} ++BTS.DATA.data_predicted_depth=false ++SYNTHETIC_GT.DEPTH_PREDICTOR_NAME="UniDepth" ++BTS.MASTER_PORT=12876 ++SYNTHETIC_GT.N_RETRIES_TO_SAMPLE_VALID_NOVEL_VIEW=2 ++SYNTHETIC_GT.MIN_MEAN_OCCLUDED_PIXELS=0.03 ++SYNTHETIC_GT.MIN_MEAN_VALID_PIXELS=0.5
# python train.py -cn exp_bts_synthetic ++NAME="EXP84" ${SHARED} ++BTS.DATA.data_predicted_depth=false ++SYNTHETIC_GT.DEPTH_PREDICTOR_NAME="UniDepth" ++BTS.MASTER_PORT=12876 ++SYNTHETIC_GT.N_RETRIES_TO_SAMPLE_VALID_NOVEL_VIEW=2 ++SYNTHETIC_GT.MIN_MEAN_OCCLUDED_PIXELS=0.03 ++SYNTHETIC_GT.MIN_MEAN_VALID_PIXELS=0.5 ++BTS.WEIGHT_DECAY=0


# python train.py -cn exp_bts_synthetic ++NAME="EXP90" ${SHARED} ++BTS.BATCH_SIZE=32 ++BTS.DATA.data_predicted_depth=false ++SYNTHETIC_GT.DEPTH_PREDICTOR_NAME="UniDepth" ++SYNTHETIC_GT.NV_CAM_SAMPLER.NUM_NOVEL_VIEWS=1 ++CONTROLNET.EVAL.NUM_INFERENCE_STEPS=1 ++BTS.LOSSES.DepthReconstructionLoss.LAMBDA_IN=1.0
# python train.py -cn exp_bts_synthetic ++NAME="EXP91" ${SHARED} ++BTS.BATCH_SIZE=12 ++BTS.DATA.data_predicted_depth=false ++SYNTHETIC_GT.DEPTH_PREDICTOR_NAME="UniDepth" ++SYNTHETIC_GT.N_RETRIES_TO_SAMPLE_VALID_NOVEL_VIEW=2 ++SYNTHETIC_GT.MIN_MEAN_OCCLUDED_PIXELS=0.03 ++SYNTHETIC_GT.MIN_MEAN_VALID_PIXELS=0.5

# python train.py -cn exp_bts_synthetic ++NAME="EXP92" ${SHARED} ++BTS.BATCH_SIZE=32 ++BTS.LOSSES.DepthReconstructionLoss.criterion="l2" ++BTS.LOSSES.DepthReconstructionLoss.lambda_var=0 ++BTS.DATA.data_predicted_depth=false ++SYNTHETIC_GT.DEPTH_PREDICTOR_NAME="UniDepth" ++SYNTHETIC_GT.NV_CAM_SAMPLER.NUM_NOVEL_VIEWS=1 ++CONTROLNET.EVAL.NUM_INFERENCE_STEPS=1 ++BTS.LOSSES.DepthReconstructionLoss.LAMBDA_IN=1.0
# python train.py -cn exp_bts_synthetic ++NAME="EXP93" ${SHARED} ++BTS.BATCH_SIZE=32 ++BTS.LOSSES.DepthReconstructionLoss.criterion="l2" ++BTS.LOSSES.DepthReconstructionLoss.lambda_var=1 ++BTS.DATA.data_predicted_depth=false ++SYNTHETIC_GT.DEPTH_PREDICTOR_NAME="UniDepth" ++SYNTHETIC_GT.NV_CAM_SAMPLER.NUM_NOVEL_VIEWS=1 ++CONTROLNET.EVAL.NUM_INFERENCE_STEPS=1 ++BTS.LOSSES.DepthReconstructionLoss.LAMBDA_IN=1.0


# python train.py -cn exp_bts_synthetic ++NAME="EXP94" ${SHARED} ++BTS.BATCH_SIZE=16 ++SYNTHETIC_GT.NV_CAM_SAMPLER.NUM_NOVEL_VIEWS=2 ++BTS.DATA.data_predicted_depth=false ++SYNTHETIC_GT.DEPTH_PREDICTOR_NAME="UniDepth" ++SYNTHETIC_GT.N_RETRIES_TO_SAMPLE_VALID_NOVEL_VIEW=2 ++SYNTHETIC_GT.MIN_MEAN_OCCLUDED_PIXELS=0.03 ++SYNTHETIC_GT.MIN_MEAN_VALID_PIXELS=0.5
# python train.py -cn exp_bts_synthetic ++NAME="EXP95" ${SHARED} ++BTS.BATCH_SIZE=16 ++SYNTHETIC_GT.NV_CAM_SAMPLER.NUM_NOVEL_VIEWS=2 ++BTS.DATA.data_predicted_depth=false ++SYNTHETIC_GT.DEPTH_PREDICTOR_NAME="UniDepth" ++SYNTHETIC_GT.N_RETRIES_TO_SAMPLE_VALID_NOVEL_VIEW=2 ++SYNTHETIC_GT.MIN_MEAN_OCCLUDED_PIXELS=0.01 ++SYNTHETIC_GT.MIN_MEAN_VALID_PIXELS=0.4

# python train.py -cn exp_bts_synthetic ++NAME="EXP96" ${SHARED} ++BTS.BATCH_SIZE=16 ++SYNTHETIC_GT.NV_CAM_SAMPLER.NUM_NOVEL_VIEWS=12 ++SYNTHETIC_GT.TOP_K_NOVEL_VIEWS_TO_KEEP=4
# python train.py -cn exp_bts_synthetic_rig ++NAME="EXP97" ${SHARED} ++BTS.BATCH_SIZE=16 ++SYNTHETIC_GT.NV_CAM_SAMPLER.NUM_NOVEL_VIEWS=12 ++SYNTHETIC_GT.TOP_K_NOVEL_VIEWS_TO_KEEP=4
# python train.py -cn exp_bts_synthetic_rig ++NAME="EXP98" ${SHARED} ++BTS.BATCH_SIZE=16 ++SYNTHETIC_GT.NV_CAM_SAMPLER.NUM_NOVEL_VIEWS=12 ++SYNTHETIC_GT.TOP_K_NOVEL_VIEWS_TO_KEEP=4 ++BTS.LOSSES.DepthReconstructionLoss.invalid_policy="weight_guided"
# python train.py -cn exp_bts_synthetic_rig ++NAME="EXP99" ${SHARED} ++BTS.BATCH_SIZE=16 ++SYNTHETIC_GT.NV_CAM_SAMPLER.NUM_NOVEL_VIEWS=12 ++SYNTHETIC_GT.TOP_K_NOVEL_VIEWS_TO_KEEP=4 ++SYNTHETIC_GT.INFERENCE_OCCLUSIONS.USE_FLOW=false ++cfg.SYNTHETIC_GT.INFERENCE_OCCLUSIONS.POSTPROCESSING_OPS_DEPTH_GRADS=[['opening',3],['closing',15],['dilation',9]]
# python train.py -cn exp_bts_synthetic_rig ++NAME="EXP100" ${SHARED} ++BTS.BATCH_SIZE=16 ++SYNTHETIC_GT.NV_CAM_SAMPLER.NUM_NOVEL_VIEWS=12 ++SYNTHETIC_GT.TOP_K_NOVEL_VIEWS_TO_KEEP=4 ++SYNTHETIC_GT.INFERENCE_OCCLUSIONS.USE_FLOW=false ++cfg.SYNTHETIC_GT.INFERENCE_OCCLUSIONS.POSTPROCESSING_OPS_DEPTH_GRADS=[['opening',3],['closing',15],['dilation',9]] ++BTS.LOSSES.DepthReconstructionLoss.invalid_policy="weight_guided"

# python train.py -cn exp_bts_synthetic_rig ++NAME="EXP101" ${SHARED}
# python train.py -cn exp_bts_synthetic_rig ++NAME="EXP102" ${SHARED} ++BTS.LOSSES.DepthReconstructionLoss.invalid_policy="weight_guided"


# 1231899
# python train.py -cn exp_bts_synthetic_rig ++NAME="EXP201" ${SHARED}

# 1230484
# python train.py -cn exp_bts_synthetic_rig ++NAME="EXP202_weight_guided" ${SHARED} ++BTS.LOSSES.DepthReconstructionLoss.invalid_policy="weight_guided"

# 1231901 -> 1239118
# python train.py -cn exp_bts_synthetic_rig ++NAME="EXP203_l2" ${SHARED} ++BTS.LOSSES.DepthReconstructionLoss.criterion="l2" ++BTS.LOSSES.DepthReconstructionLoss.lambda_var=0

# 1231902 -> 1239120
# python train.py -cn exp_bts_synthetic_rig ++NAME="EXP204_l2_var" ${SHARED} ++BTS.LOSSES.DepthReconstructionLoss.criterion="l2" ++BTS.LOSSES.DepthReconstructionLoss.lambda_var=1

# 1239309
# python train.py -cn exp_bts_synthetic_rig ++NAME="EXP205_weight_guided_occl" ${SHARED} ++BTS.LOSSES.DepthReconstructionLoss.invalid_policy="weight_guided" SYNTHETIC_GT.ONLY_OCCLUSIONS_VALID=true

# 1244944
# python train.py -cn exp_bts_synthetic_rig ++NAME="EXP206_metric3d_gnll" ${SHARED} SYNTHETIC_GT.DEPTH_PREDICTOR_NAME="Metric3D" SYNTHETIC_GT.ONLY_OCCLUSIONS_VALID=true ++BTS.LOSSES.DepthReconstructionLoss.criterion="l2" ++BTS.LOSSES.DepthReconstructionLoss.lambda_var=0

# 1244943
# python train.py -cn exp_bts_synthetic_rig ++NAME="EXP207_metric3d_mse" ${SHARED} SYNTHETIC_GT.DEPTH_PREDICTOR_NAME="Metric3D" SYNTHETIC_GT.ONLY_OCCLUSIONS_VALID=true 


# new
# NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 
# python train.py -cn exp_bts_synthetic_rig ++NAME="EXP300" ${SHARED} SYNTHETIC_GT.DEPTH_PREDICTOR_NAME="Metric3D" ++BTS.LOSSES.DepthReconstructionLoss.invalid_policy="weight_guided" SYNTHETIC_GT.ONLY_OCCLUSIONS_VALID=true 
# python train.py -cn exp_bts_synthetic_cascade_depth ++NAME="EXP301" ${SHARED} SYNTHETIC_GT.DEPTH_PREDICTOR_NAME="Metric3D" ++BTS.LOSSES.DepthReconstructionLoss.invalid_policy="weight_guided" SYNTHETIC_GT.ONLY_OCCLUSIONS_VALID=true 
# python train.py -cn exp_bts_synthetic_cascade ++NAME="EXP301" ${SHARED} SYNTHETIC_GT.DEPTH_PREDICTOR_NAME="Metric3D"
# python train.py -cn exp_bts_synthetic_cascade_depth ++NAME="EXP302" ${SHARED} SYNTHETIC_GT.DEPTH_PREDICTOR_NAME="Metric3D"
# python train.py -cn exp_bts_synthetic_cascade ++NAME="EXP303" ${SHARED} SYNTHETIC_GT.DEPTH_PREDICTOR_NAME="Metric3D"

# python train.py -cn exp_bts_synthetic_cascade_depth ++NAME="EXP304" ${SHARED} SYNTHETIC_GT.DEPTH_PREDICTOR_NAME="Metric3D" BTS.LOSSES.DepthReconstructionLoss.invalid_policy="weight_guided" SYNTHETIC_GT.ONLY_OCCLUSIONS_VALID=true BTS.CACHE_SYNTHETIC_GT=true
# python train.py -cn exp_bts_synthetic_cascade_depth ++NAME="EXP305" ${SHARED} SYNTHETIC_GT.DEPTH_PREDICTOR_NAME="Metric3D" BTS.LOSSES.DepthReconstructionLoss.invalid_policy="weight_guided" SYNTHETIC_GT.ONLY_OCCLUSIONS_VALID=true BTS.CACHE_SYNTHETIC_GT=true BTS.LOSSES.DensityGridLoss.WEIGHT_OCCL_AND_EMPTY=3 
# python train.py -cn exp_bts_synthetic_cascade_depth ++NAME="EXP307" ${SHARED} SYNTHETIC_GT.DEPTH_PREDICTOR_NAME="Metric3D" BTS.LOSSES.DensityGridLoss.WEIGHT_OCCL_AND_EMPTY=3 BTS.BATCH_SIZE=8 SYNTHETIC_GT.CASCADE.CHOSEN_ANCHOR_TYPE=rig8of8plus8 SYNTHETIC_GT.TOP_K_NOVEL_VIEWS_TO_KEEP=4
# python train.py -cn exp_bts_synthetic_cascade_depth ++NAME="EXP308" ${SHARED} SYNTHETIC_GT.DEPTH_PREDICTOR_NAME="Metric3D" BTS.LOSSES.DensityGridLoss.WEIGHT_OCCL_AND_EMPTY=3 BTS.BATCH_SIZE=8
# python train.py -cn exp_bts_synthetic_cascade_depth NAME="EXP309" ${SHARED} SYNTHETIC_GT.DEPTH_PREDICTOR_NAME="Metric3D" BTS.LOSSES.DensityGridLoss.WEIGHT_OCCL_AND_EMPTY=3 
# python train.py -cn exp_bts_synthetic_cascade_depth NAME="EXP311" ${SHARED} SYNTHETIC_GT.DEPTH_PREDICTOR_NAME="Metric3DLocal" BTS.LOSSES.DensityGridLoss.WEIGHT_OCCL_AND_EMPTY=3
# python train.py -cn exp_bts_synthetic_cascade_depth NAME="EXP315" ${SHARED} SYNTHETIC_GT.DEPTH_PREDICTOR_NAME="Metric3DLocal" BTS.LOSSES.DensityGridLoss.WEIGHT_OCCL_AND_EMPTY=3 BTS.DATA.return_stereo=true


# python train.py -cn exp_bts_synthetic_cascade_depth NAME="EXP320" ${SHARED} BTS.LOSSES.DensityGridLoss.WEIGHT_OCCL_AND_EMPTY=3
# python train.py -cn exp_bts_synthetic_cascade_depth NAME="EXP321" ${SHARED} SYNTHETIC_GT.DEPTH_PREDICTOR_NAME="Metric3D" BTS.LOSSES.DensityGridLoss.WEIGHT_OCCL_AND_EMPTY=3 BTS.MODEL_CONF.OUTPUT_UNCERTAINTY=true BTS.MODEL_CONF.encoder.d_out=128

# python train.py -cn exp_bts_synthetic_cascade_depth NAME="EXP322" ${SHARED} BTS.LOSSES.DensityGridLoss.WEIGHT_OCCL_AND_EMPTY=3
# python train.py -cn exp_bts_synthetic_cascade_depth_uncert NAME="EXP323" ${SHARED} BTS.LOSSES.DensityGridLoss.WEIGHT_OCCL_AND_EMPTY=3

# python train.py -cn exp_bts_synthetic_cascade_depth NAME="EXP324" ${SHARED} BTS.LOSSES.DensityGridLoss.WEIGHT_OCCL_AND_EMPTY=3
# python train.py -cn exp_bts_synthetic_cascade_depth_uncert NAME="EXP325" ${SHARED} BTS.LOSSES.DensityGridLoss.WEIGHT_OCCL_AND_EMPTY=3

# TODO
# python train.py -cn exp_bts_synthetic_cascade_depth_only NAME="EXP326" ${SHARED} 

# python train.py -cn exp_bts_synthetic_cascade_depth NAME="EXP327" ${SHARED}
# python train.py -cn exp_bts_synthetic_cascade_depth_occ NAME="EXP328" ${SHARED}
# python train.py -cn exp_bts_synthetic_cascade_depth NAME="EXP329" ${SHARED} BTS.LOSSES.DensityGridLoss.ALIGN=true BTS.LOSSES.DensityGridLoss.ALIGN_SHIFT=true BTS.NUM_EPOCHS=50
# python train.py -cn exp_bts_synthetic_cascade_depth NAME="EXP330" ${SHARED} BTS.LOSSES.DensityGridLoss.ALIGN=true BTS.LOSSES.DensityGridLoss.ALIGN_SHIFT=false BTS.NUM_EPOCHS=40
# python train.py -cn exp_bts_synthetic_cascade_depth NAME="EXP331" ${SHARED} BTS.LOSSES.DensityGridLoss.ALIGN=true BTS.LOSSES.DensityGridLoss.ALIGN_SHIFT=false BTS.LOSSES.DensityGridLoss.WEIGHT=10


# python train.py -cn exp_bts_synthetic_cascade_depth_best NAME="EXP332" ${SHARED}
# python train.py -cn exp_bts_synthetic_cascade_depth_best NAME="EXP333" ${SHARED} BTS.LOSSES.DensityGridLoss.ALIGN=false
# python train.py -cn exp_bts_synthetic_cascade_depth_best NAME="EXP334" ${SHARED} BTS.MODEL_CONF.OUTPUT_UNCERTAINTY=false BTS.MODEL_CONF.encoder.d_out=64


# python train.py -cn exp_recon_full NAME="EXP334" ${SHARED}
# python train.py -cn exp_recon_full NAME="EXP335" ${SHARED} BTS.MODEL_CONF.OUTPUT_UNCERTAINTY=false BTS.MODEL_CONF.encoder.d_out=64
# python train.py -cn exp_recon_no_density_loss NAME="EXP336" ${SHARED}
# python train.py -cn exp_recon_no_depth_loss NAME="EXP337" ${SHARED}
# python train.py -cn exp_recon_full NAME="EXP338" ${SHARED} BTS.LOSSES.DensityGridLoss.WEIGHT_OCCL_AND_EMPTY=1
# python train.py -cn exp_recon_full NAME="EXP339" ${SHARED}
# python train.py -cn exp_recon_full NAME="EXP340" ${SHARED} BTS.LOSSES.DepthReconstructionLoss.criterion="l2"
# python train.py -cn exp_recon_full NAME="EXP341" ${SHARED}
# python train.py -cn exp_recon_full NAME="EXP342" ${SHARED} BTS.DATA.return_stereo=true

# python train.py -cn waymo NAME="EXP400" ${SHARED} SYNTHETIC_GT.DEPTH_PREDICTOR_NAME="Metric3D" BTS.LOSSES.DensityGridLoss.WEIGHT_OCCL_AND_EMPTY=3

# SHARED="++NPROC_PER_NODE=4 ++BACKEND="gloo" ++BTS.BATCH_SIZE=16 ++BTS.DATA.MAX_TRAIN_DATASET_LEN=10000"
# python train.py -cn exp_bts_synthetic_cascade ++NAME="EXPc2" ${SHARED} ++SYNTHETIC_GT.DEPTH_PREDICTOR_NAME="Metric3D"


# 1364368 -> 1364870 -> 1365614
# python train.py -cn exp_recon_full NAME="EXP500" ${SHARED} BTS.BATCH_SIZE_MULTIPLE_AFTER_CLEANUP=3 SYNTHETIC_GT.CASCADE.CHOSEN_ANCHOR_TYPE=empty BTS.DATA.return_stereo=true BTS.DATA.return_fisheye=true BTS.DATA.frame_count=2 BTS.DATA.fisheye_rotation=[0,-15] BTS.DATA.fisheye_offset=10 BTS.DATA.dilation=5

# 1364371 -> 1364961 -> 1365615
# python train.py -cn waymo NAME="EXP501" ${SHARED} BTS.BATCH_SIZE_MULTIPLE_AFTER_CLEANUP=3 SYNTHETIC_GT.CASCADE.CHOSEN_ANCHOR_TYPE=empty BTS.DATA.return_45=true BTS.DATA.offset_45=5 BTS.DATA.frame_count=2 BTS.DATA.dilation=5

# python train.py -cn waymo NAME="EXP501" ${SHARED} BTS.BATCH_SIZE_MULTIPLE_AFTER_CLEANUP=4 SYNTHETIC_GT.CASCADE.CHOSEN_ANCHOR_TYPE=empty BTS.DATA.return_45=true BTS.DATA.offset_45=5 BTS.DATA.frame_count=2 BTS.DATA.dilation=1
# python train.py -cn exp_recon_full NAME="EXP500" ${SHARED} BTS.BATCH_SIZE_MULTIPLE_AFTER_CLEANUP=4 SYNTHETIC_GT.CASCADE.CHOSEN_ANCHOR_TYPE=empty BTS.DATA.return_stereo=true BTS.DATA.return_fisheye=true BTS.DATA.frame_count=2 BTS.DATA.fisheye_rotation=[0,-15] BTS.DATA.fisheye_offset=10 BTS.DATA.dilation=1

# --- END BTS 10k imgs ----
# --- BEGIN BTS 100k imgs ----

# SHARED="++NPROC_PER_NODE=4 ++BACKEND="nccl" ++BTS.BATCH_SIZE=12 ++BTS.DATA.USE_TRAIN_FOR_TEST=true ++BTS.MASTER_PORT=12876 ++BTS.DATA.data_predicted_depth=false ++SYNTHETIC_GT.DEPTH_PREDICTOR_NAME="UniDepth""

# python train.py -cn exp_bts_synthetic ++NAME="EXP100" ${SHARED} ++BTS.BATCH_SIZE=32 ++SYNTHETIC_GT.NV_CAM_SAMPLER.NUM_NOVEL_VIEWS=2

# --- END BTS 100k imgs ----