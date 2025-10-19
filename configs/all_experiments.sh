# This file contains a lot of the experiments we tried out.
# You can use it as a reference for how to set different configurations.

# ----------------------------------------------------------------------------
# EVAL CASCADE
# ----------------------------------------------------------------------------

SHARED="AMP.ENABLED=true BTS.LOG_EVERY_ITERS=50 BTS.VISUALIZE_EVERY=10 BTS.BATCH_SIZE=1 SEED=42"

# Eval pseudo vol from GT multi-view data
SHARED_KITTI360="$SHARED CONTROLNET.DATA.image_size=[192,640]"
# python eval.py -cn eval_controlnet_lidar_occ ${SHARED_KITTI360} UNIQUE_EVAL_ID=gt_multi_view_data_kitti360_dilation5 SYNTHETIC_GT.CASCADE.CHOSEN_ANCHOR_TYPE=empty CONTROLNET.DATA.return_stereo=true CONTROLNET.DATA.return_fisheye=true CONTROLNET.DATA.frame_count=2 CONTROLNET.DATA.fisheye_rotation=[0,-15] CONTROLNET.DATA.fisheye_offset=10 CONTROLNET.DATA.dilation=5
# python eval.py -cn eval_controlnet_lidar_occ ${SHARED_KITTI360} UNIQUE_EVAL_ID=gt_multi_view_data_kitti360_dilation1 SYNTHETIC_GT.CASCADE.CHOSEN_ANCHOR_TYPE=empty CONTROLNET.DATA.return_stereo=true CONTROLNET.DATA.return_fisheye=true CONTROLNET.DATA.frame_count=2 CONTROLNET.DATA.fisheye_rotation=[0,-15] CONTROLNET.DATA.fisheye_offset=10 CONTROLNET.DATA.dilation=1

# Table 1:
# python eval.py -cn eval_controlnet_lidar_occ ${SHARED_KITTI360} UNIQUE_EVAL_ID=render_refine_repeat_kitti360 SYNTHETIC_GT.CASCADE.CHOSEN_ANCHOR_TYPE=rig8of8
# OG:
# /home/stud/wph/storage/user/BTS/out/controlnet/exp_recon_full/eval_controlnet_lidar_occ/full_cascade/config.yaml

# Waymo
# SHARED_WAYMO="$SHARED CONTROLNET.DATA.image_size=[320,480] EVAL_OCCUPANCY.GT_AGGREGATE_TIMESTEPS=20"
SHARED_WAYMO="$SHARED CONTROLNET.DATA.image_size=[320,480]"
# python eval.py -cn eval_controlnet_lidar_occ_waymo ${SHARED_WAYMO} UNIQUE_EVAL_ID=gt_multi_view_data_waymo_dilation5 SYNTHETIC_GT.CASCADE.CHOSEN_ANCHOR_TYPE=empty CONTROLNET.DATA.return_45=true CONTROLNET.DATA.offset_45=5 CONTROLNET.DATA.frame_count=2 CONTROLNET.DATA.dilation=5
# python eval.py -cn eval_controlnet_lidar_occ_waymo ${SHARED_WAYMO} UNIQUE_EVAL_ID=gt_multi_view_data_waymo_dilation1 SYNTHETIC_GT.CASCADE.CHOSEN_ANCHOR_TYPE=empty CONTROLNET.DATA.return_45=true CONTROLNET.DATA.offset_45=5 CONTROLNET.DATA.frame_count=2 CONTROLNET.DATA.dilation=1
# python eval.py -cn eval_controlnet_lidar_occ_waymo ${SHARED_WAYMO} UNIQUE_EVAL_ID=gt_multi_view_data_waymo_dilation1_rig8of8 SYNTHETIC_GT.CASCADE.CHOSEN_ANCHOR_TYPE=rig8of8_waymo CONTROLNET.DATA.return_45=true CONTROLNET.DATA.offset_45=0 CONTROLNET.DATA.frame_count=2 CONTROLNET.DATA.dilation=1

# Table 1:
# python eval.py -cn eval_controlnet_lidar_occ_waymo ${SHARED_WAYMO} UNIQUE_EVAL_ID=render_refine_repeat_waymo SYNTHETIC_GT.CASCADE.CHOSEN_ANCHOR_TYPE=rig8of8_waymo
# OG:
# /home/stud/wph/storage/user/BTS/out/controlnet/waymo/eval_controlnet_lidar_occ/full_cascade/config.yaml

# ----------------------------------------------------------------------------
# EVAL CONTROLNET OCCLUSIONS
# ----------------------------------------------------------------------------

SHARED="AMP.ENABLED=true BTS.LOG_EVERY_ITERS=50 BTS.VISUALIZE_EVERY=10 BTS.BATCH_SIZE=1 SEED=42 CONTROLNET.DATA.image_size=[192,640]"
# SHARED_LIDAR=""

# python eval.py -cn eval_controlnet_novel_view ${SHARED} JOB_TYPE=eval_controlnet_novel_view UNIQUE_EVAL_ID=occlusions_grads SYNTHETIC_GT.INFERENCE_OCCLUSIONS.USE_DEPTH_GRADS=true SYNTHETIC_GT.INFERENCE_OCCLUSIONS.USE_FLOW=false
# python eval.py -cn eval_controlnet_lidar_occ ${SHARED} UNIQUE_EVAL_ID=occlusions_grads SYNTHETIC_GT.INFERENCE_OCCLUSIONS.USE_DEPTH_GRADS=true SYNTHETIC_GT.INFERENCE_OCCLUSIONS.USE_FLOW=false

# python eval.py -cn eval_controlnet_novel_view ${SHARED} JOB_TYPE=eval_controlnet_novel_view UNIQUE_EVAL_ID=occlusions_flow SYNTHETIC_GT.INFERENCE_OCCLUSIONS.USE_DEPTH_GRADS=false SYNTHETIC_GT.INFERENCE_OCCLUSIONS.USE_FLOW=true
# python eval.py -cn eval_controlnet_lidar_occ ${SHARED} UNIQUE_EVAL_ID=occlusions_flow SYNTHETIC_GT.INFERENCE_OCCLUSIONS.USE_DEPTH_GRADS=false SYNTHETIC_GT.INFERENCE_OCCLUSIONS.USE_FLOW=true

# python eval.py -cn eval_controlnet_novel_view ${SHARED} JOB_TYPE=eval_controlnet_novel_view UNIQUE_EVAL_ID=occlusions_grads_flow SYNTHETIC_GT.INFERENCE_OCCLUSIONS.USE_DEPTH_GRADS=true SYNTHETIC_GT.INFERENCE_OCCLUSIONS.USE_FLOW=true
# python eval.py -cn eval_controlnet_lidar_occ ${SHARED} UNIQUE_EVAL_ID=occlusions_grads_flow SYNTHETIC_GT.INFERENCE_OCCLUSIONS.USE_DEPTH_GRADS=true SYNTHETIC_GT.INFERENCE_OCCLUSIONS.USE_FLOW=true

# ----------------------------------------------------------------------------
# EVAL CONTROLNET PSEUDO
# ----------------------------------------------------------------------------

SHARED="AMP.ENABLED=false BTS.LOG_EVERY_ITERS=50 BTS.VISUALIZE_EVERY=10 BTS.BATCH_SIZE=1 SEED=42 CONTROLNET.DATA.image_size=[192,640] CONTROLNET.DATA.return_stereo=true"


pseudoVolArgs=(
    "pseudo_mean SYNTHETIC_GT.PSEUDO_VOLUME.COLOR_SAMPLING_MODE=mean SYNTHETIC_GT.PSEUDO_VOLUME.SET_ONLY_OCCLUSIONS_EMPTY=true"
    "pseudo_mean_valid SYNTHETIC_GT.PSEUDO_VOLUME.COLOR_SAMPLING_MODE=mean_valid SYNTHETIC_GT.PSEUDO_VOLUME.SET_ONLY_OCCLUSIONS_EMPTY=true"
    # "pseudo_mean_valid_no_occl_mask SYNTHETIC_GT.PSEUDO_VOLUME.COLOR_SAMPLING_MODE=mean_valid SYNTHETIC_GT.PSEUDO_VOLUME.SET_ONLY_OCCLUSIONS_EMPTY=false"
    "pseudo_mean_valid_surface_05 SYNTHETIC_GT.PSEUDO_VOLUME.COLOR_SAMPLING_MODE=mean_valid_surface SYNTHETIC_GT.PSEUDO_VOLUME.SURFACE_THRESH=0.5 SYNTHETIC_GT.PSEUDO_VOLUME.SET_ONLY_OCCLUSIONS_EMPTY=true"
    "pseudo_mean_valid_surface_05_no_occl_mask SYNTHETIC_GT.PSEUDO_VOLUME.COLOR_SAMPLING_MODE=mean_valid_surface SYNTHETIC_GT.PSEUDO_VOLUME.SURFACE_THRESH=0.5 SYNTHETIC_GT.PSEUDO_VOLUME.SET_ONLY_OCCLUSIONS_EMPTY=false"
    # "pseudo_mean_valid_surface_1 SYNTHETIC_GT.PSEUDO_VOLUME.COLOR_SAMPLING_MODE=mean_valid_surface SYNTHETIC_GT.PSEUDO_VOLUME.SURFACE_THRESH=1.0 SYNTHETIC_GT.PSEUDO_VOLUME.SET_ONLY_OCCLUSIONS_EMPTY=true"
    # "pseudo_mean_valid_surface_2 SYNTHETIC_GT.PSEUDO_VOLUME.COLOR_SAMPLING_MODE=mean_valid_surface SYNTHETIC_GT.PSEUDO_VOLUME.SURFACE_THRESH=2.0 SYNTHETIC_GT.PSEUDO_VOLUME.SET_ONLY_OCCLUSIONS_EMPTY=true"
)

for arg in "${pseudoVolArgs[@]}"; do
    echo "Running eval: $arg"
    # python eval.py -cn eval_pseudo_vol $SHARED JOB_TYPE=eval_controlnet_novel_view UNIQUE_EVAL_ID=$arg
    python eval.py -cn eval_pseudo_vol $SHARED JOB_TYPE=eval_controlnet_lidar_occ EVAL_OCCUPANCY.MODE=cascade UNIQUE_EVAL_ID=$arg
done


# ----------------------------------------------------------------------------
# EVAL CONTROLNET SBATCH
# ----------------------------------------------------------------------------

export NCCL_DEBUG=INFO
pwd; hostname; date
nvidia-smi
# Launching in the right conda environment automatically uses it in the job
echo $CONDA_DEFAULT_ENV

# Need to set the UNIQUE_EVAL_ID for every individual exp.
SHARED="AMP.ENABLED=false BTS.LOG_EVERY_ITERS=50 BTS.VISUALIZE_EVERY=10 BTS.BATCH_SIZE=1 SEED=42"

################ CONTROLNET EVAL KITTI-360 ################

controlnetExps=(
    "controlnet_rgb"
    "controlnet_rgb_unclip"
    # "controlnet_rgb_unclip_small"
    # "controlnet_rgb_unclip_noise"
    # "controlnet_rgb_unclip_img_embed"
    # "controlnet_rgb_SD1-5"
    # "controlnet_rgb_unmasked"
    # "controlnet_rgb_prompt"
    # "controlnet_rgb_cnoise"
    # "controlnet_rgb_invclosed"
    # "controlnet_rgbd"
    # "controlnet_rgbm"
    # "controlnet_rgbm_unmasked"
    # "controlnet_rgbd_inverse"
    # "controlnet_rgbm_cnoise"

    "controlnet_full"
    # "controlnet_rgbdm_inverse"
    "controlnet_rgbdm_unmasked_inverse"

    # "controlnet_rgb_512x768"
    # "controlnet_full_512x768"
)

#  1225527
# for exp in "${controlnetExps[@]}"; do
#     echo "Running eval: $exp"
#     # input view
#     python eval.py -cn ${exp} ${SHARED} JOB_TYPE="eval_controlnet_input_view" UNIQUE_EVAL_ID="eval_run_0" SYNTHETIC_GT.NUM_SYNTHETIC_VERSIONS=4
#     # novel view
#     python eval.py -cn ${exp} ${SHARED} JOB_TYPE="eval_controlnet_novel_view" UNIQUE_EVAL_ID="eval_ver4" SYNTHETIC_GT.NUM_SYNTHETIC_VERSIONS=4
#     python eval.py -cn ${exp} ${SHARED} JOB_TYPE="eval_controlnet_novel_view" UNIQUE_EVAL_ID="eval_nv4" SYNTHETIC_GT.NUM_SYNTHETIC_VERSIONS=1 SYNTHETIC_GT.NV_CAM_SAMPLER.NUM_NOVEL_VIEWS=4
# done

# SHARED="AMP.ENABLED=false BTS.LOG_EVERY_ITERS=50 BTS.VISUALIZE_EVERY=10 BTS.BATCH_SIZE=1 SEED=42"
# python eval.py -cn controlnet_full ${SHARED} JOB_TYPE="eval_controlnet_input_view" UNIQUE_EVAL_ID="eval_run_0_kitti360" SYNTHETIC_GT.NUM_SYNTHETIC_VERSIONS=4
# python eval.py -cn controlnet_full ${SHARED} JOB_TYPE="eval_controlnet_novel_view" UNIQUE_EVAL_ID="eval_ver4_kitti360" SYNTHETIC_GT.NUM_SYNTHETIC_VERSIONS=4
# python eval.py -cn controlnet_full ${SHARED} JOB_TYPE="eval_controlnet_novel_view" UNIQUE_EVAL_ID="eval_nv4_kitti360" SYNTHETIC_GT.NUM_SYNTHETIC_VERSIONS=1 SYNTHETIC_GT.NV_CAM_SAMPLER.NUM_NOVEL_VIEWS=4

# python eval.py -cn controlnet_full_512x768 ${SHARED} JOB_TYPE="eval_controlnet_input_view" UNIQUE_EVAL_ID="eval_run_0_kitti360" SYNTHETIC_GT.NUM_SYNTHETIC_VERSIONS=4
# python eval.py -cn controlnet_full_512x768 ${SHARED} JOB_TYPE="eval_controlnet_novel_view" UNIQUE_EVAL_ID="eval_ver4_kitti360" SYNTHETIC_GT.NUM_SYNTHETIC_VERSIONS=4
# python eval.py -cn controlnet_full_512x768 ${SHARED} JOB_TYPE="eval_controlnet_novel_view" UNIQUE_EVAL_ID="eval_nv4_kitti360" SYNTHETIC_GT.NUM_SYNTHETIC_VERSIONS=1 SYNTHETIC_GT.NV_CAM_SAMPLER.NUM_NOVEL_VIEWS=4

# SHARED="AMP.ENABLED=false BTS.LOG_EVERY_ITERS=50 BTS.VISUALIZE_EVERY=10 BTS.BATCH_SIZE=1 SEED=42"
# SHARED_WAYMO="CONTROLNET.DATA.MAX_TEST_DATASET_LEN=500"

# python eval.py -cn controlnet_full_512x768 ${SHARED} ${SHARED_WAYMO} JOB_TYPE="eval_controlnet_input_view" UNIQUE_EVAL_ID="eval_run_0_waymo" SYNTHETIC_GT.NUM_SYNTHETIC_VERSIONS=4
# python eval.py -cn controlnet_full_512x768 ${SHARED} ${SHARED_WAYMO} JOB_TYPE="eval_controlnet_novel_view" UNIQUE_EVAL_ID="eval_ver4_waymo" SYNTHETIC_GT.NUM_SYNTHETIC_VERSIONS=4
# python eval.py -cn controlnet_full_512x768 ${SHARED} ${SHARED_WAYMO} JOB_TYPE="eval_controlnet_novel_view" UNIQUE_EVAL_ID="eval_nv4_waymo" SYNTHETIC_GT.NUM_SYNTHETIC_VERSIONS=1 SYNTHETIC_GT.NV_CAM_SAMPLER.NUM_NOVEL_VIEWS=4

# python eval.py -cn controlnet_full_512x768_waymo ${SHARED} ${SHARED_WAYMO} JOB_TYPE="eval_controlnet_input_view" UNIQUE_EVAL_ID="eval_run_0" SYNTHETIC_GT.NUM_SYNTHETIC_VERSIONS=4
# python eval.py -cn controlnet_full_512x768_waymo ${SHARED} ${SHARED_WAYMO} JOB_TYPE="eval_controlnet_novel_view" UNIQUE_EVAL_ID="eval_ver4" SYNTHETIC_GT.NUM_SYNTHETIC_VERSIONS=4
# python eval.py -cn controlnet_full_512x768_waymo ${SHARED} ${SHARED_WAYMO} JOB_TYPE="eval_controlnet_novel_view" UNIQUE_EVAL_ID="eval_nv4" SYNTHETIC_GT.NUM_SYNTHETIC_VERSIONS=1 SYNTHETIC_GT.NV_CAM_SAMPLER.NUM_NOVEL_VIEWS=4
# --- CO3D ---


# 1230649 1230656
controlnetCo3ds=(
    # "controlnet_co3d_rgb"

    "controlnet_co3d_full CONTROLNET.DATA.CATEGORY_NAME=hydrant"
    "controlnet_co3d_full_cake CONTROLNET.DATA.CATEGORY_NAME=cake"
    "controlnet_co3d_full_motorcycle CONTROLNET.DATA.CATEGORY_NAME=motorcycle"
    "controlnet_co3d_full_sandwich CONTROLNET.DATA.CATEGORY_NAME=sandwich"
    "controlnet_co3d_full_backpack CONTROLNET.DATA.CATEGORY_NAME=backpack"
    "controlnet_co3d_full_bench CONTROLNET.DATA.CATEGORY_NAME=bench"
)
# exp=controlnet_co3d_rgb
exp=controlnet_co3d_full

# for co3dName in "${controlnetCo3ds[@]}"; do
#     echo "Running eval: $exp $co3dName"
#     # input view
#     python eval.py -cn ${exp} ${SHARED} JOB_TYPE="eval_controlnet_input_view" NAME=$co3dName UNIQUE_EVAL_ID="eval_run_0" SYNTHETIC_GT.NUM_SYNTHETIC_VERSIONS=4 CONTROLNET.DATA.MAX_TEST_DATASET_LEN=1000
#     # novel view
#     python eval.py -cn ${exp} ${SHARED} JOB_TYPE="eval_controlnet_novel_view" NAME=$co3dName UNIQUE_EVAL_ID="eval_ver4" SYNTHETIC_GT.NUM_SYNTHETIC_VERSIONS=4 CONTROLNET.DATA.MAX_TEST_DATASET_LEN=1000
#     python eval.py -cn ${exp} ${SHARED} JOB_TYPE="eval_controlnet_novel_view" NAME=$co3dName UNIQUE_EVAL_ID="eval_nv4" SYNTHETIC_GT.NUM_SYNTHETIC_VERSIONS=1 SYNTHETIC_GT.NV_CAM_SAMPLER.NUM_NOVEL_VIEWS=4 CONTROLNET.DATA.MAX_TEST_DATASET_LEN=1000
# done

# ------------------------------

################ OCCLUSION DET EVAL KITTI-360 ################

occlusions=(
    "occlusions_grads SYNTHETIC_GT.INFERENCE_OCCLUSIONS.USE_DEPTH_GRADS=true SYNTHETIC_GT.INFERENCE_OCCLUSIONS.USE_FLOW=false"
    "occlusions_flow SYNTHETIC_GT.INFERENCE_OCCLUSIONS.USE_DEPTH_GRADS=false SYNTHETIC_GT.INFERENCE_OCCLUSIONS.USE_FLOW=true"
    "occlusions_grads_flow SYNTHETIC_GT.INFERENCE_OCCLUSIONS.USE_DEPTH_GRADS=true SYNTHETIC_GT.INFERENCE_OCCLUSIONS.USE_FLOW=true"
)

# for occlusion in "${occlusions[@]}"; do
#     echo "Running: $occlusion"
#     python eval.py -cn controlnet_rgb ${SHARED} JOB_TYPE=eval_controlnet_novel_view UNIQUE_EVAL_ID=$occlusion
#     python eval.py -cn controlnet_rgb ${SHARED} JOB_TYPE=eval_controlnet_lidar_occ UNIQUE_EVAL_ID=$occlusion
# done


################ DEPTH ALIGNMENT DET EVAL KITTI-360 ################

depthalignments=(
    "alignment_none SYNTHETIC_GT.ALIGN_DEPTH_POLICY=none"
    "alignment_direct_median SYNTHETIC_GT.ALIGN_DEPTH_POLICY=direct SYNTHETIC_GT.ALIGN_DEPTH_MODE=median"
    "alignment_direct_mean SYNTHETIC_GT.ALIGN_DEPTH_POLICY=direct SYNTHETIC_GT.ALIGN_DEPTH_MODE=mean"
    "alignment_direct_lstsq SYNTHETIC_GT.ALIGN_DEPTH_POLICY=direct SYNTHETIC_GT.ALIGN_DEPTH_MODE=lstsq SYNTHETIC_GT.ALIGN_DEPTH_SCALE_ONLY=false"
    "alignment_inverse_median SYNTHETIC_GT.ALIGN_DEPTH_POLICY=inverse SYNTHETIC_GT.ALIGN_DEPTH_MODE=median"
    "alignment_inverse_mean SYNTHETIC_GT.ALIGN_DEPTH_POLICY=inverse SYNTHETIC_GT.ALIGN_DEPTH_MODE=mean"
    "alignment_inverse_lstsq SYNTHETIC_GT.ALIGN_DEPTH_POLICY=inverse SYNTHETIC_GT.ALIGN_DEPTH_MODE=lstsq SYNTHETIC_GT.ALIGN_DEPTH_SCALE_ONLY=false"
)

# for alignment in "${depthalignments[@]}"; do
#     cmd="eval.py -cn controlnet_rgb ${SHARED} JOB_TYPE="eval_controlnet_novel_view" UNIQUE_EVAL_ID=$alignment"
#     echo "Running cmd: $cmd"
#     python $cmd
# done

################

################ SCHEDULER EVAL KITTI-360 ################

schedulers=(
    "UniPCMultistepScheduler"
    "LMSDiscreteScheduler"
    "HeunDiscreteScheduler"  
    "DPMSolverMultistepScheduler"
    "DPMSolverSinglestepScheduler"
    "DDIMScheduler"
)

steps=(30 20 15 10 7 5 3 1)

# for scheduler in "${schedulers[@]}"; do
#     for step in "${steps[@]}"; do
#         echo "Running with scheduler: $scheduler and steps: $step"
#         python eval.py -cn controlnet_rgb ${SHARED} JOB_TYPE="eval_controlnet_input_view" UNIQUE_EVAL_ID="sched_${scheduler}_${step}" CONTROLNET.EVAL.SCHEDULER_NAME="$scheduler" CONTROLNET.EVAL.NUM_INFERENCE_STEPS=$step
#     done
# done

################ GUIDANCE SCALE EVAL KITTI-360 ################

# "1" is no guidance
guidancescales=(1 2 3 4 5 6 7 8 9 10)

# for guidance in "${guidancescales[@]}"; do
#     echo "Running guidance: $guidance"
#     python eval.py -cn controlnet_rgb_unclip $SHARED JOB_TYPE="eval_controlnet_input_view" UNIQUE_EVAL_ID=guidance_$guidance CONTROLNET.EVAL.GUIDANCE_SCALE=$guidance
#     python eval.py -cn controlnet_rgb_unclip $SHARED JOB_TYPE="eval_controlnet_novel_view" UNIQUE_EVAL_ID=guidance_$guidance CONTROLNET.EVAL.GUIDANCE_SCALE=$guidance
# done

################ POSE SAMPLER EVAL ################

# --- SHIFT SAMPLER EVAL ----
xlims=(
    "-5,5"
    "-4,4"
    "-3,3"
    "-2,2"
    "-1,1"
    "0,0"
)

# for xlim in "${xlims[@]}"; do
#     xlim_eval_id=$(echo $xlim | sed 's/,/to/')
#     echo "Running eval: $xlim $xlim_eval_id"
#     python eval.py -cn sampler_eval_shift ${SHARED} JOB_TYPE="eval_controlnet_input_view" UNIQUE_EVAL_ID=shift_$xlim_eval_id SYNTHETIC_GT.NV_CAM_SAMPLER.X_LIMS="[$xlim]"
#     python eval.py -cn sampler_eval_shift ${SHARED} JOB_TYPE="eval_controlnet_novel_view" UNIQUE_EVAL_ID=shift_$xlim_eval_id SYNTHETIC_GT.NV_CAM_SAMPLER.X_LIMS="[$xlim]"
# done

# --- Orbit SAMPLER EVAL ----
ylims=(
    "-25,25"
    "-20,20"
    "-15,15"
    "-10,10"
    "-5,5"
    "-0,0"
)
# for ylim in "${ylims[@]}"; do
#     ylim_eval_id=$(echo $ylim | sed 's/,/to/')
#     echo "Running eval: $ylim $ylim_eval_id"
#     python eval.py -cn sampler_eval_orbit ${SHARED} JOB_TYPE="eval_controlnet_input_view" UNIQUE_EVAL_ID=orbit_$ylim_eval_id SYNTHETIC_GT.NV_CAM_SAMPLER.Y_LIMS="[$ylim]"
#     python eval.py -cn sampler_eval_orbit ${SHARED} JOB_TYPE="eval_controlnet_novel_view" UNIQUE_EVAL_ID=orbit_$ylim_eval_id SYNTHETIC_GT.NV_CAM_SAMPLER.Y_LIMS="[$ylim]"
# done

# ------------------------------

################ PSEUDO VOL EVAL ################


pseudoVolArgs=(
    "pseudo_mean SYNTHETIC_GT.PSEUDO_VOLUME.COLOR_SAMPLING_MODE=mean SYNTHETIC_GT.PSEUDO_VOLUME.SET_ONLY_OCCLUSIONS_EMPTY=true"
    "pseudo_mean_valid SYNTHETIC_GT.PSEUDO_VOLUME.COLOR_SAMPLING_MODE=mean_valid SYNTHETIC_GT.PSEUDO_VOLUME.SET_ONLY_OCCLUSIONS_EMPTY=true"
    "pseudo_mean_valid_no_occl_mask SYNTHETIC_GT.PSEUDO_VOLUME.COLOR_SAMPLING_MODE=mean_valid SYNTHETIC_GT.PSEUDO_VOLUME.SET_ONLY_OCCLUSIONS_EMPTY=false"
    "pseudo_mean_valid_surface_1 SYNTHETIC_GT.PSEUDO_VOLUME.COLOR_SAMPLING_MODE=mean_valid_surface SYNTHETIC_GT.PSEUDO_VOLUME.SURFACE_THRESH=1.0 SYNTHETIC_GT.PSEUDO_VOLUME.SET_ONLY_OCCLUSIONS_EMPTY=true"
    "pseudo_mean_valid_surface_2 SYNTHETIC_GT.PSEUDO_VOLUME.COLOR_SAMPLING_MODE=mean_valid_surface SYNTHETIC_GT.PSEUDO_VOLUME.SURFACE_THRESH=2.0 SYNTHETIC_GT.PSEUDO_VOLUME.SET_ONLY_OCCLUSIONS_EMPTY=true"
)

# 1224551 1225401
# for arg in "${pseudoVolArgs[@]}"; do
#     echo "Running eval: $arg"
#     python eval.py -cn sampler_eval_rig4 $SHARED JOB_TYPE=eval_controlnet_novel_view SYNTHETIC_GT.NV_CAM_SAMPLER.NUM_NOVEL_VIEWS=4 SYNTHETIC_GT.NV_CAM_SAMPLER.MAKE_PROJS_POLICY=center_crop UNIQUE_EVAL_ID=$arg
#     python eval.py -cn sampler_eval_rig4 $SHARED JOB_TYPE=eval_controlnet_lidar_occ EVAL_OCCUPANCY.MODE=cascade SYNTHETIC_GT.NV_CAM_SAMPLER.NUM_NOVEL_VIEWS=4 SYNTHETIC_GT.INPUT_CROP_POLICY="center+left+right" SYNTHETIC_GT.NV_CAM_SAMPLER.MAKE_PROJS_POLICY=center_crop UNIQUE_EVAL_ID=$arg
# done

# 1225443
# for arg in "${pseudoVolArgs[@]}"; do
#     # The same again but with 8 views and rig8
#     echo "Running eval: $arg"
#     python eval.py -cn sampler_eval_rig8 $SHARED JOB_TYPE=eval_controlnet_novel_view SYNTHETIC_GT.NV_CAM_SAMPLER.NUM_NOVEL_VIEWS=8 SYNTHETIC_GT.NV_CAM_SAMPLER.MAKE_PROJS_POLICY=center_crop UNIQUE_EVAL_ID=rig8_$arg
#     python eval.py -cn sampler_eval_rig8 $SHARED JOB_TYPE=eval_controlnet_lidar_occ EVAL_OCCUPANCY.MODE=cascade SYNTHETIC_GT.NV_CAM_SAMPLER.NUM_NOVEL_VIEWS=8 SYNTHETIC_GT.INPUT_CROP_POLICY="center+left+right" SYNTHETIC_GT.NV_CAM_SAMPLER.MAKE_PROJS_POLICY=center_crop UNIQUE_EVAL_ID=rig8_$arg
# done


# --- EXPLORATION SAMPLER EVAL ----

# 1224475 6 7
# 1224997 . 1224998
numprops=(
    # 2 4
    # 8 12
    16 24
)
numnvs=(1 2 4 8)
initstrat=stratified
zlimsfar=10

# for numnv in "${numnvs[@]}"; do
#     for numprop in "${numprops[@]}"; do
#         # Skip the iteration if the number of novel views is larger than the number of proposals
#         if [ "$numnv" -gt "$numprop" ]; then
#             continue
#         fi
#         echo "Running eval: $numnv $numprop"
#         args="SYNTHETIC_GT.NV_CAM_SAMPLER.NUM_NOVEL_VIEWS=$numnv SYNTHETIC_GT.NV_CAM_SAMPLER.NUM_PROPOSALS=$numprop SYNTHETIC_GT.NV_CAM_SAMPLER.INIT_POLICY=$initstrat SYNTHETIC_GT.NV_CAM_SAMPLER.ZLIMS=[0,$zlimsfar]"
#         python eval.py -cn sampler_eval_explore $SHARED JOB_TYPE=eval_controlnet_novel_view UNIQUE_EVAL_ID=explore_propnv_${numnv}_${numprop} $args
#         python eval.py -cn sampler_eval_explore $SHARED JOB_TYPE=eval_controlnet_lidar_occ EVAL_OCCUPANCY.MODE=cascade UNIQUE_EVAL_ID=explore_propnv_${numnv}_${numprop} $args
#     done
# done


# 1224528 new 1225662
numprop=12
numnv=4
initstrat=stratified
zlimsfar=(3 5 7 10 15 20)
# for zlim in "${zlimsfar[@]}"; do
#     echo "Running eval: $zlim"
#     args="SYNTHETIC_GT.NV_CAM_SAMPLER.NUM_NOVEL_VIEWS=$numnv SYNTHETIC_GT.NV_CAM_SAMPLER.NUM_PROPOSALS=$numprop SYNTHETIC_GT.NV_CAM_SAMPLER.INIT_POLICY=$initstrat SYNTHETIC_GT.NV_CAM_SAMPLER.ZLIMS=[0,$zlim]"
#     python eval.py -cn sampler_eval_explore $SHARED JOB_TYPE=eval_controlnet_novel_view UNIQUE_EVAL_ID=explore_zfar_${zlim} $args
#     python eval.py -cn sampler_eval_explore $SHARED JOB_TYPE=eval_controlnet_lidar_occ EVAL_OCCUPANCY.MODE=cascade UNIQUE_EVAL_ID=explore_zfar_${zlim} $args
# done


################ SAMPLER EVAL ################

# 1225583 1226022
samplers=(
    "sampler_eval_orbit"
    "sampler_eval_shift"
    "sampler_eval_choice"
    "sampler_eval_rig12"
    "sampler_eval_rig8"
    "sampler_eval_rig4"
    "sampler_eval_rig4v2"
    "sampler_eval_explore SYNTHETIC_GT.NV_CAM_SAMPLER.NUM_PROPOSALS=16 SYNTHETIC_GT.NV_CAM_SAMPLER.INIT_POLICY=stratified SYNTHETIC_GT.NV_CAM_SAMPLER.ZLIMS=[0,5]"
    "cascade_eval UNIQUE_EVAL_ID=sampler_cascade SYNTHETIC_GT.CASCADE.CHOSEN_ANCHOR_TYPE=explore_16_1_4_stratified_5"
    "cascade_eval UNIQUE_EVAL_ID=sampler_cascade_2 SYNTHETIC_GT.CASCADE.CHOSEN_ANCHOR_TYPE=explore_16_2_2_stratified_5"
)

# for sampler in "${samplers[@]}"; do
#     echo "Running eval: $sampler"
#     python eval.py -cn $sampler $SHARED JOB_TYPE=eval_controlnet_novel_view SYNTHETIC_GT.NV_CAM_SAMPLER.NUM_NOVEL_VIEWS=4 SYNTHETIC_GT.NV_CAM_SAMPLER.MAKE_PROJS_POLICY=center_crop
#     python eval.py -cn $sampler $SHARED JOB_TYPE=eval_controlnet_lidar_occ EVAL_OCCUPANCY.MODE=cascade SYNTHETIC_GT.NV_CAM_SAMPLER.NUM_NOVEL_VIEWS=4 SYNTHETIC_GT.INPUT_CROP_POLICY="center+left+right" SYNTHETIC_GT.NV_CAM_SAMPLER.MAKE_PROJS_POLICY=center_crop
# done


samplers8=(
    # "sampler_eval_shift UNIQUE_EVAL_ID=sampler_8_shift"
    # "sampler_eval_rig12 UNIQUE_EVAL_ID=sampler_8_rig"
    # "sampler_eval_explore UNIQUE_EVAL_ID=sampler_8_explore SYNTHETIC_GT.NV_CAM_SAMPLER.NUM_PROPOSALS=16 SYNTHETIC_GT.NV_CAM_SAMPLER.INIT_POLICY=stratified SYNTHETIC_GT.NV_CAM_SAMPLER.ZLIMS=[0,5]"
    # "cascade_eval UNIQUE_EVAL_ID=sampler_8_cascade SYNTHETIC_GT.CASCADE.CHOSEN_ANCHOR_TYPE=explore_16_4_2_stratified_5"
    # "cascade_eval UNIQUE_EVAL_ID=sampler_8_cascade_2 SYNTHETIC_GT.CASCADE.CHOSEN_ANCHOR_TYPE=explore_16_2_4_stratified_5"
    
    # ACTUAL CASCADE EVALS
    # "cascade_eval UNIQUE_EVAL_ID=cascade_rig_explore_short SYNTHETIC_GT.CASCADE.CHOSEN_ANCHOR_TYPE=rig12_explore_short"   # 1226039
    # "cascade_eval UNIQUE_EVAL_ID=cascade_rig_explore_mid SYNTHETIC_GT.CASCADE.CHOSEN_ANCHOR_TYPE=rig12_explore_mid"       # 1226040
    # "cascade_eval UNIQUE_EVAL_ID=cascade_rig_explore_far SYNTHETIC_GT.CASCADE.CHOSEN_ANCHOR_TYPE=rig12_explore_far"       # 1226041
    # "cascade_eval UNIQUE_EVAL_ID=cascade_rig_explore_short_filter_mod SYNTHETIC_GT.CASCADE.CHOSEN_ANCHOR_TYPE=rig12_explore_short ++cfg.SYNTHETIC_GT.INFERENCE_OCCLUSIONS.POSTPROCESSING_OPS_DEPTH_GRADS=[['opening',3],['closing',15],['dilation',3]]"   # 1226042
    # "cascade_eval UNIQUE_EVAL_ID=cascade_rig_explore_short_filter_mod2 SYNTHETIC_GT.CASCADE.CHOSEN_ANCHOR_TYPE=rig12_explore_short ++cfg.SYNTHETIC_GT.INFERENCE_OCCLUSIONS.POSTPROCESSING_OPS_DEPTH_GRADS=[['opening',3],['closing',15]]"   # 1230564
    # "cascade_eval UNIQUE_EVAL_ID=cascade_rig12_8_explore3x SYNTHETIC_GT.CASCADE.CHOSEN_ANCHOR_TYPE=rig12_8_explore3x"       # 1230697
    
    # "cascade_eval UNIQUE_EVAL_ID=cascade_rig12_8_explore3x_far SYNTHETIC_GT.CASCADE.CHOSEN_ANCHOR_TYPE=rig12_8_explore3x_far ++cfg.SYNTHETIC_GT.INFERENCE_OCCLUSIONS.POSTPROCESSING_OPS_DEPTH_GRADS=[['opening',3],['closing',15]]"   # 1235917
    
    # running 1240556 1240553 1240558
    # "cascade_eval UNIQUE_EVAL_ID=cascade_rig12_8_explore3x_mod SYNTHETIC_GT.CASCADE.CHOSEN_ANCHOR_TYPE=rig12_8_explore3x SYNTHETIC_GT.INFERENCE_OCCLUSIONS.POSTPROCESSING_OPS_DEPTH_GRADS=[['opening',3],['closing',15]]"  
    # "cascade_eval UNIQUE_EVAL_ID=cascade_rig12_8_explore3x4 SYNTHETIC_GT.CASCADE.CHOSEN_ANCHOR_TYPE=rig12_8_explore3x4"
    # "cascade_eval UNIQUE_EVAL_ID=cascade_rig12_8_explore3x4_ranged SYNTHETIC_GT.CASCADE.CHOSEN_ANCHOR_TYPE=rig12_8_explore3x4_ranged"
    
    # "eval_controlnet_cascade UNIQUE_EVAL_ID=cascade_rig12_8_explore3x_full SYNTHETIC_GT.CASCADE.CHOSEN_ANCHOR_TYPE=rig12_8_explore3x ++cfg.SYNTHETIC_GT.INFERENCE_OCCLUSIONS.POSTPROCESSING_OPS_DEPTH_GRADS=[['opening',3],['closing',15]]"   # 1235917
    
    # running 1240559 1240561 1240562 1240710
    # "eval_controlnet_cascade UNIQUE_EVAL_ID=cascade_rig12_8_explore3x4_full SYNTHETIC_GT.CASCADE.CHOSEN_ANCHOR_TYPE=rig12_8_explore3x4 SYNTHETIC_GT.INFERENCE_OCCLUSIONS.POSTPROCESSING_OPS_DEPTH_GRADS=[['opening',3],['closing',15]]"
    # "eval_controlnet_cascade UNIQUE_EVAL_ID=cascade_rig12_8_explore3x4_ranged_full SYNTHETIC_GT.CASCADE.CHOSEN_ANCHOR_TYPE=rig12_8_explore3x4_ranged SYNTHETIC_GT.INFERENCE_OCCLUSIONS.POSTPROCESSING_OPS_DEPTH_GRADS=[['opening',3],['closing',15]]"
    # "sampler_eval_rig12_full UNIQUE_EVAL_ID=sampler_rig12_full SYNTHETIC_GT.NV_CAM_SAMPLER.NUM_NOVEL_VIEWS=8"
    "sampler_eval_explore_full UNIQUE_EVAL_ID=sampler_explore_full SYNTHETIC_GT.NV_CAM_SAMPLER.NUM_PROPOSALS=20 SYNTHETIC_GT.NV_CAM_SAMPLER.NUM_NOVEL_VIEWS=10 SYNTHETIC_GT.NV_CAM_SAMPLER.INIT_POLICY=stratified SYNTHETIC_GT.NV_CAM_SAMPLER.ZLIMS=[0,10] SYNTHETIC_GT.INFERENCE_OCCLUSIONS.POSTPROCESSING_OPS_DEPTH_GRADS=[['opening',3],['closing',15]]"
)

# 1225584
# for sampler in "${samplers8[@]}"; do
#     echo "Running eval: $sampler"
#     # python eval.py -cn $sampler $SHARED JOB_TYPE=eval_controlnet_novel_view SYNTHETIC_GT.NV_CAM_SAMPLER.NUM_NOVEL_VIEWS=8 SYNTHETIC_GT.NV_CAM_SAMPLER.MAKE_PROJS_POLICY=center_crop
#     # python eval.py -cn $sampler $SHARED JOB_TYPE=eval_controlnet_lidar_occ EVAL_OCCUPANCY.MODE=cascade SYNTHETIC_GT.NV_CAM_SAMPLER.NUM_NOVEL_VIEWS=8 SYNTHETIC_GT.INPUT_CROP_POLICY="center+left+right" SYNTHETIC_GT.NV_CAM_SAMPLER.MAKE_PROJS_POLICY=center_crop
#     python eval.py -cn $sampler $SHARED JOB_TYPE=eval_controlnet_lidar_occ EVAL_OCCUPANCY.MODE=cascade SYNTHETIC_GT.INPUT_CROP_POLICY="center+left+right" SYNTHETIC_GT.NV_CAM_SAMPLER.MAKE_PROJS_POLICY=center_crop
# done


################ CASCADE EVAL KITTI-360 ################

# 1224983 4 5

numprop=12
numnvs=(
    # 1 2         # 1225658
    # 3 4           # 1225656 1225657
    6 8         # 1225659 1225660 1225661
)
# numcascs=(1 2 3 5)
numcascs=(5)
initstrat=stratified
zlimsfar=10
# for numcasc in "${numcascs[@]}"; do
#     for numnv in "${numnvs[@]}"; do
#         echo "Running with numcasc: $numcasc and numnv: $numnv"
#         python eval.py -cn cascade_eval ${SHARED} UNIQUE_EVAL_ID=cascade_nv_vs_casc_${numnv}_${numcasc} SYNTHETIC_GT.CASCADE.CHOSEN_ANCHOR_TYPE=explore_${numprop}_${numnv}_${numcasc}_${initstrat}_${zlimsfar}
#     done
# done

# ---  ----

# 1224986
numprop=12
numnv=3
numcasc=2
initstrat=stratified
zlimsfar=(3 5 7 10 15 20)
# for zlimfar in "${zlimsfar[@]}"; do
#     echo "Running with zlimfar: $zlimfar"
#     python eval.py -cn cascade_eval ${SHARED} UNIQUE_EVAL_ID=cascade_zlimfar_${zlimfar} SYNTHETIC_GT.CASCADE.CHOSEN_ANCHOR_TYPE=explore_${numprop}_${numnv}_${numcasc}_${initstrat}_${zlimfar}
# done

################


# CASCADE_SHARED="JOB_TYPE=eval_controlnet_lidar_occ EVAL_OCCUPANCY.MODE=cascade"
# python eval.py -cn exp_bts_synthetic_cascade_depth ${CASCADE_SHARED} ${SHARED} UNIQUE_EVAL_ID=rig8of8 SYNTHETIC_GT.CASCADE.CHOSEN_ANCHOR_TYPE=rig8of8
# python eval.py -cn exp_bts_synthetic_cascade_depth ${CASCADE_SHARED} ${SHARED} UNIQUE_EVAL_ID=rig8of8plus8 SYNTHETIC_GT.CASCADE.CHOSEN_ANCHOR_TYPE=rig8of8plus8
# python eval.py -cn exp_bts_synthetic_cascade_depth ${CASCADE_SHARED} ${SHARED} UNIQUE_EVAL_ID=rig12of12 SYNTHETIC_GT.CASCADE.CHOSEN_ANCHOR_TYPE=rig12of12 SYNTHETIC_GT.TOP_K_NOVEL_VIEWS_TO_KEEP=6

# python eval.py -cn exp_bts_synthetic_cascade_depth JOB_TYPE=eval_controlnet_lidar_occ EVAL_OCCUPANCY.MODE=cascade AMP.ENABLED=false BTS.LOG_EVERY_ITERS=50 BTS.VISUALIZE_EVERY=10 BTS.BATCH_SIZE=1 SEED=42


# bash eval_controlnet_pseudo.sh

# 1366480
# 1366479


# 
bash eval_sampler_vs_occlusion.sh 



SHARED="AMP.ENABLED=false BTS.LOG_EVERY_ITERS=50 BTS.VISUALIZE_EVERY=10 BTS.BATCH_SIZE=1 SEED=42"
python eval.py -cn controlnet_rgb_512x768 ${SHARED} JOB_TYPE="eval_controlnet_input_view" UNIQUE_EVAL_ID="eval_run_0_kitti360" SYNTHETIC_GT.NUM_SYNTHETIC_VERSIONS=4
python eval.py -cn controlnet_rgb_512x768 ${SHARED} JOB_TYPE="eval_controlnet_novel_view" UNIQUE_EVAL_ID="eval_ver4_kitti360" SYNTHETIC_GT.NUM_SYNTHETIC_VERSIONS=4
python eval.py -cn controlnet_rgb_512x768 ${SHARED} JOB_TYPE="eval_controlnet_novel_view" UNIQUE_EVAL_ID="eval_nv4_kitti360" SYNTHETIC_GT.NUM_SYNTHETIC_VERSIONS=1 SYNTHETIC_GT.NV_CAM_SAMPLER.NUM_NOVEL_VIEWS=4

# ----------------------------------------------------------------------------
# EVAL DEPTH PREDS
# ----------------------------------------------------------------------------

SHARED="UNIQUE_EVAL_ID="eval_run_0" AMP.ENABLED=false BTS.LOG_EVERY_ITERS=50 BTS.VISUALIZE_EVERY=10 BTS.BATCH_SIZE=1"
# On KITTI360:
SHARED="$SHARED CONTROLNET.DATA.image_size=[192,640]"
# On Waymo:
# SHARED="$SHARED CONTROLNET.DATA.image_size=[320,480]"

tasks=(
    # "eval_z20_20"
    "eval_z20_300 EVAL_OCCUPANCY.GT_AGGREGATE_TIMESTEPS=300"
)

depthmodels=(
    "UniDepth"
    "Metric3D"
)

for depthmodel in "${depthmodels[@]}"; do
    for task in "${tasks[@]}"; do
        echo "Running eval: $depthmodel $task"
        python eval.py -cn eval_depth_pred $SHARED SYNTHETIC_GT.DEPTH_PREDICTOR_NAME=$depthmodel UNIQUE_EVAL_ID=${depthmodel}_$task
    done
done

# ----------------------------------------------------------------------------
# EVAL SAMPLER VS OCCLUSIONS
# ----------------------------------------------------------------------------

SHARED="AMP.ENABLED=true BTS.LOG_EVERY_ITERS=50 BTS.VISUALIZE_EVERY=10 BTS.BATCH_SIZE=1 SEED=42 CONTROLNET.DATA.image_size=[192,640]"

xlims=(
    # "0,0"
    
    # 1368044
    # "1,1"
    # "1.25,1.25"
    # "1.5,1.5"

    # 1368077
    # "1.75,1.75"
    # "2,2"
    # "2.25,2.25"

    # 1368078
    # "2.5,2.5"
    # "2.75,2.75"
    # "3,3"

    # 1368080
    # "3.25,3.25"
    # "3.5,3.5"

    # 1368236
    "3.75,3.75"
    "4,4"

    # "5,5"
    # "6,6"
    # "7,7"
)

occlusions=(
    "SYNTHETIC_GT.INFERENCE_OCCLUSIONS.USE_DEPTH_GRADS=true SYNTHETIC_GT.INFERENCE_OCCLUSIONS.USE_FLOW=false"
    "SYNTHETIC_GT.INFERENCE_OCCLUSIONS.USE_DEPTH_GRADS=false SYNTHETIC_GT.INFERENCE_OCCLUSIONS.USE_FLOW=true"
    # "SYNTHETIC_GT.INFERENCE_OCCLUSIONS.USE_DEPTH_GRADS=true SYNTHETIC_GT.INFERENCE_OCCLUSIONS.USE_FLOW=true"
)
occlusion_names=(
    "depth_only"
    "flow_only"
    "depth_and_flow"
)

for xlim in "${xlims[@]}"; do
    xlim_eval_id=$(echo $xlim | sed 's/,/to/')
    for i in "${!occlusions[@]}"; do
        occlusion="${occlusions[$i]}"
        occl_name="${occlusion_names[$i]}"
        unique_eval_id="new_shift_occl_${xlim_eval_id}_${occl_name}"
        echo "Running eval: $xlim ($occl_name)"
        python eval.py -cn sampler_eval_shift ${SHARED} JOB_TYPE="eval_controlnet_input_view" UNIQUE_EVAL_ID=$unique_eval_id SYNTHETIC_GT.NV_CAM_SAMPLER.X_LIMS="[$xlim]" $occlusion
        python eval.py -cn sampler_eval_shift ${SHARED} JOB_TYPE="eval_controlnet_novel_view" UNIQUE_EVAL_ID=$unique_eval_id SYNTHETIC_GT.NV_CAM_SAMPLER.X_LIMS="[$xlim]" $occlusion
    done
done


# ----------------------------------------------------------------------------
# TRAIN RECONSTRUCTOR
# ----------------------------------------------------------------------------


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


# ----------------------------------------------------------------------------
# TRAIN CONTROLNET
# ----------------------------------------------------------------------------


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
