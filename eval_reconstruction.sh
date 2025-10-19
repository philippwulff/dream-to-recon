SHARED="UNIQUE_EVAL_ID="eval_run_0" AMP.ENABLED=false BTS.LOG_EVERY_ITERS=50 BTS.VISUALIZE_EVERY=10 BTS.BATCH_SIZE=1"

CHECKPOINTS=(
    # BTS and BTS-D (KITTI-360 and Waymo)
    "out/kitti360/pretrained/BTS_kitti360.pt"
    "out/waymo/pretrained/BTS_waymo.pt"
    "out/kitti360/pretrained/BTS-D_kitti360.pt"
    "out/waymo/pretrained/BTS-D_waymo.pt"
    # Ours (Distilled) (KITTI-360 and Waymo)
    "out/kitti360/pretrained/Ours-Distilled_kitti360.pt"
    "out/waymo/pretrained/Ours-Distilled_waymo.pt"
)

CONFIGS=(
    # BTS and BTS-D (KITTI-360 and Waymo)
    eval_bts_kitti360
    eval_bts_waymo
    eval_bts_kitti360
    eval_bts_waymo
    # Ours (Distilled) (KITTI-360 and Waymo)
    eval_recon_kitti369
    eval_recon_waymo
)

TASK=eval_z20_300

echo "----------------------- Evaluating pretrained reconstruction models -----------------------"

for i in "${!CHECKPOINTS[@]}"; do
    checkpoint="${CHECKPOINTS[$i]}"
    config="${CONFIGS[$i]}"
    python eval.py -cn $config $SHARED BTS.RESUME_FROM=$checkpoint UNIQUE_EVAL_ID=$TASK
done