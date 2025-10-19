SHARED="AMP.ENABLED=false BTS.LOG_EVERY_ITERS=50 BTS.VISUALIZE_EVERY=10 BTS.BATCH_SIZE=1 SEED=42"

controlnetExps=(
    "controlnet_rgb_512x768"
    "controlnet_full_512x768"
    "controlnet_full_512x768_waymo"
)

echo "----------------------- Evaluating pretrained view completion models -----------------------"

for exp in "${controlnetExps[@]}"; do
    echo "Running eval: $exp"
    # input view
    python eval.py -cn ${exp} ${SHARED} JOB_TYPE="eval_controlnet_input_view" UNIQUE_EVAL_ID="eval_run_0" SYNTHETIC_GT.NUM_SYNTHETIC_VERSIONS=4
    # novel view
    python eval.py -cn ${exp} ${SHARED} JOB_TYPE="eval_controlnet_novel_view" UNIQUE_EVAL_ID="eval_ver4" SYNTHETIC_GT.NUM_SYNTHETIC_VERSIONS=4
    python eval.py -cn ${exp} ${SHARED} JOB_TYPE="eval_controlnet_novel_view" UNIQUE_EVAL_ID="eval_nv4" SYNTHETIC_GT.NUM_SYNTHETIC_VERSIONS=1 SYNTHETIC_GT.NV_CAM_SAMPLER.NUM_NOVEL_VIEWS=4
done
