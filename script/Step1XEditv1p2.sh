python src/Step1X-Edit-v1p2/main.py \
    --model_path stepfun-ai/Step1X-Edit-v1p2 \
    --num_inference_steps 28 \
    --guidance_scale 6.0 \
    --seed 110 \
    --device cuda \
    --image_path assets/data.jsonl \
    --output_dir result/Step1X-Edit-v1p2/Pretrain


# Demo script for Step1X-Edit-v1p2
python src/Step1X-Edit-v1p2/main.py \
    --model_path stepfun-ai/Step1X-Edit-v1p2 \
    --num_inference_steps 28 \
    --use_regione \
    --warmup_step 6 \
    --post_step 2 \
    --refresh_step "16" \
    --threshold 0.88 \
    --cache_threshold 0.02 \
    --erosion_dilation \
    --guidance_scale 6.0 \
    --seed 110 \
    --device cuda \
    --image_path assets/data.jsonl \
    --output_dir result/Step1X-Edit-v1p2/Demo/RegionE


# Evaluation script for Step1X-Edit-v1p2
python src/Step1X-Edit-v1p2/main.py \
    --model_path stepfun-ai/Step1X-Edit-v1p2 \
    --num_inference_steps 28 \
    --use_regione \
    --warmup_step 6 \
    --post_step 2 \
    --refresh_step "16" \
    --threshold 0.88 \
    --cache_threshold 0.02 \
    --erosion_dilation \
    --guidance_scale 6.0 \
    --seed 110 \
    --device cuda \
    --evaluation \
    --image_path data/Processed/GEdit-Bench/en \
    --output_dir result/Step1X-Edit-v1p2/RegionE