# Demo script for Step1X-Edit
python src/Step1X-Edit/main.py \
    --model_path stepfun-ai/Step1X-Edit-v1p1-diffusers \
    --num_inference_steps 28 \
    --use_regione \
    --warmup_step 6 \
    --post_step 2 \
    --refresh_step 16 \
    --threshold 0.88 \
    --cache_threshold 0.02 \
    --erosion_dilation \
    --guidance_scale 6.0 \
    --seed 110 \
    --device cuda \
    --image_path assets/data.json \
    --output_dir result/Step1X-Edit/Demo/RegionE

# Evaluation script for Step1X-Edit
python src/Step1X-Edit/main.py \
    --model_path stepfun-ai/Step1X-Edit-v1p1-diffusers \
    --num_inference_steps 28 \
    --use_regione \
    --warmup_step 6 \
    --post_step 2 \
    --refresh_step 16 \
    --threshold 0.88 \
    --cache_threshold 0.02 \
    --erosion_dilation \
    --guidance_scale 6.0 \
    --seed 110 \
    --device cuda \
    --evaluation \
    --image_path data/Processed/GEdit-Bench/en \
    --output_dir result/Step1X-Edit/RegionE