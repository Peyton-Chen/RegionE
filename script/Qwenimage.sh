# Demo script for Qwen-Image-Edit
python src/Qwen-Image/main.py \
    --model_path Qwen/Qwen-Image-Edit \
    --num_inference_steps 28 \
    --use_regione \
    --warmup_step 6 \
    --post_step 2 \
    --refresh_step "16" \
    --threshold 0.80 \
    --cache_threshold 0.03 \
    --erosion_dilation \
    --guidance_scale 4.0 \
    --seed 110 \
    --device cuda \
    --image_path assets/data.jsonl \
    --output_dir result/Qwen-Image/Demo/RegionE

# Evaluation script for Qwen-Image-Edit
python src/Qwen-Image/main.py \
    --model_path Qwen/Qwen-Image-Edit \
    --num_inference_steps 28 \
    --use_regione \
    --warmup_step 6 \
    --post_step 2 \
    --refresh_step "16" \
    --threshold 0.80 \
    --cache_threshold 0.03 \
    --erosion_dilation \
    --guidance_scale 4.0 \
    --seed 110 \
    --device cuda \
    --evaluation \
    --image_path data/Processed/GEdit-Bench/en \
    --output_dir result/Qwen-Image/RegionE
