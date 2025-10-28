# Demo script for FLUX.1 Kontext
python src/FluxKontext/main.py \
    --model_path black-forest-labs/FLUX.1-Kontext-dev \
    --num_inference_steps 28 \
    --use_regione \
    --warmup_step 6 \
    --post_step 2 \
    --refresh_step "16" \
    --threshold 0.93 \
    --cache_threshold 0.01 \
    --erosion_dilation \
    --guidance_scale 2.5 \
    --seed 110 \
    --device cuda \
    --image_path assets/data.jsonl \
    --output_dir result/FluxKontext/Demo1/RegionE

# Evaluation script for FLUX.1 Kontext
python src/FluxKontext/main.py \
    --model_path black-forest-labs/FLUX.1-Kontext-dev \
    --num_inference_steps 28 \
    --use_regione \
    --warmup_step 6 \
    --post_step 2 \
    --refresh_step "16" \
    --threshold 0.93 \
    --cache_threshold 0.04 \
    --erosion_dilation \
    --guidance_scale 2.5 \
    --seed 110 \
    --device cuda \
    --evaluation \
    --image_path data/Processed/Kontext-Bench \
    --output_dir result/FluxKontext/RegionE
