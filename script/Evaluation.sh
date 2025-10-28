# -----------------------------------------------
# SSIM PSNR and LPIPS evaluation script
# -----------------------------------------------

# Step1X-Edit Pretrain vs RegionE
python evaluation/metric_all_task.py \
--folder1 result/Step1X-Edit/Pretrain \
--folder2 result/Step1X-Edit/RegionE

python evaluation/metric_merge.py \
--direction result/Step1X-Edit/RegionE

# FLUX.1 Kontext Pretrain vs RegionE
python evaluation/metric_all_task.py \
--folder1 result/FluxKontext/Pretrain \
--folder2 result/FluxKontext/RegionE

python evaluation/metric_merge.py \
--direction result/FluxKontext/RegionE

# Qwen-Image Pretrain vs RegionE
python evaluation/metric_all_task.py \
--folder1 result/Qwen-Image/Pretrain \
--folder2 result/Qwen-Image/RegionE

python evaluation/metric_merge.py \
--direction result/Qwen-Image/RegionE

# -----------------------------------------------
# GPT4o-Score evaluation script 
# -----------------------------------------------
# Step1X-Edit
cd evaluation/GEdit-Bench
python run_gedit_score.py \
    --backbone="gpt4o" \
    --instruction_language="en" \
    --task_type="all" \
    --edited_images_dir="../../result/Step1X-Edit/RegionE" \
    --source_img_dir="../../data/Processed/GEdit-Bench/en" \
    --model_name="RegionE"

# FLUX.1 Kontext
cd evaluation/GEdit-Bench
python run_gedit_score.py \
    --backbone="gpt4o" \
    --instruction_language="en" \
    --task_type="all" \
    --edited_images_dir="../../result/FluxKontext/RegionE" \
    --source_img_dir="../../data/Processed/Kontext-Bench" \
    --model_name="RegionE"

# Qwen-Image
cd evaluation/GEdit-Bench
python run_gedit_score.py \
    --backbone="gpt4o" \
    --instruction_language="en" \
    --task_type="all" \
    --edited_images_dir="../../result/Qwen-Image/RegionE" \
    --source_img_dir="../../data/Processed/GEdit-Bench/en" \
    --model_name="RegionE"
