from viescore import VIEScore
import PIL
import os
import megfile
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset, load_from_disk
import sys
import csv
import time
import argparse
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from collections import defaultdict
import json

def calculate_dimensions(target_area, ratio):
    width = math.sqrt(target_area * ratio)
    height = width / ratio
    new_area = width * height
    return int(width), int(height), int(new_area)

def process_single_item(item, vie_score, max_retries=10000):

    instruction = item['instruction']
    key = item['key']
    instruction_language = item['instruction_language']
    edit_image_path = item['edited_image_path']
    
    for retry in range(max_retries):
        try:
            pil_image = Image.open(megfile.smart_open(item['input_image_raw'], 'rb')).convert("RGB")
            pil_image_edited = Image.open(megfile.smart_open(edit_image_path, 'rb')).convert("RGB")
            source_img_width, source_img_height, source_img_area = calculate_dimensions(512 * 512, pil_image.width / pil_image.height)
            edited_img_width, edited_img_height, edited_img_area = calculate_dimensions(512 * 512, pil_image_edited.width / pil_image_edited.height)
            pil_image = pil_image.resize((int(source_img_width), int(source_img_height)))
            pil_image_edited = pil_image_edited.resize((int(edited_img_width), int(edited_img_height)))
            text_prompt = instruction
            score_list = vie_score.evaluate([pil_image, pil_image_edited], text_prompt)
            sementics_score, quality_score, overall_score = score_list

            print(f"sementics_score: {sementics_score}, quality_score: {quality_score}, overall_score: {overall_score}, instruction_language: {instruction_language}, instruction: {instruction}")
            
            return {
                "key": key,
                "edited_image": edit_image_path,
                "instruction": instruction,
                "sementics_score": sementics_score,
                "quality_score": quality_score,
                "instruction_language" : item['instruction_language']
            }
        except Exception as e:
            if retry < max_retries - 1:
                wait_time = (retry + 1) * 2  # 指数退避：2秒, 4秒, 6秒...
                print(f"Error processing (attempt {retry + 1}/{max_retries}): {e}")
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"Failed to process after {max_retries} attempts: {e}")
                return

def find_files_with_given_basename(folder_path, basename):
    pattern = os.path.join(folder_path, f"{basename}.*")
    matched_files = megfile.smart_glob(pattern)
    return [os.path.basename(f) for f in matched_files]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="RegionE", help="edit model name")
    parser.add_argument("--edited_images_dir", type=str, default="result/Step1X-Edit/RegionE", help="path to edited images")
    parser.add_argument("--instruction_language", type=str, default="en", choices=["en"])
    parser.add_argument("--task_type", type=str, default="all",  choices=["all"])
    parser.add_argument("--backbone", type=str, default="gpt4o", choices=["qwen25vl", "gpt4o"])
    parser.add_argument("--source_img_dir", type=str, default="data/Processed/GEdit-Bench/en")
    args = parser.parse_args()
    model_name = args.model_name
    edited_images_dir = args.edited_images_dir
    backbone = args.backbone

    if "Step1X-Edit" or "Qwen-Image" in edited_images_dir:
        groups = ["background_change", "color_alter", "material_alter", "motion_change", "ps_human", 
        "style_change", "subject-add", "subject-remove", "subject-replace", "text_change", "tone_transfer"]
    elif "FluxKontext" in edited_images_dir:
        groups = ['CR', 'IEG', 'IEL', 'SR', 'TE']
    else:
        NotImplementedError("Error Direction!")

    # Load GEdit-Bench dataset and group by task type
    vie_score = VIEScore(backbone=backbone, task="tie", key_path='secret.env')
    dataset_by_group = defaultdict(list)

    for i, task in enumerate(groups):
        with open(os.path.join(edited_images_dir, model_name, task, "metadata.json"), 'r') as f:
            data = json.load(f)
        for key, item in data.items():
            dataset_by_group[task].append({"task_type": task, "key": key, "instruction_language": "en", "instruction": item, "input_image_raw": os.path.join(args.source_img_dir, task, "img")})
    
    for k, v in dataset_by_group.items():
        print(f"Number of samples in {k}:", len(v))

    # Evaluate each group
    save_path_new = os.path.join(edited_images_dir, model_name)
    for group_name in groups:
        group_csv_list = []
        group_dataset_list = dataset_by_group[group_name]

        # Load existing group CSV if it exists, if csv esists, skip this group
        group_csv_path = os.path.join(save_path_new, group_name, f"{backbone}_vie_score.csv")
        if megfile.smart_exists(group_csv_path):
            with megfile.smart_open(group_csv_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                group_results = list(reader)
            print(f"{model_name} - {group_name} exsits, skip this group")
            continue

        if backbone == 'qwen25vl':
            for item in tqdm(group_dataset_list, desc=f"Processing {model_name} - {group_name}"):
                key = item['key']
                try:
                    # Should organize edited image directory, please refer EVAL.md for details
                    edited_images_path = os.path.join(edited_images_dir, model_name, group_name, "generation")
                    item['edited_image_path'] = os.path.join(edited_images_path, find_files_with_given_basename(edited_images_path, key)[0])
                    item['input_image_raw'] = os.path.join(item['input_image_raw'], find_files_with_given_basename(item['input_image_raw'], key)[0])
                except:
                    print(key, "not found in", edited_images_path)
                    continue

                result = process_single_item(item, vie_score)
                if result:
                    group_csv_list.append(result)

        elif backbone == 'gpt4o':
            with ThreadPoolExecutor(max_workers=6) as executor:
                futures = []
                for item in group_dataset_list:
                    key = item['key']
                    try:
                        # Should organize edited image directory, please refer EVAL.md for details
                        edited_images_path = os.path.join(edited_images_dir, model_name, group_name, "generation")
                        item['edited_image_path'] = os.path.join(edited_images_path, find_files_with_given_basename(edited_images_path, key)[0])
                        item['input_image_raw'] = os.path.join(item['input_image_raw'], find_files_with_given_basename(item['input_image_raw'], key)[0])
                    except:
                        print(key, "not found in", edited_images_path)
                        continue

                    # Check if this sample has already been processed
                    future = executor.submit(process_single_item, item, vie_score)
                    futures.append(future)
                
                for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {model_name} - {group_name}"):
                    result = future.result()
                    if result:
                        group_csv_list.append(result)

        else:
            NotImplementedError('Error backbone')

        # Save group-specific CSV
        with megfile.smart_open(group_csv_path, 'w', newline='') as f:
            fieldnames = ["key", "edited_image", "instruction", "sementics_score", "quality_score", "instruction_language"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in group_csv_list:  
                writer.writerow(row)
        print(f"Saved group CSV for {group_name}, length： {len(group_csv_list)}")
