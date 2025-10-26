import megfile
import os
import pandas as pd
from collections import defaultdict
import sys
import numpy as np
import math



def analyze_scores(save_path_dir, evaluate_group, language, backbone, GROUPS):

    group_scores_semantics = defaultdict(lambda: defaultdict(list))
    group_scores_quality = defaultdict(lambda: defaultdict(list))
    group_scores_overall = defaultdict(lambda: defaultdict(list))

    for group_name in GROUPS:
        csv_path = os.path.join(save_path_dir, group_name, f"{backbone}_vie_score.csv")
        csv_file = megfile.smart_open(csv_path)
        df = pd.read_csv(csv_file)
        
        filtered_semantics_scores = []
        filtered_quality_scores = []
        filtered_overall_scores = []
        
        for _, row in df.iterrows():
            source_image = row['key']
            edited_image = row['edited_image']
            instruction = row['instruction']
            semantics_score = row['sementics_score']
            quality_score = row['quality_score']
            instruction_language = row['instruction_language']

            if instruction_language == language:
                pass
            else:
                continue
            
            overall_score = math.sqrt(semantics_score * quality_score)

            filtered_semantics_scores.append(semantics_score)
            filtered_quality_scores.append(quality_score)
            filtered_overall_scores.append(overall_score)
        
        avg_semantics_score = np.mean(filtered_semantics_scores)
        avg_quality_score = np.mean(filtered_quality_scores)
        avg_overall_score = np.mean(filtered_overall_scores)
        group_scores_semantics[evaluate_group[0]][group_name] = avg_semantics_score
        group_scores_quality[evaluate_group[0]][group_name] = avg_quality_score
        group_scores_overall[evaluate_group[0]][group_name] = avg_overall_score

    # --- Overall Model Averages ---

    # Semantics
    for model_name in evaluate_group:
        model_scores = [group_scores_semantics[model_name][group] for group in GROUPS]
        model_avg = np.mean(model_scores)
        group_scores_semantics[model_name]["avg_semantics"] = model_avg

    # Quality
    for model_name in evaluate_group:
        model_scores = [group_scores_quality[model_name][group] for group in GROUPS]
        model_avg = np.mean(model_scores)
        group_scores_quality[model_name]["avg_quality"] = model_avg

    # Overall
    for model_name in evaluate_group:
        model_scores = [group_scores_overall[model_name][group] for group in GROUPS]
        model_avg = np.mean(model_scores)
        group_scores_overall[model_name]["avg_overall"] = model_avg

    return group_scores_semantics, group_scores_quality, group_scores_overall


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="RegionE")
    parser.add_argument("--backbone", type=str, default="gpt4o", choices=["qwen25vl", "gpt4o"])
    parser.add_argument("--save_path", type=str, default="result/Step1X-Edit")
    parser.add_argument("--language", type=str, default="en", choices=["en"])
    args = parser.parse_args()
    model_name = args.model_name
    backbone = args.backbone
    save_path_dir = args.save_path
    if args.language == "all":
        languages = ["en", "cn"]
    else:
        languages = [args.language]

    if "FluxKontext" in save_path_dir:
        GROUPS = ['CR', 'IEG', 'IEL', 'SR', 'TE']
    elif "Step1X-Edit" or "Qwen-Image" in save_path_dir:
        GROUPS = [
        "background_change", "color_alter", "material_alter", "motion_change", "ps_human", "style_change", "subject-add", "subject-remove", "subject-replace", "text_change", "tone_transfer"
        ]
    else:
        NotImplementedError("Error Path!")

    for language in languages:
        print("="*10 + f" backbone:{backbone} - model_name:{model_name} - language:{language} " + "="*10)
        
        save_path_new = os.path.join(save_path_dir, model_name)
        group_scores_semantics, group_scores_quality, group_scores_overall = analyze_scores(save_path_new, [model_name], language=language, backbone=backbone, GROUPS=GROUPS)
        
        print("\nOverall:")
        for group_name in GROUPS:
            print(f"{group_name}: {group_scores_semantics[model_name][group_name]:.3f}, {group_scores_quality[model_name][group_name]:.3f}, {group_scores_overall[model_name][group_name]:.3f}")
            with open(os.path.join(save_path_new, group_name, f'{backbone}_voe_score_mean.txt'), 'w') as f:
                f.write(f"{group_scores_semantics[model_name][group_name]:.3f}, {group_scores_quality[model_name][group_name]:.3f}, {group_scores_overall[model_name][group_name]:.3f}")    

        with open(os.path.join(save_path_new, f'{backbone}_voe_score_merged.txt'), 'w') as f:      
            f.write(f"Average: {group_scores_semantics[model_name]['avg_semantics']:.3f}, {group_scores_quality[model_name]['avg_quality']:.3f}, {group_scores_overall[model_name]['avg_overall']:.3f}")
        print(f"Average: {group_scores_semantics[model_name]['avg_semantics']:.3f}, {group_scores_quality[model_name]['avg_quality']:.3f}, {group_scores_overall[model_name]['avg_overall']:.3f}")
