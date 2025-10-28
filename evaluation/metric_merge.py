import os
import json
import argparse
import pandas as pd

def merge_kontext(path):

    ref = ['CR', 'IEG', 'IEL', 'SR', 'TE']
    dir_list = os.listdir(path)
    if_raise = not all(s in dir_list for s in ref)
    if if_raise: NotImplementedError('direction is not right')
    PSNR_sum, SSIM_sum, LPIPS_sum, item_sum, latency_sum = 0, 0, 0, 0, 0

    if os.path.basename(path).lower() == 'pretrain':
        for task in ref:
            latency_path = f"{path}/{task}/time_consuming.json"

            with open(latency_path, 'r') as f:
                latency = json.load(f)
            num_prompt, ave_latency = latency['num_item'], latency['ave_time_consuming']
            
            item_sum += num_prompt
            latency_sum += ave_latency * num_prompt
        
        num_prompt, ave_latency =  item_sum, latency_sum/item_sum
        print(num_prompt, ave_latency)
        
        with open(f"{path}/merged_metric.txt", 'w') as f:
            f.write(f'PSNR: inf \n')
            f.write(f'SSIM: 1 \n')
            f.write(f'LPIPS: 0 \n')
            f.write(f'Prompts: {num_prompt} \n')
            f.write(f'Latency: {ave_latency} \n')
    else:

        for task in ref:
            metric_path = f"{path}/{task}/metric.csv"
            latency_path = f"{path}/{task}/time_consuming.json"

            metric = pd.read_csv(metric_path).tail(1).to_dict(orient='records')[0]
            PSNR, SSIM, LPIPS = metric['PSNR'], metric['SSIM'], metric['LPIPS']


            with open(latency_path, 'r') as f:
                latency = json.load(f)
            num_prompt, ave_latency = latency['num_item'], latency['ave_time_consuming']
            
            PSNR_sum += PSNR * num_prompt
            SSIM_sum += SSIM * num_prompt
            LPIPS_sum += LPIPS * num_prompt
            item_sum += num_prompt
            latency_sum += ave_latency * num_prompt
        
        PSNR, SSIM, LPIPS, num_prompt, ave_latency = PSNR_sum/item_sum, SSIM_sum/item_sum, LPIPS_sum/item_sum, item_sum, latency_sum/item_sum
        print( PSNR, SSIM, LPIPS, num_prompt, ave_latency)
        
        with open(f"{path}/merged_metric.txt", 'w') as f:
            f.write(f'PSNR: {PSNR} \n')
            f.write(f'SSIM: {SSIM} \n')
            f.write(f'LPIPS: {LPIPS} \n')
            f.write(f'Prompts: {num_prompt} \n')
            f.write(f'Latency: {ave_latency} \n')


def merge_gedit(path):

    ref = ['motion_change', 'ps_human', 'color_alter', 'material_alter', 'subject-add',
            'subject-remove', 'style_change', 'tone_transfer', 'subject-replace',
            'text_change', 'background_change']
    dir_list = os.listdir(path)
    if_raise = not all(s in dir_list for s in ref)
    if if_raise: NotImplementedError('direction is not right')
    PSNR_sum, SSIM_sum, LPIPS_sum, item_sum, latency_sum = 0, 0, 0, 0, 0

    if os.path.basename(path).lower() == 'pretrain':
        for task in ref:
            latency_path = f"{path}/{task}/time_consuming.json"

            with open(latency_path, 'r') as f:
                latency = json.load(f)
            num_prompt, ave_latency = latency['num_item'], latency['ave_time_consuming']
            
            item_sum += num_prompt
            latency_sum += ave_latency * num_prompt
        
        num_prompt, ave_latency =  item_sum, latency_sum/item_sum
        print(num_prompt, ave_latency)
        
        with open(f"{path}/merged_metric.txt", 'w') as f:
            f.write(f'PSNR: inf \n')
            f.write(f'SSIM: 1 \n')
            f.write(f'LPIPS: 0 \n')
            f.write(f'Prompts: {num_prompt} \n')
            f.write(f'Latency: {ave_latency} \n')
    else:

        for task in ref:
            metric_path = f"{path}/{task}/metric.csv"
            latency_path = f"{path}/{task}/time_consuming.json"

            metric = pd.read_csv(metric_path).tail(1).to_dict(orient='records')[0]
            PSNR, SSIM, LPIPS = metric['PSNR'], metric['SSIM'], metric['LPIPS']


            with open(latency_path, 'r') as f:
                latency = json.load(f)
            num_prompt, ave_latency = latency['num_item'], latency['ave_time_consuming']

            PSNR_sum += PSNR * num_prompt
            SSIM_sum += SSIM * num_prompt
            LPIPS_sum += LPIPS * num_prompt
            item_sum += num_prompt
            latency_sum += ave_latency * num_prompt
        
        PSNR, SSIM, LPIPS, num_prompt, ave_latency = PSNR_sum/item_sum, SSIM_sum/item_sum, LPIPS_sum/item_sum, item_sum, latency_sum/item_sum
        print( PSNR, SSIM, LPIPS, num_prompt, ave_latency)
        
        with open(f"{path}/merged_metric.txt", 'w') as f:
            f.write(f'PSNR: {PSNR} \n')
            f.write(f'SSIM: {SSIM} \n')
            f.write(f'LPIPS: {LPIPS} \n')
            f.write(f'Prompts: {num_prompt} \n')
            f.write(f'Latency: {ave_latency} \n')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--direction", type=str, required=True)
    args = parser.parse_args() 

    if "FluxKontext" in args.direction:
        merge_kontext(args.direction)

    elif "Step1X-Edit" or "Qwen-Image" in args.direction:
        merge_gedit(args.direction)

    else:
        NotImplementedError("Diretion Error")
