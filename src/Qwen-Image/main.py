import os
import time
import json
import torch
import argparse

from PIL import Image
from utils import MANAGER
from inplace import regione_init
from diffusers import QwenImageEditPipeline


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # config for Qwen-Image-Edit
    parser.add_argument("--seed", type=int, default=110, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on (e.g., 'cuda' or 'cpu')")
    parser.add_argument("--num_inference_steps", type=int, default=28, help="Number of inference steps for the model")
    parser.add_argument("--guidance_scale", type=float, default=4.0, help="Guidance scale for the model")
    # config for RegionE
    parser.add_argument("--use_regione", action='store_true', help="Whether to use regione")
    parser.add_argument("--warmup_step", type=int, default=6, help="Step of the stablization stage")
    parser.add_argument("--post_step", type=int, default=2, help="Step of the smooth stage")
    parser.add_argument("--refresh_step", type=str, default="16", help="Steps are forcibly updated during the region-aware generation stage, format(str):16,22")
    parser.add_argument("--threshold", type=float, default=0.80, help="Threshold for adaptive region partition")
    parser.add_argument("--cache_threshold", type=float, default=0.03, help="Threshold for adaptive velocity decacy cache")
    parser.add_argument("--erosion_dilation", action='store_true', help="Whether to use dilation and erosion")
    # config for path
    parser.add_argument("--model_path", type=str, default="/mnt/jfs-test/lib/Qwen-Image-Edit", help="Path to the pre-trained model")
    parser.add_argument("--evaluation", action='store_true', help="Whether to evaluate the model on the benchmark")
    parser.add_argument("--image_path", type=str, default="assets/data.jsonl", help="Path to the input data")
    parser.add_argument("--output_dir", type=str, default="result/Qwen-Image/Demo/RegionE", help="Directory to save the output images")
    args = parser.parse_args()

    if args.use_regione:
        MANAGER.set_parameters(args)
        pipe = regione_init(args.model_path, args.device)
    else:
        pipe = QwenImageEditPipeline.from_pretrained(args.model_path, torch_dtype=torch.bfloat16).to(args.device)

    if not args.evaluation:
        # Demonstration with a single image (Demo)
        os.makedirs(args.output_dir, exist_ok=True)
        with open(args.image_path, 'r') as f:
            metadata = []
            for data in f:
                metadata.append(json.loads(data))

        # warmup
        for _ in range(3):
            _ = pipe(
                image=Image.open(f'assets/demo0.png').convert("RGB"),
                prompt='just warmup!',
                num_inference_steps=args.num_inference_steps,
                true_cfg_scale=args.guidance_scale,
                negative_prompt=" ",
                generator=torch.Generator("cpu").manual_seed(args.seed),
            ).images[0]

        for index, data in enumerate(metadata):
            print(f"[{index + 1} / {len(metadata)}] Reference Image: {data['key']}.png, Instruction: {data['instruction']}")
            torch.cuda.synchronize()
            t0 = time.time()
            image = pipe(
                image=Image.open(f"{data['key']}.png").convert("RGB"),
                prompt=data["instruction"],
                num_inference_steps=args.num_inference_steps,
                true_cfg_scale=args.guidance_scale,
                negative_prompt=" ",
                generator=torch.Generator("cpu").manual_seed(args.seed),
            ).images[0]
            torch.cuda.synchronize()
            t1 = time.time()

            print(f"Time consuming: {t1-t0}s")
            image.save(f"{args.output_dir}/{data['key']}.png")
            print(f"Image has been saved to {args.output_dir}")

    else:
        # Generate batch images based on metadata (for evaluation)
        for task in os.listdir(args.image_path):
            image_path = os.path.join(args.image_path, task)
            output_dir = os.path.join(args.output_dir, task)
            os.makedirs(f'{output_dir}/generation', exist_ok=True)

            with open(f'{image_path}/metadata.jsonl', 'r') as f:
                metadata = []
                for line in f:
                    metadata.append(json.loads(line))

            # warmup
            for _ in range(3):
                _ = pipe(
                    image=Image.open('assets/demo_0.png').convert("RGB"),
                    prompt='just warmup!',
                    num_inference_steps=args.num_inference_steps,
                    true_cfg_scale=args.guidance_scale,
                    negative_prompt=" ",
                    generator=torch.Generator("cpu").manual_seed(args.seed),
                ).images[0]

            prefix_prompt = dict()
            time_consuming = []
            for idx, data in enumerate(metadata):
                input_image = Image.open(f'{image_path}/img/{data["key"]}.png').convert("RGB")
                prompt = data['instruction']
                print(f"prompt:{prompt}")

                torch.cuda.synchronize()
                t0 = time.time()
                image = pipe(
                    image=input_image,
                    prompt=prompt,
                    num_inference_steps=args.num_inference_steps,
                    true_cfg_scale=args.guidance_scale,
                    negative_prompt=" ",
                    generator=torch.Generator("cpu").manual_seed(args.seed),
                ).images[0]
                torch.cuda.synchronize()
                t1 = time.time()

                prefix_prompt[data["key"]] = prompt
                time_consuming.append(t1-t0)
                image.save(f'{output_dir}/generation/{data["key"]}.png')
                print(f'[task:{task} {idx+1}/{len(metadata)}] {output_dir}/generation/{data["key"]}.png, save! cosuming:{t1 - t0}s')

            time_consuming_dict = {"num_item": len(time_consuming), "ave_time_consuming": sum(time_consuming) / len(time_consuming), "time_consuming_list": time_consuming}
            with open(f'{output_dir}/time_consuming.json', 'w') as f:
                json.dump(time_consuming_dict, f, indent=4)

            with open(f'{output_dir}/metadata.json', 'w') as f:
                json.dump(prefix_prompt, f, indent=4)
