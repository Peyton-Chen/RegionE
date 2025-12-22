
<!-- <div align= "center">
    <h1> Official repo for RegionE</h1>

</div> -->


<h3 align="center">
  <img src="assets/RegionE.gif" width="100" style="vertical-align: middle; margin-right: 10px;">

  <strong>RegionE: Adaptive Region-Aware Generation for Efficient Image Editing</strong>
</h3>

<div align="center">
<a href='https://arxiv.org/abs/2510.25590'><img src='https://img.shields.io/badge/arXiv-2510.25590-b31b1b.svg'></a> &nbsp;&nbsp;&nbsp;&nbsp;
</div>

## 🥳 What's New 
- [2025/12/22] Release the code for Step1X-Edit-v1p2 and Qwen-Image-Edit-2509.
- [2025/10/29] 👋 Upload [paper](https://arxiv.org/abs/2510.25590) and init project. 
RegionE losslessly accelerates SOTA instruction-based image editing models, including Step1X-Edit, FLUX.1 Kontext, and Qwen-Image-Edit, achieving acceleration factors of **2.57×**, **2.41×**, and **2.06×**.


## 🎥 Demo


https://github.com/user-attachments/assets/23cb6eda-6f2e-418d-8638-8de6c6aaf44d




## 🏃 Overview
**RegionE** is an adaptive, region-aware generation framework that accelerates instruction-based image editing tasks without additional training. Specifically, the RegionE framework consists of three main components: **1) Adaptive Region Partition.**
We observed that the trajectory of unedited regions is straight, allowing for multi-step denoised predictions to be inferred in a single step. 
Therefore, in the early denoising stages, we partition the image into edited and unedited regions based on the difference between the final estimated result and the reference image. **2) Region-Aware Generation.**  After distinguishing the regions, we replace multi-step denoising with one-step prediction for unedited areas. 
For edited regions, the trajectory is curved, requiring local iterative denoising. To improve the efficiency and quality of local iterative generation, we propose the Region-Instruction KV Cache, which reduces computational cost while incorporating global information. 
**3) Adaptive Velocity Decay Cache.**
Observing that adjacent timesteps in edited regions exhibit strong velocity similarity, we further propose an adaptive velocity decay cache to accelerate the local denoising process.
<p align="center">
    <img src="assets/pipeline.jpg" alt="Pipeline" width="890px" />
</p>
We applied RegionE to state-of-the-art instruction-based image editing models, including Step1X-Edit, FLUX.1 Kontext, and Qwen-Image-Edit. RegionE achieved acceleration factors of 2.57×, 2.41×, and 2.06×, respectively, with minimal quality loss (PSNR: 30.520–32.133). Evaluations by GPT-4o also confirmed that semantic and perceptual fidelity were well preserved.
<p align="center">
    <img src="assets/result.jpg" alt="Quantitative results" width="890px" />
</p>

<!-- ## 🎥 Demo -->


## 🛠️ Dependencies and Installation
Begin by cloning the repository:
```shell
git clone https://github.com/Peyton-Chen/RegionE.git
cd RegionE
```

We recommend CUDA versions 12.4 or 12.1 for the manual installation.
```shell
# 1. Create conda environment
conda create -n regione python==3.10.18

# 2. Activate the environment
conda activate regione

# 3. Install PyTorch and other dependencies using pip
# For CUDA 12.1
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
# For CUDA 12.4
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# 4. Install pip dependencies
python -m pip install -r requirements.txt

# 5. Install the latest version of diffusers
pip install git+https://github.com/Peyton-Chen/diffusers.git@step1xedit_v1p2

# 6. Install flash attention v2 (optional)
python -m pip install git+https://github.com/Dao-AILab/flash-attention.git@v2.8.2 --no-build-isolation
```

## 🎯 Quick Start
Here is an example for using the RegionE model to different pretrained model:

### Step1X-Edit [🤗[Download Pretrained Model](https://huggingface.co/stepfun-ai/Step1X-Edit-v1p1-diffusers) ] 
```python
import torch
from diffusers import Step1XEditPipeline
from diffusers.utils import load_image
from RegionE import RegionEHelper

# Loading the original pipeline
pipeline = Step1XEditPipeline.from_pretrained("stepfun-ai/Step1X-Edit-v1p1-diffusers", torch_dtype=torch.bfloat16)
pipeline.to("cuda")

# Import the RegionEHelper
regionehelper = RegionEHelper(pipeline)
regionehelper.set_params()   # default hyperparameter
regionehelper.enable()

# Generate Image
image = load_image("demo_0.png").convert("RGB")
prompt = "Replace the text 'SUMMER' with 'WINTER'"
image = pipeline(
    image=image,
    prompt=prompt,
    num_inference_steps=28,
    true_cfg_scale=6.0,
    generator=torch.Generator().manual_seed(42),
).images[0]
image.save("step1xeditv1p1_output_image_edit.jpg")

regionehelper.disable()
```

### Step1X-Edit-v1p2 [🤗[Download Pretrained Model](https://huggingface.co/stepfun-ai/Step1X-Edit-v1p2) ] 
```python
import torch
from diffusers import Step1XEditPipelineV1P2
from diffusers.utils import load_image
from RegionE import RegionEHelper

# Loading the original pipeline
pipeline = Step1XEditPipelineV1P2.from_pretrained("stepfun-ai/Step1X-Edit-v1p2", torch_dtype=torch.bfloat16)
pipeline.to("cuda")

# Import the RegionEHelper
regionehelper = RegionEHelper(pipeline)
regionehelper.set_params()   # default hyperparameter
regionehelper.enable()

# Generate Image
image = load_image("demo_0.png").convert("RGB")
prompt = "Replace the text 'SUMMER' with 'WINTER'"
enable_thinking_mode=True
enable_reflection_mode=True
pipe_output = pipeline(
    image=image,
    prompt=prompt,
    num_inference_steps=28,
    true_cfg_scale=6,
    generator=torch.Generator().manual_seed(42),
    enable_thinking_mode=enable_thinking_mode,
    enable_reflection_mode=enable_reflection_mode,
)
if enable_thinking_mode:
    print("Reformat Prompt:", pipe_output.reformat_prompt)
for image_idx in range(len(pipe_output.images)):
    pipe_output.images[image_idx].save(f"step1xeditv1p2_output_image_edit-{image_idx}.jpg", lossless=True)
    if enable_reflection_mode:
        print(pipe_output.think_info[image_idx])
        print(pipe_output.best_info[image_idx])
pipe_output.final_images[0].save(f"step1xeditv1p2_output_image_edit-final.jpg", lossless=True)

regionehelper.disable()
```

### FLUX.1 Kontext [🤗[Download Pretrained Model](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) ] 
```python
import torch
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image
from RegionE import RegionEHelper

# Loading the original pipeline
pipeline = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16)
pipeline.to("cuda")

# Import the RegionEHelper
regionehelper = RegionEHelper(pipeline)
regionehelper.set_params()   # default hyperparameter
regionehelper.enable()

# Generate Image
input_image = load_image("demo_0.png")
image = pipeline(
  image=input_image,
  prompt="Replace the text 'SUMMER' with 'WINTER'",
  guidance_scale=2.5
).images[0]
image.save("fluxkontext_output_image_edit.png")

regionehelper.disable()
```

### Qwen-Image-Edit [🤗[Download Pretrained Model](https://huggingface.co/Qwen/Qwen-Image-Edit) ] 
```python
import torch
from PIL import Image
from RegionE import RegionEHelper
from diffusers import QwenImageEditPipeline

# Loading the original pipeline
pipeline = QwenImageEditPipeline.from_pretrained("Qwen/Qwen-Image-Edit", torch_dtype=torch.bfloat16)

# Import the RegionEHelper
regionehelper = RegionEHelper(pipeline)
regionehelper.set_params()   # default hyperparameter
regionehelper.enable()

# Generate Image
pipeline.to("cuda")
pipeline.set_progress_bar_config(disable=None)
image = Image.open("demo_0.png").convert("RGB")
prompt = "Replace the text 'SUMMER' with 'WINTER'"
inputs = {
    "image": image,
    "prompt": prompt,
    "generator": torch.manual_seed(0),
    "true_cfg_scale": 4.0,
    "negative_prompt": " ",
    "num_inference_steps": 28,
}

with torch.inference_mode():
    output = pipeline(**inputs)
    output_image = output.images[0]
    output_image.save("qwenimageedit_output_image_edit.png")

regionehelper.disable()
```


### Qwen-Image-Edit-2509 [🤗[Download Pretrained Model](https://huggingface.co/Qwen/Qwen-Image-Edit-2509) ] 
```python
import torch
from PIL import Image
from RegionE import RegionEHelper
from diffusers import QwenImageEditPlusPipeline

# Loading the original pipeline
pipeline = QwenImageEditPlusPipeline.from_pretrained("Qwen/Qwen-Image-Edit-2509", torch_dtype=torch.bfloat16)

# Import the RegionEHelper
regionehelper = RegionEHelper(pipeline)
regionehelper.set_params()   # default hyperparameter
regionehelper.enable()

# Generate Image
pipeline.to('cuda')
pipeline.set_progress_bar_config(disable=None)
image1 = Image.open("demo_0.png")
prompt = "Replace the text 'SUMMER' with 'WINTER'"
inputs = {
    "image": [image1],
    "prompt": prompt,
    "generator": torch.manual_seed(0),
    "true_cfg_scale": 4.0,
    "negative_prompt": " ",
    "num_inference_steps": 28,
    "guidance_scale": 1.0,
    "num_images_per_prompt": 1,
}
with torch.inference_mode():
    output = pipeline(**inputs)
    output_image = output.images[0]
    output_image.save("output_image_edit_plus.png")

regionehelper.disable()
```


## 💫 Acknowledgments
We thank the following excellent open-source works: [Step1X-Edit](https://github.com/stepfun-ai/Step1X-Edit), [FLUX.1 Kontext](https://github.com/black-forest-labs/flux), [Qwen-Image](https://github.com/QwenLM/Qwen-Image) and [RAS](https://github.com/microsoft/RAS).


## 🔗 Citation

```BibTeX
@article{chen2025regione,
  title   = {RegionE: Adaptive Region-Aware Generation for Efficient Image Editing},
  author  = {Pengtao Chen and Xianfang Zeng and Maosen Zhao and Mingzhu Shen and Peng Ye and Bangyin Xiang and Zhibo Wang and Wei Cheng and Gang Yu and Tao Chen},
  journal = {arXiv preprint arXiv:2510.25590},
  year    = {2025}
}
```