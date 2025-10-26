# EVAL

Our evaluation process consists of the following steps:
1. Prepare the Environment and Dataset
   - Install required dependencies:
     ```bash
     conda env create -f qwen25vl_environment.yml
     conda activate qwen25vl
     ```
   - Then download our dataset stepfun-ai/GEdit-Bench:
     ```python
     from datasets import load_dataset
     dataset = load_dataset("stepfun-ai/GEdit-Bench")
     ```

2. Generate and Organize Your Images
   - Generate and Organize your generated images in the following directory structure (as a reference, you could find the step1x-edit results in [here](https://huggingface.co/datasets/Shiyu95/gedit_results)):
     ```
     results/
     ├── {method_name}/
     │   └── fullset/
     │       └── {edit_task}/
     │           ├── cn/  # Chinese instructions
     │           │   ├── key1.png
     │           │   ├── key2.png
     │           │   └── ...
     │           └── en/  # English instructions
     │               ├── key1.png
     │               ├── key2.png
     │               └── ...
     ```

3. Evaluate using GPT4.1/Qwen2.5VL-72B-Instruct-AWQ
   - For GPT-4.1 evaluation, set up your API keys in secret_t2.env for GPT4.1 access, and run the following command:
     ```bash
     python run_gedit_score.py --model_name your_model --save_dir score_dir --backbone gpt4o --edited_images_dir your_edited_images_dir
     ```
   - For Qwen evaluation:
     ```bash
     python run_gedit_score.py --model_name your_model --save_dir score_dir --backbone qwen25vl --edited_images_dir your_edited_images_dir
     ```

4. Analyze your results and obtain scores across all dimensions
   - Run the analysis script to get scores for semantics, quality, and overall performance:
     ```bash
     python calculate_statistics.py --model_name your_model --save_path score_dir --backbone gpt4o --language all
     ```
   - This will output scores broken down by edit category and provide aggregate metrics

# Notice
We observed that the evaluation scores from GPT-4o exhibit a degree of volatility. Even for the same input image at the same time point, the scores may fluctuate, with a typical variation of around 0.1. To facilitate reproducibility, we have released the [intermediate results](https://huggingface.co/datasets/Shiyu95/gedit_results) of our model evaluations. Evaluating these results using the Qwen model should allow for full reproduction of the values reported in our paper.

As a reference, we report the scores for the quantitative evaluation on GEdit-Bench-EN (Full set). G_SC, G_PQ, and G_O refer to the metrics evaluated by GPT-4.1, while Q_SC, Q_PQ, and Q_O refer to the metrics evaluated by Qwen2.5-VL-72B. All metrics are reported as higher-is-better.

|     Models    |     G_SC ⬆️   |  G_PQ ⬆️ | G_O ⬆️   |  Q_SC ⬆️ | Q_PQ ⬆️   |  Q_O ⬆️ |
|:------------:|:------------:|:------------:| :------------:|:------------:| :------------:|:------------:|
| Step1X-Edit (v1.0)  |    7.13   | 7.00 |   6.44   | 7.39 |    7.28   | 7.07 | 
| Step1X-Edit (v1.1)  |    7.66   | 7.35 |   6.97   | 7.65 |    7.41   | 7.35 | 


# Acknowledgements

This project builds upon and adapts code from the following excellent repositories:

- [VIEScore](https://github.com/TIGER-AI-Lab/VIEScore): A visual instruction-guided explainable metric for evaluating conditional image synthesis

We thank the authors of these repositories for making their code publicly available.

