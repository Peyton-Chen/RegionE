import os
import json
from pathlib import Path
from typing import Dict, List, Any
from datasets import load_from_disk
from PIL import Image
from tqdm import tqdm


# # download Kontext Bench
# # huggingface-cli download --repo-type dataset --resume-download black-forest-labs/kontext-bench --local-dir data/Kontext-Bench

# # download GEdit Bench
# # huggingface-cli download --repo-type dataset --resume-download GEdit --local-dir data/GEdit-Bench


class DatasetProcessor:
    """Base class for dataset processing"""
    
    def __init__(self, base_output_dir: str = 'data/Processed'):
        self.base_output_dir = Path(base_output_dir)
    
    def create_directories(self, dataset_name: str, tasks: List[str]):
        """Create directory structure for all tasks in batch"""
        for task in tasks:
            img_dir = f"{self.base_output_dir}/{dataset_name}/{task}/img"
            os.makedirs(img_dir, exist_ok=True)
    
    def save_metadata(self, dataset_name: str, task: str, items: List[Dict[str, Any]]):
        """Save metadata to JSONL file"""
        metadata_path = f"{self.base_output_dir}/{dataset_name}/{task}/metadata.jsonl"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            for item in items:
                if dataset_name == 'Kontext-Bench':
                    item.pop('file_name')
                elif dataset_name == 'GEdit-Bench/en':
                    item.pop('input_image_raw')
                    item.pop('input_image')
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    def save_image(self, image: Image.Image, dataset_name: str, task: str, key: str):
        """Save image file"""
        img_path = f"{self.base_output_dir}/{dataset_name}/{task}/img/{key}.png"
        image.save(img_path)


class KontextBenchProcessor(DatasetProcessor):
    """Processor for Kontext Bench dataset"""
    
    CATEGORY_MAPPING = {
        'Character Reference': 'CR',
        'Style Reference': 'SR',
        'Instruction Editing - Global': 'IEG',
        'Text Editing': 'TE',
        'Instruction Editing - Local': 'IEL'
    }
    
    def __init__(self, data_dir: str = 'data/Kontext-Bench', **kwargs):
        super().__init__(**kwargs)
        self.data_dir = Path(data_dir)
        self.dataset_name = 'Kontext-Bench'
    
    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load Kontext Bench dataset"""
        print("Loading Kontext Bench dataset...")
        metadata_path = f"{self.data_dir}/test/metadata.jsonl"
        
        dataset = []
        with open(metadata_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                img_path = f"{self.data_dir}/test/{data['file_name']}"
                data['file_name'] = Image.open(img_path)
                dataset.append(data)
        
        print(f"Dataset length: {len(dataset)}")
        return dataset
    
    def process(self):
        """Process Kontext Bench dataset"""
        dataset = self.load_dataset()
        
        # Create directories
        tasks = list(self.CATEGORY_MAPPING.values())
        self.create_directories(self.dataset_name, tasks)
        
        # Classify by task
        task_dict = {task: [] for task in tasks}
        
        for item in tqdm(dataset, desc="Processing Kontext Bench"):
            category = item.get('category')
            task_abbr = self.CATEGORY_MAPPING.get(category)
            
            if task_abbr:
                task_dict[task_abbr].append(item)
                self.save_image(item['file_name'], self.dataset_name, task_abbr, item['key'])
            else:
                print(f"Unknown category: {category} for key: {item['key']}")
        
        # Save metadata
        for task, items in task_dict.items():
            self.save_metadata(self.dataset_name, task, items)
        
        print(f"Kontext Bench processing complete. Processed {len(dataset)} items.")


class GEditBenchProcessor(DatasetProcessor):
    """Processor for GEdit Bench dataset"""
    
    TASK_TYPES = [
        'motion_change', 'ps_human', 'color_alter', 'material_alter',
        'subject-add', 'subject-remove', 'style_change', 'tone_transfer',
        'subject-replace', 'text_change', 'background_change'
    ]
    
    def __init__(self, data_dir: str = 'data/GEdit-Bench', language: str = 'en', **kwargs):
        super().__init__(**kwargs)
        self.data_dir = Path(data_dir)
        self.language = language
        self.dataset_name = f'GEdit-Bench/{language}'
    
    def load_dataset(self):
        """Load GEdit Bench dataset"""
        print("Loading GEdit Bench dataset...")
        dataset = load_from_disk(str(self.data_dir))
        print(f"Dataset length: {len(dataset)}")
        return dataset
    
    def process(self):
        """Process GEdit Bench dataset"""
        dataset = self.load_dataset()
        
        # Create directories
        self.create_directories(self.dataset_name, self.TASK_TYPES)
        
        # Classify by task
        task_dict = {task: [] for task in self.TASK_TYPES}
        
        for item in tqdm(dataset, desc="Processing GEdit Bench"):
            # Only process specified language
            if item.get('instruction_language') != self.language:
                continue
            
            task_type = item.get('task_type')
            
            if task_type in self.TASK_TYPES:
                task_dict[task_type].append(item)
                self.save_image(item['input_image'], self.dataset_name, task_type, item['key'])
            else:
                print(f"Unknown task_type: {task_type} for key: {item['key']}")
        
        # Save metadata
        for task, items in task_dict.items():
            self.save_metadata(self.dataset_name, task, items)
        
        total_processed = sum(len(items) for items in task_dict.values())
        print(f"GEdit Bench processing complete. Processed {total_processed} items.")


def main():
    """Main entry point"""
    # Process Kontext Bench
    kontext_processor = KontextBenchProcessor()
    kontext_processor.process()
    
    print("\n" + "="*50 + "\n")
    
    # Process GEdit Bench
    gedit_processor = GEditBenchProcessor()
    gedit_processor.process()


if __name__ == '__main__':
    main()