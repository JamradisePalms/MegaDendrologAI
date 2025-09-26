import json
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(self):
        self.samples = []

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        image_encoded = Image.open(image_path).convert('RGB')
        return image_encoded, label

class ImageDatasetJson(BaseDataset):
    def __init__(self, json_path: Path, image_filepath_field: str, target_field: str):
        self.json_path = json_path
        self.image_filepath_field = image_filepath_field
        self.target_field = target_field
        self.samples = []
        
        with open(self.json_path, 'r', encoding='Utf-8') as f:
            images_dict = json.load(f)
        
        for image_classified in images_dict:
            image_filepath = image_classified[self.image_filepath_field]
            label = image_classified[self.target_field]
            self.samples.append((image_filepath, label))
    
