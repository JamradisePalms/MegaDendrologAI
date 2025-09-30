import json
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from typing import List, Dict, Iterable
import pickle

class BaseDataset(Dataset):
    def __init__(self):
        self.samples = []
        self.label_encoders = {}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, labels_dict = self.samples[index]
        image_encoded = Image.open(image_path).convert('RGB')
        return image_encoded, labels_dict

class ImageDatasetJson(BaseDataset):
    def __init__(
        self, 
        json_path: Path, 
        image_filepath_field: str, 
        target_fields: Iterable[str],
        label_encoders: Dict[str, LabelEncoder] = None
    ):
        self.json_path = json_path
        self.image_filepath_field = image_filepath_field
        self.target_fields = target_fields
        self.samples = []
        
        with open(self.json_path, 'r', encoding='utf-8') as f:
            images_dict = json.load(f)
        
        field_values = {field: [] for field in self.target_fields}
        for image_classified in images_dict:
            for field in self.target_fields:
                field_values[field].append(image_classified[field])
        
        if label_encoders is None:
            self.label_encoders = {}
            for field in self.target_fields:
                le = LabelEncoder()
                le.fit(field_values[field])
                self.label_encoders[field] = le
        else:
            self.label_encoders = label_encoders
        
        for image_classified in images_dict:
            image_filepath = image_classified[self.image_filepath_field]
            
            labels_dict = {}
            for field in self.target_fields:
                value = image_classified[field]
                encoded_value = self.label_encoders[field].transform([value])[0]
                labels_dict[field] = encoded_value
            
            self.samples.append((image_filepath, labels_dict))
    
    def save_label_encoders(self, filepath: Path):
        """Сохраняет LabelEncoder'ы для использования при инференсе"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.label_encoders, f)