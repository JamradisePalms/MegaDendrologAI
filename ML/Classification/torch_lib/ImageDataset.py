import json
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
import torch
from sklearn.preprocessing import LabelEncoder
from typing import List, Dict, Callable
import pickle

class BaseDataset(Dataset):
    def __init__(self, preprocessor):
        self.samples = []
        self.label_encoders = {}
        self.preprocessor = preprocessor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, labels_dict = self.samples[index]
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.preprocessor(image)
        return image_tensor, labels_dict

class ImageDatasetJson(BaseDataset):
    def __init__(
        self, 
        json_path: Path, 
        image_filepath_field: str, 
        target_fields: List[str],
        preprocessor: Callable,
        label_mappings: Dict[str, Dict[str, int]] = None,
    ):
        super().__init__(preprocessor)
        self.json_path = json_path
        self.image_filepath_field = image_filepath_field
        self.target_fields = target_fields
        self.samples = []
        
        with open(self.json_path, 'r', encoding='utf-8') as f:
            images_dict = json.load(f)
        
        field_values = {field: set() for field in self.target_fields}
        for image_classified in images_dict:
            for field in self.target_fields:
                field_values[field].add(image_classified[field])
        
        if label_mappings is None:
            self.label_mappings = {}
            for field in self.target_fields:
                unique_values = sorted(list(field_values[field]))
                mapping = {value: idx for idx, value in enumerate(unique_values)}
                self.label_mappings[field] = mapping
        else:
            self.label_mappings = label_mappings

        self.reverse_mappings = {}
        for field, mapping in self.label_mappings.items():
            self.reverse_mappings[field] = {v: k for k, v in mapping.items()}
        
        for image_classified in images_dict:
            image_filepath = image_classified[self.image_filepath_field]
            
            labels_dict = {}
            for field in self.target_fields:
                value = image_classified[field]
                if value in self.label_mappings[field]:
                    encoded_value = self.label_mappings[field][value]
                else:
                    new_id = len(self.label_mappings[field])
                    self.label_mappings[field][value] = new_id
                    self.reverse_mappings[field][new_id] = value
                    encoded_value = new_id
                
                labels_dict[field] = encoded_value
            
            self.samples.append((image_filepath, labels_dict))
    
    def encode_value(self, field: str, value: str) -> int:
        """Кодирует строковое значение в числовой ID"""
        if field not in self.label_mappings:
            raise ValueError(f"Field {field} not found in label mappings")
        return self.label_mappings[field].get(value, -1)
    
    def decode_value(self, field: str, encoded_value: int) -> str:
        """Декодирует числовой ID обратно в строковое значение"""
        if field not in self.reverse_mappings:
            raise ValueError(f"Field {field} not found in reverse mappings")
        return self.reverse_mappings[field].get(encoded_value, "UNKNOWN")
    
    def get_num_classes(self, field: str) -> int:
        """Возвращает количество классов для указанного поля"""
        if field not in self.label_mappings:
            raise ValueError(f"Field {field} not found in label mappings")
        return len(self.label_mappings[field])
    
    def save_label_mappings(self, filepath: Path):
        """Сохраняет маппинги в pickle файл"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'label_mappings': self.label_mappings,
                'reverse_mappings': self.reverse_mappings
            }, f)
    
    @classmethod
    def load_label_mappings(cls, filepath: Path) -> Dict:
        """Загружает маппинги из pickle файла"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)

