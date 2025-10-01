import albumentations as A
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from transformers import AutoImageProcessor
from torchvision.models import EfficientNet_B0_Weights, EfficientNet_B1_Weights, EfficientNet_B3_Weights, EfficientNet_B4_Weights
from typing import Optional

class TreeImagePreprocessor:
    def __init__(self, model_name: str, backbone_type: str, image_size: int = 224, is_train: bool = False):
        self.model_name = model_name
        self.backbone_type = backbone_type.lower()
        self.image_size = image_size
        self.is_train = is_train
        
        self.base_transform = self._get_base_transform()
        self.augmentations = self._get_strong_augmentations() if is_train else None
        
    def _get_base_transform(self):
        if self.backbone_type == "efficientnet":
            if 'b0' in self.model_name.lower():
                return EfficientNet_B0_Weights.DEFAULT.transforms()
            elif 'b1' in self.model_name.lower():
                return EfficientNet_B1_Weights.DEFAULT.transforms()
            elif 'b3' in self.model_name.lower():
                return EfficientNet_B3_Weights.DEFAULT.transforms()
            elif 'b4' in self.model_name.lower():
                return EfficientNet_B4_Weights.DEFAULT.transforms()
            else:
                return transforms.Compose([
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ])
        
        elif self.backbone_type == "resnet":
            try:
                image_processor = AutoImageProcessor.from_pretrained(self.model_name, use_fast=True)
                return image_processor
            except:
                return transforms.Compose([
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ])
        
        else:
            return transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
    
    def _get_augmentations(self):
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.OneOf([
                A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
                A.ToGray(p=0.7),
            ], p=0.8),
            A.OneOf([
                A.GaussianBlur(blur_limit=(1, 3)),
                A.GaussNoise(var_limit=(10, 50)),
            ], p=0.5),
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.5),
        ])
    
    def _get_strong_augmentations(self):
        return A.Compose([
            A.HorizontalFlip(p=0.5), 
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.3), 
            A.Rotate(limit=20, p=0.4), 
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=0, p=0.4),

            A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
            A.RandomGamma(gamma_limit=(80, 120), p=0.2),

            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5)),
                A.MotionBlur(blur_limit=5),
                A.MedianBlur(blur_limit=3),
            ], p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),

            A.CoarseDropout(
                max_holes=8, max_height=30, max_width=30, 
                min_holes=4, min_height=15, min_width=15, p=0.3
            ),
        ])

    
    def _apply_albumentations(self, image: Image.Image) -> Image.Image:
        image_np = np.array(image)
        
        if image_np.dtype != np.uint8:
            image_np = image_np.astype(np.uint8)
            
        augmented = self.augmentations(image=image_np)
        augmented_image_np = augmented['image']
        
        return Image.fromarray(augmented_image_np.astype(np.uint8))

    def _apply_base_transform(self, image: Image.Image):
        if isinstance(self.base_transform, transforms.Compose):
            return self.base_transform(image)
        elif hasattr(self.base_transform, '__call__'):
            result = self.base_transform(image)
            if isinstance(result, dict) and 'pixel_values' in result:
                return result['pixel_values'].squeeze(0)
            return result
        else:
            return self.base_transform(image, return_tensors="pt")['pixel_values'].squeeze(0)
    
    def __call__(self, image: Image.Image):
        if self.is_train and self.augmentations is not None:
            image = self._apply_albumentations(image)
        
        return self._apply_base_transform(image)
    
    def set_train_mode(self, is_train: bool = True):
        self.is_train = is_train
        if is_train and self.augmentations is None:
            self.augmentations = self._get_augmentations()