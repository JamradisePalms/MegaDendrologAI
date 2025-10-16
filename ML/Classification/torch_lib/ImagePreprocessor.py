import albumentations as A
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch
from transformers import AutoImageProcessor, MobileViTImageProcessor
from torchvision.models import (
    EfficientNet_B0_Weights,
    EfficientNet_B1_Weights,
    EfficientNet_B3_Weights,
    EfficientNet_B4_Weights,
)
from typing import Optional


class TreeImagePreprocessor:
    """
    Preprocessor that applies albumentations (optional) and then backbone-specific base transforms.
    Always returns a torch.Tensor with shape [C, H, W].
    """

    def __init__(
        self,
        model_name: str,
        backbone_type: str,
        image_size: int = 224,
        is_train: bool = False,
        augmentation_strength: str = "strong",  # "default" or "strong"
    ):
        self.model_name = model_name
        self.backbone_type = backbone_type.lower()
        self.image_size = image_size
        self.is_train = is_train
        self.augmentation_strength = augmentation_strength

        self._is_hf_processor = False

        self.base_transform = self._get_base_transform()
        self.augmentations = self._get_augmentations(strength=augmentation_strength) if is_train else None

    def _get_base_transform(self):
        """Return either a torchvision transform (callable) or a HuggingFace ImageProcessor instance.
           Also sets self._is_hf_processor when appropriate.
        """
        if self.backbone_type == "efficientnet":
            name_low = self.model_name.lower()
            try:
                if "b0" in name_low:
                    return EfficientNet_B0_Weights.DEFAULT.transforms()
                elif "b1" in name_low:
                    return EfficientNet_B1_Weights.DEFAULT.transforms()
                elif "b3" in name_low:
                    return EfficientNet_B3_Weights.DEFAULT.transforms()
                elif "b4" in name_low:
                    return EfficientNet_B4_Weights.DEFAULT.transforms()
            except Exception:
                pass

            return transforms.Compose(
                [
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )

        if self.backbone_type == "resnet":
            try:
                proc = AutoImageProcessor.from_pretrained(self.model_name)
                self._is_hf_processor = True
                return proc
            except Exception:
                return transforms.Compose(
                    [
                        transforms.Resize((self.image_size, self.image_size)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ]
                )

        if self.backbone_type == "mobilevit":
            mobilevit_mapping = {
                "mobilevit_xxs": "apple/mobilevit-xx-small",
                "mobilevit_xs": "apple/mobilevit-x-small",
                "mobilevit_s": "apple/mobilevit-small",
            }
            hf_model_name = mobilevit_mapping.get(self.model_name.lower(), "apple/mobilevit-small")
            try:
                proc = MobileViTImageProcessor.from_pretrained(hf_model_name)
                self._is_hf_processor = True
                return proc
            except Exception:
                return transforms.Compose(
                    [
                        transforms.Resize((self.image_size, self.image_size)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ]
                )

        return transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _get_augmentations(self, strength: str = "default"):
        """Return albumentations Compose. 'strength' can be 'default' or 'strong'."""
        if strength == "default":
            return A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.OneOf(
                        [
                            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
                            A.ToGray(p=0.7),
                        ],
                        p=0.8,
                    ),
                    A.OneOf([A.GaussianBlur(blur_limit=(1, 3)), A.GaussNoise(var_limit=(10, 50))], p=0.5),
                    A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.5),
                ]
            )

        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomRotate90(p=0.3),
                A.Rotate(limit=20, p=0.4),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=0, p=0.4),
                A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.5),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
                A.RandomGamma(gamma_limit=(80, 120), p=0.2),
                A.OneOf(
                    [A.GaussianBlur(blur_limit=(3, 5)), A.MotionBlur(blur_limit=5), A.MedianBlur(blur_limit=3)],
                    p=0.3,
                ),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
                A.CoarseDropout(
                    max_holes=8, max_height=30, max_width=30, min_holes=4, min_height=15, min_width=15, p=0.3
                ),
            ]
        )

    def _pil_to_clean_numpy(self, image: Image.Image) -> np.ndarray:
        """Convert PIL Image to RGB uint8 numpy array with 3 channels (H, W, C)."""
        if not isinstance(image, Image.Image):
            raise TypeError("Expected PIL.Image.Image")

        if image.mode == "RGB":
            img = image
        elif image.mode == "L":
            img = image.convert("RGB")
        elif image.mode == "RGBA":
            img = image.convert("RGB")
        else:
            img = image.convert("RGB")

        arr = np.array(img)

        if arr.dtype != np.uint8:
            if np.issubdtype(arr.dtype, np.floating):
                arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)

        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.shape[-1] == 4:
            arr = arr[..., :3]

        return arr
    
    def _apply_albumentations(self, image: Image.Image) -> Image.Image:
        """Apply albumentations and return PIL.Image in RGB (uint8)."""
        image_np = self._pil_to_clean_numpy(image)
        augmented = self.augmentations(image=image_np)
        augmented_np = augmented["image"]

        return Image.fromarray(augmented_np.astype(np.uint8))

    def _apply_base_transform(self, image: Image.Image) -> torch.Tensor:
        """Apply base_transform and return torch.Tensor [C, H, W]."""
        if self._is_hf_processor:
            result = self.base_transform(images=image, return_tensors="pt")
            if isinstance(result, dict) and "pixel_values" in result:
                pix = result["pixel_values"]
                if isinstance(pix, torch.Tensor):
                    return pix.squeeze(0)  # [1,C,H,W] -> [C,H,W]
            if "pixel_values" in result:
                val = result["pixel_values"]
                if isinstance(val, (list, tuple)):
                    t = torch.as_tensor(val[0])
                    if t.ndim == 3:
                        return t.permute(2, 0, 1)  # HWC -> CHW
            raise RuntimeError("HF processor returned unexpected structure")

        if isinstance(self.base_transform, transforms.Compose):
            out = self.base_transform(image)
            if isinstance(out, torch.Tensor):
                return out
            if isinstance(out, Image.Image):
                return transforms.ToTensor()(out)
            return torch.as_tensor(out)

        if callable(self.base_transform):
            out = self.base_transform(image)
            if isinstance(out, torch.Tensor):
                return out
            if isinstance(out, dict) and "pixel_values" in out:
                pv = out["pixel_values"]
                if isinstance(pv, torch.Tensor):
                    return pv.squeeze(0)
            if isinstance(out, Image.Image):
                return transforms.ToTensor()(out)
            try:
                arr = np.asarray(out)
                if arr.ndim == 3 and arr.shape[-1] in (1, 3):
                    arr = arr.astype(np.float32) / 255.0
                    t = torch.from_numpy(arr).permute(2, 0, 1)
                    return t
            except Exception:
                pass

        fallback = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        return fallback(image)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        """Process the input PIL image and return torch.Tensor [C,H,W]."""
        if not isinstance(image, Image.Image):
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image.astype(np.uint8))
            else:
                raise TypeError("Input must be PIL.Image.Image or numpy.ndarray")

        if image.mode != "RGB":
            image = image.convert("RGB")

        if self.is_train and self.augmentations is not None:
            image = self._apply_albumentations(image)

        return self._apply_base_transform(image)

    def set_train_mode(self, is_train: bool = True):
        """Switch train mode and (re)create augmentations if needed."""
        self.is_train = is_train
        if is_train and self.augmentations is None:
            self.augmentations = self._get_augmentations(strength=self.augmentation_strength)
        if not is_train:
            self.augmentations = None
