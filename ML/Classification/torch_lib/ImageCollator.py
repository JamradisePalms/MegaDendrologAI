import torch
from PIL import Image, ImageFile
from typing import List, Tuple, Any, Callable, Dict


class ImageCollator:
    def __init__(
        self,
        processor: Callable[[List[ImageFile], Any], Dict[str, torch.Tensor]],
    ):
        self.processor = processor

    def __call__(self, batch: List[Tuple[Image, int]]):
        images, labels = zip(*batch)
        inputs = self.processor(images, return_tensors="pt")
        inputs["labels"] = torch.tensor(labels)

        return inputs
