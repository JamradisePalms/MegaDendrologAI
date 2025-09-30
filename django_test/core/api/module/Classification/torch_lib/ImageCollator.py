import torch
from PIL import Image, ImageFile
from typing import List, Tuple, Any, Callable, Dict


class ImageCollator:
    def __init__(
        self,
        processor: Callable[[List[ImageFile], Any], Dict[str, torch.Tensor]],
        task_names: List[str]
    ):
        self.processor = processor
        self.task_names = task_names

    def __call__(self, batch: List[Tuple[Image, Dict[str, int]]]):
        images, labels_dict = zip(*batch)
        inputs = self.processor(images, return_tensors="pt")
        
        labels = {}
        for task_name in self.task_names:
            task_labels = [label_dict[task_name] for label_dict in labels_dict]
            labels[task_name] = torch.tensor(task_labels)
        
        inputs["labels"] = labels
        return inputs
    