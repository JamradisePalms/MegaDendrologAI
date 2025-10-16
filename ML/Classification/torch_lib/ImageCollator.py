import torch
from PIL import Image
from typing import List, Dict

class ImageCollator:
    def __init__(
        self,
        task_names: List[str]
    ):
        self.task_names = task_names

    def __call__(self, batch):
        images, labels_dict = zip(*batch)
        pixel_values = torch.stack(images)
        
        labels = {}
        for task_name in self.task_names:
            task_labels = [label_dict[task_name] for label_dict in labels_dict]
            labels[task_name] = torch.tensor(task_labels)
        
        return {
            "pixel_values": pixel_values,
            "labels": labels
        }
