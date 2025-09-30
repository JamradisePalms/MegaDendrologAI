import torch
from PIL import Image, ImageFile
from typing import List, Tuple, Any, Callable, Dict, Iterable


class ImageCollator:
    def __init__(self, processor, task_names):
        self.processor = processor
        self.task_names = task_names

    def __call__(self, batch):
        images, labels_dict = zip(*batch)
        images = list(images)

        inputs = self.processor(images, return_tensors="pt")

        labels = {}
        for task_name in self.task_names:
            task_labels = [label_dict[task_name] for label_dict in labels_dict]
            labels[task_name] = torch.tensor(task_labels, dtype=torch.long)

        inputs["labels"] = labels
        return inputs

    