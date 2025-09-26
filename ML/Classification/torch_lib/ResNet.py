import torch
import torch.nn as nn
from transformers import ResNetForImageClassification

class ResNet(nn.Module):
    def __init__(self, num_labels: int = 2):
        super().__init__()
        # Made dynamic number of labels for reusing resnet on different tree classifications
        self.num_labels = num_labels

        self.resnet = ResNetForImageClassification.from_pretrained(
            'microsoft/resnet-50',
            num_labels=self.num_labels,
            ignore_mismatched_sizes=True
        )

    def forward(self, pixel_values: torch.Tensor):
        outputs = self.resnet(pixel_values=pixel_values)
        return outputs.logits
