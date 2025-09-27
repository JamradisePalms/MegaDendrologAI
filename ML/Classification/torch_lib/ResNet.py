import torch
import torch.nn as nn
from transformers import ResNetForImageClassification

class ResNet(nn.Module):
    def __init__(self, num_labels: int = 2, model_name: str = 'microsoft/resnet-50'):
        super().__init__()
        self.num_labels = num_labels
        self.model_name = model_name

        self.resnet = ResNetForImageClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            ignore_mismatched_sizes=True
        )

    def forward(self, pixel_values: torch.Tensor):
        outputs = self.resnet(pixel_values=pixel_values)
        return outputs.logits
