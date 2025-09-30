import torch
import torch.nn as nn
from transformers import ResNetModel, ResNetForImageClassification


class ResNetBackbone(nn.Module):
    def __init__(self, model_name: str = "microsoft/resnet-50"):
        super().__init__()
        self.model_name = model_name

        self.resnet = ResNetModel.from_pretrained(self.model_name)

        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.forward(dummy_input)
            self.feature_size = features.shape[-1]

    def forward(self, pixel_values: torch.Tensor):
        outputs = self.resnet(pixel_values)
        return outputs.pooler_output


class ResNet(nn.Module):
    def __init__(self, output_size: int = 2, model_name: str = "microsoft/resnet-50"):
        super().__init__()
        self.output_size = output_size
        self.model_name = model_name

        self.resnet = ResNetForImageClassification.from_pretrained(
            self.model_name, num_labels=self.output_size, ignore_mismatched_sizes=True
        )

    def forward(self, pixel_values):
        output = self.resnet(pixel_values=pixel_values)
        return output.logits
