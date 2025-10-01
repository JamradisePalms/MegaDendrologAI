import torch
import torch.nn as nn
import torchvision
from transformers import ResNetModel

class BackboneWrapper(nn.Module):
    def __init__(self):
        super(BackboneWrapper, self).__init__()
    
    def forward(self, x):
        raise NotImplementedError
    
class EfficientNetBackbone(BackboneWrapper):
    def __init__(self, model_name: str, pretrained: bool = True):
        super(EfficientNetBackbone, self).__init__()
        
        efficientnet_versions = {
            'efficientnet-b0': (torchvision.models.efficientnet_b0, 1280),
            'efficientnet-b1': (torchvision.models.efficientnet_b1, 1280),
            'efficientnet-b2': (torchvision.models.efficientnet_b2, 1408),
            'efficientnet-b3': (torchvision.models.efficientnet_b3, 1536),
            'efficientnet-b4': (torchvision.models.efficientnet_b4, 1792),
            'efficientnet-b5': (torchvision.models.efficientnet_b5, 2048),
            'efficientnet-b6': (torchvision.models.efficientnet_b6, 2304),
            'efficientnet-b7': (torchvision.models.efficientnet_b7, 2560),
        }
        
        if model_name.lower() in efficientnet_versions:
            model_fn, self.feature_size = efficientnet_versions[model_name.lower()]
            self.model = model_fn(pretrained=pretrained)
            self.model.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported EfficientNet model: {model_name}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    

class ResNetBackbone(nn.Module):
    def __init__(self, model_name: str = 'microsoft/resnet-50'):
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
