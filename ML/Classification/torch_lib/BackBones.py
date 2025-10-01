import torch
import torch.nn as nn
import torchvision
from transformers import ResNetModel
import timm

class BackboneWrapper(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        raise NotImplementedError
    
    def freeze_layers(self, freeze=True, freeze_until=None):
        """
        freeze: bool - заморозить или разморозить
        freeze_until: int или None - заморозить первые N слоев (по индексу)
        """
        if freeze_until is None:
            for param in self.parameters():
                param.requires_grad = not freeze
        else:
            children = list(self.children())
            for i, child in enumerate(children):
                for param in child.parameters():
                    param.requires_grad = not (freeze and i < freeze_until)
    
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

class MobileTransformerBackbone(BackboneWrapper):
    def __init__(self, model_name: str, pretrained: bool = True):
        super().__init__()
        self.model_name = model_name
        mobile_transformers = {
            'mobilevit_xxs': 320,
            'mobilevit_xs': 384,
            'mobilevit_s': 640,
        }
        if model_name.lower() not in mobile_transformers:
            raise ValueError(f"Unsupported model: {model_name}")
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.feature_size = mobile_transformers[model_name.lower()]

    def forward(self, x):
        return self.model(x)

class SwinBackbone(BackboneWrapper):
    def __init__(self, model_name='swin_tiny_patch4_window7_224', pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.feature_size = 768  # для swin tiny
    def forward(self, x):
        return self.model(x)

class DeiTBackbone(BackboneWrapper):
    def __init__(self, model_name='deit_small_patch16_224', pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.feature_size = 384  # DeiT-Small
    def forward(self, x):
        return self.model(x)

