import torch
import torch.nn as nn
from typing import Dict, List, Union, Optional
from ML.Classification.torch_lib.ResNet import ResNetBackbone

class ResNetWrapper(nn.Module):
    def __init__(
        self, 
        resnet_model: str,
        hidden_size: int = 256,
        num_output_features: Union[int, Dict[str, int]] = 2,
        num_hidden_layers: int = 1,
        dropout: float = 0.1,
        activation: str = "relu",
        freeze_resnet: bool = False,
    ):
        super(ResNetWrapper, self).__init__()
        
        self.resnet = ResNetBackbone(model_name=resnet_model)
        
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            dummy_output = self.resnet(dummy_input)
            dummy_output = dummy_output.view(dummy_output.size(0), -1)
            self.feature_size = dummy_output.shape[-1]
        
        if freeze_resnet:
            for param in self.resnet.parameters():
                param.requires_grad = False
        
        self.task_names = []
        self.output_layers = nn.ModuleDict()
        
        if isinstance(num_output_features, int):
            self.task_names = ['main']
            self.output_layers['main'] = self._create_mlp(
                input_size=self.feature_size,
                hidden_size=hidden_size,
                output_size=num_output_features,
                num_hidden_layers=num_hidden_layers,
                dropout=dropout,
                activation=activation
            )
        else:
            self.task_names = list(num_output_features.keys())
            for task_name, num_classes in num_output_features.items():
                self.output_layers[task_name] = self._create_mlp(
                    input_size=self.feature_size,
                    hidden_size=hidden_size,
                    output_size=num_classes,
                    num_hidden_layers=num_hidden_layers,
                    dropout=dropout,
                    activation=activation
                )
    
    def _create_mlp(
        self, 
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_hidden_layers: int,
        dropout: float,
        activation: str
    ) -> nn.Sequential:
        layers = []
        current_input_size = input_size
        
        for i in range(num_hidden_layers):
            layers.extend([
                nn.Linear(current_input_size, hidden_size),
                self._get_activation(activation),
                nn.Dropout(dropout)
            ])
            current_input_size = hidden_size
        
        layers.append(nn.Linear(current_input_size, output_size))
        
        return nn.Sequential(*layers)
    
    def _get_activation(self, activation: str) -> nn.Module:
        activations = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(0.1),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid()
        }
        return activations.get(activation, nn.ReLU())
    
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        features = self.resnet(x)
        if features.dim() == 4:  # [batch, channels, height, width]
            features = features.view(features.size(0), -1)

        if len(self.task_names) == 1:
            return self.output_layers[self.task_names[0]](features)
        else:
            return {task_name: self.output_layers[task_name](features) 
                   for task_name in self.task_names}
    
    def unfreeze_resnet(self, unfreeze: bool = True):
        for param in self.resnet.parameters():
            param.requires_grad = unfreeze