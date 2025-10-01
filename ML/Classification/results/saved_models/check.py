import torch
from ML.Classification.torch_lib.ResNet import ResNet
from transformers import AutoImageProcessor
from configs.train_config_classification import TrainConfigs
from PIL import Image

cfg = TrainConfigs.TreeClassificationModelWithMultiHeadMLP
device = 'cuda' if torch.cuda.is_available() else 'cpu'

num_classes = cfg.TASK_TO_OUTPUT_SIZE['has_hollow']
ckpt_path = 'ML/Classification/results/saved_models/hollow_classification_small_has_hollow_best.pt'

model = ResNet(output_size=num_classes, model_name='microsoft/resnet-18').to(device)
state = torch.load(ckpt_path, map_location=device)
model.load_state_dict(state, strict=True)
model.eval()

image = Image.open('ML/Classification/results/saved_models/debug_images/not_hollow.png')
processor = AutoImageProcessor.from_pretrained('microsoft/resnet-18')
inputs = processor(image, return_tensors="pt")

import torch.nn.functional as F

logits = model(inputs["pixel_values"].to(device))
probs = F.softmax(logits, dim=1)

print("Вероятности:", probs)
print("Класс:", torch.argmax(probs, dim=1).item())