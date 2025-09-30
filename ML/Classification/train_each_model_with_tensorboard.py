import warnings
warnings.filterwarnings("ignore", message="Some weights of")

import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from ML.Classification.torch_lib.ImageDataset import ImageDatasetJson
from ML.Classification.torch_lib.ImageCollator import ImageCollator
from ML.Classification.torch_lib.ResNet import ResNetBackbone, ResNet
from configs.train_config_classification import TrainConfigs
import json
from pathlib import Path
import os

CURRENT_CONFIG = TrainConfigs.TreeClassificationModelWithMultiHeadMLP
device = 'cuda' if torch.cuda.is_available() else 'cpu'

log_dir = CURRENT_CONFIG.PATH_TO_SAVE_MODEL.parent.parent / 'tensorboard_logs'
log_dir.mkdir(exist_ok=True)

writer = SummaryWriter(log_dir=str(log_dir))

task_to_output_size = CURRENT_CONFIG.TASK_TO_OUTPUT_SIZE

task_to_model = {
    task: ResNet(output_size=output_size, model_name='microsoft/resnet-18').to(device).train()
    for task, output_size in task_to_output_size.items()
}

train_dataset = ImageDatasetJson(
    CURRENT_CONFIG.TRAIN_JSON_FILEPATH,
    CURRENT_CONFIG.IMAGE_JSON_FIELD,
    target_fields=list(task_to_output_size.keys())
)

encoders_path = CURRENT_CONFIG.PATH_TO_SAVE_MODEL.parent / 'label_encoders.pkl'
train_dataset.save_label_encoders(encoders_path)

data_collator = ImageCollator(
    CURRENT_CONFIG.IMAGE_PROCESSOR,
    task_names=list(task_to_output_size.keys())
)

train_loader = DataLoader(
    train_dataset,
    batch_size=CURRENT_CONFIG.BATCH_SIZE,
    shuffle=True,
    collate_fn=data_collator,
    num_workers=0,
)

task_to_criterion = {
    task_name: nn.CrossEntropyLoss() 
    for task_name in task_to_output_size.keys()
}

task_to_optimizer = {
    task: optim.AdamW(model.parameters(), lr=CURRENT_CONFIG.LR)
    for task, model in task_to_model.items()
}

global_step = 0

for epoch in range(CURRENT_CONFIG.NUM_EPOCHS):
    progress_bar = tqdm(
        train_loader, desc=f'Epoch: {epoch + 1} of {CURRENT_CONFIG.NUM_EPOCHS}'
    )
    
    for batch in progress_bar:
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['labels']
        labels = {k: v.to(device) for k, v in labels.items()}
        for task, model in task_to_model.items():
            optimizer = task_to_optimizer[task]
            criterion = task_to_criterion[task]
            optimizer.zero_grad()
            output = model(pixel_values)
            
            total_loss = 0.0
            task_losses_dict = {}
            
            task_loss = criterion(
                output,
                labels[task]
            )
            
            task_loss.backward()
            optimizer.step()

            global_step += 1
            writer.add_scalar(f'Loss/{task}_Batch', task_loss.item(), global_step)

torch.save(model.state_dict(), CURRENT_CONFIG.PATH_TO_SAVE_MODEL)

writer.close()

print(f'Model saved to: {CURRENT_CONFIG.PATH_TO_SAVE_MODEL}')
print(f'Label encoders saved to: {encoders_path}')
print(f'TensorBoard logs saved to: {log_dir}')
print(f'To view TensorBoard, run: tensorboard --logdir={log_dir}')
