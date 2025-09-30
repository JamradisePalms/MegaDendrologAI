import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
from ML.Classification.torch_lib.ImageDataset import ImageDatasetJson
from ML.Classification.torch_lib.ImageCollator import ImageCollator
from configs.train_config_classification import TrainConfigs
from ML.Classification.torch_lib.ResNetWrapper import ResNetWrapper
import json
from pathlib import Path

CURRENT_CONFIG = TrainConfigs.TreeClassificationModelWithMultiHeadMLP


task_names = list(CURRENT_CONFIG.TARGET_JSON_FIELD.keys())
num_classes_per_task = CURRENT_CONFIG.TARGET_JSON_FIELD


train_dataset = ImageDatasetJson(
    CURRENT_CONFIG.TRAIN_JSON_FILEPATH,
    CURRENT_CONFIG.IMAGE_JSON_FIELD,
    target_fields=task_names
)


encoders_path = CURRENT_CONFIG.PATH_TO_SAVE_MODEL.parent / "label_encoders.pkl"
train_dataset.save_label_encoders(encoders_path)

data_collator = ImageCollator(
    CURRENT_CONFIG.IMAGE_PROCESSOR,
    task_names=task_names
)

train_loader = DataLoader(
    train_dataset,
    batch_size=CURRENT_CONFIG.BATCH_SIZE,
    shuffle=True,
    collate_fn=data_collator,
    num_workers=0,
)


model = ResNetWrapper(
    resnet_model=CURRENT_CONFIG.MODEL_NAME,
    num_output_features=num_classes_per_task,
    freeze_resnet=False
)
model.train()

criterions = {
    task_name: nn.CrossEntropyLoss() 
    for task_name in task_names
}

optimizer = optim.AdamW(model.parameters(), lr=CURRENT_CONFIG.LR)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

batch_losses = []
epoch_losses = []
task_losses = {task_name: [] for task_name in task_names}

for epoch in range(CURRENT_CONFIG.NUM_EPOCHS):
    progress_bar = tqdm(
        train_loader, desc=f"Epoch: {epoch + 1} of {CURRENT_CONFIG.NUM_EPOCHS}"
    )
    epoch_loss_sum = 0.0
    epoch_batch_count = 0
    
    for batch in progress_bar:
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['labels']
        labels = {k: v.to(device) for k, v in labels.items()}
        
        optimizer.zero_grad()
        logits = model(pixel_values)
        
        total_loss = 0
        task_current_losses = {}
        
        for task_name in task_names:
            task_loss = criterions[task_name](
                logits[task_name], 
                labels[task_name]
            )
            total_loss += task_loss
            task_current_losses[task_name] = task_loss.item()
        
        total_loss.backward()
        optimizer.step()
        
        batch_loss_value = total_loss.item()
        batch_losses.append(batch_loss_value)
        
        for task_name, loss_value in task_current_losses.items():
            task_losses[task_name].append(loss_value)
        
        epoch_loss_sum += batch_loss_value
        epoch_batch_count += 1
        
        loss_info = {f'loss_{task}': f'{loss:.4f}' for task, loss in task_current_losses.items()}
        loss_info['total_loss'] = f'{batch_loss_value:.4f}'
        progress_bar.set_postfix(loss_info)
    
    if epoch_batch_count > 0:
        epoch_avg_loss = epoch_loss_sum / epoch_batch_count
        epoch_losses.append(epoch_avg_loss)

torch.save(model.state_dict(), CURRENT_CONFIG.PATH_TO_SAVE_MODEL)

with open(str(CURRENT_CONFIG.PATH_TO_SAVE_MODEL) + ".losses.json", 'w') as f:
    json.dump({
        'batch_losses': batch_losses,
        'epoch_losses': epoch_losses,
        'task_losses': task_losses,
    }, f, indent=2)

print(f"Label encoders saved to: {encoders_path}")