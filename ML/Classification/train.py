import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
from ML.Classification.torch_lib.ImageDataset import ImageDatasetJson
from ML.Classification.torch_lib.ImageCollator import ImageCollator
from ML.Classification.torch_lib.config import TrainConfigs
from ML.Classification.torch_lib.ResNet import ResNet

CURRENT_CONFIG = TrainConfigs.HollowClassification

train_dataset = ImageDatasetJson(
    CURRENT_CONFIG.TRAIN_JSON_FILEPATH,
    CURRENT_CONFIG.IMAGE_JSON_FIELD,
    CURRENT_CONFIG.TARGET_JSON_FIELD,
)
data_collator = ImageCollator(CURRENT_CONFIG.IMAGE_PROCESSOR)
train_loader = DataLoader(
    train_dataset,
    batch_size=CURRENT_CONFIG.BATCH_SIZE,
    shuffle=True,
    collate_fn=data_collator,
    num_workers=4,
)

model = ResNet(num_labels=CURRENT_CONFIG.NUM_LABELS)
model.train()
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=CURRENT_CONFIG.LR)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

for epoch in range(CURRENT_CONFIG.NUM_EPOCHS):
    progress_bar = tqdm(
        train_loader, desc=f"Epoch: {epoch + 1} of {CURRENT_CONFIG.NUM_EPOCHS}"
    )
    for batch in progress_bar:
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['labels'].to(device)
        optimizer.zero_grad()
        logits = model(pixel_values)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
torch.save(model.state_dict(), CURRENT_CONFIG.PATH_TO_SAVE_MODEL)