import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
from ML.Classification.torch_lib.ImageDataset import ImageDatasetJson
from ML.Classification.torch_lib.ImageCollator import ImageCollator
from configs.train_config_classification import TrainConfigs
from ML.Classification.torch_lib.ClassificationWrappers import MultiHeadCNNWrapper
import json

CURRENT_CONFIG = TrainConfigs.TreeClassificationModelWithMultiHeadMLP

image_processor = CURRENT_CONFIG.get_image_processor()
train_preprocessor = CURRENT_CONFIG.get_image_preprocessor(is_train=True)
val_preprocessor = CURRENT_CONFIG.get_image_preprocessor(is_train=False)

task_names = list(CURRENT_CONFIG.TARGET_JSON_FIELD.keys())
num_classes_per_task = CURRENT_CONFIG.TARGET_JSON_FIELD

train_dataset = ImageDatasetJson(
    CURRENT_CONFIG.TRAIN_JSON_FILEPATH,
    CURRENT_CONFIG.IMAGE_JSON_FIELD,
    target_fields=task_names,
    preprocessor=train_preprocessor
)

val_dataset = ImageDatasetJson(
    CURRENT_CONFIG.VAL_JSON_FILEPATH,
    CURRENT_CONFIG.IMAGE_JSON_FIELD,
    target_fields=task_names,
    preprocessor=val_preprocessor
)

encoders_path = CURRENT_CONFIG.PATH_TO_SAVE_MODEL.parent / "label_encoders.pkl"
train_dataset.save_label_mappings(encoders_path)

data_collator = ImageCollator(task_names=task_names)

train_loader = DataLoader(
    train_dataset,
    batch_size=CURRENT_CONFIG.BATCH_SIZE,
    shuffle=True,
    collate_fn=data_collator,
    num_workers=0,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=CURRENT_CONFIG.BATCH_SIZE,
    shuffle=False,
    collate_fn=data_collator,
    num_workers=0,
)

model = MultiHeadCNNWrapper(
    backbone_model=CURRENT_CONFIG.MODEL_NAME,
    backbone_type='efficientnet',
    num_output_features=num_classes_per_task
)
model.train()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def create_loss_weights(weights_list):
    """Вспомогательная функция для создания тензоров весов"""
    return torch.tensor(weights_list, dtype=torch.float32).to(device)

criterions = {
    task_name: nn.CrossEntropyLoss() 
    for task_name in task_names
}

# criterions['tree_type'] = nn.CrossEntropyLoss(weight=create_loss_weights([0.694229112833764, 0.1139383658467628, 7.462962962962963, 3.7314814814814814, 9.950617283950617, 0.35537918871252205, 2.9851851851851854, 2.2962962962962963, 4.9753086419753085, 9.950617283950617, 0.2985185185185185, 0.23691945914168136, 4.9753086419753085, 9.950617283950617, 0.8292181069958847, 9.950617283950617, 1.6584362139917694, 3.316872427983539, 3.7314814814814814, 3.316872427983539, 4.264550264550264, 7.462962962962963, 2.4876543209876543, 7.462962962962963, 7.462962962962963, 4.264550264550264, 2.132275132275132]))
# criterions['has_hollow'] = nn.CrossEntropyLoss(weight=create_loss_weights([0.5295663600525624, 8.955555555555556]))
# criterions['has_cracks'] = nn.CrossEntropyLoss(weight=create_loss_weights([0.6096822995461422, 2.779310344827586]))
# criterions['has_fruits_or_flowers'] = nn.CrossEntropyLoss(weight=create_loss_weights([0.5394912985274432, 6.830508474576271]))
# criterions['overall_condition'] = nn.CrossEntropyLoss(weight=create_loss_weights([19.19047619047619, 4.9753086419753085, 3.124031007751938, 0.23608670181605154, 2.066666666666667, 1.4140350877192982]))
# criterions['has_crown_damage'] = nn.CrossEntropyLoss(weight=create_loss_weights([0.5247395833333334, 10.605263157894736]))
# criterions['has_trunk_damage'] = nn.CrossEntropyLoss(weight=create_loss_weights([0.6818950930626058, 1.8744186046511628]))
# criterions['has_rot'] = nn.CrossEntropyLoss(weight=create_loss_weights([0.5081967213114754, 31.0]))

optimizer = optim.AdamW(model.parameters(), lr=CURRENT_CONFIG.LR)
model.to(device)

PATIENCE = CURRENT_CONFIG.PATIENCE
MIN_DELTA = CURRENT_CONFIG.MIN_DELTA
LOSS_WEIGHTS = CURRENT_CONFIG.LOSS_WEIGHTS
best_val_loss = float('inf')
patience_counter = 0
best_model_state = None

batch_losses = []
epoch_losses = []
task_losses = {task_name: [] for task_name in task_names}
val_losses = []

for epoch in range(CURRENT_CONFIG.NUM_EPOCHS):
    model.train()
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
            task_loss = criterions[task_name](logits[task_name], labels[task_name])
            weighted_task_loss = task_loss * LOSS_WEIGHTS[task_name]
            total_loss += weighted_task_loss
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
    
    model.eval()
    val_loss_sum = 0.0
    val_batch_count = 0
    val_task_losses = {task_name: 0.0 for task_name in task_names}
    
    val_progress_bar = tqdm(
        val_loader, desc=f"Validation Epoch: {epoch + 1}"
    )
    
    with torch.no_grad():
        for batch in val_progress_bar:
            pixel_values = batch['pixel_values'].to(device)
            labels = {k: v.to(device) for k, v in batch['labels'].items()}
            
            logits = model(pixel_values)
            
            batch_val_loss = 0
            for task_name in task_names:
                task_loss = criterions[task_name](logits[task_name], labels[task_name])
                weighted_task_loss = task_loss * LOSS_WEIGHTS[task_name]
                batch_val_loss += weighted_task_loss
                val_task_losses[task_name] += task_loss.item()
            
            val_loss_sum += batch_val_loss.item()
            val_batch_count += 1
            
            val_progress_bar.set_postfix({
                'val_loss': f'{batch_val_loss.item():.4f}'
            })
    
    if epoch_batch_count > 0:
        epoch_avg_loss = epoch_loss_sum / epoch_batch_count
        epoch_losses.append(epoch_avg_loss)
    
    if val_batch_count > 0:
        val_avg_loss = val_loss_sum / val_batch_count
        val_losses.append(val_avg_loss)
        
        print(f"\nEpoch {epoch + 1} - Train Loss: {epoch_avg_loss:.4f}, Val Loss: {val_avg_loss:.4f}")
        for task_name in task_names:
            task_avg_loss = val_task_losses[task_name] / val_batch_count
            print(f"  {task_name} Val Loss: {task_avg_loss:.4f}")
        
        if val_avg_loss < best_val_loss - MIN_DELTA:
            best_val_loss = val_avg_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            
            torch.save(best_model_state, CURRENT_CONFIG.PATH_TO_SAVE_MODEL)
            print(f"Best model saved to: {CURRENT_CONFIG.PATH_TO_SAVE_MODEL}")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter}/{PATIENCE} epochs")
            
            if patience_counter >= PATIENCE:
                print(f"Early stopping triggered after {epoch + 1} epochs!")
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                    print("🔄 Restored best model weights")
                break

if patience_counter < PATIENCE:
    torch.save(model.state_dict(), CURRENT_CONFIG.PATH_TO_SAVE_MODEL)
    print(f"Final model saved to: {CURRENT_CONFIG.PATH_TO_SAVE_MODEL}")

metrics_data = {
    'batch_losses': batch_losses,
    'epoch_losses': epoch_losses,
    'val_losses': val_losses,
    'task_losses': task_losses,
    'best_val_loss': best_val_loss,
    'final_epoch': epoch + 1,
    'early_stopping_triggered': patience_counter >= PATIENCE
}

metrics_path = CURRENT_CONFIG.PATH_TO_SAVE_MODEL.parent / "training_metrics.json"
with open(metrics_path, 'w') as f:
    json.dump(metrics_data, f, indent=2)

print(f"Metrics saved to: {metrics_path}")
print(f"Label encoders saved to: {encoders_path}")
print(f"Best validation loss: {best_val_loss:.4f}")