import torch
import json
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from ML.Classification.results.eval_results import ClassificationMetricsCalculator, quick_evaluation

from ML.Classification.torch_lib.ImageDataset import ImageDatasetJson
from ML.Classification.torch_lib.ImageCollator import ImageCollator
from ML.Classification.torch_lib.ClassificationWrappers import MultiHeadCNNWrapper
from configs.train_config_classification import TrainConfigs

def analyze_dataset_classes(dataset, task_name='tree_type'):
    all_labels = []
    for i in range(len(dataset)):
        _, labels_dict = dataset.samples[i]
        all_labels.append(labels_dict[task_name])
    
    unique_classes = np.unique(all_labels)
    print(f"Dataset analysis:")
    print(f"   Total samples: {len(dataset)}")
    print(f"   Unique classes: {len(unique_classes)}")
    print(f"   Classes present: {sorted(unique_classes)}")
    
    return unique_classes

def main():
    config = TrainConfigs.TreeClassificationWithMobileTransformer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("Loading model...")
    model = MultiHeadCNNWrapper(
        backbone_model=config.MODEL_NAME,
        backbone_type=config.BACKBONE_TYPE,
        num_output_features=config.TARGET_JSON_FIELD,
        hidden_size=128,
        dropout=0.3
    )
    
    checkpoint_path = config.PATH_TO_SAVE_MODEL
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    
    print(f"Model loaded: {config.MODEL_NAME}")
    print(f"Total parameters: {model.get_parameter_count():,}")
    
    print("Loading data...")
    val_preprocessor = config.get_image_preprocessor(is_train=False)
    val_dataset = ImageDatasetJson(
        config.VAL_JSON_FILEPATH,
        config.IMAGE_JSON_FIELD,
        target_fields=['tree_type'],
        preprocessor=val_preprocessor
    )
    
    encoders_path = config.PATH_TO_SAVE_MODEL.parent / "label_encoders_mobile.pkl"
    label_mappings = ImageDatasetJson.load_label_mappings(encoders_path)
    class_names = list(label_mappings['label_mappings']['tree_type'].keys())
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=ImageCollator(task_names=['tree_type']),
        num_workers=4
    )
    
    actual_classes = analyze_dataset_classes(val_dataset)
    
    encoders_path = config.PATH_TO_SAVE_MODEL.parent / "label_encoders_mobile.pkl"
    try:
        label_mappings = ImageDatasetJson.load_label_mappings(encoders_path)
        class_names = list(label_mappings['label_mappings']['tree_type'].keys())
        print(f"All class names: {len(class_names)} classes")
    except Exception as e:
        print(f"Could not load class names: {e}")
        class_names = None
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=ImageCollator(task_names=['tree_type']),
        num_workers=4
    )
    
    print(f"Data loaded: {len(val_dataset)} validation samples")
    
    print("\nQuick Evaluation:")
    quick_accuracy = quick_evaluation(model, val_loader, device)
    
    print("\nFull Evaluation:")
    metrics_calculator = ClassificationMetricsCalculator(model, val_loader, device)
    metrics = metrics_calculator.evaluate(class_names=class_names)
    
    metrics_calculator.print_detailed_report(metrics, class_names)
    
    if metrics['actual_num_classes'] > 1:
        confusion_matrix_path = config.PATH_TO_SAVE_MODEL.parent / "confusion_matrix.png"
        metrics_calculator.plot_confusion_matrix(metrics, class_names, confusion_matrix_path)
    else:
        print("Skipping confusion matrix - only one class in validation set")
    
    metrics_path = config.PATH_TO_SAVE_MODEL.parent / "detailed_metrics.json"
    
    json_metrics = {
        'accuracy': float(metrics['accuracy']),
        'top_k_accuracy': {k: float(v) for k, v in metrics['top_k_accuracy'].items()},
        'confusion_matrix': metrics['confusion_matrix'].tolist(),
        'per_class_metrics': metrics['per_class_metrics'].to_dict('records'),
        'actual_num_classes': int(metrics['actual_num_classes'])
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(json_metrics, f, indent=2)
    
    print(f"\nDetailed metrics saved to: {metrics_path}")
    
    print(f"\nFINAL RESULTS:")
    print(f"   Accuracy: {metrics['accuracy']:.4f}")
    for k, v in metrics['top_k_accuracy'].items():
        print(f"   {k}: {v:.4f}")
    print(f"   Actual classes in validation: {metrics['actual_num_classes']}")

if __name__ == "__main__":
    main()