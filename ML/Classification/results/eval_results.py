import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tqdm import tqdm
import pandas as pd
from torch.utils.data import DataLoader

class ClassificationMetricsCalculator:
    def __init__(self, model, dataloader, device, task_name='tree_type'):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.task_name = task_name
        
    def evaluate(self, class_names=None):
        """Основной метод вычисления всех метрик"""
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in tqdm(self.dataloader, desc="Evaluating"):
                pixel_values = batch['pixel_values'].to(self.device)
                labels = batch['labels'][self.task_name].to(self.device)
                
                outputs = self.model(pixel_values)
                logits = outputs[self.task_name]
                
                probabilities = torch.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        probabilities = np.array(all_probabilities)
        
        metrics = self._compute_all_metrics(targets, predictions, probabilities, class_names)
        
        return metrics
    
    def _compute_all_metrics(self, targets, predictions, probabilities, class_names):
        """Вычисляет все метрики с защитой от отсутствующих классов"""
        accuracy = accuracy_score(targets, predictions)
        
        top_k_metrics = self._compute_top_k_accuracy(probabilities, targets)
        
        unique_classes = np.unique(np.concatenate([targets, predictions]))
        num_classes = len(unique_classes) if class_names is None else len(class_names)
        
        if class_names is None:
            class_names = [f'Class_{i}' for i in range(num_classes)]
        else:
            class_names = class_names[:num_classes]
        
        try:
            classification_rep = classification_report(
                targets, predictions, 
                target_names=class_names, 
                output_dict=True,
                digits=4,
                zero_division=0
            )
        except ValueError as e:
            print(f"Warning in classification_report: {e}")
            classification_rep = {
                'accuracy': accuracy,
                'macro avg': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': len(targets)},
                'weighted avg': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': len(targets)}
            }
        
        try:
            cm = confusion_matrix(targets, predictions, labels=range(num_classes))
        except ValueError as e:
            print(f"Warning in confusion_matrix: {e}")
            cm = np.zeros((num_classes, num_classes), dtype=int)
            for true, pred in zip(targets, predictions):
                if true < num_classes and pred < num_classes:
                    cm[true, pred] += 1
        
        per_class_metrics = self._compute_per_class_metrics(targets, predictions, class_names, num_classes)
        
        return {
            'accuracy': accuracy,
            'top_k_accuracy': top_k_metrics,
            'classification_report': classification_rep,
            'confusion_matrix': cm,
            'per_class_metrics': per_class_metrics,
            'predictions': predictions,
            'targets': targets,
            'probabilities': probabilities,
            'actual_num_classes': len(np.unique(targets))
        }
    
    def _compute_top_k_accuracy(self, probabilities, targets, k_list=[2, 3, 5]):
        """Вычисляет Top-K accuracy"""
        top_k_results = {}
        
        for k in k_list:
            top_k_preds = np.argsort(probabilities, axis=1)[:, -k:]
            correct = 0
            for i, target in enumerate(targets):
                if target in top_k_preds[i]:
                    correct += 1
            top_k_results[f'top_{k}'] = correct / len(targets)
        
        return top_k_results
    
    def _compute_per_class_metrics(self, targets, predictions, class_names, num_classes):
        """Вычисляет метрики для каждого класса с защитой"""
        from sklearn.metrics import precision_recall_fscore_support
        
        try:
            precision, recall, f1, support = precision_recall_fscore_support(
                targets, predictions, 
                labels=range(num_classes),
                average=None, 
                zero_division=0
            )
        except ValueError as e:
            print(f"Warning in precision_recall_fscore_support: {e}")
            precision = np.zeros(num_classes)
            recall = np.zeros(num_classes)
            f1 = np.zeros(num_classes)
            support = np.zeros(num_classes, dtype=int)
            
            unique_targets, counts = np.unique(targets, return_counts=True)
            for cls, count in zip(unique_targets, counts):
                if cls < num_classes:
                    support[cls] = count
        
        try:
            metrics_df = pd.DataFrame({
                'Class': class_names[:num_classes],
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'Support': support
            })
        except ValueError as e:
            print(f"Error creating DataFrame: {e}")
            print(f"Shapes - class_names: {len(class_names)}, precision: {len(precision)}")
            metrics_df = pd.DataFrame({
                'Class': [f'Class_{i}' for i in range(len(precision))],
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'Support': support
            })
        
        return metrics_df
    
    def print_detailed_report(self, metrics, class_names):
        accuracy = metrics['accuracy']
        actual_classes = metrics['actual_num_classes']
        
        print("=" * 70)
        print("CLASSIFICATION METRICS REPORT")
        print("=" * 70)
        print(f"Overall Accuracy: {accuracy:.4f}")
        print(f"Actual classes in validation: {actual_classes}")
        
        print(f"\nTop-K Accuracy:")
        for k, acc in metrics['top_k_accuracy'].items():
            print(f"  {k}: {acc:.4f}")
        
        print(f"\nDetailed Classification Report:")
        try:
            if class_names and len(class_names) > actual_classes:
                class_names = class_names[:actual_classes]
            
            print(classification_report(
                metrics['targets'], 
                metrics['predictions'], 
                target_names=class_names, 
                digits=4,
                zero_division=0
            ))
        except Exception as e:
            print(f"Error printing classification report: {e}")
            print("Using fallback report...")
            print(f"Accuracy: {accuracy:.4f}")
        
        print(f"\nPer-Class Metrics:")
        print(metrics['per_class_metrics'].round(4))
    
    def plot_confusion_matrix(self, metrics, class_names, save_path=None):
        """Визуализирует матрицу ошибок с защитой"""
        cm = metrics['confusion_matrix']
        accuracy = metrics['accuracy']
        actual_classes = metrics['actual_num_classes']
        
        if class_names and len(class_names) > actual_classes:
            class_names = class_names[:actual_classes]
        elif class_names is None:
            class_names = [f'Class_{i}' for i in range(actual_classes)]
        
        plt.figure(figsize=(max(10, actual_classes), max(8, actual_classes)))
        
        try:
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_normalized = np.nan_to_num(cm_normalized)
            
            sns.heatmap(cm_normalized, 
                        annot=True, 
                        fmt='.2f',
                        cmap='Blues',
                        xticklabels=class_names,
                        yticklabels=class_names,
                        cbar_kws={'label': 'Fraction of True Class'})
            
            plt.title(f'Confusion Matrix (Accuracy: {accuracy:.4f})', fontsize=16, pad=20)
            plt.xlabel('Predicted Label', fontsize=12)
            plt.ylabel('True Label', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            print(f"Error plotting confusion matrix: {e}")
            print("CM shape:", cm.shape)
            print("Class names length:", len(class_names))

def quick_evaluation(model, dataloader, device, task_name='tree_type'):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'][task_name].to(device)
            
            outputs = model(pixel_values)
            predictions = torch.argmax(outputs[task_name], dim=1)
            
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total
    print(f"Quick Accuracy: {accuracy:.4f} ({correct}/{total})")
    return accuracy