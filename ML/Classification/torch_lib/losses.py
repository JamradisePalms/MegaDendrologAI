import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean', label_smoothing=0.0):
        """
        Args:
            alpha: Вес для балансировки классов (можно использовать tensor весов)
            gamma: Параметр фокусировки (чем больше, тем больше внимание на сложных примерах)
            reduction: 'mean', 'sum' или 'none'
            label_smoothing: Сглаживание меток для регуляризации
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        if self.label_smoothing > 0:
            num_classes = inputs.size(-1)
            targets = F.one_hot(targets, num_classes=num_classes)
            targets = targets * (1 - self.label_smoothing) + self.label_smoothing / num_classes
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        else:
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        pt = torch.exp(-ce_loss)
        
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss