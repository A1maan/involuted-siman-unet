import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    Paper: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    
    FL(pt) = -αt(1-pt)^γ log(pt)
    
    Args:
        alpha (float): Weighting factor in [0, 1] to balance positive/negative examples
        gamma (float): Exponent of the modulating factor (1-pt)^γ to balance easy/hard examples
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Predictions from model, shape [B, C, H, W]
            targets (torch.Tensor): Ground truth, shape [B, C, H, W] or [B, 1, H, W]
        Returns:
            torch.Tensor: Focal loss value
        """
        # Convert to probabilities
        p = torch.sigmoid(inputs)
        
        # Focal loss computation
        # BCE = -[y*log(p) + (1-y)*log(1-p)]
        # FL = -[α*y*(1-p)^γ*log(p) + (1-α)*(1-y)*p^γ*log(1-p)]
        
        ce_loss = F.binary_cross_entropy(p, targets, reduction='none')
        
        # Focal term
        pt = torch.where(targets == 1, p, 1 - p)
        focal_weight = (1 - pt) ** self.gamma
        
        focal_loss = self.alpha * targets * focal_weight * ce_loss + \
                     (1 - self.alpha) * (1 - targets) * focal_weight * ce_loss
        
        return focal_loss.mean()


class DiceLoss(nn.Module):
    """
    Dice Loss (F1 Loss) for binary segmentation.
    Dice = 2*TP / (2*TP + FP + FN)
    
    DiceLoss = 1 - Dice
    """
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Predictions from model, shape [B, C, H, W]
            targets (torch.Tensor): Ground truth, shape [B, C, H, W] or [B, 1, H, W]
        Returns:
            torch.Tensor: Dice loss value
        """
        # Convert to probabilities
        probs = torch.sigmoid(inputs)
        
        # Flatten
        probs = probs.view(-1)
        targets = targets.view(-1)
        
        # Dice coefficient
        intersection = (probs * targets).sum()
        dice_coef = (2.0 * intersection + self.smooth) / (
            probs.sum() + targets.sum() + self.smooth
        )
        
        return 1.0 - dice_coef


class CombinedDeepSupervisionLoss(nn.Module):
    """
    Combined loss for deep supervision strategy.
    
    Loss_i = FocalLoss(GT_i, Output_i) + DiceLoss(GT_i, Output_i)
    Loss_total = Σ(λ_i × Loss_i) for i=1 to 5
    
    Where:
    - GT_i is resized to match Output_i spatial dimensions
    - λ_i = [0.5, 0.4, 0.3, 0.2, 0.1] by default
    - Output_i are predictions from different deep supervision levels
    """
    def __init__(self, lambdas=None, alpha=0.25, gamma=2.0, dice_smooth=1.0):
        super(CombinedDeepSupervisionLoss, self).__init__()
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        self.dice_loss = DiceLoss(smooth=dice_smooth)
        
        # Default lambda values: [0.5, 0.4, 0.3, 0.2, 0.1]
        if lambdas is None:
            self.lambdas = [0.5, 0.4, 0.3, 0.2, 0.1]
        else:
            self.lambdas = lambdas

    def forward(self, outputs, gt):
        """
        Args:
            outputs (list): List of predictions from deep supervision heads
                           Each element shape: [B, C, H_i, W_i]
            gt (torch.Tensor): Ground truth, shape [B, 1, H, W]
        Returns:
            torch.Tensor: Combined weighted loss
        """
        total_loss = 0.0
        
        for i, output in enumerate(outputs):
            # Get lambda for this level (cycle if needed)
            lambda_i = self.lambdas[i % len(self.lambdas)]
            
            # Get output spatial size
            _, _, h_out, w_out = output.shape
            
            # Resize GT to match output spatial dimensions
            gt_resized = F.interpolate(
                gt,
                size=(h_out, w_out),
                mode='nearest'
            )
            
            # Calculate focal loss
            focal = self.focal_loss(output, gt_resized)
            
            # Calculate dice loss
            dice = self.dice_loss(output, gt_resized)
            
            # Combined loss for this level: Eq. (1)
            loss_i = focal + dice
            
            # Weighted accumulation: Eq. (2)
            total_loss += lambda_i * loss_i
        
        return total_loss

    def set_lambdas(self, lambdas):
        """Customize lambda weights for each deep supervision level"""
        self.lambdas = lambdas
