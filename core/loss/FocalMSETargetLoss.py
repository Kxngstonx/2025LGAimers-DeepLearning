import torch

class FocalTargetColumnLoss(torch.nn.Module):
    """Focal Loss 기반 - 어려운 예측에 집중"""

    def __init__(self, target_idx=0, alpha=2.0, gamma=1.0):
        super(FocalTargetColumnLoss, self).__init__()
        self.target_idx = target_idx
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, predictions, targets):
        pred_target = predictions[:, :, self.target_idx]
        true_target = targets[:, :, self.target_idx]

        # MSE 계산
        mse_loss = (pred_target - true_target) ** 2

        # 예측 어려움 정도 계산 (정규화된 MSE)
        normalized_error = mse_loss / (true_target.abs() + 1e-8)

        # Focal weight 계산
        focal_weight = self.alpha * torch.pow(normalized_error, self.gamma)

        return (focal_weight * mse_loss).mean()
