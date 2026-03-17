import torch

class QuantileTargetColumnLoss(torch.nn.Module):
    """분위수 기반 손실 함수"""

    def __init__(self, target_idx=0, quantiles=[0.1, 0.5, 0.9]):
        super(QuantileTargetColumnLoss, self).__init__()
        self.target_idx = target_idx
        self.quantiles = quantiles

    def forward(self, predictions, targets):
        pred_target = predictions[:, :, self.target_idx]
        true_target = targets[:, :, self.target_idx]

        total_loss = 0
        for q in self.quantiles:
            residual = true_target - pred_target
            loss = torch.max(q * residual, (q - 1) * residual)
            total_loss += loss.mean()

        return total_loss / len(self.quantiles)
