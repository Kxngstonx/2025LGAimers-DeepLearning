import torch

class AsymmetricTargetColumnLoss(torch.nn.Module):
    """비대칭 손실 함수 - 과소예측에 더 큰 페널티"""

    def __init__(self, target_idx=0, alpha=1.5):
        super(AsymmetricTargetColumnLoss, self).__init__()
        self.target_idx = target_idx
        self.alpha = alpha  # 과소예측 페널티 배수

    def forward(self, predictions, targets):
        pred_target = predictions[:, :, self.target_idx]
        true_target = targets[:, :, self.target_idx]

        residual = pred_target - true_target

        # 과소예측(residual < 0)에 더 큰 페널티
        loss = torch.where(
            residual < 0,
            self.alpha * (residual ** 2),  # 과소예측: 높은 페널티
            residual ** 2  # 과대예측: 기본 페널티
        )

        return loss.mean()
