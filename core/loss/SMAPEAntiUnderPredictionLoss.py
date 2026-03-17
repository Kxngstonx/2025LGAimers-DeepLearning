import torch
import torch.nn as nn


class AntiUnderPredictionLoss(nn.Module):
    """
    과소예측을 강력하게 페널티를 주는 손실함수
    under-prediction에 더 큰 가중치 적용
    """

    def __init__(self, target_idx=0, under_weight=3.0, over_weight=1.0,
                 zero_tolerance=1e-6, reduction='mean'):
        super().__init__()
        self.target_idx = target_idx
        self.under_weight = under_weight  # 과소예측 페널티 가중치
        self.over_weight = over_weight  # 과대예측 페널티 가중치
        self.zero_tolerance = zero_tolerance
        self.reduction = reduction

    def forward(self, predictions, targets):
        """
        Args:
            predictions: [B, T, C]
            targets: [B, T, C]
        """
        pred_target = predictions[:, :, self.target_idx]
        true_target = targets[:, :, self.target_idx]

        # 실제값이 0이 아닌 경우만 고려
        valid_mask = (true_target.abs() > self.zero_tolerance)

        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)

        pred_valid = pred_target[valid_mask]
        true_valid = true_target[valid_mask]

        # 오차 계산
        error = pred_valid - true_valid

        # 과소예측(negative error)과 과대예측(positive error) 분리
        under_pred_mask = (error < 0)  # 예측 < 실제 (과소예측)
        over_pred_mask = (error >= 0)  # 예측 >= 실제 (과대예측)

        # Asymmetric penalty 적용
        loss = torch.zeros_like(error)

        if under_pred_mask.any():
            # 과소예측에 강한 페널티 (SMAPE 기반)
            under_pred = pred_valid[under_pred_mask]
            under_true = true_valid[under_pred_mask]
            under_smape = 2 * torch.abs(under_pred - under_true) / (
                    torch.abs(under_pred) + torch.abs(under_true) + 1e-8
            ) * 100
            loss[under_pred_mask] = self.under_weight * under_smape

        if over_pred_mask.any():
            # 과대예측에 일반적인 페널티
            over_pred = pred_valid[over_pred_mask]
            over_true = true_valid[over_pred_mask]
            over_smape = 2 * torch.abs(over_pred - over_true) / (
                    torch.abs(over_pred) + torch.abs(over_true) + 1e-8
            ) * 100
            loss[over_pred_mask] = self.over_weight * over_smape

        return loss.mean() if self.reduction == 'mean' else loss.sum()