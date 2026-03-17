import torch
import torch.nn as nn


class SMAPELoss(nn.Module):
    """
    SMAPE (Symmetric Mean Absolute Percentage Error) 손실 함수
    실제값이 0인 경우는 손실 계산에서 제외 (대회 규칙 반영)
    """

    def __init__(self, target_idx=0, epsilon=1e-8, reduction='mean'):
        super(SMAPELoss, self).__init__()
        self.target_idx = target_idx
        self.epsilon = epsilon  # 분모가 0이 되는 것을 방지
        self.reduction = reduction

    def forward(self, predictions, targets):
        """
        predictions: [batch_size, seq_len, features]
        targets: [batch_size, seq_len, features]
        """
        pred_target = predictions[:, :, self.target_idx]
        true_target = targets[:, :, self.target_idx]

        # 실제값이 0이 아닌 경우만 마스킹 (대회 규칙)
        valid_mask = (true_target != 0)

        if valid_mask.sum() == 0:
            # 유효한 값이 없으면 0 반환
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)

        # 유효한 값들만 추출
        pred_valid = pred_target[valid_mask]
        true_valid = true_target[valid_mask]

        # SMAPE 계산: 2 * |pred - true| / (|pred| + |true|) * 100
        numerator = torch.abs(pred_valid - true_valid)
        denominator = torch.abs(pred_valid) + torch.abs(true_valid) + self.epsilon

        smape = 2 * numerator / denominator * 100  # 백분율로 변환

        if self.reduction == 'mean':
            return smape.mean()
        elif self.reduction == 'sum':
            return smape.sum()
        else:
            return smape
