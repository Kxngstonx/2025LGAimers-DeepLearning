import torch

class TargetColumnLoss(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.target_idx = kwargs.get('target_idx', 0)
        self.loss_fn = kwargs.get('loss_fn', torch.nn.MSELoss())

    def forward(self, predictions, targets):
        # predictions, targets: [batch, seq, feature_size]
        # 🎯 타겟 칼럼만 추출하여 로스 계산
        pred_target = predictions[:, :, self.target_idx]  # [batch, seq]
        true_target = targets[:, :, self.target_idx]  # [batch, seq]

        return self.loss_fn(pred_target, true_target)
