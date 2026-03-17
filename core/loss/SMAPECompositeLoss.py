import torch
import torch.nn as nn
import math

class SMAPEInspiredLoss(nn.Module):
    """
    - nonzero(타깃!=0): SMAPE-like
    - zero(타깃==0): 약한 규제(lambda_zero), 전체 손실의 일정 비율(cap) 이하로 제한
    - Quantile L1 보조항(언더슛 억제)
    - Top-K 에러 가중(피크 반응 강화)
    """
    def __init__(self, **kwargs):
        """
            Args:
                target_idx: int = 0,
                epsilon: float = 1e-8,
                zero_tol: float = 0.0,
                lambda_zero: float = 0.02,
                detach_pred_in_den: bool = True,
                zero_penalty: str = "huber",    # "l1" | "huber"
                huber_delta: float = 0.5,
                max_zero_share: float = 0.15,

                # Quantile aux loss
                use_quantile: bool = True,
                quantile_q: float = 0.6,
                quantile_weight: float = 0.2,
                quantile_nonzero_only: bool = True,

                # Top-K weighting
                use_topk: bool = True,
                topk_frac: float = 0.2,          # nonzero 중 상위 20%
                topk_boost: float = 1.5,         # 상위 k 가중치
                topk_metric: str = "abs_error"     # "abs_error" | "smape"
        """
        super().__init__()
        self.target_idx = kwargs.get('target_idx', 0)
        self.epsilon = kwargs.get('epsilon', 1e-8)
        self.zero_tol = kwargs.get('zero_tol', 0.0)
        self.lambda_zero = kwargs.get('lambda_zero', 0.02)
        self.detach_pred_in_den = kwargs.get('detach_pred_in_den', True)
        self.zero_penalty = kwargs.get('zero_penalty', "huber")
        self.huber_delta = kwargs.get('huber_delta', 0.5)
        self.max_zero_share = kwargs.get('max_zero_share', 0.15)

        self.use_quantile = kwargs.get('use_quantile', True)
        self.quantile_q = kwargs.get('quantile_q', 0.6)
        self.quantile_weight = kwargs.get('quantile_weight', 0.2)
        self.quantile_nonzero_only = kwargs.get('quantile_nonzero_only', True)

        self.use_topk = kwargs.get('use_topk', True)
        self.topk_frac = kwargs.get('topk_frac', 0.2)
        self.topk_boost = kwargs.get('topk_boost', 1.5)
        self.topk_metric = kwargs.get('topk_metric', "abs_error")

    def _huber(self, x, d):
        ax = x.abs()
        return torch.where(ax <= d, 0.5*(ax**2)/d, ax - 0.5*d)

    def _quantile_l1(self, pred, true):
        e = true - pred
        return (self.quantile_q * torch.clamp(e, min=0) +
                (1 - self.quantile_q) * torch.clamp(-e, min=0)).mean()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # predictions/targets: [B, T', C]
        pred = predictions[:, :, self.target_idx]
        true = targets[:, :, self.target_idx]

        nonzero_mask = (true.abs() > self.zero_tol)
        zero_mask    = ~nonzero_mask

        # --- SMAPE-like ---
        denom_pred = pred.abs().detach() if self.detach_pred_in_den else pred.abs()
        denom = true.abs() + denom_pred + self.epsilon  # self.eps -> self.epsilon으로 수정
        smape_full = 2.0 * (pred - true).abs() / denom

        # nonzero에서 평균(Top-K 가중 가능)
        if nonzero_mask.any():
            smape_nz = smape_full[nonzero_mask]  # [N_nz]
            if self.use_topk and self.topk_boost > 1.0 and self.topk_frac > 0.0:
                if self.topk_metric == "abs_error":
                    score = (pred - true).abs()[nonzero_mask]
                else:
                    score = smape_nz
                N = score.numel()
                k = max(1, int(math.ceil(self.topk_frac * N)))
                _, topk_idx = torch.topk(score, k, largest=True, sorted=False)
                weights = torch.ones_like(smape_nz)
                weights[topk_idx] = self.topk_boost
                loss_nonzero = (smape_nz * weights).sum() / (weights.sum() + self.epsilon)
            else:
                loss_nonzero = smape_nz.mean()
        else:
            loss_nonzero = pred.new_tensor(0.0)

        # --- 약한 zero 규제 (cap) ---
        zero_term = pred.new_tensor(0.0)
        if self.lambda_zero > 0.0 and zero_mask.any():
            pz = pred[zero_mask]
            base = self._huber(pz, self.huber_delta) if self.zero_penalty == "huber" else pz.abs()
            loss_zero = base.mean()
            allowed = self.max_zero_share * (loss_nonzero.detach() + self.epsilon)
            raw_zero = self.lambda_zero * loss_zero
            cap_factor = torch.clamp(allowed / (raw_zero + self.epsilon), max=1.0)
            zero_term = cap_factor * raw_zero

        total = loss_nonzero + zero_term

        # --- Quantile L1 (언더슛 억제) ---
        if self.use_quantile and self.quantile_weight > 0:
            q_loss = (self._quantile_l1(pred[nonzero_mask], true[nonzero_mask])
                      if self.quantile_nonzero_only and nonzero_mask.any()
                      else self._quantile_l1(pred, true))
            total = total + self.quantile_weight * q_loss

        return total