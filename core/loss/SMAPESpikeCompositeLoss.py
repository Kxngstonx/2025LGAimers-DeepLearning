import torch
import torch.nn as nn
import math


class SMAPESpikeLoss(nn.Module):
    """
    판매 재개 패턴을 고려한 개선된 SMAPE Inspired Loss

    주요 개선사항:
    1. 판매 재개 감지 및 가중치 부여
    2. 급격한 증가 패턴에 대한 특별 처리
    3. 과소예측에 대한 강화된 패널티
    4. 비시즌→시즌 전환 구간 특별 가중치
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

                # Quantile aux loss (과소예측 억제)
                use_quantile: bool = True,
                quantile_q: float = 0.65,      # 0.6 -> 0.65로 증가
                quantile_weight: float = 0.3,  # 0.2 -> 0.3으로 증가
                quantile_nonzero_only: bool = True,

                # Top-K weighting
                use_topk: bool = True,
                topk_frac: float = 0.2,
                topk_boost: float = 1.5,
                topk_metric: str = "abs_error",

                # 새로운 판매 재개 감지 파라미터
                use_resumption_detection: bool = True,
                resumption_threshold: float = 10.0,    # 판매 재개로 간주할 최소 값
                resumption_boost: float = 2.5,         # 판매 재개 시점 가중치
                severe_underpredict_threshold: float = 0.5,  # 심각한 과소예측 기준
                severe_underpredict_boost: float = 3.0,      # 심각한 과소예측 추가 패널티

                # 급격한 변화 감지
                use_spike_detection: bool = True,
                spike_multiplier: float = 5.0,         # 급증 감지 배수
                spike_boost: float = 2.0,              # 급증 구간 가중치
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

        # 기존 파라미터들 (일부 기본값 조정)
        self.use_quantile = kwargs.get('use_quantile', True)
        self.quantile_q = kwargs.get('quantile_q', 0.65)  # 증가
        self.quantile_weight = kwargs.get('quantile_weight', 0.3)  # 증가
        self.quantile_nonzero_only = kwargs.get('quantile_nonzero_only', True)

        self.use_topk = kwargs.get('use_topk', True)
        self.topk_frac = kwargs.get('topk_frac', 0.2)
        self.topk_boost = kwargs.get('topk_boost', 1.5)
        self.topk_metric = kwargs.get('topk_metric', "abs_error")

        # 새로운 판매 재개 감지 파라미터
        self.use_resumption_detection = kwargs.get('use_resumption_detection', True)
        self.resumption_threshold = kwargs.get('resumption_threshold', 10.0)
        self.resumption_boost = kwargs.get('resumption_boost', 2.5)
        self.severe_underpredict_threshold = kwargs.get('severe_underpredict_threshold', 0.5)
        self.severe_underpredict_boost = kwargs.get('severe_underpredict_boost', 3.0)

        # 급격한 변화 감지
        self.use_spike_detection = kwargs.get('use_spike_detection', True)
        self.spike_multiplier = kwargs.get('spike_multiplier', 5.0)
        self.spike_boost = kwargs.get('spike_boost', 2.0)

    def _huber(self, x, d):
        ax = x.abs()
        return torch.where(ax <= d, 0.5 * (ax ** 2) / d, ax - 0.5 * d)

    def _quantile_l1(self, pred, true):
        e = true - pred
        return (self.quantile_q * torch.clamp(e, min=0) +
                (1 - self.quantile_q) * torch.clamp(-e, min=0)).mean()

    def _detect_sales_resumption(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """
        판매 재개 구간을 감지하여 추가 가중치 적용

        Args:
            pred: 예측값 [B, T']
            true: 실제값 [B, T']

        Returns:
            torch.Tensor: 판매 재개 가중치 [B, T']
        """
        batch_size, seq_len = true.shape
        resumption_weights = torch.ones_like(true)

        if not self.use_resumption_detection:
            return resumption_weights

        for b in range(batch_size):
            for t in range(1, seq_len):
                current_true = true[b, t]
                prev_true = true[b, t - 1]
                current_pred = pred[b, t]

                # 판매 재개 감지: 이전이 거의 0이고 현재가 임계값 이상
                is_resumption = (prev_true <= self.zero_tol and
                                 current_true >= self.resumption_threshold)

                # 심각한 과소예측 감지: 실제값 대비 예측값이 너무 낮음
                if current_true > 0:
                    pred_ratio = current_pred / current_true
                    is_severe_underpredict = pred_ratio < self.severe_underpredict_threshold
                else:
                    is_severe_underpredict = False

                # 급격한 증가 감지
                is_spike = (prev_true > 0 and
                            current_true >= prev_true * self.spike_multiplier) if self.use_spike_detection else False

                # 가중치 적용
                weight = 1.0
                if is_resumption:
                    weight *= self.resumption_boost
                if is_severe_underpredict:
                    weight *= self.severe_underpredict_boost
                if is_spike:
                    weight *= self.spike_boost

                resumption_weights[b, t] = weight

        return resumption_weights

    def _apply_value_based_weighting(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """
        실제값 구간별 차등 가중치 적용

        Args:
            pred: 예측값 [B, T']
            true: 실제값 [B, T']

        Returns:
            torch.Tensor: 구간별 가중치 [B, T']
        """
        # 구간별 가중치 설정
        # 0~10: 높은 가중치 (판매 재개 초기)
        # 10~50: 중간 가중치
        # 50~200: 기본 가중치
        # 200+: 낮은 가중치 (SMAPE 특성상 이미 안정적)

        weights = torch.ones_like(true)

        # 구간 1: 0~10 (판매 재개 초기, 높은 중요도)
        mask_low = (true > 0) & (true <= 10)
        weights[mask_low] = 2.0

        # 구간 2: 10~50 (성장 초기, 중간 중요도)
        mask_medium_low = (true > 10) & (true <= 50)
        weights[mask_medium_low] = 1.5

        # 구간 3: 50~200 (안정 구간, 기본 중요도)
        mask_medium = (true > 50) & (true <= 200)
        weights[mask_medium] = 1.0

        # 구간 4: 200+ (고값 구간, 낮은 중요도)
        mask_high = true > 200
        weights[mask_high] = 0.7

        return weights

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # predictions/targets: [B, T', C]
        pred = predictions[:, :, self.target_idx]  # [B, T']
        true = targets[:, :, self.target_idx]  # [B, T']

        nonzero_mask = (true.abs() > self.zero_tol)
        zero_mask = ~nonzero_mask

        # --- SMAPE-like 계산 ---
        denom_pred = pred.abs().detach() if self.detach_pred_in_den else pred.abs()
        denom = true.abs() + denom_pred + self.epsilon
        smape_full = 2.0 * (pred - true).abs() / denom

        # --- 판매 재개 및 특수 상황 감지 ---
        resumption_weights = self._detect_sales_resumption(pred, true)
        value_weights = self._apply_value_based_weighting(pred, true)

        # 복합 가중치 적용
        combined_weights = resumption_weights * value_weights

        # nonzero에서 가중 평균 (기존 Top-K + 새로운 가중치)
        if nonzero_mask.any():
            smape_nz = smape_full[nonzero_mask]
            additional_weights = combined_weights[nonzero_mask]

            if self.use_topk and self.topk_boost > 1.0 and self.topk_frac > 0.0:
                if self.topk_metric == "abs_error":
                    score = (pred - true).abs()[nonzero_mask]
                else:
                    score = smape_nz
                N = score.numel()
                k = max(1, int(math.ceil(self.topk_frac * N)))
                _, topk_idx = torch.topk(score, k, largest=True, sorted=False)
                topk_weights = torch.ones_like(smape_nz)
                topk_weights[topk_idx] = self.topk_boost

                # 최종 가중치 = Top-K 가중치 × 추가 가중치
                final_weights = topk_weights * additional_weights
                loss_nonzero = (smape_nz * final_weights).sum() / (final_weights.sum() + self.epsilon)
            else:
                # Top-K 미사용 시 추가 가중치만 적용
                loss_nonzero = (smape_nz * additional_weights).sum() / (additional_weights.sum() + self.epsilon)
        else:
            loss_nonzero = pred.new_tensor(0.0)

        # --- 기존 zero 규제 (변경 없음) ---
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

        # --- 강화된 Quantile L1 (과소예측 억제) ---
        if self.use_quantile and self.quantile_weight > 0:
            if self.quantile_nonzero_only and nonzero_mask.any():
                # nonzero 구간에서만 적용하되, 가중치도 함께 고려
                nz_pred = pred[nonzero_mask]
                nz_true = true[nonzero_mask]
                nz_weights = combined_weights[nonzero_mask]

                # 가중치를 고려한 Quantile L1
                e = nz_true - nz_pred
                quantile_loss = (self.quantile_q * torch.clamp(e, min=0) +
                                 (1 - self.quantile_q) * torch.clamp(-e, min=0))
                weighted_quantile_loss = (quantile_loss * nz_weights).mean()

                total = total + self.quantile_weight * weighted_quantile_loss
            else:
                q_loss = self._quantile_l1(pred, true)
                total = total + self.quantile_weight * q_loss

        return total