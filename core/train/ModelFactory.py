from typing import Dict, Type

import torch
import torch.nn as nn

from ..config.Config import ModelType, LossType, OptimizerType, SchedulerType
from ..loss.SMAPEAntiUnderPredictionLoss import AntiUnderPredictionLoss
from ..loss.SMAPESpikeCompositeLoss import SMAPESpikeLoss
from ..loss.SMAPECompositeLoss import SMAPEInspiredLoss
from ..loss.SingleTargetColumnLoss import TargetColumnLoss
from ..model.PatchTSTPretrainFinetuneModel import CustomPatchTSTForPrediction
# 기존 모델 import
from ..model.DLinearDecompModel import DLinear
from ..model.DLinearRevINDecompModel import DLinearRevIN
from ..model.PatchTSTRevINSalesHeadModel import PatchTSTRevINSalesHead
from ..model.RevINLinearModel import RLinear
from ..model.XPatchHybridDecompModel import xPatch
from ..model.PatchMixerCNNModel import PatchMixer


class ModelFactory:
    """
    모델/손실함수/옵티마이저/스케줄러를 쉽게 추가하고 사용할 수 있게 하는 팩토리
    """

    # 모델 레지스트리
    MODELS: Dict[ModelType, Type[nn.Module]] = {
        ModelType.DLinear: DLinear,
        ModelType.RLinear: RLinear,
        ModelType.PatchTSTRevINSalesHead: PatchTSTRevINSalesHead,
        ModelType.DLinearRevIN: DLinearRevIN,
        ModelType.xPatch: xPatch,
        ModelType.CustomPatchTSTForPrediction: CustomPatchTSTForPrediction,
        ModelType.PatchMixer: PatchMixer
        # 추가 모델들은 여기에 등록
        # ModelType.NLinear: NLinear,
        # ModelType.GLinear: GLinear,
    }

    # 손실함수 레지스트리
    LOSSES: Dict[LossType, Type[nn.Module]] = {
        LossType.MSELoss: nn.MSELoss,
        LossType.L1Loss: nn.L1Loss,
        LossType.SmoothL1Loss: nn.SmoothL1Loss,
        LossType.TargetColumnLoss: TargetColumnLoss,  # 커스텀 로스
        LossType.SMAPEInspiredLoss: SMAPEInspiredLoss,
        LossType.WeightedSMAPEInspiredLoss: WeightedSMAPEInspiredLoss,
        LossType.SMAPESpikeLoss: SMAPESpikeLoss,
        LossType.AntiUnderPredictionLoss: AntiUnderPredictionLoss
    }

    # 옵티마이저 레지스트리
    OPTIMIZERS: Dict[OptimizerType, Type[torch.optim.Optimizer]] = {
        OptimizerType.Adam: torch.optim.Adam,
        OptimizerType.AdamW: torch.optim.AdamW,
        OptimizerType.SGD: torch.optim.SGD,
    }

    # 스케줄러 레지스트리
    SCHEDULERS: Dict[SchedulerType, Type] = {
        SchedulerType.StepLR: torch.optim.lr_scheduler.StepLR,
        SchedulerType.CosineAnnealingLR: torch.optim.lr_scheduler.CosineAnnealingLR,
        SchedulerType.ReduceLROnPlateau: torch.optim.lr_scheduler.ReduceLROnPlateau,
    }

    @classmethod
    def create_model(cls, model_type: ModelType, **kwargs) -> nn.Module:
        """모델 생성"""
        if model_type not in cls.MODELS:
            raise ValueError(f"지원하지 않는 모델: {model_type}")

        return cls.MODELS[model_type](**kwargs)

    @classmethod
    def create_loss(cls, loss_type: LossType, **kwargs) -> nn.Module:
        """손실함수 생성"""
        if loss_type not in cls.LOSSES:
            raise ValueError(f"지원하지 않는 손실함수: {loss_type}")

        return cls.LOSSES[loss_type](**kwargs)

    @classmethod
    def create_optimizer(cls, optimizer_type: OptimizerType, model_params, **kwargs) -> torch.optim.Optimizer:
        """옵티마이저 생성"""
        if optimizer_type not in cls.OPTIMIZERS:
            raise ValueError(f"지원하지 않는 옵티마이저: {optimizer_type}")

        return cls.OPTIMIZERS[optimizer_type](model_params, **kwargs)

    @classmethod
    def create_scheduler(cls, scheduler_type: SchedulerType, optimizer, **kwargs):
        """스케줄러 생성"""
        if scheduler_type == SchedulerType.none:
            return None

        if scheduler_type not in cls.SCHEDULERS:
            raise ValueError(f"지원하지 않는 스케줄러: {scheduler_type}")

        return cls.SCHEDULERS[scheduler_type](optimizer, **kwargs)

    @classmethod
    def register_model(cls, model_type: ModelType, model_class: Type[nn.Module]):
        """새 모델 등록"""
        cls.MODELS[model_type] = model_class

    @classmethod
    def register_loss(cls, loss_type: LossType, loss_class: Type[nn.Module]):
        """새 손실함수 등록"""
        cls.LOSSES[loss_type] = loss_class

    @classmethod
    def list_available(cls) -> Dict[str, list]:
        """사용 가능한 컴포넌트 목록 반환"""
        return {
            'models': [m.value for m in cls.MODELS.keys()],
            'losses': [l.value for l in cls.LOSSES.keys()],
            'optimizers': [o.value for o in cls.OPTIMIZERS.keys()],
            'schedulers': [s.value for s in cls.SCHEDULERS.keys()],
        }