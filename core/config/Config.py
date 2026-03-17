from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any, Union

from ..utils.Logger import LogLevel


# 전처리 관련 Enum
class EncodingType(Enum):
    LabelEncoder = "LabelEncoder"
    EmbeddingEncoder = "EmbeddingEncoder"
    # OneHotEncoder = "OneHotEncoder"
    # BinaryEncoder = "BinaryEncoder"
    # TargetEncoder = "TargetEncoder"
    # CountEncoder = "CountEncoder"
    # FrequencyEncoder = "FrequencyEncoder"
    # BaseNEncoder = "BaseNEncoder"
    # CatBoostEncoder = "CatBoostEncoder"
    # WOEEncoder = "WOEEncoder"
    # TargetDummiesEncoder = "TargetDummiesEncoder"
    
    def __eq__(self, other):
        """값 기반 비교를 위한 메서드 (모듈 재로드 시 Enum 비교 문제 해결)"""
        if isinstance(other, EncodingType):
            return self.value == other.value
        return self.value == other
    
    def __hash__(self):
        """해시 값은 value 기반으로 생성"""
        return hash(self.value)

class ScalingType(Enum):
    none = "none"
    StandardScaler = "StandardScaler"
    MinMaxScaler = "MinMaxScaler"
    LogTransform = "LogTransform"
    
    def __eq__(self, other):
        """값 기반 비교를 위한 메서드 (모듈 재로드 시 Enum 비교 문제 해결)"""
        if isinstance(other, ScalingType):
            return self.value == other.value
        return self.value == other
    
    def __hash__(self):
        """해시 값은 value 기반으로 생성"""
        return hash(self.value)

#모델 관련 Enum
class ModelType(Enum):
    DLinear = "DLinear"
    NLinear = "NLinear"
    RLinear = "RLinear"
    GLinear = "GLinear"
    PatchTSTRevINSalesHead = "PatchTSTRevINSalesHead"
    DLinearRevIN = "DLinearRevIN"
    xPatch = "xPatch"
    CustomPatchTSTForPrediction = "CustomPatchTSTForPrediction"
    PatchMixer = "PatchMixer"
    HDMixer = "HDMixer"
    # ML Models
    LGBM = "LGBM"
    LightGBM = "LightGBM"
    RandomForest = "RandomForest"
    GradientBoosting = "GradientBoosting"
    LinearRegression = "LinearRegression"
    Ridge = "Ridge"
    Lasso = "Lasso"
    
    def __eq__(self, other):
        """값 기반 비교를 위한 메서드 (모듈 재로드 시 Enum 비교 문제 해결)"""
        if isinstance(other, ModelType):
            return self.value == other.value
        return self.value == other
    
    def __hash__(self):
        """해시 값은 value 기반으로 생성"""
        return hash(self.value)


class LossType(Enum):
    MSELoss = "MSELoss"
    L1Loss = "L1Loss"
    SmoothL1Loss = "SmoothL1Loss"
    TargetColumnLoss = "TargetColumnLoss"
    SMAPEInspiredLoss = "SMAPEInspiredLoss"
    WeightedSMAPEInspiredLoss = "WeightedSMAPEInspiredLoss"
    SMAPESpikeLoss = "SMAPESpikeLoss"
    SMAPELoss = "SMAPELoss"
    AntiUnderPredictionLoss = "AntiUnderPredictionLoss"

    def __eq__(self, other):
        """값 기반 비교를 위한 메서드 (모듈 재로드 시 Enum 비교 문제 해결)"""
        if isinstance(other, LossType):
            return self.value == other.value
        return self.value == other
    
    def __hash__(self):
        """해시 값은 value 기반으로 생성"""
        return hash(self.value)


class OptimizerType(Enum):
    Adam = "Adam"
    AdamW = "AdamW"
    SGD = "SGD"
    
    def __eq__(self, other):
        """값 기반 비교를 위한 메서드 (모듈 재로드 시 Enum 비교 문제 해결)"""
        if isinstance(other, OptimizerType):
            return self.value == other.value
        return self.value == other
    
    def __hash__(self):
        """해시 값은 value 기반으로 생성"""
        return hash(self.value)


class SchedulerType(Enum):
    none = "none"
    StepLR = "StepLR"
    CosineAnnealingLR = "CosineAnnealingLR"
    ReduceLROnPlateau = "ReduceLROnPlateau"
    
    def __eq__(self, other):
        """값 기반 비교를 위한 메서드 (모듈 재로드 시 Enum 비교 문제 해결)"""
        if isinstance(other, SchedulerType):
            return self.value == other.value
        return self.value == other
    
    def __hash__(self):
        """해시 값은 value 기반으로 생성"""
        return hash(self.value)

class PostProcessType(Enum):
    clip_to_zero = "clip_to_zero"
    clip_to_one = "clip_to_one"
    clip_to_mode = "clip_to_mode"
    none = "none"
    
    def __eq__(self, other):
        """값 기반 비교를 위한 메서드 (모듈 재로드 시 Enum 비교 문제 해결)"""
        if isinstance(other, PostProcessType):
            return self.value == other.value
        return self.value == other
    
    def __hash__(self):
        """해시 값은 value 기반으로 생성"""
        return hash(self.value)


# Config 구조
@dataclass
class PathConfig:
    train_csv: str = "data/train/train.csv"
    test_pattern: str = "data/test/TEST_*.csv"
    static_csv: str = "data/static_feature.csv"
    submit_csv: str = "data/sample_submission.csv"
    result_dir: str = field(default_factory=lambda: f"result/{datetime.now().strftime('%Y%m%d_%H%M%S')}_default")


@dataclass
class DatasetConfig:
    # 컬럼 정의
    target_col: str = "sales"
    date_col: str = "days_since_reference"
    group_col: str = "store_menu"

    # 시퀀스 설정
    train_window: int = 28
    forecast_window: int = 7
    validation_ratio: float = 0.2

    # 데이터로더 설정
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = False
    prefetch_factor: int = 2
    persistent_workers: bool = True

    # 샘플링 설정
    sampling_method: str = "temporal_stratified"
    temporal_bin: int = 12
    random_seed: int = 42


@dataclass
class PreProcessConfig:
    # 특성 엔지니어링
    timestamp_ext: bool = True

    outlier_iqr_replacement: bool = False
    new_menu_zero_dropper: bool = False
    is_sales: bool = False
    historical_season_avg: bool = False

    lag: bool = False
    rolling_mean: bool = False
    season: bool = False
    is_holiday: bool = False
    zero_streak: bool = False

    # 전처리
    encoding_features: List[str] = field(default_factory=lambda: ["store_menu", 'is_group_menu', 'store', 'category', 'season'])
    encoding_method: EncodingType = EncodingType.LabelEncoder

    scaling_features: List[str] = field(default_factory=lambda: ["sales"])
    scaling_method: ScalingType = ScalingType.none


@dataclass
class TrainConfig:
    epochs: int = 50
    learning_rate: float = 0.0001
    optimizer: OptimizerType = OptimizerType.AdamW
    scheduler: SchedulerType = SchedulerType.ReduceLROnPlateau

    # 정규화/클리핑
    weight_decay: float = 0.001
    gradient_clip: Optional[float] = None

    # 얼리스탑
    early_stopping: bool = True
    patience: int = 10
    
    # Cross-validation 설정
    use_cross_validation: bool = False
    cv_n_splits: int = 5
    cv_test_size_ratio: float = 0.2


@dataclass
class PostProcessConfig:
    method: PostProcessType = PostProcessType.clip_to_one
    round_to_int: bool = True


@dataclass
class SettingConfig:
    name: str = "default"
    device: str = "auto"  # auto, cuda, cpu
    seed: int = 42
    log_level: LogLevel = LogLevel.LEVEL2  # 로그 레벨 (NONE, LEVEL1, LEVEL2, LEVEL3)
    plot: bool = True
    save_plot: bool = True
    save_model_best: bool = True
    save_model_final: bool = True

# 새로운 유연한 모델/손실함수 설정 구조
@dataclass
class ModelVariant:
    """모델 변형 설정"""
    name: str
    params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ModelConfig:
    """유연한 모델 설정"""
    base_params: Dict[str, Any] = field(default_factory=dict)
    variants: List[ModelVariant] = field(default_factory=list)

@dataclass
class LossVariant:
    """손실함수 변형 설정"""
    name: str
    params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LossConfig:
    """유연한 손실함수 설정"""
    base_params: Dict[str, Any] = field(default_factory=dict)
    variants: List[LossVariant] = field(default_factory=list)

@dataclass
class ExperimentSpec:
    """실험 사양 - 모델과 손실함수 조합"""
    model_type: str
    model_variant: str
    loss_type: str
    loss_variant: str
    train_params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Config:
    """새로운 유연한 설정 구조"""
    # 기본 설정들
    path: PathConfig = field(default_factory=PathConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    preprocess: PreProcessConfig = field(default_factory=PreProcessConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    postprocess: PostProcessConfig = field(default_factory=PostProcessConfig)
    setting: SettingConfig = field(default_factory=SettingConfig)
    
    # 새로운 유연한 구조
    models: Dict[str, ModelConfig] = field(default_factory=dict)
    losses: Dict[str, LossConfig] = field(default_factory=dict)
    experiments: List[ExperimentSpec] = field(default_factory=list)
    
    def get_model_params(self, model_type: str, variant_name: str) -> Dict[str, Any]:
        """모델 타입과 변형명으로 최종 모델 파라미터 반환"""
        if model_type not in self.models:
            raise ValueError(f"모델 타입 '{model_type}'이 설정에 없습니다.")
        
        model_config = self.models[model_type]
        base_params = model_config.base_params.copy()
        
        # 변형 파라미터 찾기
        variant_params = {}
        for variant in model_config.variants:
            if variant.name == variant_name:
                variant_params = variant.params
                break
        else:
            raise ValueError(f"모델 변형 '{variant_name}'이 '{model_type}'에 없습니다.")
        
        # base_params에 variant_params 오버라이드
        base_params.update(variant_params)
        return base_params
    
    def get_loss_params(self, loss_type: str, variant_name: str) -> Dict[str, Any]:
        """손실함수 타입과 변형명으로 최종 손실함수 파라미터 반환"""
        if loss_type not in self.losses:
            raise ValueError(f"손실함수 타입 '{loss_type}'이 설정에 없습니다.")
        
        loss_config = self.losses[loss_type]
        base_params = loss_config.base_params.copy()
        
        # 변형 파라미터 찾기
        variant_params = {}
        for variant in loss_config.variants:
            if variant.name == variant_name:
                variant_params = variant.params
                break
        else:
            raise ValueError(f"손실함수 변형 '{variant_name}'이 '{loss_type}'에 없습니다.")
        
        # base_params에 variant_params 오버라이드
        base_params.update(variant_params)
        return base_params

    def __post_init__(self):
        """Config 초기화 후 실행되는 메서드"""
        # ExperimentConfig의 name을 사용하여 result_dir 업데이트
        if self.path.result_dir.endswith("_default"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.path.result_dir = f"result/{timestamp}_{self.setting.name}"