import yaml
from datetime import datetime
from typing import Dict, Any, List
from dataclasses import asdict

from .Config import (
    Config, ModelVariant, ModelConfig,
    LossVariant, LossConfig, ExperimentSpec,
    PathConfig, DatasetConfig, PreProcessConfig, TrainConfig,
    PostProcessConfig, SettingConfig,
    EncodingType, ScalingType, OptimizerType, SchedulerType, PostProcessType
)
from ..utils.Logger import LogLevel


class Configurator:
    """
    새로운 유연한 YAML 설정을 파싱하여 Config 객체를 생성하는 클래스
    """
    
    @staticmethod
    def load_from_yaml(yaml_path: str) -> Config:
        """YAML 파일에서 Config 로드"""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            yaml_data = yaml.safe_load(f)
        
        return Configurator._parse_yaml_data(yaml_data)
    
    @staticmethod
    def _parse_yaml_data(yaml_data: Dict[str, Any]) -> Config:
        """YAML 데이터를 FlexibleConfig로 변환"""
        
        # 기본 설정들 파싱
        path_config = Configurator._parse_path_config(yaml_data.get('path', {}))
        dataset_config = Configurator._parse_dataset_config(yaml_data.get('dataset', {}))
        preprocess_config = Configurator._parse_preprocess_config(yaml_data.get('preprocess', {}))
        train_config = Configurator._parse_train_config(yaml_data.get('train', {}))
        postprocess_config = Configurator._parse_postprocess_config(yaml_data.get('postprocess', {}))
        experiment_config = Configurator._parse_setting_config(yaml_data.get('setting', {}))
        
        # 새로운 유연한 구조 파싱
        models = Configurator._parse_models(yaml_data.get('models', {}))
        losses = Configurator._parse_losses(yaml_data.get('losses', {}))
        experiments = Configurator._parse_experiments(yaml_data.get('experiments', []))
        
        return Config(
            path=path_config,
            dataset=dataset_config,
            preprocess=preprocess_config,
            train=train_config,
            postprocess=postprocess_config,
            setting=experiment_config,
            models=models,
            losses=losses,
            experiments=experiments
        )
    
    @staticmethod
    def _parse_path_config(path_data: Dict[str, Any]) -> PathConfig:
        """PathConfig 파싱"""
        return PathConfig(
            train_csv=path_data.get('train_csv', "data/train/train.csv"),
            test_pattern=path_data.get('test_pattern', "data/test/TEST_*.csv"),
            static_csv=path_data.get('static_csv', "data/static_feature.csv"),
            submit_csv=path_data.get('submit_csv', "data/sample_submission.csv"),
            result_dir=path_data.get('result_dir', f"result/{datetime.now().strftime('%Y%m%d_%H%M%S')}_default")
        )
    
    @staticmethod
    def _parse_dataset_config(dataset_data: Dict[str, Any]) -> DatasetConfig:
        """DatasetConfig 파싱"""
        return DatasetConfig(
            target_col=dataset_data.get('target_col', 'sales'),
            date_col=dataset_data.get('date_col', 'days_since_reference'),
            group_col=dataset_data.get('group_col', 'store_menu'),
            train_window=dataset_data.get('train_window', 28),
            forecast_window=dataset_data.get('forecast_window', 7),
            validation_ratio=dataset_data.get('validation_ratio', 0.05),
            batch_size=dataset_data.get('batch_size', 32),
            num_workers=dataset_data.get('num_workers', 4),
            pin_memory=dataset_data.get('pin_memory', False),
            sampling_method=dataset_data.get('sampling_method', 'temporal_stratified'),
            random_seed=dataset_data.get('random_seed', 42)
        )
    
    @staticmethod
    def _parse_preprocess_config(preprocess_data: Dict[str, Any]) -> PreProcessConfig:
        """PreProcessConfig 파싱"""
        return PreProcessConfig(
            timestamp_ext=preprocess_data.get('timestamp_ext', True),
            outlier_iqr_replacement=preprocess_data.get('outlier_iqr_replacement', False),
            new_menu_zero_dropper=preprocess_data.get('new_menu_zero_dropper', False),
            is_sales=preprocess_data.get('is_sales', False),
            historical_season_avg=preprocess_data.get('historical_season_avg', False),
            lag=preprocess_data.get('lag', False),
            rolling_mean=preprocess_data.get('rolling_mean', False),
            season=preprocess_data.get('season', False),
            is_holiday=preprocess_data.get('is_holiday', False),
            zero_streak=preprocess_data.get('zero_streak', False),

            encoding_features=preprocess_data.get('encoding_features', ["store_menu", 'is_group_menu', 'store', 'category', 'season']),
            encoding_method=EncodingType(preprocess_data.get('encoding_method', 'LabelEncoder')),

            scaling_features=preprocess_data.get('scaling_features', ["sales"]),
            scaling_method=ScalingType(preprocess_data.get('scaling_method', 'none'))
        )
    
    @staticmethod
    def _parse_train_config(train_data: Dict[str, Any]) -> TrainConfig:
        """TrainConfig 파싱"""
        return TrainConfig(
            epochs=train_data.get('epochs', 50),
            learning_rate=train_data.get('learning_rate', 0.0001),
            optimizer=OptimizerType(train_data.get('optimizer', 'AdamW')),
            scheduler=SchedulerType(train_data.get('scheduler', 'ReduceLROnPlateau')),
            weight_decay=train_data.get('weight_decay', 0.001),
            gradient_clip=train_data.get('gradient_clip'),
            early_stopping=train_data.get('early_stopping', True),
            patience=train_data.get('patience', 10),
            use_cross_validation=train_data.get('use_cross_validation', False),
            cv_n_splits=train_data.get('cv_n_splits', 5),
            cv_test_size_ratio=train_data.get('cv_test_size_ratio', 0.2),
        )
    
    @staticmethod
    def _parse_postprocess_config(postprocess_data: Dict[str, Any]) -> PostProcessConfig:
        """PostProcessConfig 파싱"""
        return PostProcessConfig(
            method=PostProcessType(postprocess_data.get('method', 'clip_to_one')),
            round_to_int=postprocess_data.get('round_to_int', True)
        )
    
    @staticmethod
    def _parse_setting_config(setting_data: Dict[str, Any]) -> SettingConfig:
        """SettingConfig 파싱"""
        return SettingConfig(
            name=setting_data.get('name', 'default'),
            device=setting_data.get('device', 'auto'),
            seed=setting_data.get('seed', 42),
            log_level=LogLevel(setting_data.get('log_level', 2)),
            plot=setting_data.get('plot', True),
            save_plot=setting_data.get('save_plot', True),
            save_model_best=setting_data.get('save_model_best', False),
            save_model_final=setting_data.get('save_model_final', True),
        )
    
    @staticmethod
    def _parse_models(models_data: Dict[str, Any]) -> Dict[str, ModelConfig]:
        """모델 설정 파싱"""
        models = {}
        
        for model_type, model_config in models_data.items():
            base_params = model_config.get('base_params', {})
            variants_data = model_config.get('variants', [])
            
            variants = []
            for variant_data in variants_data:
                variant = ModelVariant(
                    name=variant_data['name'],
                    params=variant_data.get('params', {})
                )
                variants.append(variant)
            
            models[model_type] = ModelConfig(
                base_params=base_params,
                variants=variants
            )
        
        return models
    
    @staticmethod
    def _parse_losses(losses_data: Dict[str, Any]) -> Dict[str, LossConfig]:
        """손실함수 설정 파싱"""
        losses = {}
        
        for loss_type, loss_config in losses_data.items():
            base_params = loss_config.get('base_params', {})
            variants_data = loss_config.get('variants', [])
            
            variants = []
            for variant_data in variants_data:
                variant = LossVariant(
                    name=variant_data['name'],
                    params=variant_data.get('params', {})
                )
                variants.append(variant)
            
            losses[loss_type] = LossConfig(
                base_params=base_params,
                variants=variants
            )
        
        return losses
    
    @staticmethod
    def _parse_experiments(experiments_data: List[Dict[str, Any]]) -> List[ExperimentSpec]:
        """실험 사양 파싱"""
        experiments = []
        
        for exp_data in experiments_data:
            experiment = ExperimentSpec(
                model_type=exp_data['model_type'],
                model_variant=exp_data['model_variant'],
                loss_type=exp_data['loss_type'],
                loss_variant=exp_data['loss_variant'],
                train_params=exp_data.get('train_params', {})
            )
            experiments.append(experiment)
        
        return experiments
    
    @staticmethod
    def save_to_yaml(config: Config, yaml_path: str):
        """FlexibleConfig를 YAML 파일로 저장"""
        
        # FlexibleConfig를 딕셔너리로 변환
        config_dict = {
            'setting': asdict(config.setting),
            'path': asdict(config.path),
            'dataset': asdict(config.dataset),
            'preprocess': asdict(config.preprocess),
            'train': asdict(config.train),
            'postprocess': asdict(config.postprocess),
            'models': {},
            'losses': {},
            'experiments': []
        }
        
        # 모델 설정 변환
        for model_type, model_config in config.models.items():
            config_dict['models'][model_type] = {
                'base_params': model_config.base_params,
                'variants': [asdict(variant) for variant in model_config.variants]
            }
        
        # 손실함수 설정 변환
        for loss_type, loss_config in config.losses.items():
            config_dict['losses'][loss_type] = {
                'base_params': loss_config.base_params,
                'variants': [asdict(variant) for variant in loss_config.variants]
            }
        
        # 실험 사양 변환
        for experiment in config.experiments:
            config_dict['experiments'].append(asdict(experiment))
        
        # Enum 값들을 문자열로 변환
        Configurator._convert_enums_to_strings(config_dict)
        
        # YAML 파일로 저장
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True, indent=2)
    
    @staticmethod
    def _convert_enums_to_strings(data):
        """딕셔너리 내의 Enum 값들을 문자열로 변환"""
        if isinstance(data, dict):
            for key, value in data.items():
                if hasattr(value, 'value'):  # Enum 객체인 경우
                    data[key] = value.value
                elif isinstance(value, (dict, list)):
                    Configurator._convert_enums_to_strings(value)
        elif isinstance(data, list):
            for i, item in enumerate(data):
                if hasattr(item, 'value'):  # Enum 객체인 경우
                    data[i] = item.value
                elif isinstance(item, (dict, list)):
                    Configurator._convert_enums_to_strings(item)


# 편의 함수들
def load_flexible_config(yaml_path: str) -> Config:
    """YAML 파일에서 Config 로드하는 편의 함수"""
    return Configurator.load_from_yaml(yaml_path)

def save_flexible_config(config: Config, yaml_path: str):
    """FlexibleConfig를 YAML 파일로 저장하는 편의 함수"""
    Configurator.save_to_yaml(config, yaml_path)