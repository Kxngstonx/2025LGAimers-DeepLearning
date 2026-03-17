from typing import List, Dict, Any, Optional, Union
import os

import numpy as np
import torch
from tqdm import tqdm

from ..train.Trainer import Trainer
from ..train.ModelFactory import ModelFactory
from ..data_processing.DataProcessor import DataProcessor
from ..config.Config import Config, Config
from ..utils.Logger import get_logger, LogLevel


class Predictor:
    """
    학습된 모델을 사용한 예측 인터페이스
    """

    def __init__(self, trainer: Trainer = None):
        self.trainer = trainer
        self.experiment_results = None
        self.logger = get_logger()
        
        # 사전 훈련된 모델 관련 속성
        self.pretrained_model = None
        self.pretrained_config = None
        self.pretrained_preprocessing_components = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def run_full_pipeline(self) -> Dict[str, Any]:
        """전체 파이프라인 실행 (훈련 → 예측 → 후처리)"""
        self.logger.info("전체 예측 파이프라인 실행", level=LogLevel.LEVEL1)

        # 훈련 실행
        self.experiment_results = self.trainer.run_all_experiments()

        return self.experiment_results

    def load_pretrained_model(self, model_path: str) -> bool:
        """
        FlexibleTrainer로 저장된 포괄적 모델 정보를 로드
        
        .pt 파일은 다음 구조를 가져야 합니다:
        {
            'model_state_dict': model.state_dict(),
            'best_val_score': best_val_score,
            'model_info': {...},
            'loss_info': {...},
            'preprocessing_components': {...},
            'experiment_config': {...},
            'training_info': {...}
        }
        
        Args:
            model_path: .pt 파일 경로
            
        Returns:
            bool: 로드 성공 여부
        """
        try:
            if not os.path.exists(model_path):
                self.logger.error(f"모델 파일을 찾을 수 없습니다: {model_path}")
                return False
            
            self.logger.info(f"FlexibleTrainer로 저장된 모델 로드 중: {model_path}", level=LogLevel.LEVEL1)
            
            # .pt 파일 로드
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # 필수 키 확인 (새로운 Trainer 형식)
            required_keys = ['model_state_dict', 'best_val_score', 'model_info', 'preprocessing_components', 'experiment_config']
            missing_keys = [key for key in required_keys if key not in checkpoint]
            
            if missing_keys:
                self.logger.error(f"모델 파일에 필수 키가 없습니다: {missing_keys}")
                self.logger.error("FlexibleTrainer로 저장된 모델이 아닙니다. 새로운 형식의 모델을 사용하세요.")
                return False
            
            # 저장된 정보 로드
            model_info = checkpoint['model_info']
            loss_info = checkpoint.get('loss_info', {})
            best_val_score = checkpoint['best_val_score']
            training_info = checkpoint.get('training_info', {})
            
            # 실험 설정을 FlexibleConfig로 재구성
            from ..config.Configurator import Configurator
            experiment_config_dict = checkpoint['experiment_config']
            self.pretrained_config = Configurator._parse_yaml_data(experiment_config_dict)
            
            # 전처리 컴포넌트 로드
            self.pretrained_preprocessing_components = checkpoint['preprocessing_components']
            self.logger.info(f"전처리 컴포넌트 로드 완료:")
            self.logger.info(f"  - 라벨 인코더: {len(self.pretrained_preprocessing_components['label_encoders'])}개")
            self.logger.info(f"  - 스케일러: {len(self.pretrained_preprocessing_components['scalers'])}개")
            
            # 각 컴포넌트 상세 정보
            if self.pretrained_preprocessing_components['label_encoders']:
                encoder_info = list(self.pretrained_preprocessing_components['label_encoders'].keys())
                self.logger.info(f"  - 인코더 대상 컬럼: {encoder_info}")
            if self.pretrained_preprocessing_components['scalers']:
                scaler_info = list(self.pretrained_preprocessing_components['scalers'].keys())
                self.logger.info(f"  - 스케일러 대상 컬럼: {scaler_info}")
            
            # 모델 정보 로깅
            self.logger.info(f"저장된 모델 정보:")
            self.logger.info(f"  - 모델 타입: {model_info['model_type']}")
            self.logger.info(f"  - 모델 변형: {model_info['model_variant']}")
            self.logger.info(f"  - 손실함수: {loss_info.get('loss_type', 'Unknown')}({loss_info.get('loss_variant', 'Unknown')})")
            self.logger.info(f"  - 최고 검증 SMAPE: {best_val_score:.4f}")
            if training_info:
                self.logger.info(f"  - 훈련 에포크: {training_info.get('epochs_trained', 'Unknown')}")
                self.logger.info(f"  - 훈련 일시: {training_info.get('timestamp', 'Unknown')}")
            
            # 모델 생성을 위한 파라미터 준비
            model_params = model_info.get('model_params', {})
            self.logger.debug(f"모델 파라미터: {model_params}")
            
            # 모델 생성
            self.logger.info("모델 인스턴스 생성 중...")
            from ..config.Config import ModelType
            model_type_enum = ModelType(model_info['model_type'])
            
            self.pretrained_model = ModelFactory.create_model(
                model_type_enum,
                **model_params
            )
            
            # 모델 상태 로드
            self.logger.info("모델 가중치 로드 중...")
            self.pretrained_model.load_state_dict(checkpoint['model_state_dict'])
            
            # 모델을 디바이스로 이동 및 평가 모드 설정
            self.pretrained_model.to(self.device)
            self.pretrained_model.eval()
            
            # 모델 정보 출력
            total_params = sum(p.numel() for p in self.pretrained_model.parameters())
            self.logger.success(f"모델 로드 완료!")
            self.logger.info(f"  - 총 파라미터 수: {total_params:,}")
            self.logger.info(f"  - 디바이스: {self.device}")
            self.logger.info(f"  - 모델 모드: eval")
            
            return True
            
        except Exception as e:
            self.logger.error(f"모델 로드 실패: {str(e)}")
            self.pretrained_model = None
            self.pretrained_config = None
            return False

    def predict_from_pretrained(self, save_results: bool = True) -> Optional[List[np.ndarray]]:
        """
        사전 훈련된 모델을 사용하여 테스트 데이터에 대한 예측 수행
        
        모델에 저장된 config를 사용하여 데이터 처리 및 예측을 수행합니다.
        
        Args:
            save_results: 결과를 파일로 저장할지 여부
            
        Returns:
            List[np.ndarray]: 예측 결과 리스트 (각 테스트 파일별)
        """
        if self.pretrained_model is None:
            self.logger.error("먼저 load_pretrained_model()을 실행하여 모델을 로드하세요")
            return None
            
        if self.pretrained_config is None:
            self.logger.error("모델에 저장된 설정이 없습니다")
            return None
            
        try:
            self.logger.info("사전 훈련된 모델로 예측 시작", level=LogLevel.LEVEL1)
            self.logger.info("모델에 저장된 설정 사용")
            
            # 데이터 프로세서 초기화 (저장된 전처리 컴포넌트 전달)
            self.logger.info("데이터 처리 준비 중...", level=LogLevel.LEVEL2)
            data_processor = DataProcessor(self.pretrained_config, self.pretrained_preprocessing_components)
            
            # 테스트 데이터 준비
            self.logger.info("테스트 데이터 로드 및 전처리 중...", level=LogLevel.LEVEL2)
            test_loaders = data_processor.prepare_test_data()
            
            # 예측 수행
            self.logger.info("예측 수행 중...", level=LogLevel.LEVEL2)
            predictions = self._predict_with_model(self.pretrained_model, test_loaders)
            
            # 후처리 및 저장
            if save_results:
                self.logger.info("예측 결과 후처리 및 저장 중...", level=LogLevel.LEVEL2)
                submission = data_processor.postprocess_and_save(predictions, save=True)
                self.logger.success(f"예측 결과 저장 완료")
            
            self.logger.success(f"사전 훈련된 모델 예측 완료 - {len(predictions)}개 테스트 파일 처리")
            return predictions
            
        except Exception as e:
            self.logger.error(f"예측 실패: {str(e)}")
            return None

    def _predict_with_model(self, model: torch.nn.Module, test_loaders: List) -> List[np.ndarray]:
        """
        주어진 모델로 테스트 데이터에 대한 예측 수행
        
        Args:
            model: 예측에 사용할 모델
            test_loaders: 테스트 데이터로더 리스트
            
        Returns:
            List[np.ndarray]: 예측 결과 리스트
        """
        model.eval()
        predictions = []

        with torch.no_grad():
            test_pbar = tqdm(test_loaders, desc="테스트 예측")
            for test_loader in test_pbar:
                loader_predictions = []
                for X in test_loader:
                    X = X.to(self.device)
                    outputs = model(X)
                    loader_predictions.append(outputs[:, :, 0].cpu().numpy())

                # 예측 결과 합치기
                predictions.append(np.concatenate(loader_predictions, axis=0))

        return predictions

    def get_best_model_predictions(self) -> Optional[List[np.ndarray]]:
        """최고 성능 모델의 예측 결과 반환"""
        if not self.experiment_results:
            self.logger.warning("먼저 run_full_pipeline()을 실행하세요", level=LogLevel.LEVEL1)
            return None

        best_result = None
        best_score = float('inf')

        for result in self.experiment_results['all_results']:
            if result['best_val_score'] < best_score:
                best_score = result['best_val_score']
                best_result = result

        return best_result['predictions'] if best_result else None

    def get_experiment_summary(self) -> Optional[Dict[str, Any]]:
        """실험 결과 요약 반환"""
        if not self.experiment_results:
            return None

        return {
            'best_model': self.experiment_results['best_model'],
            'best_score': self.experiment_results['best_score'],
            'model_rankings': sorted(
                [
                    {
                        'model': r['model_type'],
                        'score': r['best_val_score'],
                        'epochs': r['epochs_trained']
                    }
                    for r in self.experiment_results['all_results']
                ],
                key=lambda x: x['score']
            )
        }

    def print_results(self):
        """결과를 예쁘게 출력"""
        summary = self.get_experiment_summary()
        if not summary:
            self.logger.warning("실험 결과가 없습니다", level=LogLevel.LEVEL1)
            return

        self.logger.section("실험 결과 요약")

        self.logger.success(f"최고 성능 모델: {summary['best_model']}")
        self.logger.success(f"최고 검증 SMAPE: {summary['best_score']:.4f}")

        self.logger.info("모델 성능 순위:", level=LogLevel.LEVEL1)
        for i, model_result in enumerate(summary['model_rankings'], 1):
            self.logger.info(f"{i}. {model_result['model']:<10} - "
                  f"SMAPE: {model_result['score']:.4f} "
                  f"({model_result['epochs']} epochs)", level=LogLevel.LEVEL1)