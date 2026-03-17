import copy
import json
import os
from dataclasses import asdict
from datetime import datetime
from typing import List

import numpy as np
import torch
from tqdm import tqdm

from .Plotter import Plotter
from ..config.Config import Config, ModelType, LossType
from ..data_processing.DataProcessor import DataProcessor
from ..data_processing.TimeSeriesCrossValidator import TimeSeriesCrossValidator
from ..metric.WeightedSMAPEMetric import WeightedSMAPEMetric
from ..train.ModelFactory import ModelFactory
from ..utils.Logger import initialize_logger, get_logger, LogLevel
from ..utils.Seed import set_all_seeds


class Trainer:
    """
    새로운 유연한 설정 구조를 사용하는 트레이너
    """

    def __init__(self, config: Config):
        self.config = config
        self.device = self._setup_device()

        # 결과 디렉토리 생성
        os.makedirs(self.config.path.result_dir, exist_ok=True)

        # 로거 설정
        initialize_logger(
            log_file=os.path.join(self.config.path.result_dir, f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            level=self.config.setting.log_level,
            force_reinit=True
        )
        self.logger = get_logger()

        # 시드 설정
        set_all_seeds(self.config.setting.seed)

        # 데이터 프로세서 초기화
        self.data_processor = DataProcessor(self.config)

        # 메트릭 초기화
        self.metric = WeightedSMAPEMetric(device=self.device, post_processor=self.data_processor.postprocessor)

        # 플로터 구성
        self.plotter = Plotter()

        # 다중 실험 결과 저장
        self.experiment_results = []

        # 실험 이름
        self.experiment_name = ""

    def _setup_device(self):
        """디바이스 설정"""
        if self.config.setting.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self.config.setting.device

        self.logger.info(f"디바이스 설정: {device}", level=LogLevel.LEVEL1) if hasattr(self, 'logger') else None
        return device

    def _setup_model(self, experiment_spec):
        """실험 사양에 따라 모델 생성 및 설정"""
        model_type = experiment_spec.model_type
        model_variant = experiment_spec.model_variant

        # 모델 파라미터 동적 추출
        model_params = self.config.get_model_params(model_type, model_variant)

        # ModelType enum으로 변환
        try:
            model_type_enum = ModelType(model_type)
        except ValueError:
            raise ValueError(f"지원하지 않는 모델 타입: {model_type}")

        model = ModelFactory.create_model(model_type_enum, **model_params).to(self.device)

        self.logger.info(f"{model_type} ({model_variant}) 모델 생성 완료", level=LogLevel.LEVEL2)
        return model

    def _setup_training_components(self, experiment_spec):
        """실험 사양에 따라 훈련 컴포넌트 설정"""
        loss_type = experiment_spec.loss_type
        loss_variant = experiment_spec.loss_variant

        # 손실함수 파라미터 동적 추출
        loss_params = self.config.get_loss_params(loss_type, loss_variant)

        # LossType enum으로 변환
        try:
            loss_type_enum = LossType(loss_type)
        except ValueError:
            raise ValueError(f"지원하지 않는 손실함수 타입: {loss_type}")

        # 손실함수 생성
        criterion = ModelFactory.create_loss(loss_type_enum, **loss_params)

        # 훈련 파라미터 (기본값 + 실험별 오버라이드)
        train_params = asdict(self.config.train)
        train_params.update(experiment_spec.train_params)

        optimizer = ModelFactory.create_optimizer(
            optimizer_type=self.config.train.optimizer,
            model_params=self.model.parameters(),
            lr=train_params.get('learning_rate', self.config.train.learning_rate),
            weight_decay=train_params.get('weight_decay', self.config.train.weight_decay)
        )

        # 스케줄러
        scheduler = ModelFactory.create_scheduler(
            scheduler_type=self.config.train.scheduler,
            optimizer=optimizer
        )

        # 가중치 메트릭 설정
        store_weights = loss_params.get("store_weights")
        if store_weights:
            self.logger.debug(f"손실함수에 매장 별 가중치가 적용됨에 따라 메트릭 값도 가중치로 조정, 가중치: {store_weights}")
            weights = [1.0] * 9
            for idx, weight in store_weights.items():
                weights[idx] = weight
            weights = weights / np.sum(weights)
            self.metric.set_weights(weights)
        else:
            self.metric.clear_weights()

        self.logger.info(f"{loss_type} ({loss_variant}) 손실함수 설정 완료", level=LogLevel.LEVEL2)

        return criterion, optimizer, scheduler, train_params

    def train_single_experiment_with_cv(self, experiment_spec, n_splits, test_size_ratio=0.2):
        """Cross-validation을 사용한 단일 실험 실행"""
        self.logger.section(f"Cross-Validation 실험 시작: {self.experiment_name} ({n_splits} folds)", level=LogLevel.LEVEL1)

        # 현재 실험 사양 저장
        self.current_experiment_spec = experiment_spec

        # 전처리된 데이터 준비
        self.logger.info("Cross-validation용 데이터 준비 중...", level=LogLevel.LEVEL2)
        processed_data = self.data_processor.preprocessor.preprocess(
            self.data_processor.train_data,
            self.data_processor.static_feature
        )

        # TimeSeriesCrossValidator 초기화
        cv_validator = TimeSeriesCrossValidator(
            config=self.config,
            n_splits=n_splits,
            test_size_ratio=test_size_ratio
        )

        # Cross-validation 데이터로더 생성
        cv_loaders = cv_validator.create_cv_dataloaders(processed_data)
        fold_info = cv_validator.get_fold_info(processed_data)

        # 각 fold별 결과 저장
        fold_results = []
        fold_val_scores = []

        for fold_idx, (train_loader, val_loader) in enumerate(cv_loaders):
            self.logger.section(f"Fold {fold_idx + 1}/{len(cv_loaders)} 훈련 시작", level=LogLevel.LEVEL1)

            # 모델 및 훈련 컴포넌트 재초기화 (각 fold마다 새로운 모델)
            self.model = self._setup_model(experiment_spec)
            self.criterion, self.optimizer, self.scheduler, train_params = self._setup_training_components(experiment_spec)

            # 훈련 상태 초기화
            best_val_score = float('inf')
            train_losses = []
            val_scores = []

            # 훈련 루프
            epochs = train_params.get('epochs', self.config.train.epochs)
            patience = train_params.get('patience', self.config.train.patience)
            early_stopping = train_params.get('early_stopping', self.config.train.early_stopping)

            patience_counter = 0

            # Fold 정보 로깅
            info = fold_info[fold_idx]
            self.logger.info(f"Fold {fold_idx + 1} 정보:", level=LogLevel.LEVEL2)
            self.logger.info(f"  - 훈련 데이터: {info['train_size']}행, {info['train_products']}개 제품", level=LogLevel.LEVEL2)
            self.logger.info(f"  - 검증 데이터: {info['val_size']}행, {info['val_products']}개 제품", level=LogLevel.LEVEL2)
            self.logger.info(f"  - 훈련 기간: {info['train_date_range'][0]} ~ {info['train_date_range'][1]}", level=LogLevel.LEVEL2)
            self.logger.info(f"  - 검증 기간: {info['val_date_range'][0]} ~ {info['val_date_range'][1]}", level=LogLevel.LEVEL2)

            # 에폭 별 진행률 표시
            epoch_pbar = tqdm(range(epochs), desc=f"Fold {fold_idx + 1} 진행률")

            for epoch in epoch_pbar:
                # 훈련
                train_loss = self.train_epoch(train_loader)
                train_losses.append(train_loss)

                # 검증
                val_loss, val_score = self.validate(val_loader)
                val_scores.append(val_score['postprocessed'])

                # 스케줄러 업데이트
                if self.scheduler:
                    if hasattr(self.scheduler, 'step'):
                        if 'ReduceLROnPlateau' in str(type(self.scheduler)):
                            self.scheduler.step(val_score['postprocessed'])
                        else:
                            self.scheduler.step()

                # 로깅
                self.logger.info(f"Epoch {epoch + 1:3d}/{epochs}: "
                                 f"Train_Loss={train_loss:.6f}, Val_Loss={val_loss:.6f}, Val_SMAPE={val_score['original']:.4f}, Val_SMAPE(PostProcessed)={val_score['postprocessed']:.4f}",
                                 level=LogLevel.LEVEL1)

                # 최고 성능 업데이트
                if val_score['postprocessed'] < best_val_score:
                    best_val_score = val_score['postprocessed']
                    patience_counter = 0
                else:
                    patience_counter += 1

                # 얼리 스탑핑
                if early_stopping and patience_counter >= patience:
                    self.logger.info(f"Fold {fold_idx + 1} 얼리스탑: {patience} 에포크 동안 개선 없음", level=LogLevel.LEVEL2)
                    break

            # Fold 결과 저장
            fold_result = {
                'fold': fold_idx + 1,
                'best_val_score': best_val_score,
                'final_train_loss': train_losses[-1] if train_losses else None,
                'epochs_trained': len(train_losses),
                'fold_info': info
            }
            fold_results.append(fold_result)
            fold_val_scores.append(best_val_score)

            # Fold 결과 로깅
            self.logger.success(f"Fold {fold_idx + 1} 완료 - 최고 검증 SMAPE: {best_val_score:.4f}", level=LogLevel.LEVEL1)

        # Cross-validation 결과 요약
        mean_cv_score = np.mean(fold_val_scores)
        std_cv_score = np.std(fold_val_scores)

        self.logger.section("Cross-Validation 결과 요약", level=LogLevel.LEVEL1)
        self.logger.info(f"평균 검증 SMAPE: {mean_cv_score:.4f} ± {std_cv_score:.4f}", level=LogLevel.LEVEL1)

        # 각 fold별 상세 결과 로깅
        for result in fold_results:
            self.logger.info(f"Fold {result['fold']}: {result['best_val_score']:.4f} "
                             f"({result['epochs_trained']} epochs)", level=LogLevel.LEVEL1)

        # 최종 모델 훈련 (전체 데이터 사용)
        self.logger.section("최종 모델 훈련 (전체 데이터)", level=LogLevel.LEVEL1)
        final_result = self._train_final_model(experiment_spec)

        # 실험 결과 저장 (CV 정보 포함)
        experiment_result = {
            # 기본 실험 정보
            'experiment_name': self.experiment_name,
            'model_type': experiment_spec.model_type,
            'model_variant': experiment_spec.model_variant,
            'loss_type': experiment_spec.loss_type,
            'loss_variant': experiment_spec.loss_variant,
            'best_val_score': final_result['best_val_score'],  # 최종 모델 성능
            'final_train_loss': final_result['final_train_loss'],
            'epochs_trained': final_result['epochs_trained'],

            # Cross-validation 결과
            'cv_mean_score': mean_cv_score,
            'cv_std_score': std_cv_score,
            'cv_fold_scores': fold_val_scores,
            'cv_fold_results': fold_results,
            'cv_n_splits': n_splits,

            # 재현성을 위한 상세 설정 정보
            'model_params': self.config.get_model_params(experiment_spec.model_type, experiment_spec.model_variant),
            'loss_params': self.config.get_loss_params(experiment_spec.loss_type, experiment_spec.loss_variant),
            'train_params': final_result['train_params'],

            # 데이터셋 설정
            'dataset_config': asdict(self.config.dataset),
            'preprocess_config': asdict(self.config.preprocess),
            'train_config': asdict(self.config.train),

            # 시드 정보
            'seed': self.config.setting.seed,

            # 타임스탬프
            'timestamp': datetime.now().isoformat()
        }

        self.experiment_results.append(experiment_result)

        self.logger.success(f"Cross-Validation 실험 완료: {self.experiment_name}", level=LogLevel.LEVEL1)
        return experiment_result

    def _train_final_model(self, experiment_spec):
        """전체 데이터를 사용한 최종 모델 훈련"""
        self.logger.info("전체 데이터로 최종 모델 훈련 중...", level=LogLevel.LEVEL2)

        # 모델 및 훈련 컴포넌트 재초기화
        self.model = self._setup_model(experiment_spec)
        self.criterion, self.optimizer, self.scheduler, train_params = self._setup_training_components(experiment_spec)

        # 전체 데이터로 훈련/검증 데이터로더 준비
        train_loader, val_loader = self.data_processor.prepare_train_data()

        # 훈련 상태 초기화
        self.best_model = None
        self.best_val_score = float('inf')
        self.train_losses = []
        self.val_scores = []

        # 훈련 루프 (기존 로직과 동일하지만 간소화된 로깅)
        epochs = train_params.get('epochs', self.config.train.epochs)
        patience = train_params.get('patience', self.config.train.patience)
        early_stopping = train_params.get('early_stopping', self.config.train.early_stopping)

        patience_counter = 0
        epoch_pbar = tqdm(range(epochs), desc="최종 모델 훈련")

        for epoch in epoch_pbar:
            # 훈련
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)

            # 검증
            val_loss, val_score = self.validate(val_loader)
            self.val_scores.append(val_score)

            # 스케줄러 업데이트
            if self.scheduler:
                if hasattr(self.scheduler, 'step'):
                    if 'ReduceLROnPlateau' in str(type(self.scheduler)):
                        self.scheduler.step(val_score['postprocessed'])
                    else:
                        self.scheduler.step()

            # 로깅
            self.logger.info(f"Epoch {epoch + 1:3d}/{epochs}: "
                             f"Train_Loss={train_loss:.6f}, Val_Loss={val_loss:.6f}, Val_SMAPE={val_score['original']:.4f}, Val_SMAPE(PostProcessed)={val_score['postprocessed']:.4f}",
                             level=LogLevel.LEVEL1)

            # 플롯 출력
            if self.config.setting.plot:
                save_plot = False
                if self.config.setting.save_plot and epoch % 10 == 0:
                    save_plot = True
                self.plotter.plot(epoch, self.config.dataset.train_window,
                                  self.config.dataset.forecast_window,
                                  save_plot,
                                  os.path.join(self.config.path.result_dir, self.experiment_name))

            # 최고 성능 모델 저장
            if val_score['postprocessed'] < self.best_val_score:
                self.best_val_score = val_score['postprocessed']
                self.best_model = copy.deepcopy(self.model)
                patience_counter = 0
                if train_params.get('save_model_best', self.config.setting.save_model_best):
                    self._save_model('best_model.pth', experiment_spec)
            else:
                patience_counter += 1

            # 얼리 스탑핑
            if early_stopping and patience_counter >= patience:
                self.logger.info(f"최종 모델 얼리스탑: {patience} 에포크 동안 개선 없음", level=LogLevel.LEVEL2)
                break

        # 테스트 예측
        self.logger.info("테스트 데이터 예측 중...", level=LogLevel.LEVEL2)
        test_loaders = self.data_processor.prepare_test_data()
        predictions = self.predict(test_loaders)

        # 후처리 및 저장
        self.logger.info("예측 결과 후처리 및 저장 중...", level=LogLevel.LEVEL2)
        submission = self.data_processor.postprocess_and_save(predictions, save=True, filename=self.experiment_name)

        # 최종 모델 저장
        if train_params.get('save_model_final', self.config.setting.save_model_final):
            self._save_model('final_model.pth', experiment_spec)

        return {
            'best_val_score': self.best_val_score,
            'final_train_loss': self.train_losses[-1] if self.train_losses else None,
            'epochs_trained': len(self.train_losses),
            'train_params': train_params
        }

    def train_single_experiment(self, experiment_spec):
        """단일 실험 실행"""
        self.logger.section(f"실험 시작: {self.experiment_name})", level=LogLevel.LEVEL1)

        # 현재 실험 사양 저장 (모델 저장 시 사용)
        self.current_experiment_spec = experiment_spec

        # 모델 및 훈련 컴포넌트 설정
        self.model = self._setup_model(experiment_spec)
        self.criterion, self.optimizer, self.scheduler, train_params = self._setup_training_components(experiment_spec)

        # 데이터 로더 준비
        self.logger.info("훈련/검증 데이터 준비 중...", level=LogLevel.LEVEL2)
        train_loader, val_loader = self.data_processor.prepare_train_data()

        # 훈련 상태 초기화
        self.best_model = None
        self.best_val_score = float('inf')
        self.train_losses = []
        self.val_scores = []

        # 훈련 루프
        epochs = train_params.get('epochs', self.config.train.epochs)
        patience = train_params.get('patience', self.config.train.patience)
        early_stopping = train_params.get('early_stopping', self.config.train.early_stopping)

        patience_counter = 0

        # 에폭 별 진행률 표시
        epoch_pbar = tqdm(range(epochs), desc="전체 진행률")

        self.logger.info("모델 훈련 시작...", level=LogLevel.LEVEL2)
        for epoch in epoch_pbar:
            # 훈련
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)

            # 검증
            val_loss, val_score = self.validate(val_loader)
            self.val_scores.append(val_score['original'])

            # 스케줄러 업데이트
            if self.scheduler:
                if hasattr(self.scheduler, 'step'):
                    if 'ReduceLROnPlateau' in str(type(self.scheduler)):
                        self.scheduler.step(val_score['original'])
                    else:
                        self.scheduler.step()

            # 로깅
            self.logger.info(f"Epoch {epoch + 1:3d}/{epochs}: "
                             f"Train_Loss={train_loss:.6f}, Val_Loss={val_loss:.6f}, Val_SMAPE={val_score['original']:.4f}, Val_SMAPE(PostProcessed)={val_score['postprocessed']:.4f}",
                             level=LogLevel.LEVEL1)

            # 플롯 출력
            if self.config.setting.plot:
                save_plot = False
                if self.config.setting.save_plot and epoch % 10 == 0:
                    save_plot = True
                self.plotter.plot(epoch, self.config.dataset.train_window,
                                  self.config.dataset.forecast_window,
                                  save_plot,
                                  os.path.join(self.config.path.result_dir, self.experiment_name))

            # 최고 성능 모델 저장
            if val_score['original'] < self.best_val_score:
                self.best_val_score = val_score['original']
                patience_counter = 0
                if train_params.get('save_model_best', self.config.setting.save_model_best):
                    self._save_model('best_model.pth', experiment_spec)
                self.logger.success(f"실험 내 새로운 최고 성능 모델: {self.experiment_name} (점수: {self.best_val_score:.4f})")

            else:
                patience_counter += 1

            # 얼리 스탑핑
            if early_stopping and patience_counter >= patience:
                self.logger.info(f"얼리스탑: {self.config.train.patience} 에포크 동안 개선 없음", level=LogLevel.LEVEL1)
                break

        # 테스트 예측
        self.logger.info("테스트 데이터 예측 중...", level=LogLevel.LEVEL2)
        test_loaders = self.data_processor.prepare_test_data()
        predictions = self.predict(test_loaders)

        # 후처리 및 저장
        self.logger.info("예측 결과 후처리 및 저장 중...", level=LogLevel.LEVEL2)
        submission = self.data_processor.postprocess_and_save(predictions, save=True, filename=self.experiment_name)

        # 최종 모델 저장
        if train_params.get('save_model_final', self.config.setting.save_model_final):
            self._save_model('final_model.pth', experiment_spec)

        # 실험 결과 저장 (재현성을 위한 전체 설정 정보 포함)
        experiment_result = {
            # 기본 실험 정보
            'experiment_name': self.experiment_name,
            'model_type': experiment_spec.model_type,
            'model_variant': experiment_spec.model_variant,
            'loss_type': experiment_spec.loss_type,
            'loss_variant': experiment_spec.loss_variant,
            'best_val_score': self.best_val_score,
            'final_train_loss': self.train_losses[-1] if self.train_losses else None,
            'epochs_trained': len(self.train_losses),

            # 재현성을 위한 상세 설정 정보
            'model_params': self.config.get_model_params(experiment_spec.model_type, experiment_spec.model_variant),
            'loss_params': self.config.get_loss_params(experiment_spec.loss_type, experiment_spec.loss_variant),
            'train_params': train_params,

            # 데이터셋 설정
            'dataset_config': asdict(self.config.dataset),

            # 전처리 설정
            'preprocess_config': asdict(self.config.preprocess),

            # 후처리 설정
            'postprocess_config': asdict(self.config.postprocess),

            # 실험 환경 정보
            'setting_info': {
                'device': str(self.device),
                'seed': self.config.setting.seed,
                'timestamp': datetime.now().isoformat(),
                'name': self.config.setting.name
            },

            # 전처리 컴포넌트 정보 (재현성을 위해)
            'preprocessing_components_info': {
                'label_encoders_count': len(self.data_processor.get_preprocessing_components().get('label_encoders', {})),
                'scalers_count': len(self.data_processor.get_preprocessing_components().get('scalers', {})),
                'label_encoder_features': list(self.data_processor.get_preprocessing_components().get('label_encoders', {}).keys()),
                'scaler_features': list(self.data_processor.get_preprocessing_components().get('scalers', {}).keys())
            },

        }

        self.experiment_results.append(experiment_result)
        self.logger.info(f"실험 완료 - 최고 검증 점수: {self.best_val_score:.6f}", level=LogLevel.LEVEL1)

        return experiment_result

    def train_epoch(self, train_loader) -> float:
        """한 에포크 훈련"""
        self.model.train()
        total_loss = 0.0

        for batch_idx, (X, y) in enumerate(train_loader):
            X, y = X.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()

            # 순전파
            outputs = self.model(X)
            loss = self.criterion(outputs, y)

            # 역전파
            loss.backward()

            # 그래디언트 클리핑
            if self.config.train.gradient_clip:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.train.gradient_clip
                )

            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(train_loader)

    def validate(self, val_loader) -> (float, float):
        """검증"""
        if val_loader is None:
            return float('inf'), float('inf')

        self.model.eval()
        val_losses = []
        self.metric.reset()

        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(self.device), y.to(self.device)
                outputs = self.model(X)

                # 로스 계산
                loss = self.criterion(outputs, y)
                val_losses.append(loss.item())

                # 플로팅을 위한 데이터 수집
                if self.config.setting.plot:
                    self.plotter.collect_data(X, y, outputs)

                # 메트릭 계산
                self.metric(outputs, y)

        # 🔍 SMAPE 분석 수행 (에포크가 끝날 때마다)
        self.logger.info("SMAPE 분석 수행 중...", level=LogLevel.LEVEL3)
        self.metric.print_worst_smape_analysis(top_k=30, detailed=True)

        return np.mean(val_losses), self.metric.get_weighted_smape()

    def predict(self, test_loaders: List) -> List[np.ndarray]:
        """테스트 데이터에 대한 예측"""

        # 베스트 모델이 있으면 사용, 없으면 현재 모델 사용
        model_to_use = self.best_model if self.best_model is not None else self.model
        model_info = "베스트 모델" if self.best_model is not None else "현재 모델"

        if self.best_model is not None:
            self.logger.debug(f"🏆 베스트 모델로 예측", level=LogLevel.LEVEL3)
            # 베스트 모델을 디바이스로 이동 및 평가 모드 설정
            model_to_use = model_to_use.to(self.device).eval()
        else:
            self.logger.debug("현재 모델로 예측", level=LogLevel.LEVEL3)

        model_to_use.eval()

        predictions = []

        with torch.no_grad():
            test_pbar = tqdm(test_loaders, desc="테스트")
            for test_loader in test_pbar:
                loader_predictions = []
                for X in test_loader:
                    X = X.to(self.device)
                    outputs = self.model(X)
                    loader_predictions.append(outputs[:, :, 0].cpu().numpy())

                # 예측 결과 합치기
                predictions.append(np.concatenate(loader_predictions, axis=0))

        return predictions

    def run_all_experiments(self):
        """모든 실험 실행"""
        self.logger.info(f"총 {len(self.config.experiments)}개의 실험을 시작합니다.", level=LogLevel.LEVEL1)

        # Cross-validation 설정 확인
        use_cv = getattr(self.config.train, 'use_cross_validation', False)
        cv_n_splits = getattr(self.config.train, 'cv_n_splits', 5)
        cv_test_size_ratio = getattr(self.config.train, 'cv_test_size_ratio', 0.2)

        if use_cv:
            self.logger.info(f"Cross-Validation 모드 활성화: {cv_n_splits} folds, test_size_ratio: {cv_test_size_ratio}",
                             level=LogLevel.LEVEL1)
        else:
            self.logger.info("일반 훈련 모드 사용", level=LogLevel.LEVEL1)

        for i, experiment_spec in enumerate(self.config.experiments):
            self.experiment_name = f"{experiment_spec.model_type}({experiment_spec.model_variant}) + {experiment_spec.loss_type}({experiment_spec.loss_variant})"
            self.logger.info(f"실험 {i + 1}/{len(self.config.experiments)} 시작", level=LogLevel.LEVEL1)

            if use_cv:
                self.train_single_experiment_with_cv(experiment_spec, cv_n_splits, cv_test_size_ratio)
            else:
                self.train_single_experiment(experiment_spec)

        # 결과 요약
        self._summarize_results()

        return self.experiment_results

    def _make_json_serializable(self, obj):
        """
        재귀적으로 객체를 JSON 직렬화 가능한 형태로 변환
        모든 주요 데이터 타입을 처리
        """
        import numpy as np
        import torch
        from enum import Enum
        from datetime import datetime, date

        # None, bool, int, float, str은 이미 JSON 직렬화 가능
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return obj

        # Enum 처리
        elif isinstance(obj, Enum):
            return obj.value

        # NumPy 정수 타입들
        elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8,
                              np.uint64, np.uint32, np.uint16, np.uint8)):
            return int(obj)

        # NumPy 부동소수점 타입들
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)

        # NumPy 불린 타입
        elif isinstance(obj, np.bool_):
            return bool(obj)

        # NumPy 배열
        elif isinstance(obj, np.ndarray):
            return obj.tolist()

        # PyTorch 텐서
        elif isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()

        # Pandas 데이터 타입들
        elif hasattr(obj, 'dtype') and 'int' in str(obj.dtype):
            return int(obj)
        elif hasattr(obj, 'dtype') and 'float' in str(obj.dtype):
            return float(obj)

        # 날짜/시간 객체
        elif isinstance(obj, (datetime, date)):
            return obj.isoformat()

        # 딕셔너리
        elif isinstance(obj, dict):
            return {str(key): self._make_json_serializable(value) for key, value in obj.items()}

        # 리스트, 튜플
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]

        # 세트
        elif isinstance(obj, set):
            return list(obj)

        # 객체의 속성들
        elif hasattr(obj, '__dict__'):
            return {key: self._make_json_serializable(value) for key, value in obj.__dict__.items()}

        # 마지막 수단: 문자열로 변환
        else:
            try:
                return str(obj)
            except:
                return f"<비직렬화 객체: {type(obj).__name__}>"

    def _summarize_results(self):
        """실험 결과 요약"""
        self.logger.section("실험 결과 요약", level=LogLevel.LEVEL1)

        # 결과를 성능순으로 정렬
        sorted_results = sorted(self.experiment_results, key=lambda x: x['best_val_score'])

        for i, result in enumerate(sorted_results):
            # Cross-validation 결과가 있는지 확인
            if 'cv_mean_score' in result:
                self.logger.info(
                    f"{i + 1}. {result['model_type']}({result['model_variant']}) + "
                    f"{result['loss_type']}({result['loss_variant']}) - "
                    f"Final Score: {result['best_val_score']:.6f}, "
                    f"CV Score: {result['cv_mean_score']:.6f} ± {result['cv_std_score']:.6f}",
                    level=LogLevel.LEVEL1
                )
            else:
                self.logger.info(
                    f"{i + 1}. {result['model_type']}({result['model_variant']}) + "
                    f"{result['loss_type']}({result['loss_variant']}) - "
                    f"Score: {result['best_val_score']:.6f}",
                    level=LogLevel.LEVEL1
                )

        # Cross-validation 상세 결과 로깅
        cv_results = [r for r in sorted_results if 'cv_mean_score' in r]
        if cv_results:
            self.logger.section("Cross-Validation 상세 결과", level=LogLevel.LEVEL1)
            for result in cv_results:
                self.logger.info(f"{result['experiment_name']}:", level=LogLevel.LEVEL1)
                self.logger.info(f"  CV 평균: {result['cv_mean_score']:.6f} ± {result['cv_std_score']:.6f}", level=LogLevel.LEVEL1)
                self.logger.info(f"  각 Fold 점수: {[f'{score:.4f}' for score in result['cv_fold_scores']]}", level=LogLevel.LEVEL1)

        # JSON으로 결과 저장
        results_file = os.path.join(self.config.path.result_dir, "experiment_results.json")
        # Enum 값들을 JSON 직렬화 가능한 형태로 변환
        serializable_results = self._make_json_serializable(self.experiment_results)
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)

        # 실험 완료 로그
        self.logger.section("실험 완료!")
        self.logger.success(f"최고 성능 모델: {sorted_results[0]['model_type']}({sorted_results[0]['model_variant']})")
        self.logger.success(f"최고 검증 SMAPE: {sorted_results[0]['best_val_score']:.4f}")

        if 'cv_mean_score' in sorted_results[0]:
            self.logger.success(f"최고 CV SMAPE: {sorted_results[0]['cv_mean_score']:.4f} ± {sorted_results[0]['cv_std_score']:.4f}")

        self.logger.info(f"요약 파일: {results_file}", level=LogLevel.LEVEL1)

    def _save_model(self, filename, experiment_spec=None):
        """
        모델과 관련 정보를 포괄적으로 저장

        저장되는 정보:
        - model_state_dict: 모델 가중치
        - best_val_score: 최고 검증 점수
        - model_info: 모델 구성 정보 (타입, 변형, 파라미터)
        - loss_info: 손실함수 구성 정보 (타입, 변형, 파라미터)
        - preprocessing_components: 전처리 컴포넌트 (라벨 인코더, 스케일러)
        - experiment_config: 전체 실험 설정 (재현성을 위해)
        - training_info: 훈련 관련 정보
        """

        os.makedirs(self.config.path.result_dir, exist_ok=True)
        model_name = self.experiment_name + f"_{filename}"
        model_path = os.path.join(self.config.path.result_dir, model_name)

        # 현재 실험 사양 정보 (train_single_experiment에서 전달받거나 현재 상태에서 추출)
        if experiment_spec is None:
            # 현재 실행 중인 실험 정보가 있다면 사용
            if hasattr(self, 'current_experiment_spec'):
                experiment_spec = self.current_experiment_spec
            else:
                self.logger.warning("실험 사양 정보가 없어 기본 정보만 저장됩니다.")

        # 저장할 정보 구성
        checkpoint = {
            # 모델 가중치
            'model_state_dict': self.model.state_dict(),

            # 성능 정보
            'best_val_score': self.best_val_score,

            # 모델 구성 정보
            'model_info': {
                'model_type': experiment_spec.model_type if experiment_spec else 'unknown',
                'model_variant': experiment_spec.model_variant if experiment_spec else 'unknown',
                'model_params': self.config.get_model_params(
                    experiment_spec.model_type,
                    experiment_spec.model_variant
                ) if experiment_spec else {}
            },

            # 손실함수 구성 정보
            'loss_info': {
                'loss_type': experiment_spec.loss_type if experiment_spec else 'unknown',
                'loss_variant': experiment_spec.loss_variant if experiment_spec else 'unknown',
                'loss_params': self.config.get_loss_params(
                    experiment_spec.loss_type,
                    experiment_spec.loss_variant
                ) if experiment_spec else {}
            },

            # 전처리 컴포넌트
            'preprocessing_components': self.data_processor.get_preprocessing_components(),

            # 전체 실험 설정 (재현성을 위해)
            'experiment_config': asdict(self.config),

            # 훈련 정보
            'training_info': {
                'epochs_trained': len(self.train_losses) if hasattr(self, 'train_losses') else 0,
                'final_train_loss': self.train_losses[-1] if hasattr(self, 'train_losses') and self.train_losses else None,
                'train_params': experiment_spec.train_params if experiment_spec else {},
                'device': str(self.device),
                'timestamp': datetime.now().isoformat()
            }
        }

        # 모델 저장
        torch.save(checkpoint, model_path)

        # 저장 정보 로깅
        self.logger.info(f"모델 정보가 {model_path}에 저장되었습니다.", level=LogLevel.LEVEL2)
        if experiment_spec:
            self.logger.info(f"  - 모델: {experiment_spec.model_type}({experiment_spec.model_variant})", level=LogLevel.LEVEL3)
            self.logger.info(f"  - 손실함수: {experiment_spec.loss_type}({experiment_spec.loss_variant})", level=LogLevel.LEVEL3)
        self.logger.info(f"  - 최고 검증 점수: {self.best_val_score:.6f}", level=LogLevel.LEVEL3)

        # 전처리 컴포넌트 정보
        preprocessing_components = self.data_processor.get_preprocessing_components()
        encoder_count = len(preprocessing_components.get('label_encoders', {}))
        scaler_count = len(preprocessing_components.get('scalers', {}))
        self.logger.info(f"  - 전처리 컴포넌트: 인코더 {encoder_count}개, 스케일러 {scaler_count}개", level=LogLevel.LEVEL3)