import glob
import os
from typing import Tuple, List, Dict, Any, Optional, Union

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from .DatasetManager import DatasetManager
from .PostProcessor import PostProcessor
from .PreProcessor import PreProcessor
from ..config.Config import Config, Config
from ..utils.Logger import get_logger, LogLevel
from ..utils.Seed import set_all_seeds

class DataProcessor:
    """
    데이터 전처리/후처리를 통합 관리하는 클래스
    - PreProcessor와 PostProcessor를 통합
    - 훈련/검증/테스트 데이터로더 제공
    - 결과 후처리 및 저장
    """

    def __init__(self, config: Config, preprocessing_components: Dict = None):
        self.config = config
        self.preprocessing_components = preprocessing_components
        self.logger = get_logger()

        self.logger.info("DataProcessor 초기화 중...", level=LogLevel.LEVEL2)
        
        if preprocessing_components:
            self.logger.info("저장된 전처리 컴포넌트 사용", level=LogLevel.LEVEL2)

        # 기존 컴포넌트 초기화
        self._setup_preprocessor()
        self._setup_postprocessor()
        self._setup_data_manager()

        # 데이터 로드
        self._load_data()

        self.logger.success("DataProcessor 초기화 완료", level=LogLevel.LEVEL2)

    def _setup_preprocessor(self):
        """전처리기 설정"""
        # PreProcessor용 config와 저장된 전처리 컴포넌트 전달
        self.preprocessor = PreProcessor(self.config, self.preprocessing_components)
        self.logger.debug("PreProcessor 설정 완료", level=LogLevel.LEVEL3)

    def _setup_postprocessor(self):
        """후처리기 설정"""
        # PostProcessor용 config와 preprocessor 전달
        self.postprocessor = PostProcessor(self.config, self.preprocessor)
        self.logger.debug("PostProcessor 설정 완료", level=LogLevel.LEVEL3)

    def _setup_data_manager(self):
        """데이터 매니저 설정"""
        # DatasetManager용 config
        self.dataset_manager = DatasetManager(self.config)
        self.logger.debug("DatasetManager 설정 완료", level=LogLevel.LEVEL3)

    def _load_data(self):
        """데이터 로드"""
        self.logger.info("데이터 로딩 중...", level=LogLevel.LEVEL2)

        # 훈련 데이터
        self.train_data = pd.read_csv(self.config.path.train_csv)

        # 정적 특성
        if os.path.exists(self.config.path.static_csv):
            self.static_feature = pd.read_csv(self.config.path.static_csv)
        else:
            self.static_feature = None
            self.logger.warning("정적 특성 파일을 찾을 수 없습니다.", level=LogLevel.LEVEL1)

        # 테스트 데이터들
        self.test_data_list = []
        test_files = sorted(glob.glob(self.config.path.test_pattern))
        for file_path in test_files:
            test_data = pd.read_csv(file_path)
            self.test_data_list.append(test_data)

        self.logger.info(f"훈련 데이터: {len(self.train_data)} 행", level=LogLevel.LEVEL2)
        self.logger.info(f"테스트 데이터: {len(self.test_data_list)} 파일", level=LogLevel.LEVEL2)

    def prepare_train_data(self) -> Tuple[DataLoader, Optional[DataLoader]]:
        """훈련/검증 데이터로더 준비"""
        self.logger.info("훈련 데이터 전처리 중...", level=LogLevel.LEVEL2)

        # 전처리 적용
        processed_data = self.preprocessor.preprocess(
            self.train_data,
            self.static_feature
        )
        # 데이터셋 생성
        self.logger.debug("훈련/검증 데이터셋 생성 중...", level=LogLevel.LEVEL3)
        train_dataset, val_dataset = self.dataset_manager.make_train_dataset(processed_data)
        # 데이터로더 생성
        self.logger.debug("데이터로더 생성 중...", level=LogLevel.LEVEL3)
        train_loader, val_loader = self.dataset_manager.make_train_dataLoader(train_dataset, val_dataset)

        self.logger.success("훈련 데이터로더 준비 완료", level=LogLevel.LEVEL2)
        self.logger.info(f"훈련 배치 수: {len(train_loader)}", level=LogLevel.LEVEL2)
        if val_loader:
            self.logger.info(f"검증 배치 수: {len(val_loader)}", level=LogLevel.LEVEL2)

        return train_loader, val_loader

    def get_data_loaders(self) -> Tuple[DataLoader, Optional[DataLoader]]:
        """훈련/검증 데이터로더 준비 (Trainer 호환성을 위한 메서드)"""
        return self.prepare_train_data()

    def prepare_train_datasets(self) -> Tuple[Any, Optional[Any]]:
        """훈련/검증 데이터셋 준비 (MLTrainer용 - Direct Strategy용 28일→7일 데이터셋)"""
        self.logger.info("Direct Strategy용 훈련 데이터 전처리 중 (28일 입력 → 7일 예측)...", level=LogLevel.LEVEL2)

        # 전처리 적용
        processed_data = self.preprocessor.preprocess(
            self.train_data,
            self.static_feature
        )
        
        # 데이터셋 생성 - 28일 입력, 7일 출력 시퀀스
        self.logger.debug("28일→7일 시퀀스 데이터셋 생성 중...", level=LogLevel.LEVEL3)
        train_dataset, val_dataset = self.dataset_manager.make_train_dataset(processed_data)

        self.logger.success("Direct Strategy 데이터셋 준비 완료", level=LogLevel.LEVEL2)
        self.logger.info(f"훈련 데이터셋 크기: {len(train_dataset)}", level=LogLevel.LEVEL2)
        if val_dataset:
            self.logger.info(f"검증 데이터셋 크기: {len(val_dataset)}", level=LogLevel.LEVEL2)

        return train_dataset, val_dataset

    def prepare_test_data(self) -> List[DataLoader]:
        """테스트 데이터로더 리스트 준비"""
        self.logger.info("테스트 데이터 전처리 중...", level=LogLevel.LEVEL2)

        test_loaders = []

        for i, test_data in enumerate(self.test_data_list):
            self.logger.debug(f"테스트 데이터 {i+1}/{len(self.test_data_list)} 처리 중...", level=LogLevel.LEVEL3)
            
            # 전처리 적용 (훈련된 스케일러 사용)
            processed_test = self.preprocessor.preprocess(
                test_data,
                self.static_feature,
                for_train=False
            )

            # 테스트 데이터셋 생성
            test_dataset = self.dataset_manager.make_test_dataset(processed_test)

            # 테스트 데이터로더 생성
            test_loader = self.dataset_manager.make_test_dataLoader(test_dataset)

            test_loaders.append(test_loader)

        self.logger.success(f"테스트 데이터로더 {len(test_loaders)}개 준비 완료", level=LogLevel.LEVEL2)

        return test_loaders

    def postprocess_and_save(self, predictions: List[np.ndarray], save: bool = True, filename: str = None):
        """예측 결과 후처리 및 저장"""
        return self.postprocessor.save_predictions(predictions, save=save, filename=filename)

    def get_feature_info(self) -> Dict[str, Any]:
        """전처리된 특성 정보 반환"""
        # 샘플 데이터로 특성 정보 추출
        sample_features = self.preprocessor.preprocess(
            self.train_data.head(100),
            self.static_feature
        )

        return {
            'n_features': len(sample_features.columns),
            'feature_names': [col for col in sample_features.columns],
            'target_index': sample_features.columns.get_loc(self.config.dataset.target_col),
        }

    def get_preprocessing_components(self) -> Dict[str, Any]:
        """전처리 컴포넌트(라벨 인코더, 스케일러) 반환"""
        return self.preprocessor.get_preprocessing_components()