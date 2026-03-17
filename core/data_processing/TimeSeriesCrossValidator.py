import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
from torch.utils.data import DataLoader

from .DatasetManager import DatasetManager, Data
from ..config.Config import Config
from ..utils.Logger import get_logger, LogLevel


class TimeSeriesCrossValidator:
    """
    시계열 데이터를 위한 Cross-Validation 클래스
    시간 순서를 고려한 분할을 수행하여 데이터 누출을 방지
    """
    
    def __init__(self, config: Config, n_splits: int = 5, test_size_ratio: float = 0.2):
        """
        Args:
            config: 설정 객체
            n_splits: Cross-validation fold 수
            test_size_ratio: 각 fold에서 validation 데이터 비율
        """
        self.config = config
        self.n_splits = n_splits
        self.test_size_ratio = test_size_ratio
        self.logger = get_logger()
        
        # DatasetManager 초기화
        self.dataset_manager = DatasetManager(config)
        
        self.logger.info(f"TimeSeriesCrossValidator 초기화: {n_splits} folds, validation ratio: {test_size_ratio}", 
                        level=LogLevel.LEVEL2)
    
    def create_time_series_splits(self, data: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        시계열 데이터를 시간 순서를 고려하여 분할
        
        Args:
            data: 전처리된 시계열 데이터
            
        Returns:
            List of (train_data, val_data) tuples for each fold
        """
        self.logger.info("시계열 Cross-Validation 분할 생성 중...", level=LogLevel.LEVEL2)
        
        splits = []
        
        # 날짜 컬럼으로 정렬
        date_col = self.config.dataset.date_col
        data_sorted = data.sort_values(date_col).reset_index(drop=True)

        # 전체 데이터 개수
        total_samples = len(data_sorted)

        self.logger.info(f"전체 데이터 기간: ({total_samples}일)", level=LogLevel.LEVEL2)
        
        # 각 fold별로 시간 기반 분할
        for fold in range(self.n_splits):
            self.logger.info(f"Fold {fold + 1}/{self.n_splits} 생성 중...", level=LogLevel.LEVEL3)
            
            # 시간 기반 분할 계산
            # 각 fold는 점진적으로 더 많은 과거 데이터를 사용
            val_end_ratio = 0.6 + (fold * 0.1)  # 60%, 70%, 80%, 90%, 100%
            train_end_ratio = max(0.4, val_end_ratio - self.test_size_ratio)

            # # 타임스탬프 정수값 기준으로 분할점 계산
            # train_end_timestamp = min_date + int(total_days * train_end_ratio)
            # val_end_timestamp = min_date + int(total_days * val_end_ratio)
            #
            # # 데이터 분할 (정수 타임스탬프로 직접 비교)
            # train_mask = data_sorted[date_col] <= train_end_timestamp
            # val_mask = (data_sorted[date_col] > train_end_timestamp) & \
            #            (data_sorted[date_col] <= val_end_timestamp)
            #
            # train_data = data_sorted[train_mask].copy()
            # val_data = data_sorted[val_mask].copy()
            #

            # 데이터 개수 기준으로 분할점 계산
            train_end_idx = int(total_samples * train_end_ratio)
            val_end_idx = int(total_samples * val_end_ratio)

            # 인덱스 범위 확인 및 조정
            train_end_idx = max(0, min(train_end_idx, total_samples - 1))
            val_end_idx = max(train_end_idx, min(val_end_idx, total_samples))

            # 데이터 분할 (인덱스 기반)
            train_data = data_sorted.iloc[:train_end_idx].copy()
            val_data = data_sorted.iloc[train_end_idx:val_end_idx].copy()

            # 빈 데이터셋 체크
            if len(train_data) == 0 or len(val_data) == 0:
                self.logger.warning(f"Fold {fold + 1}: 빈 데이터셋이 생성되었습니다. 건너뜁니다.", level=LogLevel.LEVEL1)
                self.logger.warning(f"  - 훈련 데이터: {len(train_data)}행, 검증 데이터: {len(val_data)}행", level=LogLevel.LEVEL1)
                continue

            # 최소 데이터 요구사항 확인
            min_sequences_per_product = self.config.dataset.train_window + self.config.dataset.forecast_window
            
            # 각 제품별로 충분한 데이터가 있는지 확인
            group_col = self.config.dataset.group_col
            train_valid_products = []
            val_valid_products = []
            
            for product_id in train_data[group_col].unique():
                product_train_data = train_data[train_data[group_col] == product_id]
                if len(product_train_data) >= min_sequences_per_product:
                    train_valid_products.append(product_id)
            
            for product_id in val_data[group_col].unique():
                product_val_data = val_data[val_data[group_col] == product_id]
                if len(product_val_data) >= min_sequences_per_product:
                    val_valid_products.append(product_id)
            
            # 공통 제품만 유지
            common_products = list(set(train_valid_products) & set(val_valid_products))
            
            if len(common_products) == 0:
                self.logger.warning(f"Fold {fold + 1}: 유효한 제품이 없습니다. 건너뜁니다.", level=LogLevel.LEVEL1)
                continue
            
            train_data_filtered = train_data[train_data[group_col].isin(common_products)]
            val_data_filtered = val_data[val_data[group_col].isin(common_products)]

            # 실제 데이터에서 날짜 범위 계산
            train_date_range = (train_data_filtered[date_col].min(), train_data_filtered[date_col].max())
            val_date_range = (val_data_filtered[date_col].min(), val_data_filtered[date_col].max())

            self.logger.info(f"Fold {fold + 1}: 훈련 {len(train_data_filtered):,}행, "
                             f"검증 {len(val_data_filtered):,}행, "
                             f"제품 {len(common_products):,}개", level=LogLevel.LEVEL3)
            self.logger.info(f"  - 훈련 인덱스 범위: 0 ~ {train_end_idx:,} ({train_end_ratio:.1%})", level=LogLevel.LEVEL3)
            self.logger.info(
                f"  - 검증 인덱스 범위: {train_end_idx:,} ~ {val_end_idx:,} ({(val_end_idx - train_end_idx) / total_samples:.1%})",
                level=LogLevel.LEVEL3)
            self.logger.info(f"  - 훈련 날짜 범위: {train_date_range[0]} ~ {train_date_range[1]}", level=LogLevel.LEVEL3)
            self.logger.info(f"  - 검증 날짜 범위: {val_date_range[0]} ~ {val_date_range[1]}", level=LogLevel.LEVEL3)

            splits.append((train_data_filtered, val_data_filtered))
        
        self.logger.success(f"총 {len(splits)}개 fold 생성 완료", level=LogLevel.LEVEL2)
        return splits
    
    def create_cv_dataloaders(self, data: pd.DataFrame) -> List[Tuple[DataLoader, DataLoader]]:
        """
        Cross-validation용 데이터로더 생성
        
        Args:
            data: 전처리된 시계열 데이터
            
        Returns:
            List of (train_loader, val_loader) tuples for each fold
        """
        self.logger.info("Cross-validation 데이터로더 생성 중...", level=LogLevel.LEVEL2)
        
        splits = self.create_time_series_splits(data)
        cv_loaders = []
        
        for fold_idx, (train_data, val_data) in enumerate(splits):
            self.logger.info(f"Fold {fold_idx + 1} 데이터로더 생성 중...", level=LogLevel.LEVEL3)
            
            # 각 fold별로 시퀀스 생성
            train_X, train_Y = self.dataset_manager.create_sequences(train_data)
            val_X, val_Y = self.dataset_manager.create_sequences(val_data)
            
            # Dataset 생성
            train_dataset = Data(train_X, train_Y)
            val_dataset = Data(val_X, val_Y)
            
            # DataLoader 생성
            train_loader, val_loader = self.dataset_manager.make_train_dataLoader(train_dataset, val_dataset)
            
            cv_loaders.append((train_loader, val_loader))
            
            self.logger.info(f"Fold {fold_idx + 1}: 훈련 배치 {len(train_loader)}, "
                           f"검증 배치 {len(val_loader)}", level=LogLevel.LEVEL3)
        
        self.logger.success(f"총 {len(cv_loaders)}개 fold 데이터로더 생성 완료", level=LogLevel.LEVEL2)
        return cv_loaders
    
    def get_fold_info(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        각 fold의 정보 반환
        
        Args:
            data: 전처리된 시계열 데이터
            
        Returns:
            List of fold information dictionaries
        """
        splits = self.create_time_series_splits(data)
        fold_info = []
        
        date_col = self.config.dataset.date_col
        group_col = self.config.dataset.group_col
        
        for fold_idx, (train_data, val_data) in enumerate(splits):
            info = {
                'fold': fold_idx + 1,
                'train_size': len(train_data),
                'val_size': len(val_data),
                'train_products': len(train_data[group_col].unique()),
                'val_products': len(val_data[group_col].unique()),
                'train_date_range': (train_data[date_col].min(), train_data[date_col].max()),
                'val_date_range': (val_data[date_col].min(), val_data[date_col].max())
            }
            fold_info.append(info)
        
        return fold_info