import random
from typing import Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from ..config.Config import Config, Config
from ..utils.Logger import get_logger, LogLevel
from ..utils.Seed import worker_init_fn


class DatasetManager:
    """
    시계열 데이터를 위한 데이터 관리 클래스
    - 시퀀스 생성, 훈련/검증 데이터셋 분할, 테스트 데이터 처리 담당
    """

    def __init__(self, config: Union[Config, Config]):
        """DatasetManager 초기화"""
        self.config = config
        self.logger = get_logger()
        
        self.logger.info("DatasetManager 초기화 중...", level=LogLevel.LEVEL2)

        # 시퀀스 생성 설정
        self.train_window = self.config.dataset.train_window
        self.forecast_window = self.config.dataset.forecast_window

        self.sampling_method = self.config.dataset.sampling_method
        self.validation_ratio = self.config.dataset.validation_ratio

        # DataLoader 설정
        self.num_workers = self.config.dataset.num_workers
        self.pin_memory = self.config.dataset.pin_memory
        self.batch_size = self.config.dataset.batch_size
        self.prefatch_factor = self.config.dataset.prefetch_factor
        self.persistent_workers = self.config.dataset.persistent_workers

        # 기타 설정
        self.target_col = self.config.dataset.target_col
        self.date_col = self.config.dataset.date_col
        self.group_col = self.config.dataset.group_col

        self.logger.success("DatasetManager 초기화 완료", level=LogLevel.LEVEL2)

    def make_train_dataset(self, data):
        """데이터셋 생성 메인 함수"""
        self.logger.info("훈련 / 검증 데이터셋 생성 시작...", level=LogLevel.LEVEL2)
        train_X, train_Y, val_X, val_Y = self.create_sequences_with_sampling(data)
        train_ds = Data(train_X, train_Y)
        val_ds = Data(val_X, val_Y)
        self.logger.success(f"훈련 데이터셋: {len(train_ds)}개, 검증 데이터셋: {len(val_ds)}개 생성 완료", level=LogLevel.LEVEL2)
        return train_ds, val_ds


    def make_test_dataset(self, data):
        """
        테스트 데이터셋 생성 메인 함수

        Args:
            data: 단일 데이터프레임 또는 데이터프레임들의 리스트

        Returns:
            test_ds: 단일 데이터셋 (data가 단일 데이터프레임인 경우)
            test_datasets: 데이터셋들의 리스트 (data가 리스트인 경우)
        """

        # data가 리스트인지 확인
        if isinstance(data, list):
            self.logger.info(f"{len(data)}개의 테스트 데이터셋 생성 시작...", level=LogLevel.LEVEL2)

            test_datasets = []
            for i, single_data in enumerate(data):
                self.logger.info(f"테스트 데이터 {i + 1}/{len(data)} 처리 중...", level=LogLevel.LEVEL2)

                # 테스트 시퀀스 생성
                test_X = self.create_test_sequences(single_data)
                test_ds = Data(test_X, None)  # 테스트 데이터는 타겟이 없음

                test_datasets.append(test_ds)
                self.logger.success(f"테스트 데이터셋 {i + 1}: {len(test_ds)}개 시퀀스 생성 완료", level=LogLevel.LEVEL2)

            self.logger.success(f"총 {len(test_datasets)}개의 테스트 데이터셋 생성 완료", level=LogLevel.LEVEL2)
            return test_datasets

        else:
            # 단일 데이터프레임인 경우
            self.logger.info(f"테스트 데이터셋 생성 시작...", level=LogLevel.LEVEL2)

            # 테스트 데이터셋 생성
            test_X = self.create_test_sequences(data)
            test_ds = Data(test_X, None)  # 테스트 데이터는 타겟이 없음

            self.logger.success(f"테스트 데이터셋: {len(test_ds)}개 생성 완료", level=LogLevel.LEVEL2)
            return test_ds

    def make_train_dataLoader(self, train_ds, val_ds):
        """데이터로더 생성 메인 함수"""
        self.logger.info("훈련 / 검증 데이터로더 생성 시작...", level=LogLevel.LEVEL2)

        train_loader = DataLoader(train_ds,
                                  batch_size=self.batch_size,
                                  shuffle=True,
                                  num_workers=self.num_workers,
                                  pin_memory=self.pin_memory,
                                  prefetch_factor=self.prefatch_factor,
                                  persistent_workers=self.persistent_workers,
                                  worker_init_fn=worker_init_fn if self.num_workers > 0 else None)

        val_loader = DataLoader(val_ds,
                                batch_size=self.batch_size,
                                shuffle=False,
                                num_workers=self.num_workers,
                                pin_memory=self.pin_memory,
                                prefetch_factor=self.prefatch_factor,
                                persistent_workers=self.persistent_workers,
                                worker_init_fn=worker_init_fn if self.num_workers > 0 else None)

        self.logger.success("훈련 / 검증 데이터로더 생성 완료", level=LogLevel.LEVEL2)
        return train_loader, val_loader

    def make_test_dataLoader(self, test_ds):
        """
        테스트 데이터로더 생성 메인 함수

        Args:
            test_ds: 단일 데이터셋 또는 데이터셋들의 리스트

        Returns:
            test_loader: 단일 데이터로더 (test_ds가 단일 데이터셋인 경우)
            test_loaders: 데이터로더들의 리스트 (test_ds가 리스트인 경우)
        """

        # test_ds가 리스트인지 확인
        if isinstance(test_ds, list):
            self.logger.info(f"{len(test_ds)}개의 테스트 데이터로더 생성 시작...", level=LogLevel.LEVEL2)

            test_loaders = []
            for i, ds in enumerate(test_ds):
                test_loader = DataLoader(
                    ds,
                    batch_size=self.batch_size,
                    shuffle=False,  # 순서 유지를 위해 shuffle=False
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                    prefetch_factor=self.prefatch_factor,
                    persistent_workers=self.persistent_workers,
                    worker_init_fn=worker_init_fn if self.num_workers > 0 else None
                )
                test_loaders.append(test_loader)
                self.logger.success(f"테스트 데이터로더 {i + 1}/{len(test_ds)} 생성 완료", level=LogLevel.LEVEL2)

            self.logger.success(f"총 {len(test_loaders)}개의 테스트 데이터로더 생성 완료", level=LogLevel.LEVEL2)
            return test_loaders

        else:
            # 단일 데이터셋인 경우
            self.logger.info(f"테스트 데이터로더 생성 시작...", level=LogLevel.LEVEL2)

            test_loader = DataLoader(
                test_ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                worker_init_fn=worker_init_fn if self.num_workers > 0 else None
            )

            self.logger.success(f"테스트 데이터로더 생성 완료", level=LogLevel.LEVEL2)
            return test_loader

    def create_test_sequences(self, data):
        """테스트용 시퀀스 생성 - 각 제품의 마지막 train_window만큼의 데이터로 시퀀스 생성"""
        self.logger.info("테스트 시퀀스 생성 중...", level=LogLevel.LEVEL2)
        test_sequences = []
        product_groups = data.groupby(self.group_col)

        # tqdm으로 진행률 표시
        for product_id, product_data in tqdm(product_groups, desc="테스트 시퀀스 생성"):
            product_data = product_data.sort_values(self.date_col)

            # 최소 길이 체크
            if len(product_data) < self.train_window:
                continue

            # 각 제품의 마지막 train_window만큼의 데이터를 시퀀스로 생성
            sequence = product_data.iloc[-self.train_window:]

            # 제품 ID 일관성 체크
            seq_products = sequence[self.group_col].unique()

            if len(seq_products) == 1:
                test_sequences.append(sequence.values)

        self.logger.success(f"총 테스트 시퀀스: {len(test_sequences)}개 생성 완료", level=LogLevel.LEVEL2)
        return np.array(test_sequences, dtype='float32')

    def create_sequences(self, data):
        """시퀀스 생성"""
        self.logger.info("시퀀스 생성 중...", level=LogLevel.LEVEL2)

        all_sequences = []
        all_targets = []
        product_groups = data.groupby(self.group_col)

        # 모든 제품에 대해 가능한 시퀀스 생성
        for product_id, product_data in tqdm(product_groups, desc="시퀀스 생성"):
            product_data = product_data.sort_values(self.date_col)

            if len(product_data) < self.train_window + self.forecast_window:
                continue

            # 데이터 시퀀스 생성
            for i in range(len(product_data) - self.train_window - self.forecast_window + 1):
                sequence = product_data.iloc[i:(i + self.train_window)]
                target = product_data.iloc[(i + self.train_window):(i + self.train_window + self.forecast_window)]

                seq_products = sequence[self.group_col].unique()
                target_products = target[self.group_col].unique()

                # 제품 ID 일관성 체크
                if len(seq_products) == 1 and len(target_products) == 1 and seq_products[0] == target_products[0]:
                    all_sequences.append(sequence.values)
                    all_targets.append(target.values)

        self.logger.info(f"총 {len(all_sequences)}개 시퀀스 생성 완료", level=LogLevel.LEVEL2)

        X = np.array(all_sequences, dtype='float32')
        Y = np.array(all_targets, dtype='float32')

        return X, Y

    def create_sequences_with_sampling(self, data):
        """샘플링 방법으로 시퀀스 생성"""
        self.logger.info("시퀀스 생성 중...", level=LogLevel.LEVEL2)

        all_sequences = []
        all_targets = []
        product_sequence_info = []
        product_groups = data.groupby(self.group_col)

        # 모든 제품에 대해 가능한 시퀀스 생성
        for product_id, product_data in tqdm(product_groups, desc="시퀀스 생성"):
            product_data = product_data.sort_values(self.date_col)

            if len(product_data) < self.train_window + self.forecast_window:
                continue

            # 데이터 시퀀스 생성
            for i in range(len(product_data) - self.train_window - self.forecast_window + 1):
                sequence = product_data.iloc[i:(i + self.train_window)]
                target = product_data.iloc[(i + self.train_window):(i + self.train_window + self.forecast_window)]

                seq_products = sequence[self.group_col].unique()
                target_products = target[self.group_col].unique()

                # 제품 ID 일관성 체크
                if len(seq_products) == 1 and len(target_products) == 1 and seq_products[0] == target_products[0]:
                    seq_idx = len(all_sequences)
                    all_sequences.append(sequence.values)
                    all_targets.append(target.values)

                    start_date = sequence[self.date_col].iloc[0]
                    product_sequence_info.append((product_id, seq_idx, start_date))

        self.logger.info(f"총 {len(all_sequences)}개 시퀀스 생성", level=LogLevel.LEVEL2)

        # 샘플링 방법에 따른 validation 인덱스 선택
        val_indices = self._select_validation_indices(product_sequence_info, len(all_sequences))

        # train/validation 분할
        train_indices = [i for i in range(len(all_sequences)) if i not in val_indices]

        train_X = np.array([all_sequences[i] for i in train_indices], dtype='float32')
        train_Y = np.array([all_targets[i] for i in train_indices], dtype='float32')
        val_X = np.array([all_sequences[i] for i in val_indices], dtype='float32')
        val_Y = np.array([all_targets[i] for i in val_indices], dtype='float32')

        self.logger.info(f"훈련: {len(train_X)}개, 검증: {len(val_X)}개 시퀀스 분할 완료", level=LogLevel.LEVEL2)
        return train_X, train_Y, val_X, val_Y

    def _select_validation_indices(self, product_sequence_info, total_sequences):
        """Validation 인덱스 선택"""
        self.logger.info(f"{self.sampling_method} 방법으로 검증 데이터 선택 중...", level=LogLevel.LEVEL2)

        if self.sampling_method == 'random':
            n_val = int(total_sequences * self.validation_ratio)
            # random.sample() 대신 numpy 사용하여 시드 고정 보장
            indices = np.random.choice(total_sequences, n_val, replace=False)
            return indices.tolist()

        elif self.sampling_method == 'stratified':
            val_indices = []
            product_groups_info = {}

            for product_id, seq_idx, start_date in product_sequence_info:
                if product_id not in product_groups_info:
                    product_groups_info[product_id] = []
                product_groups_info[product_id].append(seq_idx)

            for product_id, seq_indices in product_groups_info.items():
                n_samples = max(1, int(len(seq_indices) * self.validation_ratio))

                # random.sample() 대신 numpy 사용하여 시드 고정 보장
                sampled = np.random.choice(seq_indices, n_samples, replace=False)
                val_indices.extend(sampled.tolist())

            return val_indices

        elif self.sampling_method == 'recent':
            product_groups_info = {}

            for product_id, seq_idx, start_date in product_sequence_info:
                if product_id not in product_groups_info:
                    product_groups_info[product_id] = []
                product_groups_info[product_id].append((seq_idx, start_date))

            val_indices = []
            for product_id, seq_info in product_groups_info.items():
                seq_info.sort(key=lambda x: x[1], reverse=True)

                n_samples = max(1, int(len(seq_info) * self.validation_ratio))

                recent_indices = [seq_idx for seq_idx, _ in seq_info[:n_samples]]
                val_indices.extend(recent_indices)

            return val_indices

        elif self.sampling_method == 'distributed':
            product_groups_info = {}

            for product_id, seq_idx, start_date in product_sequence_info:
                if product_id not in product_groups_info:
                    product_groups_info[product_id] = []
                product_groups_info[product_id].append((seq_idx, start_date))

            val_indices = []
            for product_id, seq_info in product_groups_info.items():
                seq_info.sort(key=lambda x: x[1])

                n_samples = max(1, int(len(seq_info) * self.validation_ratio))

                if n_samples == 1:
                    indices = [len(seq_info) // 2]
                else:
                    step = len(seq_info) / n_samples
                    indices = [int(i * step) for i in range(n_samples)]

                distributed_indices = [seq_info[i][0] for i in indices]
                val_indices.extend(distributed_indices)

            return val_indices

        elif self.sampling_method == 'temporal_stratified':
            """
            시간 축을 균등 분할하고, 각 구간에서 제품별로 비례적으로 랜덤 샘플링
            """
            return self._temporal_stratified_sampling(product_sequence_info, total_sequences)

        else:
            raise ValueError(f'해당 샘플링 방법을 지원하지 않습니다.: {self.sampling_method}')

    def _temporal_stratified_sampling(self, product_sequence_info, total_sequences):
        """
        시간 축 균등 분할 + 제품별 비례 랜덤 샘플링
        """
        self.logger.info("시간 축 균등 분할 + 제품별 비례 샘플링 시작", level=LogLevel.LEVEL3)

        # 1. 전체 시간 범위 계산
        all_dates = [start_date for _, _, start_date in product_sequence_info]
        min_date = min(all_dates)
        max_date = max(all_dates)

        self.logger.info(f"전체 시간 범위: {min_date} ~ {max_date}", level=LogLevel.LEVEL3)

        # 2. 시간 구간 수 결정 (설정 가능)
        n_temporal_bins = getattr(self.config.dataset, 'temporal_bins', 5)  # 기본 5개 구간

        # 3. 시간 축을 균등하게 분할
        time_range = max_date - min_date
        bin_size = time_range / n_temporal_bins

        temporal_bins = []
        for i in range(n_temporal_bins):
            bin_start = min_date + i * bin_size
            bin_end = min_date + (i + 1) * bin_size if i < n_temporal_bins - 1 else max_date
            temporal_bins.append((bin_start, bin_end))

        self.logger.info(f"시간 구간 {n_temporal_bins}개 생성: {bin_size:.1f}일씩", level=LogLevel.LEVEL3)

        # 4. 각 시간 구간별로 제품-시퀀스 그룹화
        bin_product_sequences = [{}] * n_temporal_bins
        for i in range(n_temporal_bins):
            bin_product_sequences[i] = {}

        for product_id, seq_idx, start_date in product_sequence_info:
            # 어느 시간 구간에 속하는지 찾기
            for bin_idx, (bin_start, bin_end) in enumerate(temporal_bins):
                if bin_start <= start_date <= bin_end:
                    if product_id not in bin_product_sequences[bin_idx]:
                        bin_product_sequences[bin_idx][product_id] = []
                    bin_product_sequences[bin_idx][product_id].append(seq_idx)
                    break

        # 5. 각 구간에서 제품별 비례 샘플링
        val_indices = []
        total_target_samples = int(total_sequences * self.validation_ratio)
        samples_per_bin = total_target_samples // n_temporal_bins   # 빈 별 선택해야 하는 샘플 개수
        remaining_samples = total_target_samples % n_temporal_bins

        for bin_idx in range(n_temporal_bins):
            # 현재 구간의 목표 샘플 수
            current_bin_target = samples_per_bin
            if bin_idx < remaining_samples:  # 나머지 샘플 분배
                current_bin_target += 1

            bin_sequences = bin_product_sequences[bin_idx]  # 빈 별 { 제품 ID_1: [시퀀스], ... }
            if not bin_sequences:  # 빈 구간은 건너뛰기
                continue

            # 현재 구간의 전체 시퀀스 수
            total_bin_sequences = sum(len(seqs) for seqs in bin_sequences.values())

            if total_bin_sequences == 0:
                continue

            # 실제 샘플 수 (구간 내 시퀀스가 목표보다 적을 수 있음)
            actual_bin_samples = min(current_bin_target, total_bin_sequences)

            # 제품별 샘플 수 계산 (비례 배분)
            bin_val_indices = []
            for product_id, seq_indices in bin_sequences.items():
                product_ratio = len(seq_indices) / total_bin_sequences
                product_samples = max(1, int(actual_bin_samples * product_ratio))

                # 제품의 시퀀스 수가 목표 샘플 수보다 적으면 조정
                product_samples = min(product_samples, len(seq_indices))

                # 랜덤 샘플링 (시드 고정됨)
                if product_samples > 0:
                    sampled = np.random.choice(seq_indices, product_samples, replace=False)
                    bin_val_indices.extend(sampled.tolist())

            val_indices.extend(bin_val_indices)

            self.logger.info(
                f"구간 {bin_idx + 1}: {len(bin_sequences)}개 제품, "
                f"{total_bin_sequences}개 시퀀스 → {len(bin_val_indices)}개 선택",
                level=LogLevel.LEVEL3
            )

        self.logger.info(f"전체 선택된 검증 샘플: {len(val_indices)}개", level=LogLevel.LEVEL2)
        return val_indices

    def get_test_info(self, data):
        """테스트 데이터의 제품 정보 반환"""
        self.logger.info("테스트 데이터 정보 수집 중...", level=LogLevel.LEVEL2)

        test_info = []
        product_groups = data.groupby(self.group_col)

        for product_id, product_data in product_groups:
            product_data = product_data.sort_values(self.date_col)

            if len(product_data) >= self.train_window:
                last_date = product_data[self.date_col].iloc[-1]
                test_info.append({
                    'product_id': product_id,
                    'last_date': last_date,
                    'data_length': len(product_data)
                })

        self.logger.success(f"{len(test_info)}개 제품 정보 수집 완료", level=LogLevel.LEVEL2)
        return test_info


class Data(Dataset):
    """PyTorch Dataset 클래스"""

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X) if self.Y is None else len(self.Y)

    def __getitem__(self, idx):
        if self.Y is not None:
            return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.Y[idx], dtype=torch.float32)
        return torch.tensor(self.X[idx], dtype=torch.float32)