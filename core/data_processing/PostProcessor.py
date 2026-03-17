import os
from typing import Union, Optional
import torch
import numpy as np
import pandas as pd

from .PreProcessor import PreProcessor
from ..config.Config import Config, Config, PostProcessType
from ..utils.Logger import get_logger, LogLevel


class PostProcessor:
    """
    예측 결과 후처리 및 제출 파일 생성을 담당하는 클래스

    주요 기능:
    - 예측값 클리핑 (최소값 보장, 최빈값 기반 클리핑)
    - 제출 파일 형식으로 변환 및 저장
    """

    def __init__(self, config: Union[Config, Config], preprocessor: PreProcessor):
        # 설정 및 로거 초기화
        self.config = config
        self.logger = get_logger()
        self.preprocessor = preprocessor
        self.preprocessing_components = {}

        self.logger.debug("PostProcessor 초기화 중...", level=LogLevel.LEVEL3)

        # 경로 설정 추출
        self.train_csv = self.config.path.train_csv
        self.submit_csv = self.config.path.submit_csv
        self.result_dir = self.config.path.result_dir

        # 후처리 방법 설정
        self.postprocess_method = self.config.postprocess.method

        # 최빈값 클리핑을 위한 모드 값들 미리 계산
        if self.postprocess_method == PostProcessType.clip_to_mode:
            self._calculate_modes()

    def _check_log_transform_used(self, enable_logging: bool = True):
        """
        전처리 컴포넌트에서 로그 변환이 사용되었는지 확인
        
        Returns:
            bool: 로그 변환이 사용되었으면 True, 아니면 False
        """
        self.preprocessing_components = self.preprocessor.get_preprocessing_components()

        if not self.preprocessing_components:
            return False
            
        scalers = self.preprocessing_components.get('scalers', {})
        
        # 스케일러 중에 LogTransform이 있는지 확인
        for feature, scaler in scalers.items():
            if scaler == "LogTransform":
                if enable_logging:
                    self.logger.debug(f"로그 변환 감지: {feature} 특성에서 LogTransform 사용됨", level=LogLevel.LEVEL3)
                return True
                
        return False

    def _calculate_modes(self):
        """
        train_data에서 각 제품별 sales 값의 최빈값 계산

        최빈값을 기반으로 예측값의 상한선을 설정하기 위해 사용
        """
        self.logger.info("각 제품별 sales 최빈값 계산 중...", level=LogLevel.LEVEL2)

        # 훈련 데이터 로드
        train_data = pd.read_csv(self.train_csv)
        self.logger.debug(f"훈련 데이터 로드 완료: {len(train_data)} 행", level=LogLevel.LEVEL3)

        # 각 제품별 최빈값 계산 및 저장
        self.product_modes = {}
        unique_products = train_data['영업장명_메뉴명'].unique()
        self.logger.debug(f"고유 제품 수: {len(unique_products)}", level=LogLevel.LEVEL3)

        for product in unique_products:
            # 해당 제품의 매출 데이터 필터링
            product_sales = train_data[train_data['영업장명_메뉴명'] == product]['매출수량']

            # 최빈값 계산 (가장 자주 나오는 값)
            mode_value = product_sales.mode().iloc[0] if not product_sales.mode().empty else 1
            self.product_modes[product] = max(1, int(mode_value))  # 최소값 1로 보장

        # 제품 순서 맞추기 (sample_submission 파일 기준)
        submit = pd.read_csv(self.submit_csv)
        product_columns = submit.columns[1:]  # 첫 번째 컬럼(ID) 제외

        # 193개 제품의 최빈값을 순서대로 배열로 저장
        self.mode_values = np.array([
            self.product_modes.get(product, 1) for product in product_columns
        ])

        self.logger.success(f"{len(self.mode_values)}개 제품의 최빈값 계산 완료", level=LogLevel.LEVEL2)
        self.logger.info(f"최빈값 범위: {self.mode_values.min()} ~ {self.mode_values.max()}", level=LogLevel.LEVEL2)

    def _clip_to_mode(self, prediction: Union[np.ndarray, torch.Tensor], menu_encoding: Optional[torch.Tensor] = None):
        """
        각 제품별 최빈값을 기준으로 예측값을 클리핑하는 함수 (텐서/배열 호환)
        0 이하의 값만 해당 제품의 최빈값으로 교체

        Args:
            prediction (Union[np.ndarray, torch.Tensor]): (193, 7) 형태의 예측값 / (Batch, seq, feature) 형태의 검증 셋 예측값
            menu_encoding: (Batch, seq, feature) 형태가 들어왔을 때 메뉴 값을 계산하기 위한 값
        Returns:
            tuple: (클리핑된 예측값, 클리핑된 요소 개수)
        """
        is_tensor = isinstance(prediction, torch.Tensor)

        # 클리핑 전 원본 저장 및 복사본 생성
        if is_tensor:
            clipped_prediction = prediction.clone()
            device = prediction.device
            mode_values = torch.tensor(self.mode_values, device=device, dtype=prediction.dtype)

            clipped_count = 0

            # 0 이하인 값들을 찾기
            zero_or_negative_mask = prediction < 0.5

            # 0 이하인 값들에 대해 해당 메뉴의 최빈값으로 교체
            for i in range(prediction.shape[0]):
                for j in range(prediction.shape[1]):
                    if zero_or_negative_mask[i, j]:
                        menu_id = menu_encoding[i, j].long()
                        if 0 <= menu_id < 193:
                            clipped_prediction[i, j] = mode_values[menu_id]
                        else:
                            clipped_prediction[i, j] = 1

            clipped_count += torch.sum(zero_or_negative_mask).item()

            return clipped_prediction, int(clipped_count)

        else:
            clipped_prediction = prediction.copy()
            mode_values = self.mode_values

            clipped_count = 0

            for product_idx in range(min(193, prediction.shape[0])):
                # 해당 제품의 최빈값 가져오기
                mode_value = mode_values[product_idx] if product_idx < len(mode_values) else 1
                zero_or_negative_mask = prediction[product_idx] < 0.5
                clipped_prediction[product_idx][zero_or_negative_mask] = mode_value
                clipped_count += np.sum(zero_or_negative_mask)

            return clipped_prediction, int(clipped_count)

    def _clip_to_one(self, prediction: Union[np.ndarray, torch.Tensor]):
        """
        최소값을 1로 클리핑하는 함수 (텐서/배열 호환)

        Args:
            prediction (Union[np.ndarray, torch.Tensor]): 예측값

        Returns:
            tuple: (클리핑된 예측값, 클리핑된 요소 개수)
        """
        is_tensor = isinstance(prediction, torch.Tensor)

        if is_tensor:
            # 텐서 처리
            clipped_count = torch.sum(prediction < 1).item()  # .item()으로 스칼라 값 추출
            clipped_prediction = torch.clamp(prediction, min=1)
        else:
            # NumPy 배열 처리
            import numpy as np
            clipped_count = np.sum(prediction < 1)
            clipped_prediction = np.clip(prediction, 1, None)

        return clipped_prediction, int(clipped_count)

    def _clip_to_zero(self, prediction: Union[np.ndarray, torch.Tensor]):
        """
        최소값을 0으로 클리핑하는 함수 (텐서/배열 호환)

        Args:
            prediction (Union[np.ndarray, torch.Tensor]): 예측값

        Returns:
            tuple: (클리핑된 예측값, 클리핑된 요소 개수)
        """
        is_tensor = isinstance(prediction, torch.Tensor)

        if is_tensor:
            # 텐서 처리
            clipped_count = torch.sum(prediction < 0).item()
            clipped_prediction = torch.clamp(prediction, min=0)
        else:
            # NumPy 배열 처리
            import numpy as np
            clipped_count = np.sum(prediction < 0)
            clipped_prediction = np.clip(prediction, 0, None)

        return clipped_prediction, int(clipped_count)

    def _inverse_log_transform(self, prediction: Union[np.ndarray, torch.Tensor]):
        """
        Log(x+1) 변환의 역변환을 적용하는 함수
        exp(x) - 1을 사용하여 원래 스케일로 복원

        Args:
            prediction (Union[np.ndarray, torch.Tensor]): 로그 변환된 예측값 배열 또는 텐서

        Returns:
            tuple: (역변환된 예측값, 변환된 요소 개수)
                   - 반환 타입은 입력과 동일 (numpy → numpy, tensor → tensor)
        """

        # 입력 타입 확인
        is_tensor = isinstance(prediction, torch.Tensor)
        original_device = prediction.device if is_tensor else None

        if is_tensor:
            # PyTorch 텐서 처리
            # 역변환 적용: exp(x) - 1
            inverse_transformed = torch.expm1(prediction)  # expm1(x) = exp(x) - 1

            # 음수값을 0으로 클리핑 (안전장치)
            inverse_transformed = torch.clamp(inverse_transformed, min=0)

            # 모든 요소가 변환되므로 전체 요소 개수 반환
            transformed_count = prediction.numel()  # tensor의 경우 numel() 사용

        else:
            # NumPy 배열 처리
            # 역변환 적용: exp(x) - 1
            inverse_transformed = np.expm1(prediction)  # expm1(x) = exp(x) - 1

            # 음수값을 0으로 클리핑 (안전장치)
            inverse_transformed = np.clip(inverse_transformed, 0, None)

            # 모든 요소가 변환되므로 전체 요소 개수 반환
            transformed_count = prediction.size

        return inverse_transformed, transformed_count

    def postprocess(self, prediction: Union[np.ndarray, torch.Tensor], menu_encoding: Optional[torch.Tensor] = None):
        """
        설정된 방법에 따라 예측 결과를 후처리하는 함수 (텐서/배열 호환)
        텐서 입력 시 모든 로그를 비활성화하여 성능 최적화
        """

        # 입력 타입 확인
        is_tensor = isinstance(prediction, torch.Tensor)
        original_device = prediction.device if is_tensor else None

        # 🔇 로그 활성화 플래그 (텐서면 False)
        enable_logging = not is_tensor

        # 전처리에서 로그 변환이 사용되었는지 확인
        self.log_transform_used = self._check_log_transform_used(enable_logging)

        if enable_logging:
            self.logger.debug(f"PostProcessor 초기화 완료 - 후처리 방법: {self.postprocess_method.value}", level=LogLevel.LEVEL3)
            if self.log_transform_used:
                self.logger.debug("전처리에서 로그 변환이 사용됨 - 자동으로 역변환 적용", level=LogLevel.LEVEL3)

        # 텐서/배열에 따른 복사 방법
        processed_prediction = prediction.clone() if is_tensor else prediction.copy()
        total_processed_count = 0

        # 1단계: 로그 변환 역변환
        if self.log_transform_used:
            if enable_logging:
                self.logger.info("1단계: 역 로그 변환 자동 적용", level=LogLevel.LEVEL2)

            processed_prediction, inverse_count = self._inverse_log_transform(processed_prediction)

            if enable_logging:
                self.logger.info(f"역 로그 변환 완료: {inverse_count}개 요소 처리", level=LogLevel.LEVEL2)

        # 2단계: 클리핑 적용
        if enable_logging:
            self.logger.info(f"2단계: 클리핑 방법 적용 - {self.postprocess_method.value}", level=LogLevel.LEVEL2)

        if self.postprocess_method == PostProcessType.clip_to_one:
            if enable_logging:
                self.logger.info("최소값 1로 클리핑 적용", level=LogLevel.LEVEL2)
            processed_prediction, clip_count = self._clip_to_one(processed_prediction)

        elif self.postprocess_method == PostProcessType.clip_to_mode:
            if enable_logging:
                self.logger.info("최빈값 기준 클리핑 적용", level=LogLevel.LEVEL2)
            processed_prediction, clip_count = self._clip_to_mode(processed_prediction, menu_encoding)

        elif self.postprocess_method == PostProcessType.clip_to_zero:
            if enable_logging:
                self.logger.info("최소값 0으로 클리핑 적용", level=LogLevel.LEVEL2)
            processed_prediction, clip_count = self._clip_to_zero(processed_prediction)

        elif self.postprocess_method == PostProcessType.none:
            if enable_logging:
                self.logger.info("클리핑 없음", level=LogLevel.LEVEL2)
            clip_count = 0
        else:
            error_msg = f"해당 후처리 메소드는 지원되지 않습니다.: {self.postprocess_method.value}"
            if enable_logging:
                self.logger.error(error_msg, level=LogLevel.LEVEL1)
            raise ValueError(error_msg)

        total_processed_count += clip_count

        if enable_logging:
            self.logger.info(f"클리핑 완료: {clip_count}개 요소 처리", level=LogLevel.LEVEL2)

        # 라운딩 적용
        round_to_int = self.config.postprocess.round_to_int

        if round_to_int:
            if is_tensor:
                rounded_prediction = torch.round(processed_prediction).to(original_device)
            else:
                rounded_prediction = np.round(processed_prediction).astype(int)
        else:
            rounded_prediction = processed_prediction

        return rounded_prediction, total_processed_count

    def save_predictions(self, predictions, save=True, filename=None):
        """
        예측 결과를 제출 형식으로 변환하여 CSV 파일로 저장하는 함수

        Args:
            predictions (list): 각 테스트 기간별 예측 결과 리스트
                               각 원소는 (193, 7) 형태의 numpy 배열
            save (bool): 파일로 저장할지 여부
        """
        if save:
            self.logger.info("예측 결과 저장 시작...", level=LogLevel.LEVEL2)
        else:
            self.logger.info("예측 결과 후처리 시작...", level=LogLevel.LEVEL2)

        self.logger.info(f"처리할 예측 결과 수: {len(predictions)}개", level=LogLevel.LEVEL2)

        # 제출 파일 템플릿 로드
        submit = pd.read_csv(self.submit_csv)
        pred = []

        # 전체 클리핑 통계
        total_clipped_count = 0
        total_elements = 0

        # 각 예측 결과에 대해 후처리 적용
        for i, prediction in enumerate(predictions):
            self.logger.debug(f"예측 결과 {i+1}/{len(predictions)} 처리 중...", level=LogLevel.LEVEL3)

            # 설정된 후처리 방법 적용
            processed_prediction, clipped_count = self.postprocess(prediction)

            # 클리핑 통계 업데이트
            total_clipped_count += clipped_count
            total_elements += prediction.size

            # 제출 형식에 맞게 전치 (193, 7) -> (7, 193)
            pred.append(processed_prediction.T)

        # 클리핑 통계 출력
        clipping_percentage = (total_clipped_count / total_elements) * 100
        self.logger.info("클리핑 통계:", level=LogLevel.LEVEL2)
        self.logger.info(f"클리핑된 요소 수: {total_clipped_count:,}개", level=LogLevel.LEVEL2)
        self.logger.info(f"전체 요소 수: {total_elements:,}개", level=LogLevel.LEVEL2)
        self.logger.info(f"클리핑 비율: {clipping_percentage:.2f}%", level=LogLevel.LEVEL2)

        # 제출 파일에 예측 결과 삽입
        for i, p in enumerate(pred):
            start_idx = (i * 7)  # 시작 인덱스 (각 테스트 기간당 7일)
            end_idx = start_idx + 7  # 끝 인덱스
            submit.iloc[start_idx:end_idx, 1:] = p[:7]  # ID 컬럼 제외하고 삽입

        if save:
            # CSV 파일로 저장
            if filename is not None:
                result_csv = os.path.join(self.result_dir, filename + "_submit.csv")
            else:
                result_csv = os.path.join(self.result_dir, f"{self.result_dir.split('/')[-1]}_{self.config.setting.name}_submit.csv")

            submit.to_csv(result_csv, index=False)
            self.logger.success("예측 결과 후처리 완료", level=LogLevel.LEVEL2)

            return submit