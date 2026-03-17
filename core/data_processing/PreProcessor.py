from typing import Dict, Union
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

from .FeatureProcessor import *
from ..config.Config import Config, Config, EncodingType, ScalingType
from ..utils.Logger import get_logger, LogLevel


class PreProcessor:
    """
        데이터 전처리 클래스
        default:
        - timestamp_to_numeric: 타임스탬프를 숫자로 변환, reference_date = 2023-01-01
        - negative_clipping: 음수 값 0으로 클리핑, 테스트 데이터에 음수 값 없음 확인(DACON FAQ)
        - encoding, scaling: 피처 엔지니어링 후 Config의 메서드로 인코딩 및 스케일링 진행
        - static_feature: 정적 특성 데이터(데이터 셋 만들 때 사용할 칼럼 필수)

        Parameters:
        - data: 원본 데이터
        - timestamp_ext: 타임스탬프를 전처리하여 순환형 시간 피처 추출(year, month, day, weekday, day_of_year, week_of_year)
        - lag: True면 lag 피처 생성
        - rolling_mean: True면 rolling mean 피처 생성
        - season: True면 계절 피처 생성
        - is_holiday: True면 휴일 피처 생성

    """

    def __init__(self, config: Union[Config, Config], preprocessing_components: Dict = None):
        # 기본 피처 전처리 컨피그 불러오기
        self.config = config
        self.logger = get_logger()

        self.logger.debug("PreProcessor 초기화 중...", level=LogLevel.LEVEL3)

        # 인코딩 및 스케일링 설정
        self.encoding_method = self.config.preprocess.encoding_method
        self.encoding_features = self.config.preprocess.encoding_features
        self.scaling_method = self.config.preprocess.scaling_method
        self.scaling_features = self.config.preprocess.scaling_features

        # 인코더 및 스케일러 파라미터 저장용
        if preprocessing_components:
            # 저장된 전처리 컴포넌트 사용
            self.label_encoders = preprocessing_components.get('label_encoders', {})
            self.scalers = preprocessing_components.get('scalers', {})
            self.logger.info(f"저장된 전처리 컴포넌트 로드: 인코더 {len(self.label_encoders)}개, 스케일러 {len(self.scalers)}개", level=LogLevel.LEVEL3)
        else:
            # 새로운 전처리 컴포넌트 생성 (훈련 시)
            self.label_encoders = {}
            self.scalers = {}
            self.logger.debug("새로운 전처리 컴포넌트 생성", level=LogLevel.LEVEL3)

        # FeatureProcessor 등록
        self._initialize_processors()

        # 칼럼 순서 재조정을 위한 설정
        self.target_col = self.config.dataset.target_col
        self.date_col = self.config.dataset.date_col
        self.group_col = self.config.dataset.group_col

        self.logger.debug("PreProcessor 초기화 완료", level=LogLevel.LEVEL3)

    def _initialize_processors(self):
        """FeatureProcessor들을 초기화하고 등록합니다."""

        self.processor_config_map = {
            # default
            'static_feature_merge': True,
            'timestamp_to_numeric': True,
            'negative_clipping': True,
            # feature engineering
            'timestamp_ext': self.config.preprocess.timestamp_ext,

            'outlier_iqr_replacement': self.config.preprocess.outlier_iqr_replacement,  # IQR + 4.5* 이상치 대체
            'new_menu_zero_dropper': self.config.preprocess.new_menu_zero_dropper,
            'is_sales': self.config.preprocess.is_sales,  # 영업 중단 기간 및 정기 휴무일 기반 피처
            'historical_season_avg': self.config.preprocess.historical_season_avg,  # 작년 동일 시기 평균 판매량 피처
            'lag': self.config.preprocess.lag,
            'rolling_mean': self.config.preprocess.rolling_mean,
            'season': self.config.preprocess.season,
            'is_holiday': self.config.preprocess.is_holiday,  # 휴일 피처 (공휴일 + 주말만)
            'zero_streak': self.config.preprocess.zero_streak,
        }

        # 프로세서 인스턴스 생성
        self.processors = {
            # default
            'static_feature_merge': StaticFeatureMerger(),
            'timestamp_to_numeric': TimestampToNumericConverter(),
            'negative_clipping': NegativeClipper(),
            'outlier_iqr_replacement': OutlierClipper(),  # IQR + 4.5* 이상치 대체
            'new_menu_zero_dropper': NewMenuInitialZeroDropper(),
            'is_sales': SalesFeatureProcessor(),  # 영업 중단 기간 및 정기 휴무일 기반 피처
            'historical_season_avg': HistoricalSeasonAvgFeatureProcessor(),  # 작년 동일 시기 평균 판매량 피처
            # feature engineering
            'timestamp_ext': TimestampExtractor(),
            'lag': LagFeatureCreator(),
            'rolling_mean': RollingFeatureCreator(),
            'season': SeasonFeatureCreator(),
            'is_holiday': HolidayFeatureCreator(),  # 휴일 피처 (공휴일 + 주말만)
            'zero_streak': ZeroStreakFeatureCreator(),
        }

        # 🔄 처리 순서 정의 (중요: 의존성 고려)
        self.processing_order = [
            'negative_clipping',  # 기본 클리핑
            'outlier_iqr_replacement',  # IQR + 4.5* 이상치 대체
            'timestamp_ext',  # 타임스탬프 기반 특성 생성
            'static_feature_merge',  # 먼저 정적 특성 병합
            'new_menu_zero_dropper',
            'is_sales',  # 영업 중단 기간 및 정기 휴무일 기반 피처 생성
            'historical_season_avg',  # 작년 동일 시기 평균 판매량 피처 생성
            'lag',  # 지연 특성
            'rolling_mean',  # 이동평균 특성
            'season',  # 계절 특성
            'is_holiday',  # 휴일 특성 (공휴일 + 주말만)
            'timestamp_to_numeric',  # 타임스탬프 숫자형 변환
            'zero_streak',  # 연속 0 특성 (timestamp 필요)
        ]

    def preprocess(self, data: pd.DataFrame, static_feature: pd.DataFrame,
                   for_train: bool = True) -> pd.DataFrame:
        """
        메인 전처리 함수

        Args:
            data: 원본 데이터
            static_feature: 정적 특성 데이터
            for_train: 훈련용인지 여부 (인코딩/스케일링 fit 결정)

        Returns:
            전처리된 데이터
        """
        # 피처 전처리 실행
        preprocessed = self.preprocess_features(data, static_feature, fit=for_train)

        # 범주형 인코딩
        preprocessed = self.encode_categorical(preprocessed, fit=for_train)

        # 스케일링
        preprocessed = self.scale_features(preprocessed, fit=for_train)

        # 칼럼 순서 재조정(sales;target_idx = 0, date = 1, store_menu = 2)
        preprocessed = self._reorder_data(preprocessed)

        return preprocessed

    def preprocess_features(self, data: pd.DataFrame, static_feature: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        피처 전처리 파이프라인 실행

        설정에 따라 활성화된 FeatureProcessor들을 순차적으로 실행합니다.
        """
        data = data.copy()

        # 컬럼명 표준화
        data.columns = ['timestamp', 'store_menu', 'sales']

        # 타임스탬프 변환 및 정렬
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data = data.sort_values(['store_menu', 'timestamp'])

        self.logger.info("피처 전처리 파이프라인 실행 중...", level=LogLevel.LEVEL2)

        # 🎯 설정된 프로세서들을 순서대로 실행
        for processor_name in self.processing_order:
            if self.processor_config_map[processor_name]:
                self.logger.debug(f"{processor_name} 처리 중...", level=LogLevel.LEVEL3)
                processor = self.processors[processor_name]
                data = processor.process(data, static_feature=static_feature, fit=fit)

        self._print_preprocess_features(data, static_feature)

        return data

    def _print_preprocess_features(self, data, static_feature):
        # 📋 최종 생성된 데이터의 컬럼 정보 출력
        self.logger.success("전처리 완료 - 최종 데이터 정보", level=LogLevel.LEVEL2)
        self.logger.info(f"데이터 형태: {data.shape}", level=LogLevel.LEVEL2)
        self.logger.info(f"총 컬럼 수: {len(data.columns)}개", level=LogLevel.LEVEL2)

        # 컬럼을 카테고리별로 그룹핑하여 출력
        basic_columns = ['timestamp', 'store_menu', 'sales']
        feature_columns = [col for col in data.columns if col not in basic_columns]

        self.logger.info("컬럼 리스트:", level=LogLevel.LEVEL2)
        self.logger.info(f"기본 컬럼 ({len(basic_columns)}개): {basic_columns}", level=LogLevel.LEVEL2)

        if feature_columns:
            # 특성별로 그룹화
            lag_features = [col for col in feature_columns if 'lag_' in col]
            rolling_features = [col for col in feature_columns if 'rolling_' in col]
            time_features = [col for col in feature_columns if
                             any(x in col for x in ['sin_', 'cos_', 'day_', 'week_', 'month_'])]
            static_features = [col for col in feature_columns if col in static_feature.columns if
                               'static_feature' in locals()]
            other_features = [col for col in feature_columns if
                              col not in lag_features + rolling_features + time_features + (
                                  static_features if 'static_features' in locals() else [])]

            if lag_features:
                self.logger.info(
                    f"지연 특성 ({len(lag_features)}개): {lag_features[:5]}{'...' if len(lag_features) > 5 else ''}",
                    level=LogLevel.LEVEL3)
            if rolling_features:
                self.logger.info(
                    f"이동평균 특성 ({len(rolling_features)}개): {rolling_features[:5]}{'...' if len(rolling_features) > 5 else ''}",
                    level=LogLevel.LEVEL3)
            if time_features:
                self.logger.info(
                    f"시간 특성 ({len(time_features)}개): {time_features[:5]}{'...' if len(time_features) > 5 else ''}",
                    level=LogLevel.LEVEL3)
            if 'static_features' in locals() and static_features:
                self.logger.info(
                    f"정적 특성 ({len(static_features)}개): {static_features[:5]}{'...' if len(static_features) > 5 else ''}",
                    level=LogLevel.LEVEL3)
            if other_features:
                self.logger.info(
                    f"기타 특성 ({len(other_features)}개): {other_features[:5]}{'...' if len(other_features) > 5 else ''}",
                    level=LogLevel.LEVEL3)

        # 전체 컬럼 리스트 (디버깅용)
        self.logger.debug(f"전체 컬럼: {list(data.columns)}", level=LogLevel.LEVEL3)

    def encode_categorical(self, data: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """범주형 변수 인코딩"""
        self.logger.info(f"범주형 변수 인코딩 시작 (방법: {self.encoding_method.value})", level=LogLevel.LEVEL2)

        encoded_columns = []
        encoding_info = {}

        for column in self.encoding_features:
            if column in data.columns:
                original_unique_count = data[column].nunique()

                if fit:
                    # 훈련 시에는 fit_transform
                    if self.encoding_method == EncodingType.LabelEncoder:
                        self.label_encoders[column] = LabelEncoder()
                        self.logger.debug(f"LabelEncoder 적용: {column}", level=LogLevel.LEVEL3)
                        data[column] = self.label_encoders[column].fit_transform(data[column])
                        
                        # 인코딩 정보 저장
                        encoding_info[column] = {
                            'original_unique': original_unique_count,
                            'encoded_range': f"0~{len(self.label_encoders[column].classes_) - 1}",
                            'classes_count': len(self.label_encoders[column].classes_),
                            'encoding_type': 'LabelEncoder'
                        }
                    elif self.encoding_method == EncodingType.EmbeddingEncoder:
                        # EmbeddingEncoder: 카테고리 매핑 정보를 보존하면서 인덱스로 변환
                        self.label_encoders[column] = LabelEncoder()
                        self.logger.debug(f"EmbeddingEncoder 적용: {column}", level=LogLevel.LEVEL3)
                        data[column] = self.label_encoders[column].fit_transform(data[column])
                        
                        # 임베딩을 위한 추가 정보 저장
                        encoding_info[column] = {
                            'original_unique': original_unique_count,
                            'encoded_range': f"0~{len(self.label_encoders[column].classes_) - 1}",
                            'classes_count': len(self.label_encoders[column].classes_),
                            'encoding_type': 'EmbeddingEncoder',
                            'category_mapping': dict(enumerate(self.label_encoders[column].classes_)),
                            'embedding_dim': min(50, (len(self.label_encoders[column].classes_) + 1) // 2)  # 임베딩 차원 계산
                        }
                        
                        # 임베딩 정보를 별도로 저장 (모델에서 사용할 수 있도록)
                        if not hasattr(self, 'embedding_info'):
                            self.embedding_info = {}
                        self.embedding_info[column] = encoding_info[column]
                else:
                    # 테스트 시에는 transform만
                    if column in self.label_encoders:
                        data[column] = self.label_encoders[column].transform(data[column])

                        # 테스트 데이터의 인코딩 정보
                        if self.encoding_method == EncodingType.LabelEncoder:
                            encoding_info[column] = {
                                'original_unique': original_unique_count,
                                'encoded_range': f"0~{len(self.label_encoders[column].classes_) - 1}",
                                'classes_count': len(self.label_encoders[column].classes_),
                                'encoding_type': 'LabelEncoder'
                            }
                        elif self.encoding_method == EncodingType.EmbeddingEncoder:
                            encoding_info[column] = {
                                'original_unique': original_unique_count,
                                'encoded_range': f"0~{len(self.label_encoders[column].classes_) - 1}",
                                'classes_count': len(self.label_encoders[column].classes_),
                                'encoding_type': 'EmbeddingEncoder',
                                'category_mapping': dict(enumerate(self.label_encoders[column].classes_)),
                                'embedding_dim': min(50, (len(self.label_encoders[column].classes_) + 1) // 2)
                            }
                    else:
                        error_msg = f"인코더가 {column}에 대해 훈련되지 않았습니다."
                        self.logger.error(error_msg, level=LogLevel.LEVEL1)
                        raise ValueError(error_msg)

                encoded_columns.append(column)
            else:
                self.logger.warning(f"컬럼 '{column}'이 데이터에 존재하지 않습니다.", level=LogLevel.LEVEL1)

        self._print_encode_features(data, encoded_columns, encoding_info)

        return data

    def _print_encode_features(self, data, encoded_columns, encoding_info):
        # 📊 인코딩 결과 출력
        if encoded_columns:
            self.logger.info(f"인코딩 완료된 컬럼 ({len(encoded_columns)}개):", level=LogLevel.LEVEL2)
            for column in encoded_columns:
                info = encoding_info[column]
                if info.get('encoding_type') == 'EmbeddingEncoder':
                    self.logger.info(
                        f"{column}: {info['original_unique']}개 고유값 → {info['encoded_range']} ({info['classes_count']}개 클래스, 임베딩 차원: {info['embedding_dim']})",
                        level=LogLevel.LEVEL3)
                else:
                    self.logger.info(
                        f"{column}: {info['original_unique']}개 고유값 → {info['encoded_range']} ({info['classes_count']}개 클래스)",
                        level=LogLevel.LEVEL3)

            # 인코딩 대상이었지만 처리되지 않은 컬럼들 확인
            missing_columns = [col for col in self.encoding_features if col not in data.columns]
            if missing_columns:
                self.logger.warning(f"누락된 컬럼 ({len(missing_columns)}개): {missing_columns}", level=LogLevel.LEVEL2)
        else:
            self.logger.info("인코딩할 수 있는 컬럼이 없습니다.", level=LogLevel.LEVEL2)

        self.logger.success("범주형 변수 인코딩 완료!", level=LogLevel.LEVEL2)


    def scale_features(self, data: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """피처 스케일링"""
        self.logger.info(f"피처 스케일링 시작 (방법: {self.scaling_method.value})", level=LogLevel.LEVEL2)

        if self.scaling_method == ScalingType.none:
            self.logger.info("스케일링이 비활성화되어 있습니다.", level=LogLevel.LEVEL2)
            return data

        scaled_columns = []
        scaling_info = {}

        for column in self.scaling_features:
            if column in data.columns:
                # 원본 데이터 통계 정보
                original_mean = data[column].mean()
                original_std = data[column].std()
                original_min = data[column].min()
                original_max = data[column].max()

                if fit:
                    # 훈련 시에는 fit_transform
                    if self.scaling_method == ScalingType.StandardScaler:
                        self.scalers[column] = StandardScaler()
                        scaler_name = "StandardScaler"
                        self.logger.debug(f"StandardScaler 적용: {column}", level=LogLevel.LEVEL3)
                    elif self.scaling_method == ScalingType.MinMaxScaler:
                        self.scalers[column] = MinMaxScaler()
                        scaler_name = "MinMaxScaler"
                        self.logger.debug(f"MinMaxScaler 적용: {column}", level=LogLevel.LEVEL3)
                    elif self.scaling_method == ScalingType.LogTransform:
                        # Log(x+1) 변환 - 음수값을 0으로 클리핑 후 변환
                        data[column] = data[column].clip(lower=0)
                        data[column] = np.log1p(data[column])  # log(1+x)
                        self.scalers[column] = "LogTransform"  # 변환 타입 저장
                        scaler_name = "LogTransform"
                        self.logger.debug(f"LogTransform 적용: {column}", level=LogLevel.LEVEL3)
                    else:
                        continue  # 스케일링 안함

                    # LogTransform은 이미 변환이 완료되었으므로 sklearn scaler만 적용
                    if self.scaling_method != ScalingType.LogTransform:
                        data[column] = self.scalers[column].fit_transform(data[[column]]).flatten()

                    # 스케일링 후 통계 정보
                    scaled_mean = data[column].mean()
                    scaled_std = data[column].std()
                    scaled_min = data[column].min()
                    scaled_max = data[column].max()

                    scaling_info[column] = {
                        'scaler': scaler_name,
                        'original_stats': {
                            'mean': original_mean,
                            'std': original_std,
                            'min': original_min,
                            'max': original_max
                        },
                        'scaled_stats': {
                            'mean': scaled_mean,
                            'std': scaled_std,
                            'min': scaled_min,
                            'max': scaled_max
                        }
                    }

                else:
                    # 테스트 시에는 transform만
                    if column in self.scalers:
                        if self.scalers[column] == "LogTransform":
                            # LogTransform의 경우 직접 변환 적용
                            data[column] = data[column].clip(lower=0)
                            data[column] = np.log1p(data[column])  # log(1+x)
                            scaler_name = "LogTransform"
                        else:
                            # sklearn scaler의 경우 transform 적용
                            data[column] = self.scalers[column].transform(data[[column]]).flatten()
                            scaler_name = type(self.scalers[column]).__name__

                        # 테스트 데이터의 스케일링 후 통계 정보
                        scaled_mean = data[column].mean()
                        scaled_std = data[column].std()
                        scaled_min = data[column].min()
                        scaled_max = data[column].max()

                        scaling_info[column] = {
                            'scaler': scaler_name,
                            'original_stats': {
                                'mean': original_mean,
                                'std': original_std,
                                'min': original_min,
                                'max': original_max
                            },
                            'scaled_stats': {
                                'mean': scaled_mean,
                                'std': scaled_std,
                                'min': scaled_min,
                                'max': scaled_max
                            }
                        }
                    else:
                        error_msg = f"스케일러가 {column}에 대해 훈련되지 않았습니다."
                        self.logger.error(error_msg, level=LogLevel.LEVEL1)
                        raise ValueError(error_msg)

                scaled_columns.append(column)
            else:
                self.logger.warning(f"컬럼 '{column}'이 데이터에 존재하지 않습니다.", level=LogLevel.LEVEL1)

        self._print_scaling_features(data, scaled_columns, scaling_info)

        return data

    def _print_scaling_features(self, data, scaled_columns, scaling_info):
        # 📊 스케일링 결과 출력
        if scaled_columns:
            self.logger.info(f"스케일링 완료된 컬럼 ({len(scaled_columns)}개):", level=LogLevel.LEVEL2)
            for column in scaled_columns:
                info = scaling_info[column]
                orig = info['original_stats']
                scaled = info['scaled_stats']

                self.logger.debug(f"{column} ({info['scaler']}):", level=LogLevel.LEVEL3)
                self.logger.debug(
                    f"  원본: 평균={orig['mean']:.4f}, 표준편차={orig['std']:.4f}, 범위=[{orig['min']:.2f}, {orig['max']:.2f}]",
                    level=LogLevel.LEVEL3)
                self.logger.debug(
                    f"  변환: 평균={scaled['mean']:.4f}, 표준편차={scaled['std']:.4f}, 범위=[{scaled['min']:.4f}, {scaled['max']:.4f}]",
                    level=LogLevel.LEVEL3)

            # 스케일링 대상이었지만 처리되지 않은 컬럼들 확인
            missing_columns = [col for col in self.scaling_features if col not in data.columns]
            if missing_columns:
                self.logger.warning(f"누락된 컬럼 ({len(missing_columns)}개): {missing_columns}", level=LogLevel.LEVEL2)
        else:
            self.logger.info("스케일링할 수 있는 컬럼이 없습니다.", level=LogLevel.LEVEL2)

        self.logger.success("피처 스케일링 완료!", level=LogLevel.LEVEL2)


    def _reorder_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
            데이터의 컬럼 순서를 target_col, date_col, group_col 순으로 재조정
             - target_col: features[0]
             - date_col: features[1]
             - group_col: features[2]
        """
        self.logger.debug("데이터 컬럼 순서 재조정 중...", level=LogLevel.LEVEL3)

        # 필요한 컬럼들 정의 (실제로 존재하는 컬럼만 포함)
        potential_primary_columns = [
            self.target_col,  # sales
            self.date_col,  # timestamp (또는 변환된 날짜 컬럼)
            self.group_col  # store_menu
        ]
        
        # 실제로 존재하는 primary 컬럼들만 선택
        primary_columns = [col for col in potential_primary_columns if col in data.columns]
        
        # 존재하지 않는 컬럼들 로깅
        missing_columns = [col for col in potential_primary_columns if col not in data.columns]
        if missing_columns:
            self.logger.debug(f"누락된 primary 컬럼들: {missing_columns}", level=LogLevel.LEVEL3)

        # 기존 데이터에서 primary_columns를 제외한 나머지 컬럼들
        remaining_columns = [col for col in data.columns if col not in primary_columns]

        # 최종 컬럼 순서: primary_columns + remaining_columns
        new_column_order = primary_columns + remaining_columns

        # 컬럼 순서 재조정
        reordered_data = data[new_column_order]

        self.logger.debug(f"컬럼 순서 재조정 완료: {primary_columns} + {len(remaining_columns)}개 추가 컬럼", level=LogLevel.LEVEL3)
        self.logger.debug(f"새로운 컬럼 순서: {list(reordered_data.columns[:5])}{'...' if len(reordered_data.columns) > 5 else ''}", level=LogLevel.LEVEL3)

        return reordered_data


    def get_active_processors(self) -> list:
        """현재 활성화된 프로세서 목록을 반환합니다."""
        return [name for name in self.processing_order if self.processor_config_map[name]]
    
    def get_preprocessing_components(self) -> Dict:
        """전처리 컴포넌트(라벨 인코더, 스케일러, 임베딩 정보)를 반환합니다."""
        components = {
            'label_encoders': self.label_encoders,
            'scalers': self.scalers
        }
        
        # 임베딩 정보가 있으면 추가
        if hasattr(self, 'embedding_info'):
            components['embedding_info'] = self.embedding_info
            
        return components
    
    def get_embedding_info(self) -> Dict:
        """임베딩 정보 반환 (모델에서 임베딩 레이어 생성 시 사용)"""
        if hasattr(self, 'embedding_info'):
            return self.embedding_info
        else:
            return {}

    def print_processor_info(self):
        """현재 등록된 프로세서들의 정보를 출력합니다."""
        self.logger.info("\n등록된 Feature Processors:", level=LogLevel.LEVEL2)
        self.logger.info("=" * 50, level=LogLevel.LEVEL2)

        active_processors = self.get_active_processors()

        for i, name in enumerate(self.processing_order, 1):
            status = "활성" if name in active_processors else "비활성"
            processor = self.processors.get(name)
            processor_class = processor.__class__.__name__ if processor else "Unknown"

            self.logger.info(f"{i:2d}. {status} | {name:<20} | {processor_class}", level=LogLevel.LEVEL2)

        self.logger.info("=" * 50, level=LogLevel.LEVEL2)
        self.logger.info(f"총 {len(self.processors)}개 프로세서 중 {len(active_processors)}개 활성화", level=LogLevel.LEVEL2)