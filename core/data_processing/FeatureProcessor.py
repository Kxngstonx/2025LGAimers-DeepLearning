from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd


# ============================
# Feature Processor 인터페이스 및 구현체들
# ============================

class FeatureProcessor(ABC):
    """피처 처리 인터페이스"""

    @abstractmethod
    def process(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """피처 처리를 수행합니다."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """프로세서 이름을 반환합니다."""
        pass


class NewMenuInitialZeroDropper(FeatureProcessor):
    """신메뉴 초기 0값 기간 제거 처리기"""

    def __init__(self,
                 date_col: str = 'timestamp',
                 target_col: str = 'sales',
                 group_col: str = 'store_menu',
                 max_initial_zero_days: int = 300,
                 min_consecutive_nonzero_days: int = 3):
        """
        Args:
            date_col: 날짜 컬럼명
            target_col: 매출 컬럼명
            group_col: 그룹핑 컬럼명 (매장_메뉴)
            max_initial_zero_days: 최대 초기 0값 기간 (일)
            min_consecutive_nonzero_days: 메뉴 시작으로 판단할 최소 연속 판매일
        """
        self.date_col = date_col
        self.target_col = target_col
        self.group_col = group_col
        self.max_initial_zero_days = max_initial_zero_days
        self.min_consecutive_nonzero_days = min_consecutive_nonzero_days

    def process(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """신메뉴 초기 0값 기간 제거"""
        fit = kwargs.get('fit', True)

        if fit is False:
            print(f"테스트 데이터에 신메뉴 초기 0값 제거 미처리")
            return data

        print(f"신메뉴 초기 0값 제거 처리 시작")
        print(f"입력 데이터 크기: {data.shape}")

        processed_data = data.copy()

        # 날짜 컬럼이 datetime이 아니면 변환
        if processed_data[self.date_col].dtype != 'datetime64[ns]':
            if processed_data[self.date_col].dtype in ['int64', 'float64']:
                # days_since_reference 형태인 경우
                reference_date = pd.to_datetime('2023-01-01')
                processed_data[self.date_col] = reference_date + pd.to_timedelta(processed_data[self.date_col],
                                                                                 unit='D')
            else:
                processed_data[self.date_col] = pd.to_datetime(processed_data[self.date_col])

        # 전체 데이터 시작일 찾기
        global_start_date = processed_data[self.date_col].min()

        # 각 메뉴별로 처리
        groups_to_process = []
        total_dropped_rows = 0
        processed_menus = 0

        for group_id, group_data in processed_data.groupby(self.group_col):
            # 날짜별로 정렬
            group_data = group_data.sort_values(self.date_col).reset_index(drop=True)

            # 신메뉴 시작점 찾기
            menu_start_date = self._find_menu_start_date(group_data, global_start_date)

            if menu_start_date is not None:
                # 시작점 이후 데이터만 유지
                filtered_group = group_data[group_data[self.date_col] >= menu_start_date].copy()

                dropped_rows = len(group_data) - len(filtered_group)
                total_dropped_rows += dropped_rows

                if dropped_rows > 0:
                    processed_menus += 1
                    # numpy.datetime64를 pandas Timestamp로 변환하여 strftime 사용
                    min_date = pd.to_datetime(group_data[self.date_col].min())
                    start_date = pd.to_datetime(menu_start_date)

                    print(f"    {group_id}: {dropped_rows}행 제거 "
                          f"({min_date.strftime('%Y-%m-%d')} → "
                          f"{start_date.strftime('%Y-%m-%d')})")

                groups_to_process.append(filtered_group)
            else:
                # 시작점을 찾지 못한 경우 원본 데이터 유지
                groups_to_process.append(group_data)

        # 결과 데이터 결합
        if groups_to_process:
            processed_data = pd.concat(groups_to_process, ignore_index=True)

        print(f"    처리 완료:")
        print(f"      - 처리된 메뉴: {processed_menus}개")
        print(f"      - 제거된 행: {total_dropped_rows:,}개")
        print(f"      - 남은 데이터: {len(processed_data):,}행")

        return processed_data

    def _find_menu_start_date(self, group_data: pd.DataFrame, global_start_date: pd.Timestamp) -> Optional[
        pd.Timestamp]:
        """메뉴 실제 시작일 찾기"""

        # 전체 기간이 너무 짧으면 처리하지 않음
        if len(group_data) < self.min_consecutive_nonzero_days:
            return None

        # 전역 시작일부터의 데이터만 확인
        initial_data = group_data[group_data[self.date_col] >= global_start_date].copy()
        if len(initial_data) == 0:
            return None

        initial_data = initial_data.sort_values(self.date_col).reset_index(drop=True)
        sales_values = initial_data[self.target_col].values
        dates = initial_data[self.date_col].values

        # 처음부터 판매가 있었다면 전역 시작일 반환
        if sales_values[0] > 0:
            return global_start_date

        # 초기 0값 기간의 끝을 찾기
        zero_end_idx = self._find_initial_zero_end(sales_values, dates, global_start_date)

        if zero_end_idx is None:
            # 초기 0값 기간이 너무 길거나 판매가 시작되지 않음
            return None

        # 메뉴 시작일 반환
        return dates[zero_end_idx]

    def _find_initial_zero_end(self, sales_values: np.ndarray, dates: np.ndarray,
                               global_start_date: pd.Timestamp) -> Optional[int]:
        """초기 0값 기간의 끝 인덱스 찾기"""

        n = len(sales_values)

        # 1. 첫 번째 0이 아닌 값 찾기
        first_nonzero_idx = None
        for i in range(n):
            if sales_values[i] > 0:
                first_nonzero_idx = i
                break

        if first_nonzero_idx is None:
            # 전체 기간 동안 판매가 없음
            return None

        # 2. 초기 0값 기간이 너무 긴지 확인
        first_nonzero_date = pd.to_datetime(dates[first_nonzero_idx])
        initial_zero_duration = (first_nonzero_date - global_start_date).days

        if initial_zero_duration > self.max_initial_zero_days:
            # 너무 긴 초기 0값 기간은 신메뉴가 아닐 수 있음
            return None

        # 3. 연속적인 판매 시작점 찾기
        consecutive_start_idx = self._find_consecutive_sales_start(
            sales_values, first_nonzero_idx
        )

        return consecutive_start_idx

    def _find_consecutive_sales_start(self, sales_values: np.ndarray, start_idx: int) -> Optional[int]:
        """연속적인 판매 시작점 찾기"""

        n = len(sales_values)

        # start_idx부터 연속 판매 구간 찾기
        for i in range(start_idx, n):
            if sales_values[i] > 0:
                # i부터 연속 판매일 확인
                consecutive_count = 0
                for j in range(i, min(i + self.min_consecutive_nonzero_days * 2, n)):
                    if sales_values[j] > 0:
                        consecutive_count += 1
                        if consecutive_count >= self.min_consecutive_nonzero_days:
                            return i
                    else:
                        consecutive_count = 0

        # 연속 판매 구간을 찾지 못한 경우 첫 번째 판매일 반환
        return start_idx

    def get_name(self) -> str:
        return "NewMenuInitialZeroDropper"


class StaticFeatureMerger(FeatureProcessor):
    """정적 특성 병합 처리기"""

    def process(self, data: pd.DataFrame, static_feature: Optional[pd.DataFrame] = None, **kwargs) -> pd.DataFrame:
        if static_feature is not None:
            data = pd.merge(
                data,
                static_feature,
                left_on='store_menu',
                right_on='item_id',
                how='left'
            ).drop(columns=['item_id'])
        return data

    def get_name(self) -> str:
        return "static_feature_merge"


class TimestampExtractor(FeatureProcessor):
    """시간 특성 추출 처리기"""

    def process(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        # 순환형 시간 피처
        data['month_sin'] = np.sin(2 * np.pi * data['timestamp'].dt.month / 12)
        data['month_cos'] = np.cos(2 * np.pi * data['timestamp'].dt.month / 12)
        data['weekday_sin'] = np.sin(2 * np.pi * data['timestamp'].dt.weekday / 7)
        data['weekday_cos'] = np.cos(2 * np.pi * data['timestamp'].dt.weekday / 7)
        data['day_sin'] = np.sin(2 * np.pi * data['timestamp'].dt.day / 31)
        data['day_cos'] = np.cos(2 * np.pi * data['timestamp'].dt.day / 31)
        data['dayofyear_sin'] = np.sin(2 * np.pi * data['timestamp'].dt.dayofyear / 365)
        data['dayofyear_cos'] = np.cos(2 * np.pi * data['timestamp'].dt.dayofyear / 365)

        return data

    def get_name(self) -> str:
        return "timestamp_ext"


class NegativeClipper(FeatureProcessor):
    """매출 클리핑 처리기"""

    def process(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        data['sales'] = data['sales'].clip(lower=0)
        return data

    def get_name(self) -> str:
        return "clipping"


class LagFeatureCreator(FeatureProcessor):
    """지연 특성 생성 처리기"""

    def __init__(self, lag_periods: list = None):
        self.lag_periods = lag_periods or [7, 14, 21, 28]

    def process(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        for period in self.lag_periods:
            data[f'lag_{period}'] = data.groupby('store_menu')['sales'].shift(period)
            data[f'lag_{period}'] = data[f'lag_{period}'].fillna(0)
        return data

    def get_name(self) -> str:
        return "lag"


class RollingFeatureCreator(FeatureProcessor):
    """이동평균 특성 생성 처리기"""

    def __init__(self, rolling_periods: list = None):
        self.rolling_periods = rolling_periods or [7, 14, 21, 28]

    def process(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        for period in self.rolling_periods:
            data[f'rolling_{period}mean'] = data.groupby('store_menu')['sales'].transform(
                lambda x: x.rolling(period, min_periods=1).mean())
            data[f'rolling_{period}mean'] = data[f'rolling_{period}mean'].fillna(0)
        return data

    def get_name(self) -> str:
        return "rolling_mean"


class SeasonFeatureCreator(FeatureProcessor):
    """계절 특성 생성 처리기"""

    def process(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        def get_season(date):
            month = date.month
            if month in [3, 4, 5]:
                return '봄'
            elif month in [6, 7, 8]:
                return '여름'
            elif month in [9, 10, 11]:
                return '가을'
            else:
                return '겨울'

        data['season'] = data['timestamp'].apply(get_season)
        return data

    def get_name(self) -> str:
        return "season"


class HolidayFeatureCreator(FeatureProcessor):
    """휴일 특성 생성 처리기 (공휴일과 주말만)"""

    def __init__(self):
        self.holidays = {
            # 2023년 공휴일
            "2023-01-01", "2023-01-21", "2023-01-22", "2023-01-23", "2023-01-24",
            "2023-03-01", "2023-05-05", "2023-05-27", "2023-06-06", "2023-08-15",
            "2023-09-28", "2023-09-29", "2023-09-30", "2023-10-03", "2023-10-09",
            "2023-12-25",

            # 2024년 공휴일
            "2024-01-01", "2024-02-09", "2024-02-10", "2024-02-11",
            "2024-02-12", "2024-03-01", "2024-05-05", "2024-05-15", "2024-06-06",
            "2024-08-15", "2024-09-16", "2024-09-17", "2024-09-18", "2024-10-03",
            "2024-10-09", "2024-12-25",

            # 2025년 공휴일 (1월~5월)
            "2025-01-01",  # 신정
            "2025-01-28", "2025-01-29", "2025-01-30",  # 설날 연휴 (1월 29일이 설날)
            "2025-03-01",  # 3.1절
            "2025-05-01",  # 근로자의 날
            "2025-05-05",  # 어린이날
            "2025-05-06"  # 부처님오신날
        }

    def process(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        def is_holiday(date):
            # 주말 체크
            if date.weekday() >= 5:  # 주말 (토요일=5, 일요일=6)
                return 1
            # 공휴일 체크
            if date.strftime('%Y-%m-%d') in self.holidays:
                return 1
            return 0

        print("휴일 피처 생성 중 (공휴일 + 주말)...")
        data['is_holiday'] = data['timestamp'].apply(is_holiday)

        # 통계 출력
        holiday_count = (data['is_holiday'] == 1).sum()
        total_rows = len(data)
        print(f"    처리 완료:")
        print(f"      - 전체 데이터: {total_rows:,}행")
        print(f"      - 휴일 데이터: {holiday_count:,}행 (is_holiday=1)")
        print(f"      - 평일 데이터: {total_rows - holiday_count:,}행 (is_holiday=0)")
        print(f"      - 휴일 비율: {holiday_count / total_rows * 100:.2f}%")

        return data

    def get_name(self) -> str:
        return "is_holiday"


class SalesFeatureProcessor(FeatureProcessor):
    """영업 중단 기간 및 정기 휴무일 기반 is_sales 피처 생성 처리기"""

    def __init__(self):
        """
        영업 중단 기간 데이터를 초기화합니다.
        0: 중단 기간 내 (판매 불가)
        1: 정상 영업 기간 (판매 가능)
        """
        # 매장별 전체 중단 기간
        self.store_suspensions = {
            '느티나무 셀프BBQ': [
                ('2023-03-01', '2023-03-10'),
                ('2024-03-03', '2024-03-14'),
                ('2025-03-04', '2025-03-13')
            ],
            '미라시아': [
                ('2023-03-02', '2023-03-10'),
                ('2024-03-04', '2024-03-13'),
                ('2025-03-04', '2025-03-13')
            ],
            '화담숲주막': [
                ('2023-01-01', '2023-03-31'),
                ('2023-11-27', '2024-03-29'),
                ('2024-11-27', '2025-03-28')
            ],
            '연회장': [
                ('2023-03-01', '2023-03-10'),
                ('2024-03-04', '2024-03-14'),
                ('2024-09-22', '2024-10-02'),
                ('2024-10-27', '2024-11-04'),
                ('2024-12-01', '2024-12-09'),
                ('2025-03-03', '2025-03-13'),
                ('2025-03-16', '2025-03-24'),
                ('2025-04-20', '2025-04-28')
            ],
            '라그로타': [
                ('2023-03-02', '2023-03-10'),
                ('2024-03-04', '2024-03-13'),
                ('2025-03-03', '2025-03-23')
            ],
            '화담숲카페': [
                ('2023-01-01', '2023-03-31'),
                ('2023-11-27', '2024-03-29'),
                ('2024-11-27', '2025-03-28')
            ],
            '카페테리아': [
                ('2023-03-02', '2023-03-10'),
                ('2024-03-04', '2024-03-14'),
                ('2025-03-04', '2025-03-13')
            ],
            '담하': [
                ('2023-03-02', '2023-03-10'),
                ('2024-03-04', '2024-03-13'),
                ('2025-03-04', '2025-03-13')
            ],
            '포레스트릿': [
                ('2023-03-02', '2023-03-12'),
                ('2023-03-13', '2023-04-01'),
                ('2023-11-27', '2023-12-06'),
                ('2024-03-04', '2024-03-27'),
                ('2024-11-18', '2024-12-12'),
                ('2025-03-04', '2025-03-29')
            ]
        }

        # 메뉴별 중단 기간 (store_menu 형태로 저장)
        self.menu_suspensions = {
            '느티나무 셀프BBQ_BBQ55(단체)': [
                ('2023-12-23', '2024-03-13')
            ],
            '느티나무 셀프BBQ_본삼겹 (단품,실내)': [
                ('2023-02-26', '2023-03-31'),
                ('2023-06-06', '2023-07-19')
            ],
            '느티나무 셀프BBQ_신라면': [
                ('2023-10-30', '2024-04-18'),
                ('2024-10-28', '2025-04-18')
            ],
            '느티나무 셀프BBQ_일회용 소주컵': [
                ('2024-02-09', '2024-03-22')
            ],
            '느티나무 셀프BBQ_일회용 종이컵': [
                ('2024-01-31', '2024-03-22')
            ],
            '느티나무 셀프BBQ_잔디그늘집 대여료 (12인석)': [
                ('2023-01-01', '2023-04-13'),
                ('2023-10-30', '2024-04-18'),
                ('2024-11-04', '2025-04-26')
            ],
            '느티나무 셀프BBQ_잔디그늘집 대여료 (6인석)': [
                ('2023-01-06', '2023-03-09'),
                ('2023-03-11', '2023-04-13'),
                ('2023-11-15', '2024-04-18'),
                ('2024-11-04', '2025-04-26')
            ],
            '느티나무 셀프BBQ_잔디그늘집 의자 추가': [
                ('2023-01-01', '2023-04-13'),
                ('2023-10-29', '2024-04-18'),
                ('2024-10-27', '2025-04-29')
            ],
            '느티나무 셀프BBQ_햇반': [
                ('2023-01-01', '2023-04-13')
            ],
            '느티나무 셀프BBQ_허브솔트': [
                ('2025-02-18', '2025-03-27')
            ],
            '미라시아_(오븐) 하와이안 쉬림프 피자': [
                ('2023-01-01', '2023-09-08')
            ],
            '미라시아_BBQ Platter': [
                ('2024-09-01', '2024-10-05')
            ],
            '미라시아_공깃밥': [
                ('2023-03-19', '2024-01-24')
            ],
            '미라시아_글라스와인 (레드)': [
                ('2023-04-29', '2023-06-08'),
                ('2025-03-02', '2025-04-18')
            ],
            '미라시아_버드와이저(무제한)': [
                ('2023-01-01', '2023-04-20')
            ],
            '미라시아_보일링 랍스타 플래터': [
                ('2023-01-01', '2023-06-04'),
                ('2024-09-01', '2024-10-08')
            ],
            '미라시아_보일링 랍스타 플래터(덜매운맛)': [
                ('2023-01-01', '2023-06-02'),
                ('2024-08-27', '2024-10-05')
            ],
            '미라시아_브런치 4인 패키지 ': [
                ('2024-02-24', '2024-04-10'),
                ('2024-11-25', '2025-01-24'),
                ('2025-02-23', '2025-04-12')
            ],
            '미라시아_브런치(대인) 주중': [
                ('2024-07-14', '2024-08-25')
            ],
            '미라시아_쉬림프 투움바 파스타': [
                ('2023-01-01', '2023-06-02')
            ],
            '미라시아_스텔라(무제한)': [
                ('2023-01-01', '2023-04-20')
            ],
            '미라시아_스프라이트': [
                ('2023-01-01', '2023-06-01')
            ],
            '미라시아_유자 하이볼': [
                ('2023-01-01', '2023-03-16')
            ],
            '미라시아_잭 애플 토닉': [
                ('2023-01-01', '2023-09-08')
            ],
            '미라시아_칠리 치즈 프라이': [
                ('2023-01-01', '2023-06-02')
            ],
            '미라시아_코카콜라': [
                ('2023-01-01', '2023-06-01')
            ],
            '미라시아_코카콜라(제로)': [
                ('2023-01-01', '2023-06-11')
            ],
            '미라시아_콥 샐러드': [
                ('2023-01-01', '2023-12-07')
            ],
            '미라시아_파스타면 추가(150g)': [
                ('2023-01-01', '2023-06-02')
            ],
            '미라시아_핑크레몬에이드': [
                ('2023-01-01', '2023-03-16'),
                ('2024-06-11', '2024-09-15'),
                ('2025-03-28', '2025-05-24')
            ],
            '화담숲주막_느린마을 막걸리': [
                ('2023-01-01', '2023-03-30'),
                ('2023-11-27', '2024-03-28'),
                ('2024-11-27', '2025-03-27')
            ],
            '화담숲주막_단호박 식혜 ': [
                ('2023-01-01', '2023-03-30'),
                ('2023-11-27', '2024-03-28'),
                ('2024-11-27', '2025-03-27')
            ],
            '화담숲주막_병천순대': [
                ('2023-01-01', '2023-03-30'),
                ('2023-11-27', '2024-03-28'),
                ('2024-11-27', '2025-03-27')
            ],
            '화담숲주막_스프라이트': [
                ('2023-01-01', '2023-03-30'),
                ('2023-11-27', '2024-03-28'),
                ('2024-11-25', '2025-03-27')
            ],
            '화담숲주막_참살이 막걸리': [
                ('2023-01-01', '2023-03-30'),
                ('2023-11-27', '2024-03-28'),
                ('2024-11-27', '2025-03-27')
            ],
            '화담숲주막_찹쌀식혜': [
                ('2023-01-01', '2023-03-30'),
                ('2023-11-27', '2024-03-28'),
                ('2024-11-27', '2025-03-27')
            ],
            '화담숲주막_콜라': [
                ('2023-01-01', '2023-03-30'),
                ('2023-11-27', '2024-03-28'),
                ('2024-11-27', '2025-03-27')
            ],
            '화담숲주막_해물파전': [
                ('2023-01-01', '2023-03-30'),
                ('2023-11-27', '2024-03-28'),
                ('2024-11-27', '2025-03-27')
            ],
            '연회장_Cass Beer': [
                ('2023-07-11', '2023-08-17'),
                ('2024-02-24', '2024-03-29'),
                ('2024-07-26', '2024-08-26'),
                ('2024-09-13', '2024-10-13'),
                ('2025-02-27', '2025-03-26')
            ],
            '연회장_Conference L1': [
                ('2024-09-07', '2024-10-07')
            ],
            '연회장_Conference L2': [
                ('2024-12-17', '2025-01-14')
            ],
            '연회장_Conference L3': [
                ('2024-07-06', '2024-08-24'),
                ('2024-09-10', '2024-10-07'),
                ('2024-12-15', '2025-01-12'),
                ('2025-02-25', '2025-03-26')
            ],
            '연회장_Conference M8': [
                ('2023-07-20', '2023-08-17'),
                ('2023-12-23', '2024-01-31'),
                ('2024-09-12', '2024-10-10')
            ],
            '연회장_Conference M9': [
                ('2024-09-07', '2024-10-06'),
                ('2024-12-29', '2025-03-12')
            ],
            '연회장_Convention Hall': [
                ('2024-09-13', '2024-10-10')
            ],
            '연회장_Cookie Platter': [
                ('2025-04-18', '2025-05-24')
            ],
            '연회장_OPUS 2': [
                ('2023-01-29', '2023-02-26'),
                ('2025-01-23', '2025-02-19')
            ],
            '연회장_Regular Coffee': [
                ('2023-01-01', '2023-02-23'),
                ('2023-06-09', '2023-07-06'),
                ('2023-07-27', '2023-08-29')
            ],
            '연회장_공깃밥': [
                ('2023-01-01', '2023-07-20')
            ],
            '연회장_마라샹궈': [
                ('2023-01-01', '2023-09-07')
            ],
            '연회장_삼겹살추가 (200g)': [
                ('2023-01-01', '2023-07-20')
            ],
            '연회장_왕갈비치킨': [
                ('2023-01-01', '2023-07-21')
            ],
            '라그로타_AUS (200g)': [
                ('2023-01-01', '2023-12-07')
            ],
            '라그로타_Open Food': [
                ('2023-06-03', '2023-07-05'),
                ('2023-08-23', '2023-09-25')
            ],
            '라그로타_그릴드 비프 샐러드': [
                ('2023-01-01', '2023-09-07')
            ],
            '라그로타_까르보나라': [
                ('2023-01-01', '2023-12-07'),
                ('2024-07-29', '2024-12-09')
            ],
            '라그로타_모둠 해산물 플래터': [
                ('2023-01-01', '2023-09-08'),
                ('2024-02-28', '2024-03-28')
            ],
            '라그로타_버섯 크림 리조또': [
                ('2023-01-01', '2023-12-07')
            ],
            '라그로타_시저 샐러드 ': [
                ('2023-01-01', '2023-09-07')
            ],
            '라그로타_아메리카노': [
                ('2025-02-26', '2025-03-25')
            ],
            '라그로타_알리오 에 올리오 ': [
                ('2023-01-01', '2023-09-07')
            ],
            '라그로타_양갈비 (4ps)': [
                ('2023-01-01', '2023-09-09'),
                ('2023-11-05', '2023-12-08')
            ],
            '라그로타_카스': [
                ('2025-02-27', '2025-03-27')
            ],
            '라그로타_한우 (200g)': [
                ('2023-01-01', '2023-12-08')
            ],
            '라그로타_해산물 토마토 리조또': [
                ('2024-03-29', '2024-06-14')
            ],
            '라그로타_해산물 토마토 스튜 파스타': [
                ('2023-01-01', '2023-12-07')
            ],
            '화담숲카페_메밀미숫가루': [
                ('2023-01-01', '2023-03-30'),
                ('2023-11-27', '2024-03-28'),
                ('2024-11-27', '2025-03-27')
            ],
            '화담숲카페_아메리카노 HOT': [
                ('2023-01-01', '2023-03-30'),
                ('2023-11-27', '2024-03-28'),
                ('2024-11-27', '2025-03-27')
            ],
            '화담숲카페_아메리카노 ICE': [
                ('2023-01-01', '2023-03-30'),
                ('2023-11-27', '2024-03-28'),
                ('2024-11-27', '2025-03-27')
            ],
            '화담숲카페_카페라떼 ICE': [
                ('2023-01-01', '2023-03-30'),
                ('2023-11-27', '2024-03-28'),
                ('2024-11-27', '2025-03-27')
            ],
            '화담숲카페_현미뻥스크림': [
                ('2023-01-01', '2023-03-30'),
                ('2023-11-27', '2024-03-28'),
                ('2024-11-25', '2025-03-27')
            ],
            '카페테리아_구슬아이스크림': [
                ('2023-04-25', '2023-12-05'),
                ('2024-03-04', '2024-12-11'),
                ('2025-02-25', '2025-05-24')
            ],
            '카페테리아_단체식 13000(신)': [
                ('2023-01-01', '2023-04-17')
            ],
            '카페테리아_단체식 18000(신)': [
                ('2023-01-01', '2023-04-04')
            ],
            '카페테리아_복숭아 아이스티': [
                ('2023-03-02', '2023-12-06'),
                ('2024-03-04', '2024-07-24'),
                ('2024-08-02', '2024-12-11'),
                ('2025-03-03', '2025-05-24')
            ],
            '카페테리아_새우튀김 우동': [
                ('2023-03-02', '2023-12-06'),
                ('2024-03-04', '2025-01-16'),
                ('2025-03-04', '2025-05-24')
            ],
            '카페테리아_샷 추가': [
                ('2023-03-01', '2023-12-08'),
                ('2024-03-03', '2024-12-14'),
                ('2025-03-04', '2025-05-24')
            ],
            '카페테리아_아메리카노(HOT)': [
                ('2023-03-02', '2023-12-05'),
                ('2024-03-04', '2024-07-28'),
                ('2024-07-30', '2024-09-12'),
                ('2024-09-14', '2024-10-15'),
                ('2024-11-09', '2024-12-11'),
                ('2025-03-04', '2025-05-24')
            ],
            '카페테리아_아메리카노(ICE)': [
                ('2023-03-02', '2023-12-05'),
                ('2024-03-04', '2024-07-24'),
                ('2024-11-09', '2024-12-11'),
                ('2025-03-04', '2025-05-24')
            ],
            '카페테리아_오픈푸드': [
                ('2023-03-02', '2023-04-27'),
                ('2023-04-29', '2023-06-10'),
                ('2023-07-22', '2023-09-17'),
                ('2024-04-09', '2024-07-01'),
                ('2024-07-13', '2024-08-26'),
                ('2024-09-04', '2024-10-22'),
                ('2025-03-03', '2025-04-09')
            ],
            '카페테리아_진사골 설렁탕': [
                ('2023-01-01', '2023-12-05'),
                ('2024-03-04', '2024-12-07'),
                ('2025-03-04', '2025-05-24')
            ],
            '카페테리아_카페라떼(HOT)': [
                ('2023-03-02', '2023-12-05'),
                ('2024-03-04', '2024-07-28'),
                ('2024-07-30', '2024-10-07'),
                ('2024-10-24', '2024-12-12'),
                ('2025-03-04', '2025-05-24')
            ],
            '카페테리아_카페라떼(ICE)': [
                ('2023-03-02', '2023-12-06'),
                ('2024-03-04', '2024-07-24'),
                ('2024-09-14', '2024-10-23'),
                ('2024-10-25', '2024-11-27'),
                ('2025-03-04', '2025-05-24')
            ],
            '카페테리아_한상 삼겹구이 정식(2인) 소요시간 약 15~20분': [
                ('2023-01-01', '2023-03-16'),
                ('2023-12-06', '2024-03-28'),
                ('2024-12-01', '2025-03-27')
            ],
            '담하_(단체) 공깃밥': [
                ('2023-01-01', '2023-03-12'),
                ('2023-04-28', '2023-05-30'),
                ('2024-07-12', '2024-08-12')
            ],
            '담하_(단체) 생목살 김치전골 2.0': [
                ('2023-01-01', '2023-09-17'),
                ('2024-12-17', '2025-01-13')
            ],
            '담하_(단체) 은이버섯 갈비탕': [
                ('2023-01-01', '2023-06-11'),
                ('2024-09-13', '2024-10-13')
            ],
            '담하_(단체) 한우 우거지 국밥': [
                ('2024-09-08', '2024-10-08')
            ],
            '담하_(정식) 된장찌개': [
                ('2023-01-01', '2023-06-02')
            ],
            '담하_(정식) 물냉면 ': [
                ('2023-01-01', '2023-06-02')
            ],
            '담하_(정식) 비빔냉면': [
                ('2023-01-01', '2023-06-02')
            ],
            '담하_(후식) 물냉면': [
                ('2023-01-01', '2023-06-01')
            ],
            '담하_(후식) 비빔냉면': [
                ('2023-01-01', '2023-06-01')
            ],
            '담하_갑오징어 비빔밥': [
                ('2023-01-01', '2023-03-16'),
                ('2023-09-08', '2024-03-29'),
                ('2024-09-08', '2025-03-27'),
                ('2025-03-31', '2025-05-24')
            ],
            '담하_갱시기': [
                ('2023-01-01', '2023-12-07'),
                ('2024-06-13', '2024-09-13')
            ],
            '담하_꼬막 비빔밥': [
                ('2023-01-01', '2023-09-07'),
                ('2024-03-29', '2024-09-12'),
                ('2025-03-30', '2025-05-24')
            ],
            '담하_담하 한우 불고기 정식': [
                ('2023-01-01', '2023-06-01')
            ],
            '담하_더덕 한우 지짐': [
                ('2023-01-01', '2023-09-08')
            ],
            '담하_들깨 양지탕': [
                ('2024-03-29', '2025-03-27')
            ],
            '담하_라면사리': [
                ('2023-04-06', '2023-05-13')
            ],
            '담하_명인안동소주': [
                ('2023-01-01', '2023-06-30')
            ],
            '담하_명태회 비빔냉면': [
                ('2023-01-01', '2023-06-01')
            ],
            '담하_문막 복분자 칵테일': [
                ('2023-01-01', '2023-09-11'),
                ('2024-03-03', '2024-03-30')
            ],
            '담하_봉평메밀 물냉면': [
                ('2023-01-01', '2023-06-01')
            ],
            '담하_하동 매실 칵테일': [
                ('2023-01-01', '2023-03-17')
            ],
            '포레스트릿_꼬치어묵': [
                ('2023-06-01', '2023-09-08'),
                ('2024-06-10', '2024-09-13')
            ],
            '포레스트릿_복숭아 아이스티': [
                ('2023-03-02', '2023-03-31'),
                ('2024-03-04', '2024-12-12'),
                ('2025-03-04', '2025-05-24')
            ],
            '포레스트릿_생수': [
                ('2023-03-02', '2023-03-31')
            ],
            '포레스트릿_스프라이트': [
                ('2023-03-02', '2023-03-31'),
                ('2025-03-04', '2025-04-05')
            ],
            '포레스트릿_아메리카노(HOT)': [
                ('2023-03-02', '2023-03-31')
            ],
            '포레스트릿_카페라떼(HOT)': [
                ('2023-03-02', '2023-03-31'),
                ('2024-07-07', '2024-08-31'),
                ('2025-03-04', '2025-04-05')
            ],
            '포레스트릿_카페라떼(ICE)': [
                ('2023-03-02', '2023-03-31'),
                ('2025-03-04', '2025-04-05')
            ],
            '포레스트릿_페스츄리 소시지': [
                ('2023-03-02', '2023-03-31')
            ]
        }

        # 매장별 정기 휴무일 (모든 메뉴에 적용)
        self.store_closures = {
            '화담숲카페': [0],  # 월요일 휴무
            '화담숲주막': [0],  # 월요일 휴무
            '라그로타': [0],  # 월요일 휴무
        }

        # 메뉴별 정기 휴무일 (특정 메뉴만 적용)
        self.menu_closures = {
            # 연회장 특정 메뉴들 - 토일 휴무
            '연회장_Regular Coffee': [5, 6],
            '연회장_OPUS 2': [5, 6],
            '연회장_Cookie Platter': [5, 6],
            '연회장_Convention Hall': [5, 6],
            '연회장_Conference L1': [5, 6],
            '연회장_Conference L2': [5, 6],
            '연회장_Conference L3': [5, 6],
            '연회장_Conference M8': [5, 6],
            '연회장_Conference M9': [5, 6],
            '연회장_Cass Beer': [5, 6],

            # 미라시아 브런치 메뉴들
            '미라시아_브런치(대인) 주중': [5, 6],  # 토일 휴무
            '미라시아_브런치 4인 패키지 ': [5, 6],  # 토일 휴무 (단체 브런치 주중)
            '미라시아_브런치(대인) 주말': [0, 1, 2, 3, 4],  # 월화수목금 휴무

            # 담하 단체 메뉴들 - 일요일 휴무
            '담하_(단체) 황태 해장국': [6],
            '담하_(단체) 한우 우거지 국밥': [6],
            '담하_(단체) 은이버섯 갈비탕': [6],
            '담하_(단체) 생목살 김치전골 2.0': [6],

            # 느티나무 셀프BBQ 단체 메뉴들 - 일요일 휴무
            '느티나무 셀프BBQ_BBQ55(단체)': [6],
            '느티나무 셀프BBQ_콜라(단체)': [6],
            '느티나무 셀프BBQ_카스병(단체)': [6],
            '느티나무 셀프BBQ_참이슬(단체)': [6],
            '느티나무 셀프BBQ_스프라이트(단체)': [6],
        }

        # 날짜 문자열을 datetime 객체로 변환
        self._convert_dates_to_datetime()

    def _convert_dates_to_datetime(self):
        """날짜 문자열을 datetime 객체로 변환"""
        # 매장별 중단 기간 변환
        for store in self.store_suspensions:
            self.store_suspensions[store] = [
                (pd.to_datetime(start), pd.to_datetime(end))
                for start, end in self.store_suspensions[store]
            ]

        # 메뉴별 중단 기간 변환
        for menu in self.menu_suspensions:
            self.menu_suspensions[menu] = [
                (pd.to_datetime(start), pd.to_datetime(end))
                for start, end in self.menu_suspensions[menu]
            ]

    def _extract_store_name(self, store_menu: str) -> str:
        """store_menu에서 매장명 추출"""
        if '_' in store_menu:
            return store_menu.split('_')[0]
        return store_menu

    def _is_date_in_suspension_periods(self, date: pd.Timestamp, periods) -> bool:
        """주어진 날짜가 중단 기간 내에 있는지 확인"""
        for start_date, end_date in periods:
            if start_date <= date <= end_date:
                return True
        return False

    def _is_suspension_period(self, date: pd.Timestamp, store_menu: str) -> bool:
        """주어진 날짜가 영업 중단 기간인지 확인"""
        store_name = self._extract_store_name(store_menu)

        # 1. 매장 전체 중단 기간 확인
        if store_name in self.store_suspensions:
            if self._is_date_in_suspension_periods(date, self.store_suspensions[store_name]):
                return True

        # 2. 메뉴별 중단 기간 확인
        if store_menu in self.menu_suspensions:
            if self._is_date_in_suspension_periods(date, self.menu_suspensions[store_menu]):
                return True

        return False

    def _is_regular_closure(self, date: pd.Timestamp, store_menu: str) -> bool:
        """주어진 날짜가 해당 매장/메뉴의 정기 휴무일인지 확인"""
        weekday = date.weekday()  # 0: 월요일, 6: 일요일

        # 1. 메뉴별 휴무일 확인 (우선순위 높음)
        if store_menu in self.menu_closures:
            if weekday in self.menu_closures[store_menu]:
                return True

        # 2. 매장별 휴무일 확인
        store_name = self._extract_store_name(store_menu)
        if store_name in self.store_closures:
            if weekday in self.store_closures[store_name]:
                return True

        return False

    def process(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """is_sales 피처를 추가합니다."""

        print("영업 중단 기간 및 정기 휴무일 기반 is_sales 피처 생성 중...")

        processed_data = data.copy()

        # 날짜 컬럼 확인 및 변환
        date_col = 'timestamp'
        if date_col not in processed_data.columns:
            raise ValueError(f"날짜 컬럼 '{date_col}'이 데이터에 없습니다.")

        # 날짜 컬럼이 datetime이 아니면 변환
        if processed_data[date_col].dtype != 'datetime64[ns]':
            if processed_data[date_col].dtype in ['int64', 'float64']:
                # days_since_reference 형태인 경우
                reference_date = pd.to_datetime('2023-01-01')
                processed_data[date_col] = reference_date + pd.to_timedelta(processed_data[date_col], unit='D')
            else:
                processed_data[date_col] = pd.to_datetime(processed_data[date_col])

        # store_menu 컬럼 확인
        store_menu_col = 'store_menu'
        if store_menu_col not in processed_data.columns:
            raise ValueError(f"매장_메뉴 컬럼 '{store_menu_col}'이 데이터에 없습니다.")

        # is_sales 피처 초기화 (기본값: 1 - 판매 가능)
        processed_data['is_sales'] = 1

        non_sales_count = 0
        total_rows = len(processed_data)

        # 각 행에 대해 중단/휴무 기간 확인
        for idx, row in processed_data.iterrows():
            date = row[date_col]
            store_menu = row[store_menu_col]

            is_non_sales = False

            # 1. 영업 중단 기간 확인
            if self._is_suspension_period(date, store_menu):
                is_non_sales = True

            # 2. 정기 휴무일 확인 (중단 기간이 아닌 경우에만)
            if not is_non_sales and self._is_regular_closure(date, store_menu):
                is_non_sales = True

            # 중단/휴무 기간인 경우 is_sales를 0으로 설정
            if is_non_sales:
                processed_data.at[idx, 'is_sales'] = 0
                non_sales_count += 1

        print(f"    처리 완료:")
        print(f"      - 전체 데이터: {total_rows:,}행")
        print(f"      - 중단/휴무 데이터: {non_sales_count:,}행 (is_sales=0)")
        print(f"      - 정상 영업 데이터: {total_rows - non_sales_count:,}행 (is_sales=1)")
        print(f"      - 중단/휴무 비율: {non_sales_count / total_rows * 100:.2f}%")

        return processed_data

    def get_name(self) -> str:
        return "is_sales"


class HistoricalSeasonAvgFeatureProcessor(FeatureProcessor):
    """작년 동일 시기 평균 판매량 특성 생성 처리기"""

    def __init__(self, window_days: int = 14):
        """
        Args:
            window_days: 평균 계산을 위한 윈도우 크기 (일)
        """
        self.window_days = window_days

    def process(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """historical_season_avg 피처를 추가합니다."""

        print(f"작년 동일 시기 평균 판매량 피처 생성 중 (윈도우: {self.window_days}일)...")

        processed_data = data.copy()

        # 날짜 컬럼 확인
        date_col = 'timestamp'
        if date_col not in processed_data.columns:
            raise ValueError(f"날짜 컬럼 '{date_col}'이 데이터에 없습니다.")

        # store_menu 컬럼 확인
        store_menu_col = 'store_menu'
        if store_menu_col not in processed_data.columns:
            raise ValueError(f"매장_메뉴 컬럼 '{store_menu_col}'이 데이터에 없습니다.")

        # sales 컬럼 확인
        sales_col = 'sales'
        if sales_col not in processed_data.columns:
            raise ValueError(f"매출 컬럼 '{sales_col}'이 데이터에 없습니다.")

        # 날짜 컬럼이 datetime이 아니면 변환
        if processed_data[date_col].dtype != 'datetime64[ns]':
            if processed_data[date_col].dtype in ['int64', 'float64']:
                # days_since_reference 형태인 경우
                reference_date = pd.to_datetime('2023-01-01')
                processed_data[date_col] = reference_date + pd.to_timedelta(processed_data[date_col], unit='D')
            else:
                processed_data[date_col] = pd.to_datetime(processed_data[date_col])

        # historical_season_avg 피처 초기화 (NaN으로 초기화)
        processed_data['historical_season_avg'] = np.nan

        # 각 매장_메뉴별로 처리
        total_calculated = 0
        total_rows = len(processed_data)

        for store_menu in processed_data[store_menu_col].unique():
            # 해당 매장_메뉴 데이터 추출
            menu_data = processed_data[processed_data[store_menu_col] == store_menu].copy()
            menu_data = menu_data.sort_values(date_col)

            # 각 날짜에 대해 작년 동일 시기 평균 계산
            for idx, row in menu_data.iterrows():
                current_date = row[date_col]

                # 작년 동일 시기 계산 (1년 전)
                last_year_date = current_date - pd.DateOffset(years=1)

                # 작년 동일 시기 윈도우 범위 계산
                window_start = last_year_date - pd.Timedelta(days=self.window_days // 2)
                window_end = last_year_date + pd.Timedelta(days=self.window_days // 2)

                # 작년 동일 시기 데이터 추출
                historical_data = menu_data[
                    (menu_data[date_col] >= window_start) &
                    (menu_data[date_col] <= window_end)
                    ]

                # 평균 계산
                if len(historical_data) > 0:
                    avg_sales = historical_data[sales_col].mean()
                    processed_data.at[idx, 'historical_season_avg'] = avg_sales
                    total_calculated += 1
                # 작년 데이터가 없는 경우 NaN으로 유지 (나중에 채움)

        # 각 매장_메뉴별로 결측값 채우기 (bfill -> ffill -> 전체 평균)
        for store_menu in processed_data[store_menu_col].unique():
            mask = processed_data[store_menu_col] == store_menu
            menu_series = processed_data.loc[mask, 'historical_season_avg']

            # 1. 먼저 backward fill (미래 값으로 채우기)
            menu_series = menu_series.bfill()

            # 2. 그 다음 forward fill (과거 값으로 채우기)
            menu_series = menu_series.ffill()

            # 3. 여전히 NaN인 경우 해당 메뉴의 전체 평균으로 채우기
            if menu_series.isna().any():
                menu_avg = processed_data.loc[
                    mask & processed_data['historical_season_avg'].notna(), 'historical_season_avg'].mean()
                if not np.isnan(menu_avg):
                    menu_series = menu_series.fillna(menu_avg)
                else:
                    # 해당 메뉴에 데이터가 전혀 없는 경우 전체 평균 사용
                    global_avg = processed_data['historical_season_avg'].mean()
                    if not np.isnan(global_avg):
                        menu_series = menu_series.fillna(global_avg)
                    else:
                        # 전체 데이터에도 값이 없는 경우 0으로 채우기
                        menu_series = menu_series.fillna(0.0)

            processed_data.loc[mask, 'historical_season_avg'] = menu_series

        # 통계 출력
        non_zero_count = (processed_data['historical_season_avg'] > 0).sum()
        avg_value = processed_data['historical_season_avg'].mean()

        print(f"    처리 완료:")
        print(f"      - 전체 데이터: {total_rows:,}행")
        print(f"      - 계산된 데이터: {total_calculated:,}행")
        print(f"      - 0이 아닌 값: {non_zero_count:,}행 ({non_zero_count / total_rows * 100:.2f}%)")
        print(f"      - 평균값: {avg_value:.2f}")

        return processed_data

    def get_name(self) -> str:
        return "historical_season_avg"


class TimestampToNumericConverter(FeatureProcessor):
    """타임스탬프 숫자 변환 처리기"""

    def __init__(self, reference_date: str = '2023-01-01'):
        self.reference_date = reference_date

    def process(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        reference = pd.to_datetime(self.reference_date)
        data['days_since_reference'] = (data['timestamp'] - reference).dt.days
        data.drop(columns=['timestamp'], inplace=True)
        return data

    def get_name(self) -> str:
        return "timestamp_to_numeric"


class ZeroStreakFeatureCreator(FeatureProcessor):
    """연속 0 특성 생성 처리기"""

    def process(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        def calculate_rolling_features(group):
            group = group.sort_values('days_since_reference').copy()
            sales = group['sales']

            # 최근 7일간 0 개수
            rolling_7d = sales.rolling(window=7, min_periods=1)
            zero_count_7d = rolling_7d.apply(lambda x: (x == 0).sum(), raw=False).astype(int)

            # 최근 14일간 0 개수
            rolling_14d = sales.rolling(window=14, min_periods=1)
            zero_count_14d = rolling_14d.apply(lambda x: (x == 0).sum(), raw=False).astype(int)

            # 최근 21일간 0 개수
            rolling_21d = sales.rolling(window=21, min_periods=1)
            zero_count_21d = rolling_21d.apply(lambda x: (x == 0).sum(), raw=False).astype(int)

            # 연속 0 판매일 수
            is_zero = (sales == 0).astype(int)
            consecutive_zeros = is_zero.groupby((is_zero != is_zero.shift()).cumsum()).cumcount() + 1
            consecutive_zeros = consecutive_zeros * is_zero

            # 지난 7일 중 0 판매 비율
            zero_ratio_7d = (zero_count_7d / rolling_7d.count().clip(lower=1)).fillna(0)

            # 지난 14일 중 0 판매 비율
            zero_ratio_14d = (zero_count_14d / rolling_14d.count().clip(lower=1)).fillna(0)

            # 특성을 그룹에 추가
            group['zero_count_7d'] = zero_count_7d
            group['zero_count_14d'] = zero_count_14d
            group['zero_count_21d'] = zero_count_21d
            group['consecutive_zeros'] = consecutive_zeros
            group['zero_ratio_7d'] = zero_ratio_7d
            group['zero_ratio_14d'] = zero_ratio_14d

            return group

        data = data.groupby('store_menu').apply(calculate_rolling_features)
        data = data.reset_index(drop=True)
        return data

    def get_name(self) -> str:
        return "zero_streak"


class OutlierClipper(FeatureProcessor):
    """이상치 IQR + 4.5* 대체 처리기"""

    def __init__(self):
        self.iqr_params = {}  # Store IQR parameters from training data
        self.is_fitted = False

    def process(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        # Check if this is training data (fit=True) or test data (fit=False)
        fit = kwargs.get('fit', True)

        if fit:
            print("🔧 이상치 IQR + 4.5* 대체 처리 중 (훈련 데이터)...")
            return self._fit_and_transform(data)
        else:
            print("🔧 이상치 IQR + 4.5* 대체 처리 중 (테스트 데이터)...")
            return self._transform_only(data)

    def _fit_and_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """훈련 데이터에서 IQR 파라미터를 계산하고 이상치를 대체"""
        processed_data = data.copy()
        total_replaced = 0

        # 2. 각 store_menu별로 IQR 파라미터 계산 및 이상치 대체
        outlier_replaced = 0
        self.iqr_params = {}

        def replace_outliers_per_group(group):
            nonlocal outlier_replaced
            store_menu = group['store_menu'].iloc[0]

            # IQR 파라미터 계산 (훈련 데이터에서만, 0 제외)
            non_zero_sales = group['sales'][group['sales'] > 0]
            if len(non_zero_sales) == 0:
                # 모든 값이 0인 경우, 기본값 사용
                Q1, Q3, IQR = 0, 0, 0
            else:
                Q1 = non_zero_sales.quantile(0.25)
                Q3 = non_zero_sales.quantile(0.75)
                IQR = Q3 - Q1

            if IQR == 0:
                # IQR이 0이면 이상치 없음으로 처리 - 모든 값이 동일하므로 이상치가 존재할 수 없음
                # 빈 마스크를 생성하여 어떤 값도 이상치로 처리하지 않음
                outlier_mask = pd.Series([False] * len(group), index=group.index)
                group_replaced = 0
                outlier_replaced += group_replaced
                return group

            # IQR 파라미터 저장 (테스트 데이터에서 사용)
            self.iqr_params[store_menu] = {
                'Q1': Q1,
                'Q3': Q3,
                'IQR': IQR,
                'replacement_value': Q3 + 4.5 * IQR
            }

            # 이상치 탐지 (표준 IQR 방법: Q1 - 1.5*IQR, Q3 + 1.5*IQR)
            lower_bound = Q1 - 4.5 * IQR
            upper_bound = Q3 + 4.5 * IQR

            # 이상치 마스크
            outlier_mask = (group['sales'] > upper_bound) & (group['sales'] > 0)

            # 이상치를 IQR + 4.5* 값으로 대체
            replacement_value = Q3 + 4.5 * IQR
            original_sales = group['sales'].copy()

            group_replaced = outlier_mask.sum()
            outlier_replaced += group_replaced

            if group_replaced > 0:
                print(f"    학습 데이터 IQR 통계: Q1={Q1:.1f}, Q3={Q3:.1f}, IQR={IQR:.1f}")
                print(f"    이상치 범위: [{lower_bound:.1f}, {upper_bound:.1f}] 밖의 값")
                print(f"    전체 데이터에서 탐지된 이상치: {group_replaced}개")
                print(f"    {store_menu}: {group_replaced}개 이상치를 {replacement_value:.2f}로 대체")

            return group

        processed_data = processed_data.groupby('store_menu', group_keys=False).apply(replace_outliers_per_group)
        processed_data = processed_data.reset_index(drop=True)

        total_replaced += outlier_replaced
        total_elements = len(data)
        replacement_rate = (total_replaced / total_elements) * 100

        print(f"    이상치 대체 완료:")
        print(f"      - 전체 데이터: {total_elements:,}개")
        print(f"      - 이상치 대체: {outlier_replaced:,}개")
        print(f"      - 총 대체: {total_replaced:,}개 ({replacement_rate:.2f}%)")
        print(f"      - IQR 파라미터 저장: {len(self.iqr_params)}개 store_menu")

        self.is_fitted = True
        return processed_data

    def _transform_only(self, data: pd.DataFrame) -> pd.DataFrame:
        """테스트 데이터에 훈련 데이터의 IQR 파라미터를 적용하여 이상치 대체"""
        if not self.is_fitted:
            raise ValueError("OutlierClipper가 훈련 데이터로 fit되지 않았습니다. 먼저 fit=True로 훈련 데이터를 처리하세요.")

        processed_data = data.copy()
        total_replaced = 0

        # 훈련 데이터의 IQR 파라미터를 사용하여 이상치 대체
        outlier_replaced = 0

        def replace_outliers_per_group(group):
            nonlocal outlier_replaced
            store_menu = group['store_menu'].iloc[0]

            # 훈련 데이터에서 계산된 IQR 파라미터 사용
            if store_menu not in self.iqr_params:
                print(f"   ️ {store_menu}의 IQR 파라미터가 없습니다. 건너뜁니다.")
                return group

            params = self.iqr_params[store_menu]
            Q1 = params['Q1']
            Q3 = params['Q3']
            IQR = params['IQR']
            replacement_value = params['replacement_value']

            # 이상치 탐지 (훈련 데이터와 동일한 기준)
            lower_bound = Q1 - 4.5 * IQR
            upper_bound = Q3 + 4.5 * IQR

            # 이상치 마스크
            outlier_mask = (group['sales'] < lower_bound) | (group['sales'] > upper_bound)

            # 이상치를 훈련 데이터에서 계산된 IQR + 4.5* 값으로 대체
            group.loc[outlier_mask, 'sales'] = replacement_value

            group_replaced = outlier_mask.sum()
            outlier_replaced += group_replaced

            if group_replaced > 0:
                print(f"   📊 {store_menu}: {group_replaced}개 이상치를 {replacement_value:.2f}로 대체 (훈련 IQR 사용)")

            return group

        processed_data = processed_data.groupby('store_menu', group_keys=False).apply(replace_outliers_per_group)
        processed_data = processed_data.reset_index(drop=True)

        total_replaced += outlier_replaced
        total_elements = len(data)
        replacement_rate = (total_replaced / total_elements) * 100

        print(f"   📊 이상치 대체 완료 (테스트 데이터):")
        print(f"      - 전체 데이터: {total_elements:,}개")
        print(f"      - 이상치 대체: {outlier_replaced:,}개")
        print(f"      - 총 대체: {total_replaced:,}개 ({replacement_rate:.2f}%)")

        return processed_data

    def get_name(self) -> str:
        return "outlier_iqr_replacement"
