from typing import Optional, Tuple, List, Dict

import numpy as np
import torch
import pandas as pd
from datetime import datetime, timedelta

from core import PostProcessor
from core.utils.Logger import get_logger, LogLevel


class WeightedSMAPEMetric(torch.nn.Module):
    def __init__(self, weights=None, device='cpu', post_processor: Optional[PostProcessor] = None,
                 start_date: str = "2023-01-01"):
        """
        weights: [num_stores] shape, 각 매장별 가중치 텐서 혹은 리스트
        post_processor: 후처리 후 메트릭 값이 얼마나 달라지는 확인하기 위한 객체
        start_date: 데이터 시작 날짜 (날짜 계산용)
        """
        super().__init__()
        if weights is None:
            weights = [1.0 / 9.0] * 9

        # 일반 텐서 변수로 생성하여 디바이스 지정
        self.weights = torch.tensor(weights, device=device).float()
        self.device = device
        self.post_processor = post_processor
        self.logger = get_logger()
        self.start_date = pd.to_datetime(start_date)

        # 메뉴 ID와 매장_메뉴명 매핑 테이블 생성
        self.menu_id_to_name = self._create_menu_mapping()

        # 후처리되지 않은 SMAPE 계산용
        self.smape_list = torch.zeros(193, dtype=torch.float32, device=device)  # 각 메뉴별 smape 저장 리스트(193)
        self.validated_date_per_menu = torch.zeros(193, dtype=torch.float32, device=device)  # 각 메뉴별 유효 날짜 개수 리스트
        self.store_smape = torch.zeros(9, device=device)  # 매장별 SMAPE 계산을 위한 메뉴-매장 매핑이 필요

        # 후처리된 SMAPE 계산용
        self.smape_list_postprocessed = torch.zeros(193, dtype=torch.float32, device=device)
        self.validated_date_per_menu_postprocessed = torch.zeros(193, dtype=torch.float32, device=device)
        self.store_smape_postprocessed = torch.zeros(9, device=device)

        # 🔍 SMAPE 분석용 데이터 저장
        self.worst_smape_details = []  # 가장 안좋은 SMAPE 값들의 상세 정보 저장
        self.seen_data_points = set()  # 중복 방지용 집합 추가 ✨

    def set_weights(self, weights):
        self.weights = torch.tensor(weights, device=self.device).float()

    def clear_weights(self):
        weights = [1.0 / 9.0] * 9
        self.set_weights(weights)

    def _create_menu_mapping(self):
        """메뉴 ID와 영업장_메뉴명 매핑 테이블 생성"""
        menu_mapping = {}

        # 매장별 메뉴 정보
        store_menus = {
            "느티나무 셀프BBQ": [
                "1인 수저세트", "BBQ55(단체)", "대여료 30,000원", "대여료 60,000원", "대여료 90,000원",
                "본삼겹 (단품,실내)", "스프라이트 (단체)", "신라면", "쌈야채세트", "쌈장",
                "육개장 사발면", "일회용 소주컵", "일회용 종이컵", "잔디그늘집 대여료 (12인석)", "잔디그늘집 대여료 (6인석)",
                "잔디그늘집 의자 추가", "참이슬 (단체)", "친환경 접시 14cm", "친환경 접시 23cm", "카스 병(단체)",
                "콜라 (단체)", "햇반", "허브솔트"
            ],
            "담하": [
                "(단체) 공깃밥", "(단체) 생목살 김치전골 2.0", "(단체) 은이버섯 갈비탕", "(단체) 한우 우거지 국밥", "(단체) 황태해장국 3/27까지",
                "(정식) 된장찌개", "(정식) 물냉면 ", "(정식) 비빔냉면", "(후식) 된장찌개", "(후식) 물냉면",
                "(후식) 비빔냉면", "갑오징어 비빔밥", "갱시기", "공깃밥", "꼬막 비빔밥",
                "느린마을 막걸리", "담하 한우 불고기", "담하 한우 불고기 정식", "더덕 한우 지짐", "들깨 양지탕",
                "라면사리", "룸 이용료", "메밀면 사리", "명인안동소주", "명태회 비빔냉면",
                "문막 복분자 칵테일", "봉평메밀 물냉면", "생목살 김치찌개", "스프라이트", "은이버섯 갈비탕",
                "제로콜라", "참이슬", "처음처럼", "카스", "콜라",
                "테라", "하동 매실 칵테일", "한우 떡갈비 정식", "한우 미역국 정식", "한우 우거지 국밥",
                "한우 차돌박이 된장찌개", "황태해장국"
            ],
            "라그로타": [
                "AUS (200g)", "G-Charge(3)", "Gls.Sileni", "Gls.미션 서드", "Open Food",
                "그릴드 비프 샐러드", "까르보나라", "모둠 해산물 플래터", "미션 서드 카베르네 쉬라", "버섯 크림 리조또",
                "빵 추가 (1인)", "스프라이트", "시저 샐러드 ", "아메리카노", "알리오 에 올리오 ",
                "양갈비 (4ps)", "자몽리치에이드", "제로콜라", "카스", "콜라",
                "하이네켄(생)", "한우 (200g)", "해산물 토마토 리조또", "해산물 토마토 스튜 파스타", "해산물 토마토 스파게티"
            ],
            "미라시아": [
                "(단체)브런치주중 36,000", "(오븐) 하와이안 쉬림프 피자", "(화덕) 불고기 페퍼로니 반반피자", "BBQ Platter", "BBQ 고기추가",
                "공깃밥", "글라스와인 (레드)", "레인보우칵테일(알코올)", "미라시아 브런치 (패키지)", "버드와이저(무제한)",
                "보일링 랍스타 플래터", "보일링 랍스타 플래터(덜매운맛)", "브런치 2인 패키지 ", "브런치 4인 패키지 ", "브런치(대인) 주말",
                "브런치(대인) 주중", "브런치(어린이)", "쉬림프 투움바 파스타", "스텔라(무제한)", "스프라이트",
                "애플망고 에이드", "얼그레이 하이볼", "오븐구이 윙과 킬바사소세지", "유자 하이볼", "잭 애플 토닉",
                "칠리 치즈 프라이", "코카콜라", "코카콜라(제로)", "콥 샐러드", "파스타면 추가(150g)",
                "핑크레몬에이드"
            ],
            "연회장": [
                "Cass Beer", "Conference L1", "Conference L2", "Conference L3", "Conference M1",
                "Conference M8", "Conference M9", "Convention Hall", "Cookie Platter", "Grand Ballroom",
                "OPUS 2", "Regular Coffee", "골뱅이무침", "공깃밥", "돈목살 김치찌개 (밥포함)",
                "로제 치즈떡볶이", "마라샹궈", "매콤 무뼈닭발&계란찜", "모둠 돈육구이(3인)", "삼겹살추가 (200g)",
                "야채추가", "왕갈비치킨", "주먹밥 (2ea)"
            ],
            "카페테리아": [
                "공깃밥(추가)", "구슬아이스크림", "단체식 13000(신)", "단체식 18000(신)", "돼지고기 김치찌개",
                "복숭아 아이스티", "새우 볶음밥", "새우튀김 우동", "샷 추가", "수제 등심 돈까스",
                "아메리카노(HOT)", "아메리카노(ICE)", "약 고추장 돌솥비빔밥", "어린이 돈까스", "오픈푸드",
                "진사골 설렁탕", "짜장면", "짜장밥", "짬뽕", "짬뽕밥",
                "치즈돈까스", "카페라떼(HOT)", "카페라떼(ICE)", "한상 삼겹구이 정식(2인) 소요시간 약 15~20분"
            ],
            "포레스트릿": [
                "꼬치어묵", "떡볶이", "복숭아 아이스티", "생수", "스프라이트",
                "아메리카노(HOT)", "아메리카노(ICE)", "치즈 핫도그", "카페라떼(HOT)", "카페라떼(ICE)",
                "코카콜라", "페스츄리 소시지"
            ],
            "화담숲주막": [
                "느린마을 막걸리", "단호박 식혜 ", "병천순대", "스프라이트", "참살이 막걸리",
                "찹쌀식혜", "콜라", "해물파전"
            ],
            "화담숲카페": [
                "메밀미숫가루", "아메리카노 HOT", "아메리카노 ICE", "카페라떼 ICE", "현미뻥스크림"
            ]
        }

        menu_id = 0
        for store_name, menus in store_menus.items():
            for menu_name in menus:
                menu_mapping[menu_id] = f"{store_name}_{menu_name}"
                menu_id += 1

        return menu_mapping

    def _get_date_from_days_since_reference(self, days_since_reference: float) -> str:
        """start_date 기준 경과일수를 실제 날짜로 변환"""
        actual_date = self.start_date + timedelta(days=int(days_since_reference))
        return actual_date.strftime("%Y-%m-%d")

    def forward(self, prediction, targets, mask=None):
        """
        prediction: (batch, seq, feature) - feature[0]: salse, feature[2]: store_menu
        targets: (batch, seq, feature) - feature[0]: salse, feature[1]: days_since_reference, feature[2]: store_menu
        mask: (batch, seq) - 유효한 값만 산출할 때 사용 (optional)
        """

        # 판매량 추출 (feature의 0번 인덱스)
        sales_prediction = prediction[:, :, 0]  # [batch, seq]
        sales_target = targets[:, :, 0]  # [batch, seq]

        # 날짜 정보 추출 (feature의 1번 인덱스에서 경과일수)
        days_since_reference = targets[:, :, 1]  # [batch, seq]

        # 메뉴 인코딩 값 추출
        menu_encoding = targets[:, :, 2].long()  # [batch, seq]

        # 클리핑된 예측값 생성
        sales_prediction_postprocessed, _ = self.post_processor.postprocess(sales_prediction, menu_encoding)

        # 실제값이 0이 아닌 위치만 선택하는 마스크 생성
        if mask is None:
            mask = (sales_target != 0)  # 실제값이 0이 아닌 위치

        if mask.sum().item() == 0:
            return None

        # 각 품목(메뉴)별로 SMAPE 계산
        unique_menus = torch.unique(menu_encoding[mask])

        for menu_id in unique_menus:
            # 현재 메뉴에 해당하는 마스크
            menu_mask = (menu_encoding == menu_id) & mask  # [batch, seq]

            if menu_mask.sum().item() == 0:
                continue

            # 현재 메뉴의 예측값과 실제값 추출 (원본)
            menu_pred = sales_prediction[menu_mask]  # [valid_dates]
            menu_target = sales_target[menu_mask]  # [valid_dates]

            # 현재 메뉴의 클리핑된 예측값 추출
            menu_pred_postprocessed = sales_prediction_postprocessed[menu_mask]  # [valid_dates]
            menu_target_postprocessed = torch.expm1(sales_target[menu_mask])

            # 현재 메뉴의 날짜 정보 추출
            menu_days_since_reference = days_since_reference[menu_mask]  # [valid_dates]

            # 원본 SMAPE 계산: 2 * |pred - target| / (|pred| + |target|)
            numerator = torch.abs(menu_pred - menu_target)
            denominator = torch.abs(menu_pred) + torch.abs(menu_target)
            menu_smape = 2 * (numerator / denominator)

            # 🔍 각 개별 예측에 대한 SMAPE 값 저장 (분석용) - 중복 제거 적용 ✨
            self._collect_worst_smape_data(
                menu_id.item(),
                menu_pred,
                menu_target,
                menu_smape,
                menu_pred_postprocessed,
                menu_target_postprocessed,
                menu_days_since_reference
            )

            # 클리핑된 SMAPE 계산: 2 * |pred_postprocessed - target| / (|pred_postprocessed| + |target|)
            numerator_postprocessed = torch.abs(menu_pred_postprocessed - menu_target_postprocessed)
            denominator_postprocessed = torch.abs(menu_pred_postprocessed) + torch.abs(menu_target_postprocessed)
            menu_smape_postprocessed = 2 * (numerator_postprocessed / denominator_postprocessed)

            # 원본 품목별 SMAPE 누적 (평균이 아닌 합계로 저장)
            self.smape_list[menu_id] += menu_smape.sum()
            self.validated_date_per_menu[menu_id] += menu_mask.sum().item()

            # 클리핑된 품목별 SMAPE 누적
            self.smape_list_postprocessed[menu_id] += menu_smape_postprocessed.sum()
            self.validated_date_per_menu_postprocessed[menu_id] += menu_mask.sum().item()

    def _collect_worst_smape_data(self, menu_id: int, pred: torch.Tensor, target: torch.Tensor,
                                  smape: torch.Tensor, pred_postprocessed: torch.Tensor,
                                  target_postprocessed: torch.Tensor, days_since_reference: torch.Tensor):
        """
        🔍 SMAPE 분석을 위한 상세 데이터 수집 (후처리 된 값으로 SMAPE 계산) - 날짜 수정 ✨
        """
        # 후처리 된 값을 기준으로 SMAPE 재계산
        numerator_postprocessed = torch.abs(pred_postprocessed - target_postprocessed)
        denominator_postprocessed = torch.abs(pred_postprocessed) + torch.abs(target_postprocessed)
        smape_postprocessed = 2 * (numerator_postprocessed / denominator_postprocessed)

        # 🔍 days_since_reference의 차원 확인 및 처리 ✨
        if len(days_since_reference.shape) > 1:
            # [batch_size, seq_len] 형태인 경우 첫 번째 배치만 사용
            days_since_ref = days_since_reference[0]  # [seq_len]
        else:
            # [seq_len] 형태인 경우 그대로 사용
            days_since_ref = days_since_reference

        # 각 예측값에 대한 상세 정보 저장 (후처리 된 값 기준)
        for i in range(len(pred)):
            # 🚀 정확한 날짜 정보 처리 ✨
            date_str = "날짜정보없음"

            try:
                if i < len(days_since_ref):
                    # i번째 예측일의 날짜 정보 사용
                    days_value = days_since_ref[i].item()
                    date_str = self._get_date_from_days_since_reference(days_value)
                else:
                    # 날짜 정보가 부족한 경우 기준일에서 i일 추가
                    self.logger.warning(f"날짜 정보 부족: pred 길이 {len(pred)}, days_since_ref 길이 {len(days_since_ref)}",
                                        level=LogLevel.LEVEL3)
                    # 마지막 날짜에서 추정
                    if len(days_since_ref) > 0:
                        last_days = days_since_ref[-1].item()
                        estimated_days = last_days + (i - len(days_since_ref) + 1)
                        date_str = self._get_date_from_days_since_reference(estimated_days)
                    else:
                        date_str = f"예측일_{i + 1}"
            except Exception as e:
                self.logger.warning(f"날짜 변환 오류: {e}, index={i}, days_since_ref shape={days_since_ref.shape}",
                                    level=LogLevel.LEVEL3)
                date_str = f"예측일_{i + 1}"

            # 🔍 더욱 정교한 중복 방지 키 생성 ✨
            rounded_target = round(target[i].item(), 6)
            rounded_pred = round(pred[i].item(), 6)

            unique_key = (
                menu_id,
                date_str,
                rounded_target,
                rounded_pred
            )

            # 이미 처리된 데이터 포인트인지 확인 ✨
            if unique_key in self.seen_data_points:
                continue  # 중복이면 건너뛰기

            # 새로운 데이터 포인트로 기록 ✨
            self.seen_data_points.add(unique_key)

            detail = {
                'menu_id': menu_id,
                'menu_name': self.menu_id_to_name.get(menu_id, f"메뉴ID_{menu_id}"),
                'date': date_str,
                'prediction': pred[i].item(),
                'target': target[i].item(),
                'prediction_postprocessed': pred_postprocessed[i].item(),
                'target_postprocessed': target_postprocessed[i].item(),
                'smape': smape_postprocessed[i].item(),
                'abs_error': torch.abs(pred_postprocessed[i] - target_postprocessed[i]).item(),
                'relative_error': (
                            (pred_postprocessed[i] - target_postprocessed[i]) / target_postprocessed[i]).item() if
                target_postprocessed[i] != 0 else 0.0
            }
            self.worst_smape_details.append(detail)

    def get_worst_smape_analysis(self, top_k: int = 20) -> Dict:
        """
        🔍 가장 안좋은 SMAPE 값들에 대한 분석 결과 반환 - 중복 분석 강화 ✨
        """
        if not self.worst_smape_details:
            return {"message": "분석할 데이터가 없습니다."}

        # 🚀 중복 분석: (메뉴명, 날짜) 기준으로 그룹화하여 중복 정보 수집 ✨
        grouped_cases = {}
        for detail in self.worst_smape_details:
            # 실제값도 반올림하여 키에 포함 (미세한 차이는 같은 것으로 처리)
            key = (detail['menu_name'], detail['date'], round(detail['target'], 2))

            if key not in grouped_cases:
                grouped_cases[key] = []
            grouped_cases[key].append(detail)

        # 🔍 각 그룹에서 가장 높은 SMAPE와 중복 개수 정보 생성 ✨
        unique_cases_with_duplicate_info = []
        total_duplicates = 0

        for key, cases in grouped_cases.items():
            # 가장 높은 SMAPE를 가진 케이스 선택
            worst_case = max(cases, key=lambda x: x['smape'])

            # 중복 정보 추가
            duplicate_count = len(cases)
            if duplicate_count > 1:
                total_duplicates += duplicate_count - 1  # 실제 중복 개수 (원본 제외)

                # 🔍 중복된 케이스들의 SMAPE 범위 계산 ✨
                smape_values = [case['smape'] for case in cases]
                smape_min = min(smape_values)
                smape_max = max(smape_values)
                smape_std = np.std(smape_values) if len(smape_values) > 1 else 0.0

                # 🔍 중복된 케이스들의 예측값 범위 계산 ✨
                pred_values = [case['prediction_postprocessed'] for case in cases]
                pred_min = min(pred_values)
                pred_max = max(pred_values)

                worst_case['duplicate_info'] = {
                    'count': duplicate_count,
                    'smape_range': f"{smape_min:.4f}~{smape_max:.4f}",
                    'smape_std': smape_std,
                    'pred_range': f"{pred_min:.2f}~{pred_max:.2f}",
                    'all_predictions': pred_values,
                    'all_smapes': smape_values
                }
            else:
                worst_case['duplicate_info'] = {
                    'count': 1,
                    'smape_range': f"{worst_case['smape']:.4f}",
                    'smape_std': 0.0,
                    'pred_range': f"{worst_case['prediction_postprocessed']:.2f}",
                    'all_predictions': [worst_case['prediction_postprocessed']],
                    'all_smapes': [worst_case['smape']]
                }

            unique_cases_with_duplicate_info.append(worst_case)

        # SMAPE 값 기준으로 내림차순 정렬 (가장 안좋은 값부터)
        sorted_details = sorted(unique_cases_with_duplicate_info, key=lambda x: x['smape'], reverse=True)
        worst_cases = sorted_details[:top_k]

        # 🔍 중복 제거 결과 로그 출력 ✨
        original_count = len(self.worst_smape_details)
        deduplicated_count = len(unique_cases_with_duplicate_info)

        # 나머지 분석 로직은 동일...
        menu_analysis = {}
        for detail in worst_cases:
            menu_name = detail['menu_name']
            if menu_name not in menu_analysis:
                menu_analysis[menu_name] = {
                    'count': 0,
                    'avg_smape': 0.0,
                    'cases': []
                }
            menu_analysis[menu_name]['count'] += 1
            menu_analysis[menu_name]['avg_smape'] += detail['smape']
            menu_analysis[menu_name]['cases'].append(detail)

        # 평균 계산
        for menu_name in menu_analysis:
            menu_analysis[menu_name]['avg_smape'] /= menu_analysis[menu_name]['count']

        # 매장별 분석
        store_analysis = {}
        for detail in worst_cases:
            store_name = detail['menu_name'].split('_')[0]
            if store_name not in store_analysis:
                store_analysis[store_name] = {
                    'count': 0,
                    'avg_smape': 0.0,
                    'cases': []
                }
            store_analysis[store_name]['count'] += 1
            store_analysis[store_name]['avg_smape'] += detail['smape']
            store_analysis[store_name]['cases'].append(detail)

        # 매장별 평균 계산
        for store_name in store_analysis:
            store_analysis[store_name]['avg_smape'] /= store_analysis[store_name]['count']

        return {
            'total_analyzed': len(worst_cases),
            'original_count': original_count,
            'deduplicated_count': deduplicated_count,
            'total_duplicates_removed': total_duplicates,  # 제거된 중복 개수 ✨
            'worst_overall_smape': worst_cases[0]['smape'] if worst_cases else 0.0,
            'worst_cases': worst_cases,
            'menu_analysis': menu_analysis,
            'store_analysis': store_analysis,
            'summary_stats': self._get_summary_stats(worst_cases)
        }

    def _get_summary_stats(self, cases: List[Dict]) -> Dict:
        """
        📊 요약 통계 계산 (후처리된 값 기준)
        """
        if not cases:
            return {}

        smapes = [case['smape'] for case in cases]
        abs_errors = [case['abs_error'] for case in cases]
        predictions = [case['prediction_postprocessed'] for case in cases]  # 후처리 예측값 사용
        targets = [case['target_postprocessed'] for case in cases]  # 후처리 실제값 사용

        return {
            'smape': {
                'mean': sum(smapes) / len(smapes),
                'min': min(smapes),
                'max': max(smapes),
                'median': sorted(smapes)[len(smapes) // 2]
            },
            'abs_error': {
                'mean': sum(abs_errors) / len(abs_errors),
                'min': min(abs_errors),
                'max': max(abs_errors)
            },
            'prediction_postprocessed': {
                'mean': sum(predictions) / len(predictions),
                'min': min(predictions),
                'max': max(predictions)
            },
            'target_postprocessed': {
                'mean': sum(targets) / len(targets),
                'min': min(targets),
                'max': max(targets)
            }
        }

    def print_worst_smape_analysis(self, top_k: int = 20, detailed: bool = True):
        """
        🖨️ 가장 안좋은 SMAPE 분석 결과를 로그로 출력 - 중복 정보 포함 ✨
        """
        analysis = self.get_worst_smape_analysis(top_k)

        if "message" in analysis:
            self.logger.warning(analysis["message"], level=LogLevel.LEVEL2)
            return

        self.logger.section(f"WORST SMAPE ANALYSIS (Top {top_k})", level=LogLevel.LEVEL2)

        # 🔍 중복 제거 통계 출력 ✨
        self.logger.info(f"원본 케이스: {analysis['original_count']:,}개", level=LogLevel.LEVEL2)
        self.logger.info(f"중복 제거 후: {analysis['deduplicated_count']:,}개", level=LogLevel.LEVEL2)
        self.logger.info(f"제거된 중복: {analysis['total_duplicates_removed']:,}개", level=LogLevel.LEVEL2)
        self.logger.error(f"최악 SMAPE 값: {analysis['worst_overall_smape']:.4f}", level=LogLevel.LEVEL2)

        # 요약 통계
        stats = analysis['summary_stats']
        if stats:
            self.logger.info("SMAPE 통계 (후처리 기준):", level=LogLevel.LEVEL2)
            self.logger.info(f"   평균: {stats['smape']['mean']:.4f}", level=LogLevel.LEVEL2)
            self.logger.info(f"   최소: {stats['smape']['min']:.4f}", level=LogLevel.LEVEL2)
            self.logger.info(f"   최대: {stats['smape']['max']:.4f}", level=LogLevel.LEVEL2)
            self.logger.info(f"   중앙값: {stats['smape']['median']:.4f}", level=LogLevel.LEVEL2)

        # 매장별 분석
        self.logger.info("매장별 문제 케이스 분포:", level=LogLevel.LEVEL2)
        for store_name, store_info in analysis['store_analysis'].items():
            self.logger.warning(f"   {store_name}: {store_info['count']:2d}개 (평균 SMAPE: {store_info['avg_smape']:.4f})",
                                level=LogLevel.LEVEL2)

        # 상위 문제 메뉴 분석
        menu_sorted = sorted(analysis['menu_analysis'].items(),
                             key=lambda x: x[1]['avg_smape'], reverse=True)[:top_k]
        self.logger.warning(f"문제 메뉴 Top {top_k}:", level=LogLevel.LEVEL2)
        for menu_name, menu_info in menu_sorted:
            self.logger.warning(
                f"   {menu_name}: {menu_info['count']:2d}개 케이스 (평균 SMAPE: {menu_info['avg_smape']:.4f})",
                level=LogLevel.LEVEL2)

        if detailed:
            self.logger.info(f"상위 {max(10, top_k)}개 최악 케이스 상세 (중복 정보 포함):", level=LogLevel.LEVEL2)
            # 🚀 헤더에 중복 정보 컬럼 추가 ✨
            header = (f"{'순위':^4} {'날짜':^12} {'메뉴명':^25} {'실제값':^8} {'예측값':^8} "
                      f"{'SMAPE':^8} {'절대오차':^8} {'상대오차':^9} {'중복수':^6} {'SMAPE범위':^15}")
            self.logger.info(header, level=LogLevel.LEVEL2)
            self.logger.info("-" * 140, level=LogLevel.LEVEL2)

            for i, case in enumerate(analysis['worst_cases'][:top_k], 1):
                # 메뉴명이 너무 길면 줄임
                menu_display = case['menu_name']
                if len(menu_display) > 23:
                    menu_display = menu_display[:20] + "..."

                # 🔍 중복 정보 표시 ✨
                duplicate_count = case['duplicate_info']['count']
                smape_range = case['duplicate_info']['smape_range']

                # 중복이 있는 경우 강조 표시
                duplicate_indicator = f"{duplicate_count:^6d}" if duplicate_count == 1 else f"⚠{duplicate_count:d}⚠"

                detail_line = (f"{i:^4d} "
                               f"{case['date']:^12} "
                               f"{menu_display:<25} "
                               f"{case['target_postprocessed']:^8.2f} "
                               f"{case['prediction_postprocessed']:^8.2f} "
                               f"{case['smape']:^8.4f} "
                               f"{case['abs_error']:^8.2f} "
                               f"{case['relative_error']:^9.2%} "
                               f"{duplicate_indicator:^6} "
                               f"{smape_range:^15}")

                self.logger.info(detail_line, level=LogLevel.LEVEL2)


            # 🔍 중복이 많은 케이스들 별도 분석 ✨
            high_duplicate_cases = [case for case in analysis['worst_cases']
                                    if case['duplicate_info']['count'] > 3]

            if high_duplicate_cases:
                self.logger.warning(f"\n⚠️  중복이 많은 케이스들 (4개 이상):", level=LogLevel.LEVEL2)
                for case in high_duplicate_cases[:5]:  # 상위 5개만
                    menu_display = case['menu_name'][:20] + "..." if len(case['menu_name']) > 20 else case['menu_name']
                    dup_info = case['duplicate_info']

                    self.logger.warning(
                        f"   {menu_display} ({case['date']}): "
                        f"{dup_info['count']}개 중복, "
                        f"예측값 범위 {dup_info['pred_range']}, "
                        f"SMAPE 표준편차 {dup_info['smape_std']:.4f}",
                        level=LogLevel.LEVEL2
                    )

    def get_weighted_smape(self):
        """
        모든 배치 처리 후 최종 Weighted SMAPE 계산
        클리핑된 점수와 클리핑되지 않은 점수를 모두 반환

        Returns:
            dict: {'original': float, 'postprocessed': float} 형태의 딕셔너리
        """
        # 메뉴 ID와 매장 ID의 매핑 관계
        menu_ranges = [
            (0, 23),  # 매장 0: 메뉴 0~22
            (23, 65),  # 매장 1: 메뉴 23~64
            (65, 90),  # 매장 2: 메뉴 65~89
            (90, 121),  # 매장 3: 메뉴 90~120
            (121, 144),  # 매장 4: 메뉴 121~143
            (144, 168),  # 매장 5: 메뉴 144~167
            (168, 180),  # 매장 6: 메뉴 168~179
            (180, 188),  # 매장 7: 메뉴 180~187
            (188, 193)  # 매장 8: 메뉴 188~192
        ]

        # === 원본 SMAPE 계산 ===
        # 각 품목별 평균 SMAPE 계산
        menu_avg_smape = torch.zeros_like(self.smape_list)
        valid_menus = self.validated_date_per_menu > 0

        menu_avg_smape[valid_menus] = (
                self.smape_list[valid_menus] / self.validated_date_per_menu[valid_menus]
        )

        # 각 매장별로 해당 메뉴들의 평균 SMAPE 계산
        for store_id, (start_menu, end_menu) in enumerate(menu_ranges):
            store_menu_smape = menu_avg_smape[start_menu:end_menu]
            store_valid_menus = valid_menus[start_menu:end_menu]

            if store_valid_menus.sum().item() > 0:
                self.store_smape[store_id] = store_menu_smape[store_valid_menus].mean()

        # 가중치 적용하여 최종 점수 계산
        weighted_smape_original = (self.store_smape * self.weights).sum()

        # === 클리핑된 SMAPE 계산 ===
        # 각 품목별 평균 SMAPE 계산 (후처리된 버전)
        menu_avg_smape_postprocessed = torch.zeros_like(self.smape_list_postprocessed)
        valid_menus_postprocessed = self.validated_date_per_menu_postprocessed > 0

        menu_avg_smape_postprocessed[valid_menus_postprocessed] = (
                self.smape_list_postprocessed[valid_menus_postprocessed] / self.validated_date_per_menu_postprocessed[
            valid_menus_postprocessed]
        )

        # 각 매장별로 해당 메뉴들의 평균 SMAPE 계산 (클리핑된 버전)
        for store_id, (start_menu, end_menu) in enumerate(menu_ranges):
            store_menu_smape_postprocessed = menu_avg_smape_postprocessed[start_menu:end_menu]
            store_valid_menus_postprocessed = valid_menus_postprocessed[start_menu:end_menu]

            if store_valid_menus_postprocessed.sum().item() > 0:
                self.store_smape_postprocessed[store_id] = store_menu_smape_postprocessed[
                    store_valid_menus_postprocessed].mean()

        # 가중치 적용하여 최종 점수 계산 (클리핑된 버전)
        weighted_smape_postprocessed = (self.store_smape_postprocessed * self.weights).sum()

        return {
            'original': weighted_smape_original.item(),
            'postprocessed': weighted_smape_postprocessed.item()
        }

    def reset(self):
        """
        누적된 값들 초기화 (새로운 에포크 시작 시 사용)
        """
        # 원본 SMAPE 추적 변수 초기화
        self.smape_list.zero_()
        self.validated_date_per_menu.zero_()
        self.store_smape.zero_()

        # 클리핑된 SMAPE 추적 변수 초기화
        self.smape_list_postprocessed.zero_()
        self.validated_date_per_menu_postprocessed.zero_()
        self.store_smape_postprocessed.zero_()

        # 🔍 분석용 데이터도 초기화 ✨
        self.worst_smape_details.clear()
        self.seen_data_points.clear()  # 중복 방지용 집합도 초기화 ✨