import os
import platform
from datetime import datetime, timedelta

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from ..utils.Logger import get_logger, LogLevel


class Plotter:
    def __init__(self):
        self.predictions = []
        self.targets = []
        self.inputs = []
        self.store_features = []
        self.date_features = []
        self.logger = get_logger()

    def collect_data(self, X, y, outputs):
        self.predictions.append(outputs[:, :, 0].cpu().numpy())
        self.targets.append(y[:, :, 0].cpu().numpy())

        self.inputs.append(X[:, :, 0].cpu().numpy())
        self.store_features.append(X[:, -1, 13].cpu().numpy())
        self.date_features.append(X[:, -1, 1].cpu().numpy())

    def plot(self, epoch, train_window: int = 28, forecast_window: int = 7, save: bool = True, save_dir: str = None):
        """영업장별 예측값과 실제값 플로팅 (배치 구분 및 샘플링 포함)"""

        # 한국어 폰트 설정 (macOS)
        if platform.system() == 'Darwin':  # macOS
            try:
                # AppleGothic 또는 Helvetica 폰트 사용
                font_path = '/System/Library/Fonts/AppleGothic.ttf'
                if os.path.exists(font_path):
                    plt.rcParams['font.family'] = 'AppleGothic'
                else:
                    plt.rcParams['font.family'] = 'Helvetica'
            except:
                plt.rcParams['font.family'] = 'DejaVu Sans'
        else:
            plt.rcParams['font.family'] = 'DejaVu Sans'

        # 어두운 배경 스타일 설정
        plt.style.use('dark_background')
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['text.color'] = 'white'
        plt.rcParams['axes.labelcolor'] = 'white'
        plt.rcParams['axes.edgecolor'] = 'white'
        plt.rcParams['xtick.color'] = 'white'
        plt.rcParams['ytick.color'] = 'white'

        self._plot_samples(epoch, train_window, forecast_window, save, save_dir)

    def _plot_samples(self, epoch, train_window, forecast_window, save: bool = True, save_suffix: str = None):
        # 데이터가 수집되었는지 확인
        if not self.predictions or not self.targets or not self.inputs:
            self.logger.info(f"Epoch {epoch}: 플롯할 데이터가 없습니다. 데이터 수집을 확인해주세요.", level=LogLevel.LEVEL2)
            return

        # 데이터 결합
        predictions = np.concatenate(self.predictions, axis=0)
        targets = np.concatenate(self.targets, axis=0)
        inputs = np.concatenate(self.inputs, axis=0)
        store_features = np.concatenate(self.store_features, axis=0)
        date_features = np.concatenate(self.date_features, axis=0)

        store_ids = store_features.astype(int)
        unique_stores = np.unique(store_ids)
        reference_date = datetime(2023, 1, 1)

        """콘솔 출력용 3x3 그리드 샘플 플롯"""
        self.logger.info(f"\n🎯 콘솔 출력: 9개 영업장별 랜덤 샘플", level=LogLevel.LEVEL2)

        fig, axes = plt.subplots(3, 3, figsize=(18, 12))
        fig.suptitle(f'Store-wise Random Samples (Epoch {epoch})', fontsize=16, fontweight='bold')

        axes = axes.flatten()
        unique_stores = np.unique(store_ids)

        for i in range(9):
            ax = axes[i]

            if i < len(unique_stores):
                store_id = unique_stores[i]
                store_mask = store_ids == store_id

                if np.any(store_mask):
                    store_predictions = predictions[store_mask]
                    store_targets = targets[store_mask]
                    store_input = inputs[store_mask]
                    store_dates = date_features[store_mask]

                    # 랜덤 샘플 선택
                    n_samples = len(store_predictions)
                    random_idx = np.random.choice(n_samples)

                    # 날짜 생성
                    last_input_date = reference_date + timedelta(days=int(store_dates[random_idx]))
                    input_dates = [last_input_date - timedelta(days=train_window - 1 - k) for k in
                                   range(train_window)]
                    forecast_dates = [last_input_date + timedelta(days=k + 1) for k in range(forecast_window)]

                    # 입력 데이터 (파란색, 가는 선)
                    ax.plot(input_dates, store_input[random_idx],
                            color='steelblue', linestyle='-', linewidth=2, alpha=0.7,
                            label='Input Data')

                    # 실제값 (초록색, 굵은 선)
                    ax.plot(forecast_dates, store_targets[random_idx],
                            color='forestgreen', linestyle='-', linewidth=3, alpha=0.9,
                            label='Actual')

                    # 예측값 (빨간색, 굵은 점선)
                    ax.plot(forecast_dates, store_predictions[random_idx],
                            color='crimson', linestyle='--', linewidth=3, alpha=0.9,
                            label='Prediction')

                    # 구분선
                    ax.axvline(x=last_input_date, color='gray', linestyle=':', alpha=0.6, linewidth=2)

                    ax.set_title(f'Store {store_id}', fontweight='bold')
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Sales')
                    ax.grid(True, alpha=0.3)

                    # x축 날짜 형식 설정
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
                    ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
                    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, fontsize=8)

                    # 범례 (첫 번째 서브플롯에만)
                    if i == 0:
                        ax.legend(fontsize=10)

                    # 통계 정보
                    mae = np.mean(np.abs(store_predictions[random_idx] - store_targets[random_idx]))
                    ax.text(0.02, 0.98,
                            f'MAE: {mae:.2f}\nPred Avg: {np.mean(store_predictions[random_idx]):.1f}\nActual Avg: {np.mean(store_targets[random_idx]):.1f}',
                            transform=ax.transAxes, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='darkblue', alpha=0.8),
                            fontsize=9)

                else:
                    ax.text(0.5, 0.5, f'Store {store_id}\nNo Data',
                            transform=ax.transAxes, ha='center', va='center',
                            fontsize=14, fontweight='bold', color='gray')
                    ax.set_title(f'Store {store_id}', fontweight='bold')
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                    ax.set_xticks([])
                    ax.set_yticks([])
            else:
                ax.text(0.5, 0.5, f'No Store Data',
                        transform=ax.transAxes, ha='center', va='center',
                        fontsize=14, fontweight='bold', color='gray')
                ax.set_title(f'Store {i}', fontweight='bold')
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_xticks([])
                ax.set_yticks([])

        plt.tight_layout()

        if save and save_suffix is not None:
            # 파일로 저장 (고해상도, 긴 이미지)
            save_filename = save_suffix + f'_{epoch}.png'
            plt.savefig(save_filename, dpi=300, bbox_inches='tight')
            self.logger.info(f"📁 배치별 구분 그래프 저장: {save_filename}", level=LogLevel.LEVEL2)

        plt.show()
        plt.close()

    def reset(self):
        self.predictions = []
        self.targets = []
        self.inputs = []
        self.store_features = []
        self.date_features = []