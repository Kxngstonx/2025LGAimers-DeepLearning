import os
import random

import numpy as np
import torch


def set_all_seeds(seed: int = 42):
    """
    모든 라이브러리의 랜덤 시드를 고정하여 재현 가능한 결과 보장

    Args:
        seed (int): 고정할 시드 값 (기본값: 42)
    """
    print(f"모든 랜덤 시드를 {seed}로 고정 중...")

    # Python 내장 random 모듈
    random.seed(seed)

    # NumPy 시드 고정
    np.random.seed(seed)

    # PyTorch 시드 고정
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 멀티 GPU 환경 대응

    # CUDA 결정론적 연산 보장 (성능 저하 가능성 있음)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 환경 변수 설정 (Python hash seed 고정)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # CUDA 관련 환경 변수 설정 (더 강력한 결정론적 동작)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    # PyTorch 결정론적 알고리즘 사용 (PyTorch 1.8+)
    try:
        torch.use_deterministic_algorithms(True)
    except AttributeError:
        # 이전 버전의 PyTorch에서는 이 함수가 없을 수 있음
        pass

    print("시드 고정 완료 - 재현 가능한 결과 보장됨")


def worker_init_fn(worker_id):
    """
    DataLoader worker 초기화 함수
    멀티프로세싱 환경에서 각 워커의 시드를 고정
    
    Args:
        worker_id (int): 워커 ID
    """
    # 각 워커마다 고유한 시드 설정
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)