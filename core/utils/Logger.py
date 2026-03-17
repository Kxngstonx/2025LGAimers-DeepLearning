import os
import logging
import sys
from datetime import datetime
from enum import Enum
from typing import Optional


class LogLevel(Enum):
    """로그 레벨 정의"""
    NONE = 0  # 로깅 없음
    LEVEL1 = 1  # 중요 정보만 (실험 결과, 주요 단계)
    LEVEL2 = 2  # 중간 수준 정보 (모델 구성, 데이터 통계)
    LEVEL3 = 3  # 상세 정보 (모든 처리 단계, 디버깅 정보)


class Logger:
    """
    다단계 로깅 시스템

    특징:
    - 4단계 로그 레벨 지원 (NONE, LEVEL1, LEVEL2, LEVEL3)
    - 콘솔: Config 설정에 따른 필터링
    - 파일: DEBUG 제외한 모든 로그 저장 (색상 코드 없음)
    - 이모지 및 색상 지원으로 가독성 향상
    - 여러 Jupyter 노트북에서 동시 사용 가능
    """
    
    # 로거 인스턴스 저장소 (여러 노트북 지원)
    _instances = {}

    # ANSI 색상 코드
    COLORS = {
        'RESET': '\033[0m',
        'RED': '\033[91m',
        'GREEN': '\033[92m',
        'YELLOW': '\033[93m',
        'BLUE': '\033[94m',
        'MAGENTA': '\033[95m',
        'CYAN': '\033[96m',
        'WHITE': '\033[97m',
    }

    def __init__(self,
                 level: LogLevel = LogLevel.LEVEL2,
                 log_file: Optional[str] = None,
                 log_dir: Optional[str] = None,
                 name: str = "default"):
        """
        로거 초기화

        Args:
            level: 콘솔 로그 레벨 (기본값: LEVEL2)
            log_file: 로그 파일명 (기본값: None, 자동 생성)
            log_dir: 로그 디렉토리 (기본값: None, 결과 디렉토리 사용)
            name: 로거 이름 (여러 인스턴스 구분용, 기본값: "default")
        """
        self.name = name
        self.level = level

        # 로깅 비활성화 시 초기화 중단
        if level == LogLevel.NONE:
            return

        # 로그 파일 설정
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"log_{timestamp}.log"

        # 로그 디렉토리 설정
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, log_file)
        else:
            log_path = log_file

        # 로거 설정 - 고유한 로거명 사용
        logger_name = f"LGAimers_{id(self)}"  # 인스턴스마다 고유한 로거명
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.DEBUG)

        # 기존 핸들러 모두 제거 (중복 방지)
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # 상위 로거로 전파 방지 (중복 출력 차단)
        self.logger.propagate = False

        # 파일 핸들러 설정 (DEBUG 제외, 색상 코드 없음)
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(logging.INFO)  # DEBUG 제외
        file_format = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        file_handler.setFormatter(file_format)
        self.logger.addHandler(file_handler)

        # 콘솔 핸들러 설정 (Config 레벨 따름, 색상 코드 포함)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_format = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)

        # 핸들러 참조 저장
        self.file_handler = file_handler
        self.console_handler = console_handler

        # 로그 파일 경로 저장
        self.log_path = log_path

        # 로거 초기화 로그
        self.info(f"로그 시스템 초기화 완료 (콘솔 레벨: {level.name}, 파일: {log_path})", level=LogLevel.LEVEL1)
        
        # 인스턴스 저장 (여러 노트북 지원)
        Logger._instances[name] = self

    def _should_log_console(self, level: LogLevel) -> bool:
        """
        콘솔에서 메시지를 출력해야 하는지 확인 (Config 설정 따름)

        Args:
            level: 메시지의 로그 레벨

        Returns:
            콘솔 로그 출력 여부
        """
        return self.level.value >= level.value

    def _should_log_file(self, message_type: str) -> bool:
        """
        파일에서 메시지를 출력해야 하는지 확인 (DEBUG 제외)

        Args:
            message_type: 메시지 타입 ('debug', 'info', 'warning', 'error')

        Returns:
            파일 로그 출력 여부
        """
        return message_type != 'debug'

    def _log_message(self, message: str, color: str, emoji: str, log_method, message_level: LogLevel,
                     message_type: str):
        """
        콘솔과 파일에 각각 다른 형태로 로깅

        Args:
            message: 로그 메시지
            color: 콘솔용 색상 코드
            emoji: 이모지
            log_method: 로깅 메서드 (debug, info, warning, error)
            message_level: 메시지의 로그 레벨
            message_type: 메시지 타입 ('debug', 'info', 'warning', 'error')
        """
        console_should_log = self._should_log_console(message_level)
        file_should_log = self._should_log_file(message_type)

        if console_should_log and file_should_log:
            # 둘 다 출력해야 하는 경우
            # 1. 콘솔용 (색상 + 이모지)
            colored_message = f"{color}{emoji} {message}{self.COLORS['RESET']}"
            self.file_handler.setLevel(logging.CRITICAL + 1)  # 파일 핸들러 임시 비활성화
            log_method(colored_message)

            # 2. 파일용 (색상 없음, 이모지만)
            plain_message = f"{emoji} {message}"
            self.file_handler.setLevel(logging.INFO)  # 파일 핸들러 재활성화
            self.console_handler.setLevel(logging.CRITICAL + 1)  # 콘솔 핸들러 임시 비활성화
            log_method(plain_message)

            # 레벨 복구
            self.console_handler.setLevel(logging.DEBUG)

        elif console_should_log and not file_should_log:
            # 콘솔만 출력 (DEBUG 메시지)
            colored_message = f"{color}{emoji} {message}{self.COLORS['RESET']}"
            self.file_handler.setLevel(logging.CRITICAL + 1)  # 파일 핸들러 비활성화
            log_method(colored_message)
            self.file_handler.setLevel(logging.INFO)  # 파일 핸들러 복구

        elif not console_should_log and file_should_log:
            # 파일만 출력 (콘솔 레벨이 낮은 경우)
            plain_message = f"{emoji} {message}"
            self.console_handler.setLevel(logging.CRITICAL + 1)  # 콘솔 핸들러 비활성화
            log_method(plain_message)
            self.console_handler.setLevel(logging.DEBUG)  # 콘솔 핸들러 복구

    def debug(self, message: str, level: LogLevel = LogLevel.LEVEL3):
        """
        디버그 메시지 로깅 (파일에는 저장되지 않음)

        Args:
            message: 로그 메시지
            level: 메시지의 로그 레벨 (기본값: LEVEL3)
        """
        self._log_message(message, self.COLORS['CYAN'], '🔍', self.logger.debug, level, 'debug')

    def info(self, message: str, level: LogLevel = LogLevel.LEVEL2):
        """
        정보 메시지 로깅

        Args:
            message: 로그 메시지
            level: 메시지의 로그 레벨 (기본값: LEVEL2)
        """
        self._log_message(message, self.COLORS['GREEN'], 'ℹ️', self.logger.info, level, 'info')

    def warning(self, message: str, level: LogLevel = LogLevel.LEVEL1):
        """
        경고 메시지 로깅

        Args:
            message: 로그 메시지
            level: 메시지의 로그 레벨 (기본값: LEVEL1)
        """
        self._log_message(message, self.COLORS['YELLOW'], '⚠️', self.logger.warning, level, 'warning')

    def error(self, message: str, level: LogLevel = LogLevel.LEVEL1):
        """
        오류 메시지 로깅

        Args:
            message: 로그 메시지
            level: 메시지의 로그 레벨 (기본값: LEVEL1)
        """
        self._log_message(message, self.COLORS['RED'], '❌', self.logger.error, level, 'error')

    def success(self, message: str, level: LogLevel = LogLevel.LEVEL1):
        """
        성공 메시지 로깅

        Args:
            message: 로그 메시지
            level: 메시지의 로그 레벨 (기본값: LEVEL1)
        """
        self._log_message(message, self.COLORS['GREEN'], '✅', self.logger.info, level, 'info')

    def highlight(self, message: str, level: LogLevel = LogLevel.LEVEL1):
        """
        강조 메시지 로깅

        Args:
            message: 로그 메시지
            level: 메시지의 로그 레벨 (기본값: LEVEL1)
        """
        self._log_message(message, self.COLORS['MAGENTA'], '🌟', self.logger.info, level, 'info')

    def section(self, title: str, level: LogLevel = LogLevel.LEVEL1):
        """
        섹션 구분선 로깅

        Args:
            title: 섹션 제목
            level: 메시지의 로그 레벨 (기본값: LEVEL1)
        """
        console_should_log = self._should_log_console(level)
        file_should_log = self._should_log_file('info')

        separator = "=" * 50

        if console_should_log and file_should_log:
            # 콘솔용 (색상 포함)
            colored_message = f"{self.COLORS['BLUE']}\n{separator}\n📊 {title}\n{separator}{self.COLORS['RESET']}"
            self.file_handler.setLevel(logging.CRITICAL + 1)
            self.logger.info(colored_message)

            # 파일용 (색상 없음)
            plain_message = f"\n{separator}\n📊 {title}\n{separator}"
            self.file_handler.setLevel(logging.INFO)
            self.console_handler.setLevel(logging.CRITICAL + 1)
            self.logger.info(plain_message)

            # 레벨 복구
            self.console_handler.setLevel(logging.DEBUG)

        elif console_should_log and not file_should_log:
            # 콘솔만
            colored_message = f"{self.COLORS['BLUE']}\n{separator}\n📊 {title}\n{separator}{self.COLORS['RESET']}"
            self.file_handler.setLevel(logging.CRITICAL + 1)
            self.logger.info(colored_message)
            self.file_handler.setLevel(logging.INFO)

        elif not console_should_log and file_should_log:
            # 파일만
            plain_message = f"\n{separator}\n📊 {title}\n{separator}"
            self.console_handler.setLevel(logging.CRITICAL + 1)
            self.logger.info(plain_message)
            self.console_handler.setLevel(logging.DEBUG)


# 기본 로거 인스턴스
_default_logger = None


def initialize_logger(level: LogLevel = LogLevel.LEVEL2,
                      log_file: Optional[str] = None,
                      log_dir: Optional[str] = None,
                      name: str = "default",
                      force_reinit: bool = False):
    """
    로거 초기화 또는 기존 로거 업데이트
    
    여러 Jupyter 노트북에서 사용할 경우 각 노트북마다 고유한 name을 지정하세요.

    Args:
        level: 콘솔 로그 레벨 (파일은 항상 INFO 이상)
        log_file: 로그 파일명
        log_dir: 로그 디렉토리
        name: 로거 이름 (여러 인스턴스 구분용)
        force_reinit: 기존 로거가 있어도 강제로 새로 초기화할지 여부
        
    Returns:
        Logger 인스턴스
    """
    global _default_logger
    
    # 이미 해당 이름의 로거가 있는지 확인
    if name in Logger._instances and not force_reinit:
        # 기존 로거의 레벨만 업데이트
        logger = Logger._instances[name]
        logger.level = level
        return logger
    
    # 강제 재초기화 시 기존 인스턴스 제거
    # force_reinit=True로 설정하면 기존 로거가 있어도 새로 생성
    # 이는 설정 변경 후 로거를 새로 초기화하거나 실험마다 새 로그 파일을 생성할 때 유용
    if name in Logger._instances and force_reinit:
        del Logger._instances[name]
    
    # 새 로거 생성
    logger = Logger(level, log_file, log_dir, name)
    
    # 기본 로거 설정 (첫 번째 생성된 로거)
    if _default_logger is None:
        _default_logger = logger
    
    return logger


def get_logger(name: str = "default"):
    """
    로거 인스턴스 반환
    
    여러 Jupyter 노트북에서 사용할 경우 각 노트북마다 고유한 name을 지정하세요.

    Args:
        name: 로거 이름 (여러 인스턴스 구분용)
        
    Returns:
        Logger 인스턴스
    """
    # 해당 이름의 로거가 있으면 반환
    if name in Logger._instances:
        return Logger._instances[name]
    
    # 기본 로거가 있으면 반환
    global _default_logger
    if _default_logger is not None:
        return _default_logger
    
    # 로거가 없으면 새로 생성
    return initialize_logger(name=name)


def has_logger(name: str = "default"):
    """
    해당 이름의 로거가 존재하는지 확인
    
    Args:
        name: 로거 이름
        
    Returns:
        로거 존재 여부
    """
    return name in Logger._instances