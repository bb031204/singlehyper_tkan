import logging
import os
import sys
from datetime import datetime
from typing import Optional


class FlushStreamHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()


class FlushFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()


def setup_logger(name: str = "SingleHyperTKAN", level: str = "INFO", output_dir: Optional[str] = None, console: bool = True, file: bool = True, append_mode: bool = False):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers.clear()

    # 支持控制台 ANSI 颜色输出
    class ColorFormatter(logging.Formatter):
        """支持 ANSI 颜色的日志格式化器"""
        GREY = '\033[90m'
        BLUE = '\033[94m'
        YELLOW = '\033[93m'
        RED = '\033[91m'
        GREEN = '\033[92m'
        RESET = '\033[0m'
        BOLD = '\033[1m'

        def __init__(self, fmt=None, datefmt=None, style='%'):
            super().__init__(fmt, datefmt, style)

        def format(self, record):
            levelname = record.levelname
            if levelname == 'INFO':
                levelname = f"{self.BLUE}{levelname}{self.RESET}"
            elif levelname == 'WARNING':
                levelname = f"{self.YELLOW}{levelname}{self.RESET}"
            elif levelname == 'ERROR':
                levelname = f"{self.RED}{levelname}{self.RESET}"

            # 保留消息中的 ANSI 颜色代码（不转义）
            record.levelname = levelname
            return super().format(record)

    formatter = ColorFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    if console:
        ch = FlushStreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    if file and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        if append_mode:
            logs = [f for f in os.listdir(output_dir) if f.startswith('train_') and f.endswith('.log')]
            log_file = os.path.join(output_dir, logs[0]) if logs else os.path.join(output_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
            mode = 'a'
        else:
            log_file = os.path.join(output_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
            mode = 'w'
        fh = FlushFileHandler(log_file, encoding='utf-8', mode=mode)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
