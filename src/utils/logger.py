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

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

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
