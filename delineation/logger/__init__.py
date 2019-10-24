import os
from .loggers import FileLogger


def make_logger(cfg):

    LOG_DIR = os.path.join(cfg.LOGGING.LOG_DIR)
    return FileLogger(LOG_DIR)