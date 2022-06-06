import os

from src.utils.config import get_config

cfg = get_config()


def define_log_dir():
    path = os.path.join("./logs", cfg["Agent"], cfg["Env"], str(cfg["Run_ID"]))
    return path
