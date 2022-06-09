import os

from src.utils.load_config import get_config

cfg = get_config()


def define_log_dir():
    path = os.path.join(
        "./logs",
        cfg["Base"]["Agent"],
        cfg["Base"]["Env"],
        str(cfg["Base"]["Run_ID"]),
    )
    return path
