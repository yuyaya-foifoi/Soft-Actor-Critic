import torch
from pytorch_lightning import Trainer

from src.utils.load_config import get_config

cfg = get_config()


def get_trainer():
    trainer = Trainer(
        gpus=torch.cuda.device_count(),
        max_epochs=cfg["Train"]["Epoch"],
    )
    return trainer
