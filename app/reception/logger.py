import logging
import pathlib
import json

import pathlib

# import os
import random
from typing import List
import typer
import pandas

# import pandas as pd
import torch
from torchvision.io import read_image
from torch.utils.data.dataset import Dataset
from app.ss.settings import settings
import torchvision.transforms.functional as TF
from torchvision import transforms
from pydantic import BaseModel

from PIL import Image
from torchvision.io import read_image


#
# ────────────────────────────────────────────────────────────────────── I ──────────
#   :::::: C O N S O L E   L O G G I N G : :  :   :    :     :        :          :
# ────────────────────────────────────────────────────────────────────────────────
#


class TyperLoggerHandler(logging.Handler):
    """A custom logger handler that use Typer echo."""

    def emit(self, record: logging.LogRecord) -> None:
        typer.echo(
            typer.style(
                self.format(record),
                fg=typer.colors.BRIGHT_CYAN,
                bold=False,
            )
        )


formatter = logging.Formatter(
    "%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s"
)
handler = TyperLoggerHandler()
handler.setFormatter(formatter)
log: logging.Logger = logging.getLogger("spleeter")
log.addHandler(handler)
log.setLevel(logging.INFO)
from torch.utils.tensorboard import SummaryWriter

#
# ──────────────────────────────────────────────────────────────── II ──────────
#   :::::: F I L E   L O G G I N G : :  :   :    :     :        :          :
# ──────────────────────────────────────────────────────────────────────────
#


class Obs(BaseModel):
    pass


class FileLogger:
    def __init__(
        self,
        observation_log_folder_path: str,
        epoch_log_folder_path: str,
        model_log_folder_path: str,
    ):
        self.observation_log_folder_path = observation_log_folder_path
        self.epoch_log_folder_path = epoch_log_folder_path
        self.model_log_folder_path = model_log_folder_path
        # self.obs_df = pd.read_csv()
        # self.epoch_df =
        # self.model_df =


file_logger = FileLogger(
    settings.observation_log_folder_path,
    settings.epoch_log_folder_path,
    settings.model_log_foler_path,
)


writer = SummaryWriter(f"/mutual/logs/tensorboard/{settings.model_name}")
