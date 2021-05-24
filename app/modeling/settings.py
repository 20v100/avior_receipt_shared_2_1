import logging
from enum import Enum
import sys
from typing import Tuple, Any, List
from pydantic import BaseModel, Field

log = logging.getLogger("spleeter")


class Settings(BaseModel):
    """
    Class to store pipeline parameters
    """

    x_img_folder_path: str = Field("/mutual/data/images/x/", description="")
    y_img_folder_path: str = Field("/mutual/data/images/y/", description="")
    meta_folder_path: str = Field("/mutual/data/images/meta/", description="")
    observation_log_folder_path: str = Field(
        "/mutual/logs/observations/", description=""
    )
    epoch_log_folder_path: str = Field("/mutual/logs/epochs/", description="")
    model_log_foler_path: str = Field("/mutual/logs/models/", description="")
    model_name: str = Field("default", description="")
    original_img_size: int = Field(2048, description="")
    crop_img_size: int = Field(128, description="")
    min_defect_ratio: float = Field(0.001, description="")
    channel_positions: List[int] = Field([10, 11, 12, 13], description="")
    resize_tuple: Tuple[float, float] = Field((0.08, 1.0), description="")
    batch: int = Field(4, description="")
    learning_rate: float = Field(1e-3, description="")
    epochs: int = Field(4, description="")
    # debug = True


settings = Settings()
