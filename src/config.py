import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class Config:
    """
    Class for storing configuration and constants
    """

    frequency: str = "week"

    threshold_max_sales: float = 0
    threshold_z_score: int = 10
    threshold_count_rows: int = 2

    max_date: Optional[pd.Timestamp] = None
    is_dense_data: bool = True

    heuristics_min_rows: int = 10
    ci_levels: Tuple[float, float, float] = (0.2, 0.5, 0.8)
    horizon_days: int = 8 * 7

    data_folder: str = "data"
    predict_folder: str = "predict"
    model_name: str = "prophet"

    @property
    def data_path(self) -> str:
        return os.path.join(self.data_folder, f"data_{self.frequency}.parquet")

    @property
    def predict_path(self) -> str:
        return os.path.join(self.predict_folder, f"predict_{self.frequency}_{self.model_name}.parquet")


def get_frequency_params(frequency_name: str, horizon_days: int) -> Tuple[int, str, int]:
    """
    Return sime seried frequency characterisics

    :param frequency_name: freqency written in words (e.g. week, month)
    :param horizon_days: predict interval in days
    """
    if frequency_name == "day":
        horizon_period = horizon_days
        freq_name = "D"
        seasonal_period = 365
    elif frequency_name == "week":
        horizon_period = int(horizon_days / 7)
        freq_name = "W-SUN"
        seasonal_period = 52
    elif frequency_name == "month":
        horizon_period = int(np.floor(horizon_days / 30))
        freq_name = "MS"
        seasonal_period = 12
    else:
        raise NotImplementedError(f"Unexpected frequency! Expected 'day', 'week' or 'month', got {frequency_name}")
    return (horizon_period, freq_name, seasonal_period)
