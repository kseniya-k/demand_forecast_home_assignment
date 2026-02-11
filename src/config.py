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

    @property
    def data_path(self) -> str:
        return os.path.join(self.data_folder, f"data_{self.frequency}.parquet")

    @property
    def predict_path(self) -> str:
        return os.path.join(self.predict_folder, f"predict_{self.frequency}.parquet")


def get_frequency_params(frequency_name: str, horizon_days: int) -> Tuple[int, int, str]:
    if frequency_name == "day":
        horizon_period = horizon_days
        freq_mult = 1
        freq_name = "D"
    elif frequency_name == "week":
        horizon_period = int(horizon_days / 7)
        freq_mult = 7
        freq_name = "W-SUN"
    elif frequency_name == "month":
        horizon_period = int(np.floor(horizon_days / 30))
        freq_mult = 31
        freq_name = "MS"
    else:
        raise NotImplementedError(f"Unexpected frequency! Expected 'day', 'week' or 'month', got {frequency_name}")
    return (horizon_period, freq_mult, freq_name)
