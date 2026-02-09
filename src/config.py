from dataclasses import dataclass
from typing import Optional, Tuple

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
