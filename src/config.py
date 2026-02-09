from dataclasses import dataclass


@dataclass
class Config:
    """
    Class for storing configuration and constants
    """

    frequency: str = "week"

    threshold_max_sales: float = 0
    threshold_z_score: int = 10
    threshold_count_rows: int = 2
