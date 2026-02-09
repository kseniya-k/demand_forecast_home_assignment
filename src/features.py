"""
Basic features
"""

from typing import List

import numpy as np
import pandas as pd

from config import Config


def add_weekly_stat(df: pd.DataFrame, lags: List[int] = [4, 8, 53]) -> pd.DataFrame:
    """
    Add rolling mean and sum sales
    :param lags: amount of weeks for rolling window
    """
    for lag in lags:
        num_days = lag * 7

        df_stats = (
            df.set_index("date")
            .groupby("sku")["sales"]
            .rolling(f"{num_days}D", min_periods=1)
            .agg(["mean", "sum"])
            .reset_index()
            .rename(columns={"mean": f"mean_{lag}w", "sum": f"sum_{lag}w"})
        )

        df = df.merge(df_stats, on=["sku", "date"], how="left")

        # df[f"mean_{lag}w"] = df_stats["mean"].copy()
        # df[f"sum_{lag}w"] = df_stats["sum"].copy()
    return df


def add_weekly_lag(
    df: pd.DataFrame,
    config: Config,
    lags: List[int] = [
        53,
    ],
) -> pd.DataFrame:
    """
    Add lag sales
    :param lags: amount of weeks or months for lag
    """
    if config.frequency == "month":
        lags = [int(np.ceil(x / 4.5)) for x in lags]

    for lag in lags:
        df[f"sales_lag_{lag}"] = df.groupby(["sku"])["sales"].shift(lag)

    return df


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add week number, month number, holiday flags
    """
    df["week_num"] = df["date"].dt.isocalendar().week
    df["month_num"] = df["date"].dt.month

    # TODO: add holidays
    # TODO: add amout of weeks since first and last sale by SKU

    return df
