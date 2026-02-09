"""
Data preparation, model learining and metric evaluation pipelines
"""

import logging
from typing import Optional

import pandas as pd

from config import Config
from data_preparation import (drop_anomaly_sales, drop_anomaly_sku,
                              explode_frequency, forward_fill_data,
                              sort_fillna_cast_date)
from features import add_date_features, add_weekly_lag, add_weekly_stat


def prepare_data(config: Config, input_path: str, output_path: str, n_samples: Optional[int] = None):
    logging.info(f"Config: {config}")

    df = pd.read_parquet(input_path)

    if n_samples:
        df = df.sample(n_samples)

    df = sort_fillna_cast_date(df)
    df = drop_anomaly_sku(df, config)
    df = drop_anomaly_sales(df, config)

    df = forward_fill_data(df, config)

    if config.is_dense_data:
        df = explode_frequency(df, config)

    df = add_weekly_stat(df)
    df = add_weekly_lag(df, config)
    df = add_date_features(df)

    df.to_parquet(output_path)
