"""
Data preparation, model learining and metric evaluation pipelines
"""

import logging
import sys
from typing import Optional

import pandas as pd

from config import Config
from features import add_date_features, add_weekly_lag, add_weekly_stat
from model import fit_predict
from preparation import (drop_anomaly_sales, drop_anomaly_sku,
                         explode_frequency, forward_fill_data,
                         sort_fillna_cast_date)


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


def fit_predict_model(config: Config, input_path: str, output_path: str):
    logging.info(f"Config: {config}")

    df = pd.read_parquet(input_path)
    preidct = fit_predict(config, df)

    preidct.to_parquet(output_path)


def main(argv, argc):
    path = argv[0]

    logging.info(
        f"Start predict on weekly level for path {path}. Predicts will be saved to 'data/predict_weekly.parquet'"
    )
    config = Config(frequency="week", horizon_days=7 * 8)
    prepare_data(config, path, "data/data_weekly.parquet")
    fit_predict_model(config, "data/data_weekly.parquet", "data/predict_weekly.parquet")

    logging.info(
        f"Start predict on monthly level for path {path}. Predicts will be saved to 'data/predict_weekly.parquet'"
    )
    config = Config(frequency="month", horizon_days=12 * 30)
    prepare_data(config, path, "data/data_weekly.parquet")
    fit_predict(config, "data/data_weekly.parquet", "data/predict_monthly.parquet")


if __name__ == "__main__":
    main(sys.argv, len(sys.argv))
