"""
Module that contains data preparation, model learining and metric evaluation pipelines
"""

from typing import Optional

import pandas as pd

from data_preparation import Dataset


def prepare_data(path: str, n_samples: Optional[int] = None):
    df = pd.read_parquet(path)

    if n_samples:
        df = df.sample(n_samples)

    ds = Dataset(data=df, frequency="week")
    ds.data["date"] = pd.to_datetime(ds.data["date"])

    ds.drop_anomaly_sku()
    ds.drop_anomaly_sales()
    ds.drop_new_sku()

    ds.explode_frequency()
    ds.add_weekly_stat()
    ds.add_weekly_lag()
    ds.add_date_features()
    return ds
