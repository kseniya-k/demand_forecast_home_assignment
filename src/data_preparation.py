"""
Functions for data preparation: drop irrelevant data, resample time series
"""

import logging

import numpy as np
import pandas as pd

from config import Config


def sort_fillna_cast_date(df: pd.DataFrame) -> pd.DataFrame:
    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values(["sku", "date"])
    df["sales"] = df["sales"].fillna(0)
    df = df.dropna(how="any")
    return df


def drop_anomaly_sku(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """
    Remove SKU with maximum sales <= threshold_max_sales
    """
    df_max = df.groupby("sku")["sales"].max()
    anomaly_sku = df_max[df_max < config.threshold_max_sales].index.values
    logging.warning(
        f"Found {len(anomaly_sku)} SKU with max sales < {config.threshold_max_sales}: {anomaly_sku[:5]},..."
    )

    df = df[~df["sku"].isin(anomaly_sku)]
    return df


def drop_anomaly_sales(df: pd.DataFrame, config: Config, num_iter: int = 2) -> pd.DataFrame:
    """
    Remove sales that deviates from distribution of SKU sales. Take last year for mean, std estimation
    Run num_iter loops, each time drop outliers from previous run and recompute sample mean, std

    :param num_iter: number of Z-score thresholding loops
    """
    # TODO: is it neccesary?
    threshold = config.threshold_z_score
    df_current = df
    for i in range(num_iter):
        df_stats = (
            df_current.sort_values(["sku", "date"])
            .set_index("date")
            .groupby("sku")["sales"]
            .rolling("365D", min_periods=10, closed="left")
            .agg(["mean", "std"])
            .reset_index()
        )

        df_current = df_current.merge(df_stats, on=["sku", "date"], how="left")
        df_current["z_score"] = (df_current["sales"] - df_current["mean"]) / df_current["std"]
        df_current["z_score"] = df_current["z_score"].abs().fillna(0)

        count_to_drop = df_current[df_current["z_score"] >= threshold].shape[0]
        logging.warning(f"Drop {count_to_drop} rows on {i}th iteration from {num_iter} with threshold {threshold}")

        df = df.merge(
            df_current[df_current["z_score"] >= threshold][["sku", "date"]], on=["sku", "date"], how="left_anti"
        )

        df_current = df_current[df_current["z_score"] < threshold]
        df_current = df_current.drop(columns=["mean", "std", "z_score"])
    return df


def drop_new_sku(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """
    Remove SKU with too small amout of data
    """
    df_counts = df.groupby("sku").size()
    new_sku = df_counts[df_counts < config.threshold_count_rows].index.values
    logging.warning(f"Found {len(new_sku)} SKU with < {config.threshold_count_rows} rows of data: {new_sku[:5]},...")

    df = df[~df["sku"].isin(new_sku)]
    return df


def forward_fill_data(df: pd.DataFrame, config: Config, fillna: bool = True) -> pd.DataFrame:
    """
    For each SKU add rows with empty sales to the max date in dataset
    :param fillna: if True, fill empty sales with 0
    """
    sku_max_dates = df.groupby("sku")["date"].max()

    if config.max_date is not None:
        max_date = config.max_date
    else:
        max_date = sku_max_dates.max()

    sku_to_ffill = sku_max_dates[sku_max_dates < max_date].index.values
    df_to_add = pd.DataFrame({"date": [max_date] * len(sku_to_ffill), "sku": sku_to_ffill})

    if fillna:
        df_to_add["sales"] = [0.0] * len(sku_to_ffill)
    else:
        df_to_add["sales"] = [np.nan] * len(sku_to_ffill)

    df = pd.concat([df, df_to_add], axis=0)
    df = df.sort_values(by=["sku", "date"])
    return df


def explode_frequency(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """
    Transform data to evenly-spaced time series
    """
    if config.frequency == "day":
        pd_freq_name = "D"
    elif config.frequency == "week":
        pd_freq_name = "W-SUN"
    elif config.frequency == "month":
        pd_freq_name = "MS"
    else:
        raise NotImplementedError(f"Unexpected frequency! Expected 'day', 'week' or 'month', got {config.frequency}")

    result = df.set_index("date").groupby("sku").resample(pd_freq_name, label="left", closed="left").sum().reset_index()

    # if config.is_dense_data:
    #    result = result.groupby("sku").asfreq(pd_freq_name).fillna(0).reset_index()

    return result
