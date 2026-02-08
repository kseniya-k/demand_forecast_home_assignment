"""
Class Dataset contains dataset itself and parameters and metholds for data preparation
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd


class Dataset:
    """
    Class for data store and preparation
    """

    data: pd.DataFrame
    # start_date: Union[datetime.datetime, str]
    # end_date: Union[datetime.datetime, str]

    frequency: str = "week"

    threshold_max_sales: float = 0  # threshold for maximum sales per SLU
    threshold_z_score: int = 10  # threshold for maximum Z-score
    threshold_count: int = 2  # threshold for count of rows

    def __init__(
        self,
        data: pd.DataFrame,
        # start_date: Optional[Union[datetime.datetime, str]] = None,
        # end_date: Optional[Union[datetime.datetime, str]] = None,
        frequency: Optional[str] = None,
        threshold_max_sales: Optional[int] = None,
        threshold_z_score: Optional[int] = None,
        threshold_count: Optional[int] = None,
    ):
        self.data = data
        # self.start_date = start_date or self.start_date
        # self.end_date = end_date or self.end_date
        self.frequency = frequency or self.frequency
        self.threshold_max_sales = threshold_max_sales or self.threshold_max_sales
        self.threshold_z_score = threshold_z_score or self.threshold_z_score
        self.threshold_count = threshold_count or self.threshold_count

    def sort_fillna(self):
        self.data = self.data.sort_values(["sku", "date"])
        self.data["sales"] = self.data["sales"].fillna(0)
        self.data = self.data.dropna(how="any")

    def drop_anomaly_sku(self):
        """
        Remove SKU with maximum sales <= threshold_max_sales
        """
        df_max = self.data.groupby("sku")["sales"].max()
        anomaly_sku = df_max[df_max < self.threshold_max_sales].index.values
        logging.warning(
            f"Found {len(anomaly_sku)} SKU with max sales < {self.threshold_max_sales}: {anomaly_sku[:5]},..."
        )

        self.data = self.data[~self.data["sku"].isin(anomaly_sku)]

    def drop_anomaly_sales(self, num_iter: int = 2):
        """
        Remove sales that deviates from distribution of SKU sales. Take last year for mean, std estimation
        Run num_iter loops, each time drop outliers from previous run and recompute sample mean, std

        :param num_iter: number of Z-score thresholding loops
        """
        # TODO: is it neccesary?
        threshold = self.threshold_z_score
        df_current = self.data
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

            self.data = self.data.merge(
                df_current[df_current["z_score"] >= threshold][["sku", "date"]], on=["sku", "date"], how="left_anti"
            )

            df_current = df_current[df_current["z_score"] < threshold]
            df_current = df_current.drop(columns=["mean", "std", "z_score"])

    def drop_new_sku(self):
        """
        Remove SKU with too small amout of data
        :param threshold_count: threshold for count of rows
        """
        df_counts = self.data.groupby("sku").size()
        new_sku = df_counts[df_counts < self.threshold_count].index.values
        logging.warning(f"Found {len(new_sku)} SKU with < {self.threshold_count} rows of data: {new_sku[:5]},...")

        self.data = self.data[~self.data["sku"].isin(new_sku)]

    def explode_frequency(self):
        """
        Transform data to evenly-spaced time series
        :param frequency: timestamp name, can be "week" or "month"
        """
        if self.frequency == "week":
            pd_freq_name = "W-SUN"
        elif self.frequency == "month":
            pd_freq_name = "M"
        else:
            raise NotImplementedError(f"Unexpected frequency! Expected 'week' or 'month', got {self.frequency}")

        result = (
            self.data.set_index("date")
            .groupby("sku")
            .resample(pd_freq_name, label="left", closed="left")
            .sum()
            .reset_index()
        )

        self.data = result

    def add_weekly_stat(self, lags: List[int] = [4, 8, 53]):
        """
        Add rolling mean and sum sales
        :param lags: amount of weeks for rolling window
        """
        for lag in lags:
            num_days = lag * 7

            df_stats = (
                self.data.set_index("date")
                .groupby("sku")["sales"]
                .rolling(f"{num_days}D", min_periods=1)
                .agg(["mean", "sum"])
                .reset_index()
                .rename(columns={"mean": f"mean_{lag}w", "sum": f"sum_{lag}w"})
            )

            self.data = self.data.merge(df_stats, on=["sku", "date"], how="left")

            # self.data[f"mean_{lag}w"] = df_stats["mean"].copy()
            # self.data[f"sum_{lag}w"] = df_stats["sum"].copy()

    def add_weekly_lag(
        self,
        lags: List[int] = [
            53,
        ],
    ):
        """
        Add lag sales
        :param lags: amount of weeks or months for lag
        """
        if self.frequency == "month":
            lags = [np.ceil(x / 4.5) for x in lags]

        for lag in lags:
            self.data[f"sales_lag_{lag}"] = self.data.groupby(["sku"])["sales"].shift(lag)

    def add_date_features(self):
        """
        Add week number, month number, holiday flags
        """
        self.data["week_num"] = self.data["date"].dt.isocalendar().week
        self.data["month_num"] = self.data["date"].dt.month

        # TODO: add holidays
        # TODO: add amout of weeks since first and last sale by SKU
