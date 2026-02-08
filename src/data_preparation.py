"""
Class Dataset that contains dataset and data preparation
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

    def drop_anomaly_sku(self) -> None:
        """
        Remove SKU with maximum sales <= threshold_max_sales
        """
        df_max = self.data.groupby("sku")["sales"].max()
        anomaly_sku = df_max[df_max < self.threshold_max_sales].index.values
        logging.info(f"Found {len(anomaly_sku)} SKU with max sales < {self.threshold_max_sales}: {anomaly_sku[:5]},...")

        self.data = self.data[~self.data["sku"].isin(anomaly_sku)]

    def drop_anomaly_sales(self, num_iter: int = 2):
        """
        Remove sales that deviates from distribution of SKU sales. Take last year for mean, std estimation
        Run num_iter loops, each time drop outliers from previous run and recompute sample mean, std

        :param num_iter: number of Z-score thresholding loops
        """

        def _add_z_score(df: pd.DataFrame) -> pd.DataFrame:
            df["mean_year"] = np.nan
            df["std_year"] = np.nan
            df["z_score_year"] = np.nan

            for sku, df_sku in df.groupby("sku"):
                if df_sku.shape[0] < 10 or df_sku["date"].max() - df_sku["date"].min() < pd.Timedelta(days=365):
                    continue

                index = df[df["sku"] == sku].index.values
                df.loc[index, "mean_year"] = df_sku.set_index("date").rolling(window="365D")["sales"].mean().values
                df.loc[index, "std_year"] = df_sku.set_index("date").rolling(window="365D")["sales"].std().values
                df.loc[index, "z_score_year"] = (df_sku["sales"] - df_sku["mean_year"]) / df_sku["std_year"]

            return df

        self.data = self.data.sort_values(["sku", "date"])

        df: pd.DataFrame = None
        for i in range(num_iter):
            df = _add_z_score(df if df is not None else self.data)

            count_to_drop = df[df["z_score_year"] >= self.threshold_z_score].shape[0]
            logging.info(
                f"Drop {count_to_drop} rows on {i}th iteration from {num_iter} with threshold {self.threshold_z_score}"
            )
            df = df[df["z_score_year"] < self.threshold_z_score]

        self.data = df.drop(columns=["mean_year", "std_year", "z_score_year"])

    def drop_new_sku(self):
        """
        Remove SKU with too small amout of data
        :param threshold_count: threshold for count of rows
        """
        df_counts = self.data.groupby("sku").size()
        new_sku = df_counts[df_counts < self.threshold_count]
        logging.info(f"Found {len(new_sku)} SKU with < {self.threshold_count} rows of data: {new_sku[:5]},...")

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
        self.data = self.data.sort_values(["sku", "date"])

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

        df_lags = self.data.sort_values(["sku", "date"]).groupby("sku")["sales"].shift(lags).add_prefix("sales_lag_")

        self.data = self.data.merge(df_lags, on=["sku", "date"], how="left")

    def add_date_features(self):
        """
        Add number of week in year
        """
        self.data["week_num"] = self.data["date"].dt.isocalendar().week

        # TODO: add amout of weeks since first and last sale by SKU
