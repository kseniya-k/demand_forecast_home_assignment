import datetime
import logging
from typing import Union

import numpy as np
import pandas as pd


class Dataset:
    """
    Class for data store and preparation
    """

    data: pd.DataFrame
    start_date: Union[datetime.datetime, str]
    end_date: Union[datetime.datetime, str]

    threshold_max_sales: float = 0  # threshold for maximum sales per SLU
    threshold_z_score: int = 10  # threshold for maximum Z-score
    threshold_count: int = 2  # threshold for count of rows

    def drop_anomaly_sku(self):
        """
        Remove SKU with maximum sales <= threshold_max_sales
        """
        df_max = self.data.groupby("sku").max()
        anomaly_sku = df_max[df_max["sales"] < self.threshold_max_sales].index.values
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

                index = df[df["sku"] == sku]
                df.loc[index, "mean_year"] = df_sku.set_index("date").rolling(window="365D")["sales"].mean().values
                df.loc[index, "std_year"] = df_sku.set_index("date").rolling(window="365D")["sales"].std().values
                df.loc[index, "z_score_year"] = (df_sku["sales"] - df_sku["mean_year"]) / df_sku["std_year"]

            return df

        df = None
        for i in range(num_iter):
            df = _add_z_score(df or self.data)

            count_to_drop = df[df["z_score_year"] >= self.threshold_z_score].shape[0]
            logging.info(
                f"Drop {count_to_drop} rows on {i}th iteration from {num_iter} with threshold {self.threshold_z_score}"
            )
            df = df[df["z_score_year"] < self.threshold_z_score]

        self.data = df

        def drop_new_sku(self):
            """
            Remove SKU with too small amout of data
            :param threshold_count: threshold for count of rows
            """
            df_counts = df.groupby["sku"].size()
            new_sku = df_counts[df_counts < self.threshold_count]
            logging.info(f"Found {len(new_sku)} SKU with < {self.threshold_count} rows of data: {new_sku[:5]},...")

            self.data = self.data[~self.data["sku"].isin(new_sku)]
