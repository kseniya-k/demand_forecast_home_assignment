"""
Simple interface for fit and predict
"""

from enum import Enum
from typing import List, Tuple

import numpy as np
import pandas as pd
from prophet import Prophet


class ModelType(Enum):
    sarima: str = "SARIMA"
    hw: str = "holt-winters"
    prophet: str = "Prohpet"
    lightgbm: str = "LightGBM"


class Model:
    type: ModelType
    heuristics_min_rows: int = 10
    ci_alphas: List[float] = [0.2, 0.5, 0.8]
    horizon_days: int = 8 * 7

    def fit_predict(self, train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
        """
        Fit a set of models for each SKU on train, predict on test
        """
        ci_alphas = self.ci_alphas

        forecast_all: List[pd.DataFrame] = []
        for sku, df_sku in train.groupby("sku"):
            if df_sku.shape[0] < self.heuristics_min_rows:
                const_forecast = df_sku["sales"].mean()

                # compute confidence intervals from residuals for leave-one-out strategy
                residuals = []
                for ind, row in df_sku.iterrows():
                    loo_forecast = df_sku[df_sku.index.values != ind]["sales"].mean()
                    loo_residual = row["sales"] - loo_forecast

                    residuals.append(loo_residual)

                ci_lengths = np.percentile(residuals, [x * 100 for x in ci_alphas])
                alpha_ci_length = dict(zip(ci_alphas, ci_lengths))

                test_start_date = df_sku["date"].iloc[-1]
                forecast = pd.DataFrame(
                    {
                        "ds": [test_start_date + pd.Timedelta(days=i) for i in range(1, self.horizon_days + 1)],
                        "yhat": const_forecast * self.horizon_days / 7,
                    }
                )

                # TODO: add for all alphas
                forecast["yhat_lower"] = forecast["yhat"] - alpha_ci_length[ci_alphas[-1]]
                forecast["yhat_upper"] = forecast["yhat"] + alpha_ci_length[ci_alphas[-1]]
                forecast["model_type"] = "const_mean"

                # for alpha in ci_alphas:
                #    forecast[f"yhat_lower_{alpha}"] = forecast["yhat"] - alpha_ci_length[alpha]
                #    forecast[f"yhat_upper_{alpha}"] = forecast["yhat"] + alpha_ci_length[alpha]
            else:
                # TODO: add for all alphas
                model = Prophet(interval_width=ci_alphas[-1])
                model.add_country_holidays(country_name="IL")

                train_prophet = df_sku[["date", "sales"]]
                train_prophet.columns = ["ds", "y"]
                model.fit(train_prophet)

                future = model.make_future_dataframe(periods=self.horizon_days)
                forecast = model.predict(future)
                forecast = forecast[forecast["ds"] > train_prophet["ds"].max()]
                forecast["model_type"] = "prophet"

            for c in ["yhat", "yhat_lower", "yhat_upper"]:
                forecast[c] = forecast[c].apply(lambda x: max(x, 0))

            forecast["sku"] = sku
            forecast_all.append(forecast)

        forecast_all_pd = pd.concat(forecast_all)
        test_w_forecast = test.copy()
        test_w_forecast = test.merge(
            forecast_all_pd[["ds", "sku", "yhat", "yhat_lower", "yhat_upper"]].rename(columns={"ds": "date"}),
            on=["date", "sku"],
            how="left",
        )
        return test_w_forecast


def calc_quality(
    df: pd.DataFrame, predict_colname: str, hist_bias_borders: Tuple[float, float] = (-0.2, 0.2)
) -> pd.DataFrame:
    """
    Compute metrics: WAPE, WAPE max, Bias, 3-bin relative error histogram
    :param df: input dataframe, should contain columns 'sales', `predict_colname`
    :param hist_bias_borders: left and right borders of relative error histogram, should be a number between 0 and 1
    """
    df["target"] = df["sales"].fillna(0).apply(lambda x: max(x, 0))

    df["err"] = df[predict_colname] - df["target"]
    df["abs_err"] = df["err"].abs()
    df["max_val"] = df[["err", "target"]].max(axis=1)
    df["rel_err"] = df["err"] / df["target"]

    result = pd.DataFrame()
    result["wape"] = [df["abs_err"].sum() / df["target"].sum() * 100]
    result["wape_max"] = [df["abs_err"].sum() / df["max_val"].sum() * 100]
    result["bias"] = [df["err"].sum() / df["target"].sum() * 100]

    count = df.shape[0]
    result[f"hist_bias_-inf_{hist_bias_borders[0]}"] = [df[df["rel_err"] < hist_bias_borders[0]].shape[0] / count * 100]
    result[f"hist_bias_{hist_bias_borders[0]}_{hist_bias_borders[1]}"] = [
        df[df["rel_err"].between(hist_bias_borders[0], hist_bias_borders[1])].shape[0] / count * 100
    ]
    result[f"hist_bias_{hist_bias_borders[1]}_inf"] = [df[df["rel_err"] > hist_bias_borders[1]].shape[0] / count * 100]
    return result
