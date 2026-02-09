"""
Simple interface for fit and predict
"""

from enum import Enum
from typing import List, Tuple

import numpy as np
import pandas as pd
from prophet import Prophet

from config import Config


class ModelType(Enum):
    sarima: str = "SARIMA"
    hw: str = "holt-winters"
    prophet: str = "Prohpet"
    lightgbm: str = "LightGBM"


def clip_forecast(df: pd.DataFrame, ci_levels: List[float]) -> pd.DataFrame:
    """
    Make predicts and confidense intervals >= 0
    """
    for col in ["lower", "upper"]:
        for level in ci_levels:
            colname_full = f"{col}_{level}"
            df[colname_full] = df[colname_full].apply(lambda x: max(x, 0))

    df["predict"] = df["predict"].apply(lambda x: max(x, 0))
    return df


def fit_predict(config: Config, train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """
    Fit a set of models for each SKU on train, predict on test
    """
    ci_levels = config.ci_levels

    if config.frequency == "day":
        horizon_periods = config.horizon_days
        freq_mult = 1
        freq_name = "D"
    elif config.frequency == "week":
        horizon_periods = int(config.horizon_days / 7)
        freq_mult = 7
        freq_name = "W-SUN"
    elif config.frequency == "month":
        horizon_periods = int(np.floor(config.horizon_days / 30))
        freq_mult = 31
        freq_name = "MS"

    forecast_all: List[pd.DataFrame] = []
    for sku, df_sku in train.groupby("sku"):
        if df_sku.shape[0] < config.heuristics_min_rows:
            const_forecast = df_sku["sales"].mean()

            # compute confidence intervals from residuals for leave-one-out strategy
            residuals = []
            for ind, row in df_sku.iterrows():
                loo_forecast = df_sku[df_sku.index.values != ind]["sales"].mean()
                loo_residual = row["sales"] - loo_forecast

                residuals.append(loo_residual)

            ci_lengths = np.percentile(residuals, [x * 100 for x in ci_levels])
            level_ci_length = dict(zip(ci_levels, ci_lengths))

            test_start_date = df_sku["date"].iloc[-1]
            forecast = pd.DataFrame(
                {
                    "data": [test_start_date + pd.Timedelta(days=i * freq_mult) for i in range(1, horizon_periods + 1)],
                    "predict": const_forecast * horizon_periods,
                }
            )

            for level in ci_levels:
                forecast[f"lower_{level}"] = forecast["yhat"] - level_ci_length[level]
                forecast[f"upper_{level}"] = forecast["yhat"] + level_ci_length[level]

            forecast["model_type"] = "const_mean"
        else:
            # TODO: add for all alphas
            model = Prophet()  # (interval_width=ci_levels[-1])
            model.add_country_holidays(country_name="IL")

            train_prophet = df_sku[["date", "sales"]]
            train_prophet.columns = ["ds", "y"]
            model.fit(train_prophet)

            future = model.make_future_dataframe(periods=horizon_periods, freq=freq_name)

            forecast: pd.DataFrame = None  # type: ignore
            for level in ci_levels:
                model.interval_width = level
                forecast_level = model.predict(future)
                forecast_level = forecast_level[forecast_level["ds"] > train_prophet["ds"].max()]
                forecast_level = forecast_level.rename(
                    columns={"yhat_lower": f"lower_{level}", "yhat_upper": f"upper_{level}"}
                )

                if forecast is None:
                    forecast = forecast_level
                else:
                    forecast = forecast.merge(forecast_level[["ds", f"lower_{level}", f"upper_{level}"]], on="ds")

            forecast = forecast.rename(columns={"ds": "date", "yhat": "predict"})
            forecast["model_type"] = "prophet"

        forecast["sku"] = sku
        forecast_all.append(forecast)

    forecast_all_pd = pd.concat(forecast_all)
    forecast_all_pd = clip_forecast(forecast_all_pd, list(config.ci_levels))

    test_w_forecast = test.copy()
    test_w_forecast = test.merge(
        forecast_all_pd[
            [
                "date",
                "sku",
                "predict",
                *[f"lower_{level}" for level in ci_levels],
                *[f"upper_{level}" for level in ci_levels],
            ]
        ],
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
    df[predict_colname] = df[predict_colname].fillna(0).apply(lambda x: max(x, 0))

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
