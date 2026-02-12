"""
Simple interface for fit and predict
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from prophet import Prophet

from config import Config, get_frequency_params


def clip_forecast(df: pd.DataFrame, ci_levels: List[float]) -> pd.DataFrame:
    """
    Make predicts and predict intervals >= 0
    """
    for col in ["lower", "upper"]:
        for level in ci_levels:
            colname_full = f"{col}_{level}"
            df[colname_full] = df[colname_full].apply(lambda x: max(x, 0))

    df["predict"] = df["predict"].apply(lambda x: max(x, 0))
    return df


def fit_predict_const_mean(
    train: pd.DataFrame, ci_levels: Tuple[float, float, float], horizon_period: int, freq_name: str
) -> pd.DataFrame:
    """
    Compute constant predict with mean train sales for the last year. Compute confidence intervals from residuals on
    leave-one-out validation

    :param train: input dataframe
    :param ci_levels: set of three (1 - alpha) levels for confidence intervals
    :param horizon_period: horizon, measured in current time series frequency units (e.g. weeks, months, ...)
    :param freq_name: pandas name for frequency (e.g. "W-SUN" for weekly frequency)
    """
    const_forecast = train[train["date"] > train["date"].max() - pd.Timedelta(days=365)]["sales"].mean()

    # compute confidence intervals from residuals for leave-one-out strategy
    residuals = []
    for ind, row in train.iterrows():
        loo_forecast = train[train.index.values != ind]["sales"].mean()
        loo_residual = row["sales"] - loo_forecast

        residuals.append(loo_residual)

    ci_lengths = np.percentile(residuals, [x * 100 for x in ci_levels])
    level_ci_length = dict(zip(ci_levels, ci_lengths))

    test_start_date = train["date"].iloc[-1]
    forecast = pd.DataFrame(
        {
            "date": pd.date_range(start=test_start_date, periods=horizon_period + 1, freq=freq_name).tolist()[1:],
            "predict": [const_forecast] * horizon_period,
        }
    )

    for level in ci_levels:
        forecast[f"lower_{level}"] = forecast["predict"] - level_ci_length[level]
        forecast[f"upper_{level}"] = forecast["predict"] + level_ci_length[level]

    forecast["model_type"] = "const_mean"
    return forecast


def fit_predict_prophet(
    train: pd.DataFrame, ci_levels: Tuple[float, float, float], horizon_period: int, freq_name: str
) -> pd.DataFrame:
    """
    Fit and predict with Prophet model

    :param train: input dataframe
    :param ci_levels: set of three (1 - alpha) levels for confidence intervals
    :param horizon_period: horizon, measured in current time series frequency units (e.g. weeks, months, ...)
    :param freq_name: pandas name for frequency (e.g. "W-SUN" for weekly frequency)
    """
    model = Prophet()
    model.add_country_holidays(country_name="IL")

    train_prophet = train[["date", "sales"]]
    train_prophet.columns = ["ds", "y"]
    model.fit(train_prophet)

    future = model.make_future_dataframe(periods=horizon_period, freq=freq_name)

    forecast: pd.DataFrame = None  # type: ignore
    for level in ci_levels:
        model.interval_width = level
        forecast_level = model.predict(future)
        forecast_level = forecast_level[forecast_level["ds"] > train_prophet["ds"].max()]
        forecast_level = forecast_level.rename(
            columns={"ds": "date", "yhat": "predict", "yhat_lower": f"lower_{level}", "yhat_upper": f"upper_{level}"}
        )

        if forecast is None:
            forecast = forecast_level
        else:
            forecast = forecast.merge(forecast_level[["date", f"lower_{level}", f"upper_{level}"]], on="date")

    forecast["model_type"] = "prophet"
    return forecast


def fit_predict_arima(
    train: pd.DataFrame,
    ci_levels: Tuple[float, float, float],
    horizon_period: int,
    freq_name: str,
    seasonal_period: int,
    order: Tuple[int, int, int] = (2, 0, 0),
    seasonal_order: Tuple[int, int, int] = (1, 0, 1),
) -> pd.DataFrame:
    """
    Fit and predict with seasonal ARIMA model with fixed hyperparameters

    :param train: input dataframe
    :param ci_levels: set of three (1 - alpha) levels for confidence intervals
    :param horizon_period: horizon, measured in current time series frequency units (e.g. weeks, months)
    :param freq_name: pandas name for frequency (e.g. "W-SUN" for weekly frequency)
    :param seasonal_period: year seasonality in current frequency units (e.g. 52 for weekly data, 12 for monthly
    :param order: p, q, d params for ARIMA
    :param seasonal_order: P, Q, D params for seasonal ARIMA
    """
    from pmdarima import ARIMA
    from scipy.special import inv_boxcox
    from scipy.stats import boxcox

    # prepare: shift to positive, apply Box-Cox with optimized lamda
    min_sales = train["sales"].min()
    if min_sales < 1e-5:
        train["sales_positive"] = train["sales"].apply(lambda x: x + 1 if x > 0 else 1)
    else:
        train["sales_positive"] = train["sales"]

    sales_bc, lmbda = boxcox(train["sales_positive"].values)

    # fit, predict
    seasonal_order_full: Tuple[int, int, int, int] = (*seasonal_order, seasonal_period)
    model = ARIMA(order=order, seasonal_order=seasonal_order_full, with_intercept=True, suppress_warnings=True)

    is_model_learned = False
    try:
        model.fit(sales_bc)
        forecast_raw = inv_boxcox(model.predict(n_periods=horizon_period), lmbda) + (-1 if min_sales < 0 else 0)
        is_model_learned = True
    except Exception:
        logging.warning(
            f"""SARIMA with params: {order} and seasonal params: {seasonal_order_full} didn't learn!
            Error: {Exception}
            Switch to heuristics"""
        )
        forecast_raw = [-1.0] * horizon_period
        is_model_learned = False

    test_start_date = train["date"].iloc[-1]
    forecast = pd.DataFrame(
        {
            "date": pd.date_range(start=test_start_date, periods=horizon_period + 1, freq=freq_name).tolist()[1:],
            "predict": forecast_raw,
        }
    )

    if is_model_learned:
        for level in ci_levels:
            _, forecast_level_raw = model.predict(n_periods=horizon_period, return_conf_int=True, alpha=1 - level)
            lower_raw = [x[0] for x in forecast_level_raw]
            upper_raw = [x[1] for x in forecast_level_raw]

            lower = inv_boxcox(lower_raw, lmbda) + (-1 if min_sales < 0 else 0)
            upper = inv_boxcox(upper_raw, lmbda) + (-1 if min_sales < 0 else 0)

            forecast[f"lower_{level}"] = lower
            forecast[f"upper_{level}"] = upper

        forecast["model_type"] = "ARIMA"
    else:
        forecast = fit_predict_const_mean(train, ci_levels, horizon_period, freq_name)
    return forecast


# TODO: save models, move predict to separate function
def fit_predict(config: Config, train: pd.DataFrame, test: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Fit a set of models for each SKU on train, predict to config.horizon_days to future

    :param train: input data
    :param test: test data. if passed, return test joined with prediction
    """
    ci_levels = config.ci_levels
    horizon_period, freq_name, seasonal_period = get_frequency_params(config.frequency, config.horizon_days)

    forecast_all: List[pd.DataFrame] = []
    for sku, df_sku in train.groupby("sku"):
        if sku % 100 == 0:
            logging.debug(f"Handle sky {sku}")

        if df_sku.shape[0] < config.heuristics_min_rows or df_sku["sales"].std() < 1e-5 or df_sku["sales"].max() < 1e-5:
            forecast = fit_predict_const_mean(df_sku, ci_levels, horizon_period, freq_name)
        else:
            if config.model_name == "prophet":
                forecast = fit_predict_prophet(df_sku, ci_levels, horizon_period, freq_name)
            elif config.model_name == "arima":
                forecast = fit_predict_arima(df_sku, ci_levels, horizon_period, freq_name, seasonal_period)
            else:
                raise NotImplementedError(f"Unexpected model type: {config.model_name}")

        forecast["sku"] = sku
        forecast_all.append(forecast)

    forecast_all_pd = pd.concat(forecast_all)
    forecast_all_pd = clip_forecast(forecast_all_pd, list(config.ci_levels))

    if test is not None:
        result = test.copy()
        result = test.merge(
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
    else:
        result = forecast_all_pd
    return result


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
