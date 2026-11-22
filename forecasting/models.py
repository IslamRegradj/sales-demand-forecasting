"""Forecasting models with one shared interface.

Every model has:
- fit(history)   history: DataFrame with date, units, promo (one product, one row per day)
- predict(future) future: DataFrame with date, promo; returns an array of units

Planned promotions are assumed to be known in advance, so `promo` is given for
the future dates too.
"""

import logging
from collections.abc import Callable
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

MAX_HORIZON = 28  # the smallest lag used by GradientBoostingForecaster


class Forecaster(Protocol):
    """What the backtest needs from a model."""

    name: str

    def fit(self, history: pd.DataFrame) -> "Forecaster": ...

    def predict(self, future: pd.DataFrame) -> np.ndarray: ...


def _check_daily(dates: pd.Series | pd.DatetimeIndex) -> None:
    if len(dates) > 1 and not (pd.Series(dates).diff().dropna() == pd.Timedelta(days=1)).all():
        raise ValueError("The dates must be consecutive days with no gaps")


class WeekdayAverage:
    """Baseline: the average of the last 4 same weekdays."""

    name = "weekday-average"

    def fit(self, history: pd.DataFrame) -> "WeekdayAverage":
        _check_daily(history["date"])
        self.history = history.tail(28).set_index("date")["units"]
        return self

    def predict(self, future: pd.DataFrame) -> np.ndarray:
        by_weekday = self.history.groupby(self.history.index.dayofweek).mean()
        return np.asarray(future["date"].dt.dayofweek.map(by_weekday), dtype=float)


class GradientBoostingForecaster:
    """Gradient boosting on calendar features, promotions and lagged sales.

    All lags are at least 28 days, so one model can forecast up to 28 days
    ahead directly, without feeding its own predictions back in.
    """

    name = "gradient-boosting"
    LAGS = (28, 35, 364)

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state

    def _features(self, dates: pd.DatetimeIndex, promo: np.ndarray) -> pd.DataFrame:
        extended = self.series.reindex(self.full_index)
        table = pd.DataFrame(index=self.full_index)
        for lag in self.LAGS:
            table[f"lag_{lag}"] = extended.shift(lag)
        table["rolling_28"] = extended.rolling(28, min_periods=14).mean().shift(28)
        table = table.loc[dates]
        table["dayofweek"] = dates.dayofweek
        table["month"] = dates.month
        angle = 2 * np.pi * dates.dayofyear / 365.25
        table["year_sin"] = np.sin(angle)
        table["year_cos"] = np.cos(angle)
        table["promo"] = np.asarray(promo, dtype=float)
        return table

    def fit(self, history: pd.DataFrame) -> "GradientBoostingForecaster":
        _check_daily(history["date"])
        self.series = history.set_index("date")["units"].astype(float)
        last = self.series.index[-1]
        self.full_index = pd.date_range(self.series.index[0], last + pd.Timedelta(days=MAX_HORIZON))

        dates = pd.DatetimeIndex(history["date"])
        X = self._features(dates, history["promo"].to_numpy())
        keep = X["lag_28"].notna().to_numpy()
        self.model = HistGradientBoostingRegressor(
            loss="poisson",
            max_iter=300,
            learning_rate=0.05,
            max_depth=5,
            random_state=self.random_state,
        )
        self.model.fit(X[keep], self.series.to_numpy()[keep])
        self.columns = list(X.columns)
        return self

    def predict(self, future: pd.DataFrame) -> np.ndarray:
        dates = pd.DatetimeIndex(future["date"])
        if len(dates) > MAX_HORIZON:
            raise ValueError(f"The horizon is limited to {MAX_HORIZON} days")
        expected_start = self.series.index[-1] + pd.Timedelta(days=1)
        if dates[0] != expected_start:
            raise ValueError("The forecast must start the day after the last observed day")
        _check_daily(future["date"])

        X = self._features(dates, future["promo"].to_numpy())[self.columns]
        return np.asarray(np.clip(self.model.predict(X), 0, None), dtype=float)


class ProphetForecaster:
    """Prophet with weekly and yearly seasonality and a promotion regressor.

    Needs the `prophet` package (pip install prophet).
    """

    name = "prophet"

    def fit(self, history: pd.DataFrame) -> "ProphetForecaster":
        try:
            from prophet import Prophet
        except ImportError as error:
            raise ImportError("Install Prophet with: pip install prophet") from error

        logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
        logging.getLogger("prophet").setLevel(logging.WARNING)
        self.model = Prophet(
            weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False
        )
        self.model.add_regressor("promo")
        frame = pd.DataFrame(
            {
                "ds": history["date"],
                "y": history["units"].astype(float),
                "promo": history["promo"].astype(float),
            }
        )
        self.model.fit(frame)
        return self

    def predict(self, future: pd.DataFrame) -> np.ndarray:
        frame = pd.DataFrame({"ds": future["date"], "promo": future["promo"].astype(float)})
        forecast = self.model.predict(frame)["yhat"].to_numpy()
        return np.asarray(np.clip(forecast, 0, None), dtype=float)


def prophet_available() -> bool:
    try:
        import prophet  # noqa: F401
    except ImportError:
        return False
    return True


def available_models() -> dict[str, Callable[[], Forecaster]]:
    """Model factories to compare. Prophet is included only when installed."""
    models: dict[str, Callable[[], Forecaster]] = {
        WeekdayAverage.name: WeekdayAverage,
        GradientBoostingForecaster.name: GradientBoostingForecaster,
    }
    if prophet_available():
        models[ProphetForecaster.name] = ProphetForecaster
    return models
