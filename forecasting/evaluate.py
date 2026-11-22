"""Error metrics and a rolling backtest that never looks into the future."""

from collections.abc import Callable

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from .models import Forecaster


def mae(actual: ArrayLike, predicted: ArrayLike) -> float:
    return float(np.mean(np.abs(np.asarray(actual) - np.asarray(predicted))))


def rmse(actual: ArrayLike, predicted: ArrayLike) -> float:
    return float(np.sqrt(np.mean((np.asarray(actual) - np.asarray(predicted)) ** 2)))


def wape(actual: ArrayLike, predicted: ArrayLike) -> float:
    """Weighted absolute percentage error: total error divided by total sales.

    Unlike MAPE, it stays defined when some days have zero sales.
    """
    actual_values = np.asarray(actual, dtype=float)
    total = actual_values.sum()
    if total == 0:
        raise ValueError("WAPE is undefined when total actual sales are zero")
    return float(np.abs(actual_values - np.asarray(predicted)).sum() / total)


def rolling_backtest(
    product_df: pd.DataFrame,
    factory: Callable[[], Forecaster],
    horizon: int = 28,
    folds: int = 4,
) -> pd.DataFrame:
    """Forecast `folds` consecutive windows of `horizon` days, training only on the past.

    Returns a DataFrame with date, actual, predicted and fold.
    """
    if horizon <= 0 or folds <= 0:
        raise ValueError("horizon and folds must be positive")
    data = product_df.sort_values("date").reset_index(drop=True)
    if len(data) < horizon * folds + 365:
        raise ValueError("Not enough history: need at least one year before the first test window")

    rows = []
    for fold in range(folds):
        cut = len(data) - (folds - fold) * horizon
        train, test = data.iloc[:cut], data.iloc[cut : cut + horizon]
        model = factory().fit(train)
        predicted = model.predict(test[["date", "promo"]])
        rows.append(
            pd.DataFrame(
                {
                    "date": test["date"].to_numpy(),
                    "actual": test["units"].to_numpy(),
                    "predicted": predicted,
                    "fold": fold + 1,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def summarize(backtest: pd.DataFrame) -> dict[str, float]:
    """MAE, RMSE and WAPE of a backtest result."""
    return {
        "MAE": mae(backtest["actual"], backtest["predicted"]),
        "RMSE": rmse(backtest["actual"], backtest["predicted"]),
        "WAPE": wape(backtest["actual"], backtest["predicted"]),
    }
