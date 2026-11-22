"""Turn a demand forecast into a stock recommendation.

Reorder point = expected demand during the supplier lead time + safety stock.
Safety stock covers forecast errors, so a worse forecast means more stock.
"""

import math
from collections.abc import Iterable
from statistics import NormalDist


def _check(lead_time_days: float, service_level: float) -> None:
    if lead_time_days <= 0:
        raise ValueError("lead_time_days must be positive")
    if not 0.5 <= service_level < 1:
        raise ValueError("service_level must be between 0.5 and 1 (for example 0.95)")


def safety_stock(
    forecast_error_std: float, lead_time_days: float, service_level: float = 0.95
) -> float:
    """Extra units to hold so demand is covered with probability `service_level`."""
    _check(lead_time_days, service_level)
    if forecast_error_std < 0:
        raise ValueError("forecast_error_std cannot be negative")
    z = NormalDist().inv_cdf(service_level)
    return z * forecast_error_std * math.sqrt(lead_time_days)


def reorder_point(
    daily_forecast: Iterable[float],
    forecast_error_std: float,
    lead_time_days: int,
    service_level: float = 0.95,
) -> int:
    """Stock level at which a new order should be placed (in units, rounded up)."""
    _check(lead_time_days, service_level)
    forecast = list(daily_forecast)
    if len(forecast) < lead_time_days:
        raise ValueError("The forecast must cover at least the lead time")
    expected = sum(forecast[:lead_time_days])
    return math.ceil(expected + safety_stock(forecast_error_std, lead_time_days, service_level))
