"""Generate and load daily sales data.

Real sales data is private, so the project ships a generator. Each product has
a base demand, a slow upward trend, a weekly pattern (busier weekends), a yearly
season and occasional promotions that lift sales. Units are Poisson counts.
"""

from os import PathLike

import numpy as np
import pandas as pd

# Average units sold per day at the start of the series.
PRODUCTS = {"coffee_beans": 120, "notebook": 60, "usb_cable": 35}

# Multiplier for Monday ... Sunday.
WEEKDAY_FACTOR = np.array([0.95, 0.95, 1.00, 1.00, 1.10, 1.30, 1.15])
PROMO_LIFT = 1.4


def generate_sales(
    start: str = "2022-01-01",
    days: int = 3 * 365,
    seed: int = 42,
    promo_rate: float = 0.05,
) -> pd.DataFrame:
    """Return a DataFrame with columns date, product, units and promo."""
    if days < 400:
        raise ValueError("Use at least 400 days so the yearly pattern can be learned")

    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=days, freq="D")
    t = np.arange(days)

    frames = []
    for index, (product, base) in enumerate(PRODUCTS.items()):
        trend = 1 + 0.0004 * t
        weekly = WEEKDAY_FACTOR[dates.dayofweek]
        yearly = 1 + 0.2 * np.sin(2 * np.pi * (dates.dayofyear - 80) / 365.25 + 0.5 * index)
        promo = rng.random(days) < promo_rate
        noise = rng.lognormal(0, 0.06, size=days)
        mean = base * trend * weekly * yearly * np.where(promo, PROMO_LIFT, 1.0) * noise
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "product": product,
                    "units": rng.poisson(mean),
                    "promo": promo.astype(int),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def load_sales(path: str | PathLike[str]) -> pd.DataFrame:
    """Read a CSV with columns date, product, units (and optionally promo)."""
    df = pd.read_csv(path, parse_dates=["date"])
    missing = [c for c in ("date", "product", "units") if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    if "promo" not in df.columns:
        df["promo"] = 0
    return df.sort_values(["product", "date"]).reset_index(drop=True)
