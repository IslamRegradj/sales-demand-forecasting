"""Compare models and recommend stock levels.

python -m forecasting.run
python -m forecasting.run --csv my_sales.csv --lead-time 10
"""

import argparse
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from .data import generate_sales, load_sales
from .evaluate import rolling_backtest, summarize
from .inventory import reorder_point, safety_stock
from .models import MAX_HORIZON, available_models


def _plot(product: str, backtest: pd.DataFrame, out_dir: Path) -> Path | None:
    """Save a forecast-versus-actual plot of the last backtest window (needs matplotlib)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    last = backtest[backtest["fold"] == backtest["fold"].max()]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(last["date"], last["actual"], label="actual", color="#222222")
    ax.plot(last["date"], last["predicted"], label="forecast", color="#0A6C86")
    ax.set_title(f"{product}: forecast vs actual (last {len(last)} days)")
    ax.set_ylabel("units per day")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    path = out_dir / f"forecast_{product}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare forecasting models and plan stock.")
    parser.add_argument("--csv", help="your own sales CSV (date, product, units, optional promo)")
    parser.add_argument(
        "--horizon", type=int, default=28, help=f"days to forecast (max {MAX_HORIZON})"
    )
    parser.add_argument("--folds", type=int, default=4, help="number of backtest windows")
    parser.add_argument("--lead-time", type=int, default=7, help="supplier lead time in days")
    parser.add_argument("--service-level", type=float, default=0.95)
    parser.add_argument("--out", default="results", help="folder for the metrics table and plots")
    args = parser.parse_args(argv)

    sales = load_sales(args.csv) if args.csv else generate_sales()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    models = available_models()
    print(f"Models compared: {', '.join(models)}")
    if "prophet" not in models:
        print("(Prophet is not installed: pip install prophet to include it)")

    metric_rows, stock_rows = [], []
    for product, history in sales.groupby("product"):
        history = history[["date", "units", "promo"]].reset_index(drop=True)
        results = {}
        for name, factory in models.items():
            backtest = rolling_backtest(history, factory, args.horizon, args.folds)
            results[name] = backtest
            metric_rows.append({"product": product, "model": name, **summarize(backtest)})

        best = min(results, key=lambda n: summarize(results[n])["WAPE"])
        errors = results[best]["actual"] - results[best]["predicted"]
        error_std = float(errors.std())

        future_dates = pd.date_range(
            history["date"].iloc[-1] + pd.Timedelta(days=1), periods=args.horizon
        )
        future = pd.DataFrame({"date": future_dates, "promo": 0})  # no promotion planned
        forecast = models[best]().fit(history).predict(future)

        stock_rows.append(
            {
                "product": product,
                "model": best,
                "forecast_next_7_days": int(round(float(np.sum(forecast[:7])))),
                "safety_stock": math.ceil(
                    safety_stock(error_std, args.lead_time, args.service_level)
                ),
                "reorder_point": reorder_point(
                    forecast, error_std, args.lead_time, args.service_level
                ),
            }
        )
        _plot(str(product), results[best], out_dir)

    metrics = pd.DataFrame(metric_rows)
    stock = pd.DataFrame(stock_rows)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    stock.to_csv(out_dir / "stock_recommendations.csv", index=False)

    pd.options.display.float_format = "{:.3f}".format
    print("\nBacktest results (lower is better):")
    print(metrics.to_string(index=False))
    print(
        f"\nStock recommendations (lead time {args.lead_time} days, "
        f"service level {args.service_level:.0%}):"
    )
    print(stock.to_string(index=False))
    print(f"\nFiles saved in {out_dir}/")


if __name__ == "__main__":
    main()
