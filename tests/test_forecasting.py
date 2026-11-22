import numpy as np
import pandas as pd
import pytest

from forecasting.data import PRODUCTS, generate_sales, load_sales
from forecasting.evaluate import mae, rmse, rolling_backtest, summarize, wape
from forecasting.inventory import reorder_point, safety_stock
from forecasting.models import (
    GradientBoostingForecaster,
    WeekdayAverage,
    available_models,
)


@pytest.fixture(scope="module")
def coffee():
    sales = generate_sales(seed=11)
    return sales[sales["product"] == "coffee_beans"][["date", "units", "promo"]].reset_index(
        drop=True
    )


# ---- data ----------------------------------------------------------------


def test_generated_data_has_all_products_and_valid_values():
    sales = generate_sales(seed=1)
    assert set(sales["product"]) == set(PRODUCTS)
    assert (sales["units"] >= 0).all()
    assert set(sales["promo"].unique()) <= {0, 1}
    assert sales.groupby("product")["date"].nunique().nunique() == 1


def test_generation_is_reproducible():
    assert generate_sales(seed=5).equals(generate_sales(seed=5))


def test_weekends_sell_more_than_mondays():
    sales = generate_sales(seed=2)
    by_day = sales.groupby(sales["date"].dt.dayofweek)["units"].mean()
    assert by_day[5] > by_day[0]


def test_promotions_increase_sales():
    sales = generate_sales(seed=3)
    assert sales[sales["promo"] == 1]["units"].mean() > sales[sales["promo"] == 0]["units"].mean()


def test_too_short_a_series_is_rejected():
    with pytest.raises(ValueError):
        generate_sales(days=100)


def test_load_sales_adds_a_missing_promo_column(tmp_path):
    path = tmp_path / "sales.csv"
    pd.DataFrame(
        {"date": ["2024-01-02", "2024-01-01"], "product": ["a", "a"], "units": [5, 4]}
    ).to_csv(path, index=False)
    loaded = load_sales(path)
    assert list(loaded["promo"]) == [0, 0]
    assert loaded["date"].is_monotonic_increasing


def test_load_sales_reports_missing_columns(tmp_path):
    path = tmp_path / "sales.csv"
    pd.DataFrame({"date": ["2024-01-01"], "units": [1]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Missing columns"):
        load_sales(path)


# ---- metrics -------------------------------------------------------------


def test_metrics_on_known_values():
    actual, predicted = [10, 20, 30], [12, 18, 30]
    assert mae(actual, predicted) == pytest.approx(4 / 3)
    assert rmse(actual, predicted) == pytest.approx(np.sqrt(8 / 3))
    assert wape(actual, predicted) == pytest.approx(4 / 60)


def test_wape_is_undefined_when_there_were_no_sales():
    with pytest.raises(ValueError):
        wape([0, 0], [1, 1])


# ---- models --------------------------------------------------------------


def test_weekday_average_is_exact_on_a_perfect_weekly_pattern():
    dates = pd.date_range("2024-01-01", periods=70, freq="D")
    units = dates.dayofweek.map(lambda d: 10 + d).to_numpy()
    history = pd.DataFrame({"date": dates, "units": units, "promo": 0})
    future_dates = pd.date_range(dates[-1] + pd.Timedelta(days=1), periods=7)
    future = pd.DataFrame({"date": future_dates, "promo": 0})
    predicted = WeekdayAverage().fit(history).predict(future)
    assert list(predicted) == [10 + d for d in future_dates.dayofweek]


def test_gradient_boosting_beats_the_baseline(coffee):
    boosted = summarize(rolling_backtest(coffee, GradientBoostingForecaster, 28, 3))
    baseline = summarize(rolling_backtest(coffee, WeekdayAverage, 28, 3))
    assert boosted["WAPE"] < baseline["WAPE"]
    assert boosted["WAPE"] < 0.2


def test_forecasts_are_never_negative(coffee):
    future = pd.DataFrame(
        {
            "date": pd.date_range(coffee["date"].iloc[-1] + pd.Timedelta(days=1), periods=28),
            "promo": 0,
        }
    )
    assert (GradientBoostingForecaster().fit(coffee).predict(future) >= 0).all()


def test_horizon_longer_than_28_days_is_rejected(coffee):
    future = pd.DataFrame(
        {
            "date": pd.date_range(coffee["date"].iloc[-1] + pd.Timedelta(days=1), periods=40),
            "promo": 0,
        }
    )
    with pytest.raises(ValueError, match="limited"):
        GradientBoostingForecaster().fit(coffee).predict(future)


def test_forecast_must_start_right_after_the_history(coffee):
    future = pd.DataFrame(
        {
            "date": pd.date_range(coffee["date"].iloc[-1] + pd.Timedelta(days=5), periods=7),
            "promo": 0,
        }
    )
    with pytest.raises(ValueError, match="day after"):
        GradientBoostingForecaster().fit(coffee).predict(future)


def test_gaps_in_the_history_are_rejected(coffee):
    with pytest.raises(ValueError, match="consecutive"):
        WeekdayAverage().fit(coffee.drop(index=[10, 11, 12]))


def test_prophet_is_listed_only_when_installed():
    names = set(available_models())
    assert {"weekday-average", "gradient-boosting"} <= names


# ---- backtest ------------------------------------------------------------


def test_backtest_windows_follow_the_past_without_overlap(coffee):
    result = rolling_backtest(coffee, WeekdayAverage, horizon=28, folds=3)
    assert len(result) == 3 * 28
    assert result["date"].is_unique
    assert result["date"].iloc[0] > coffee["date"].iloc[-1] - pd.Timedelta(days=3 * 28 + 1)
    assert result["date"].iloc[-1] == coffee["date"].iloc[-1]


def test_backtest_needs_enough_history():
    short = generate_sales(days=400)
    short = short[short["product"] == "notebook"][["date", "units", "promo"]]
    with pytest.raises(ValueError, match="Not enough history"):
        rolling_backtest(short, WeekdayAverage, horizon=28, folds=4)


# ---- inventory -----------------------------------------------------------


def test_safety_stock_matches_the_formula():
    # z(95%) = 1.6449; 1.6449 * 10 * sqrt(4) = 32.9
    assert safety_stock(10, 4, 0.95) == pytest.approx(32.9, abs=0.05)


def test_reorder_point_adds_lead_time_demand_and_safety_stock():
    forecast = [100] * 10
    assert reorder_point(forecast, 0, 7, 0.95) == 700
    assert reorder_point(forecast, 10, 4, 0.95) == 433  # 400 + 32.9 rounded up


def test_a_higher_service_level_needs_more_safety_stock():
    assert safety_stock(10, 5, 0.99) > safety_stock(10, 5, 0.90)


@pytest.mark.parametrize("lead,level", [(0, 0.95), (5, 1.0), (5, 0.3)])
def test_invalid_inventory_settings_are_rejected(lead, level):
    with pytest.raises(ValueError):
        safety_stock(5, lead, level)


def test_forecast_shorter_than_the_lead_time_is_rejected():
    with pytest.raises(ValueError, match="lead time"):
        reorder_point([10, 10], 1, 7)
