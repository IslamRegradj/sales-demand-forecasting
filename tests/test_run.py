import pandas as pd

from forecasting.data import generate_sales
from forecasting.run import main


def test_command_line_run_writes_metrics_and_stock_files(tmp_path, capsys):
    main(["--out", str(tmp_path), "--folds", "2", "--lead-time", "5"])

    output = capsys.readouterr().out
    assert "Backtest results" in output
    assert "Stock recommendations" in output

    stock = pd.read_csv(tmp_path / "stock_recommendations.csv")
    assert len(stock) == 3
    assert (stock["reorder_point"] > stock["safety_stock"]).all()
    assert (tmp_path / "metrics.csv").exists()
    assert len(list(tmp_path.glob("forecast_*.png"))) == 3


def test_command_line_run_accepts_a_csv_file(tmp_path):
    csv_path = tmp_path / "sales.csv"
    sales = generate_sales(seed=9)
    sales[sales["product"] == "notebook"].drop(columns="promo").to_csv(csv_path, index=False)

    main(["--csv", str(csv_path), "--out", str(tmp_path / "out"), "--folds", "2"])

    stock = pd.read_csv(tmp_path / "out" / "stock_recommendations.csv")
    assert list(stock["product"]) == ["notebook"]
