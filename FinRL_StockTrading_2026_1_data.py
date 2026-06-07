"""Generate FinRL train/trade CSV files from representative Taiwan stocks."""

from __future__ import annotations

import argparse
from pathlib import Path

from data_pipeline import prepare_finrl_data, save_finrl_data, split_finrl_data
from taiwan_stocks import DEFAULT_TICKERS, TAIWAN_STOCKS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and preprocess Taiwan stock data for FinRL."
    )
    parser.add_argument("--train-start", default="2018-01-01")
    parser.add_argument("--trade-start", default="2024-01-01")
    parser.add_argument("--trade-end", default="2026-06-01")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument(
        "--all-30",
        action="store_true",
        help="Use all 30 representative Taiwan stocks.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tickers = (
        [stock["symbol"] for stock in TAIWAN_STOCKS]
        if args.all_30
        else args.tickers
    )
    processed = prepare_finrl_data(
        tickers,
        args.train_start,
        args.trade_end,
        log=print,
    )
    train, trade = split_finrl_data(processed, args.trade_start)
    train_path, trade_path = save_finrl_data(train, trade, args.output_dir)
    print(f"Saved {train['date'].nunique()} training days to {train_path}")
    print(f"Saved {trade['date'].nunique()} backtest days to {trade_path}")


if __name__ == "__main__":
    main()
