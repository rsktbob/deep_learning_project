"""Download and preprocess Taiwan stock data into FinRL-compatible frames."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Callable

import pandas as pd


LogFn = Callable[[str], None]


def _log(log: LogFn | None, message: str) -> None:
    if log:
        log(message)


def download_yahoo_data(
    tickers: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Download Yahoo data without relying on FinRL's outdated column parser."""
    import yfinance as yf

    frames = []
    for ticker in tickers:
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            auto_adjust=False,
            progress=False,
        )
        if data.empty:
            continue
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data = data.reset_index()
        date_column = data.columns[0]
        close_column = "Adj Close" if "Adj Close" in data.columns else "Close"
        normalized = pd.DataFrame(
            {
                "date": pd.to_datetime(data[date_column]).dt.strftime("%Y-%m-%d"),
                "open": data["Open"].astype(float),
                "high": data["High"].astype(float),
                "low": data["Low"].astype(float),
                "close": data[close_column].astype(float),
                "volume": data["Volume"].astype(float),
                "tic": ticker,
            }
        )
        normalized["day"] = pd.to_datetime(normalized["date"]).dt.dayofweek
        frames.append(normalized)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def prepare_finrl_data(
    tickers: list[str],
    start_date: str,
    end_date: str,
    *,
    log: LogFn | None = None,
) -> pd.DataFrame:
    """Download Taiwan stocks from Yahoo and add FinRL technical indicators."""
    try:
        from finrl.config import INDICATORS
        from finrl.meta.preprocessor.preprocessors import FeatureEngineer
    except ImportError as exc:
        raise RuntimeError(
            "缺少 FinRL 或 yfinance。請先執行 pip install -r requirements.txt"
        ) from exc

    _log(log, f"下載 {len(tickers)} 檔台股資料：{start_date} 至 {end_date}")
    raw = download_yahoo_data(tickers, start_date, end_date)
    if raw.empty:
        raise RuntimeError("Yahoo Finance 沒有回傳資料，請檢查日期與股票代碼。")

    downloaded = set(raw["tic"].unique())
    missing = sorted(set(tickers).difference(downloaded))
    if missing:
        raise RuntimeError(f"以下股票沒有資料：{', '.join(missing)}")

    _log(log, "計算 MACD、RSI、布林通道、VIX 與 turbulence")
    engineer = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=INDICATORS,
        use_vix=False,
        use_turbulence=True,
        user_defined_feature=False,
    )
    processed = engineer.preprocess_data(raw)
    if processed.empty:
        raise RuntimeError("技術指標處理後沒有可用資料。")

    vix = download_yahoo_data(["^VIX"], start_date, end_date)[["date", "close"]]
    vix = vix.rename(columns={"close": "vix"}).drop_duplicates("date")
    processed = processed.merge(vix, on="date", how="left").sort_values(["date", "tic"])
    processed["vix"] = processed["vix"].ffill().bfill().fillna(0)

    trading_dates = processed["date"].drop_duplicates().tolist()
    panel = pd.DataFrame(
        itertools.product(trading_dates, tickers),
        columns=["date", "tic"],
    ).merge(processed, on=["date", "tic"], how="left")
    panel = panel.sort_values(["date", "tic"])

    price_columns = ["open", "high", "low", "close", "volume"]
    panel[price_columns] = panel.groupby("tic")[price_columns].transform(
        lambda values: values.ffill().bfill()
    )
    panel = panel.fillna(0)
    panel = panel[panel.groupby("date")["tic"].transform("nunique") == len(tickers)]
    panel = panel.sort_values(["date", "tic"]).reset_index(drop=True)

    _log(
        log,
        f"完成資料處理：{panel['date'].nunique()} 個交易日、{len(INDICATORS)} 個技術指標",
    )
    return panel


def split_finrl_data(
    processed: pd.DataFrame,
    trade_start: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a processed panel and restore FinRL's per-day integer index."""
    train = processed[processed["date"] < trade_start].copy()
    trade = processed[processed["date"] >= trade_start].copy()
    if train.empty or trade.empty:
        raise ValueError("訓練區間與回測區間都必須至少包含一個交易日。")

    for frame in (train, trade):
        date_index = {date: index for index, date in enumerate(frame["date"].unique())}
        frame.index = frame["date"].map(date_index)
        frame.index.name = ""
    return train, trade


def save_finrl_data(
    train: pd.DataFrame,
    trade: pd.DataFrame,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Save a FinRL train/trade pair."""
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train_data.csv"
    trade_path = output_dir / "trade_data.csv"
    train.to_csv(train_path)
    trade.to_csv(trade_path)
    return train_path, trade_path
