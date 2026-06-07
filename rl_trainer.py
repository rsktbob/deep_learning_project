"""Background training jobs shared by the FastAPI dashboard."""

from __future__ import annotations

import importlib.util
import math
import threading
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from algorithm_defaults import ALGORITHM_DEFAULTS
from data_pipeline import prepare_finrl_data, save_finrl_data, split_finrl_data


ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "runs"

ALGORITHMS = {
    "ppo": {"label": "PPO", "family": "On-policy", "description": "穩定的 clipped policy optimization", "defaults": ALGORITHM_DEFAULTS["ppo"]},
    "a2c": {"label": "A2C", "family": "On-policy", "description": "同步 Advantage Actor-Critic", "defaults": ALGORITHM_DEFAULTS["a2c"]},
    "td3": {"label": "TD3", "family": "Off-policy", "description": "雙 critic 的連續動作演算法", "defaults": ALGORITHM_DEFAULTS["td3"]},
    "sac": {"label": "SAC", "family": "Off-policy", "description": "最大熵 Actor-Critic", "defaults": ALGORITHM_DEFAULTS["sac"]},
    "ddpg": {"label": "DDPG", "family": "Off-policy", "description": "確定性連續動作策略", "defaults": ALGORITHM_DEFAULTS["ddpg"]},
    "ppo_lstm": {"label": "PPO + LSTM", "family": "Recurrent", "description": "保留跨交易日的 LSTM 狀態", "defaults": ALGORITHM_DEFAULTS["ppo_lstm"]},
    "ppo_mamba": {"label": "PPO + Mamba", "family": "State-space", "description": "以 selective SSM 編碼近期狀態", "defaults": ALGORITHM_DEFAULTS["ppo_mamba"]},
}

JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()


def dependency_status() -> dict[str, bool]:
    modules = ("fastapi", "finrl", "yfinance", "stable_baselines3", "sb3_contrib", "torch")
    return {name: importlib.util.find_spec(name) is not None for name in modules}


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _update(job_id: str, **values: Any) -> None:
    with JOBS_LOCK:
        JOBS[job_id].update(values)
        JOBS[job_id]["updated_at"] = _now()


def _log(job_id: str, message: str, level: str = "info") -> None:
    with JOBS_LOCK:
        JOBS[job_id]["logs"].append(
            {
                "time": datetime.now().strftime("%H:%M:%S"),
                "level": level,
                "message": message,
            }
        )
        JOBS[job_id]["logs"] = JOBS[job_id]["logs"][-200:]
        JOBS[job_id]["updated_at"] = _now()


def create_job(config: dict[str, Any]) -> dict[str, Any]:
    job_id = uuid.uuid4().hex[:12]
    job = {
        "id": job_id,
        "status": "queued",
        "stage": "等待執行",
        "progress": 0,
        "config": config,
        "logs": [],
        "result": None,
        "error": None,
        "created_at": _now(),
        "updated_at": _now(),
    }
    with JOBS_LOCK:
        JOBS[job_id] = job
    threading.Thread(target=_run_job, args=(job_id,), daemon=True).start()
    return job


def get_job(job_id: str) -> dict[str, Any] | None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        return dict(job) if job else None


def list_jobs() -> list[dict[str, Any]]:
    with JOBS_LOCK:
        jobs = [
            {
                key: value
                for key, value in job.items()
                if key not in {"result", "traceback"}
            }
            for job in JOBS.values()
        ]
    return sorted(jobs, key=lambda item: item["created_at"], reverse=True)[:20]


def _build_environment(
    data: pd.DataFrame,
    config: dict[str, Any],
    *,
    turbulence: float | None = None,
):
    from finrl.config import INDICATORS
    from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

    stock_dimension = data["tic"].nunique()
    state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension
    return StockTradingEnv(
        df=data,
        hmax=config["hmax"],
        initial_amount=config["initial_amount"],
        num_stock_shares=[0] * stock_dimension,
        buy_cost_pct=[config["transaction_cost"]] * stock_dimension,
        sell_cost_pct=[config["transaction_cost"]] * stock_dimension,
        state_space=state_space,
        stock_dim=stock_dimension,
        tech_indicator_list=INDICATORS,
        action_space=stock_dimension,
        reward_scaling=1e-4,
        turbulence_threshold=turbulence,
        risk_indicator_col="vix",
    )


def _train_standard(
    algorithm: str,
    train: pd.DataFrame,
    trade: pd.DataFrame,
    config: dict[str, Any],
    run_dir: Path,
):
    from finrl.agents.stablebaselines3.models import DRLAgent
    from stable_baselines3.common.logger import configure

    train_env = _build_environment(train, config)
    vec_env, _ = train_env.get_sb_env()
    agent = DRLAgent(env=vec_env)
    defaults = ALGORITHM_DEFAULTS[algorithm]
    common = {"learning_rate": config.get("learning_rate", defaults["learning_rate"])}
    if algorithm == "ppo":
        model_kwargs = {
            **common,
            "n_steps": config.get("n_steps", defaults["n_steps"]),
            "batch_size": config.get("batch_size", defaults["batch_size"]),
            "ent_coef": defaults["ent_coef"],
        }
    elif algorithm == "a2c":
        model_kwargs = {
            **common,
            "n_steps": config.get("n_steps", defaults["n_steps"]),
            "ent_coef": defaults["ent_coef"],
        }
    else:
        model_kwargs = {
            **common,
            "batch_size": config.get("batch_size", defaults["batch_size"]),
            "buffer_size": config.get("buffer_size", defaults["buffer_size"]),
        }
    if algorithm == "sac":
        model_kwargs["learning_starts"] = defaults["learning_starts"]
        model_kwargs["ent_coef"] = defaults["ent_coef"]

    model = agent.get_model(algorithm, model_kwargs=model_kwargs)
    model.set_logger(configure(str(run_dir / "training_log"), ["csv"]))
    trained = agent.train_model(
        model=model,
        tb_log_name=algorithm,
        total_timesteps=config["total_timesteps"],
    )
    trained.save(run_dir / f"agent_{algorithm}")
    trade_env = _build_environment(trade, config, turbulence=70)
    account_value, actions = DRLAgent.DRL_prediction(
        model=trained,
        environment=trade_env,
    )
    return account_value, actions


def _train_sequence_model(
    algorithm: str,
    train: pd.DataFrame,
    trade: pd.DataFrame,
    config: dict[str, Any],
    run_dir: Path,
):
    model_path = run_dir / f"agent_{algorithm}"
    output_dir = run_dir / "backtest"
    if algorithm == "ppo_lstm":
        from recurrent_ppo_finrl import backtest, train as train_model

        train_model(
            train,
            model_path=model_path,
            output_dir=run_dir,
            total_timesteps=config["total_timesteps"],
            n_steps=config.get("n_steps", ALGORITHM_DEFAULTS[algorithm]["n_steps"]),
            batch_size=config.get("batch_size", ALGORITHM_DEFAULTS[algorithm]["batch_size"]),
            learning_rate=config.get("learning_rate", ALGORITHM_DEFAULTS[algorithm]["learning_rate"]),
            seed=config["seed"],
            device=config["device"],
        )
    else:
        from ppo_mamba_finrl import backtest, train as train_model

        train_model(
            train,
            model_path=model_path,
            output_dir=run_dir,
            total_timesteps=config["total_timesteps"],
            n_steps=config.get("n_steps", ALGORITHM_DEFAULTS[algorithm]["n_steps"]),
            batch_size=config.get("batch_size", ALGORITHM_DEFAULTS[algorithm]["batch_size"]),
            learning_rate=config.get("learning_rate", ALGORITHM_DEFAULTS[algorithm]["learning_rate"]),
            sequence_length=config.get("sequence_length", ALGORITHM_DEFAULTS[algorithm]["sequence_length"]),
            d_model=ALGORITHM_DEFAULTS[algorithm]["d_model"],
            d_state=ALGORITHM_DEFAULTS[algorithm]["d_state"],
            n_layers=ALGORITHM_DEFAULTS[algorithm]["mamba_layers"],
            expand=ALGORITHM_DEFAULTS[algorithm]["expand"],
            conv_kernel=ALGORITHM_DEFAULTS[algorithm]["conv_kernel"],
            seed=config["seed"],
            device=config["device"],
        )
    return backtest(
        trade,
        model_path=model_path,
        output_dir=output_dir,
        turbulence_threshold=70,
        device=config["device"],
    )


def _finite(value: float) -> float:
    return round(float(value), 6) if math.isfinite(float(value)) else 0.0


def _daily_snapshots(
    actions: pd.DataFrame,
    trade: pd.DataFrame,
    tickers: list[str],
    *,
    initial_amount: float,
    transaction_cost: float,
) -> list[dict[str, Any]]:
    """Reconstruct post-trade cash, holdings, decisions, and value for each day."""
    prices = trade.pivot_table(index="date", columns="tic", values="close").sort_index()
    prices.index = prices.index.astype(str)
    prices = prices.reindex(columns=tickers)

    action_frame = actions.copy()
    if len(tickers) == 1 and tickers[0] not in action_frame.columns:
        if {"date", "actions"}.issubset(action_frame.columns):
            action_frame = action_frame.set_index("date").rename(
                columns={"actions": tickers[0]}
            )
    action_frame.index = action_frame.index.astype(str)
    action_frame = action_frame.reindex(columns=tickers, fill_value=0)

    cash = float(initial_amount)
    holdings = {ticker: 0 for ticker in tickers}
    snapshots = []

    for date, price_row in prices.iterrows():
        action_row = (
            pd.to_numeric(action_frame.loc[date], errors="coerce").fillna(0)
            if date in action_frame.index
            else pd.Series(0, index=tickers, dtype=float)
        )

        for ticker in tickers:
            quantity = int(action_row.get(ticker, 0))
            price = float(price_row[ticker])
            if quantity > 0:
                holdings[ticker] += quantity
                cash -= price * quantity * (1 + transaction_cost)
            elif quantity < 0:
                holdings[ticker] += quantity
                cash += price * abs(quantity) * (1 - transaction_cost)

        stock_value = sum(float(price_row[ticker]) * holdings[ticker] for ticker in tickers)
        total_value = cash + stock_value
        assets = []
        for ticker in tickers:
            quantity = int(action_row.get(ticker, 0))
            market_value = float(price_row[ticker]) * holdings[ticker]
            assets.append(
                {
                    "ticker": ticker,
                    "price": _finite(price_row[ticker]),
                    "shares": holdings[ticker],
                    "market_value": _finite(market_value),
                    "weight": _finite(market_value / total_value if total_value else 0),
                    "action": quantity,
                    "decision": "買進" if quantity > 0 else "賣出" if quantity < 0 else "持有",
                }
            )

        snapshots.append(
            {
                "date": date,
                "cash": _finite(cash),
                "cash_weight": _finite(cash / total_value if total_value else 0),
                "stock_value": _finite(stock_value),
                "total_value": _finite(total_value),
                "assets": assets,
            }
        )
    return snapshots


def _buy_and_hold_curve(
    prices: pd.DataFrame,
    *,
    initial_amount: float,
    transaction_cost: float,
) -> pd.Series:
    """Buy every selected stock equally on day one, then never rebalance."""
    first_prices = prices.iloc[0].astype(float)
    target_per_stock = initial_amount / len(first_prices)
    shares = np.floor(
        target_per_stock / (first_prices * (1 + transaction_cost))
    ).astype(int)
    cash = initial_amount - float(
        (shares * first_prices * (1 + transaction_cost)).sum()
    )
    return prices.astype(float).mul(shares, axis=1).sum(axis=1).add(cash)


def _result_payload(
    account_value: pd.DataFrame,
    actions: pd.DataFrame,
    trade: pd.DataFrame,
    tickers: list[str],
    *,
    initial_amount: float = 1_000_000,
    transaction_cost: float = 0.001,
) -> dict[str, Any]:
    account = account_value.copy()
    account["date"] = account["date"].astype(str)
    values = account["account_value"].astype(float)
    returns = values.pct_change().dropna()
    total_return = values.iloc[-1] / values.iloc[0] - 1
    sharpe = np.sqrt(252) * returns.mean() / returns.std() if returns.std() else 0
    drawdown = values / values.cummax() - 1
    win_rate = (returns > 0).mean() if not returns.empty else 0

    prices = trade.pivot_table(index="date", columns="tic", values="close").sort_index()
    prices = prices.reindex(columns=tickers)
    baseline = _buy_and_hold_curve(
        prices,
        initial_amount=initial_amount,
        transaction_cost=transaction_cost,
    )
    baseline.index = baseline.index.astype(str)
    baseline = baseline.reindex(account["date"]).ffill().bfill()

    snapshots = _daily_snapshots(
        actions,
        trade,
        tickers,
        initial_amount=initial_amount,
        transaction_cost=transaction_cost,
    )
    latest = snapshots[-1]
    allocation = [
        {"ticker": asset["ticker"], "weight": asset["weight"]}
        for asset in latest["assets"]
        if asset["weight"] > 0
    ]
    allocation.append({"ticker": "CASH", "weight": latest["cash_weight"]})
    allocation.sort(key=lambda item: item["weight"], reverse=True)

    return {
        "metrics": {
            "total_return": _finite(total_return),
            "sharpe": _finite(sharpe),
            "max_drawdown": _finite(drawdown.min()),
            "win_rate": _finite(win_rate),
            "final_value": _finite(values.iloc[-1]),
        },
        "curve": {
            "dates": account["date"].tolist(),
            "agent": [_finite(value) for value in values],
            "baseline": [_finite(value) for value in baseline],
            "baseline_name": "等權買入持有",
        },
        "allocation": allocation,
        "daily_snapshots": snapshots,
    }


def _run_job(job_id: str) -> None:
    job = get_job(job_id)
    if not job:
        return
    config = job["config"]
    algorithm = config["algorithm"]
    run_dir = RUNS_DIR / job_id
    try:
        _update(job_id, status="running", stage="下載與處理資料", progress=5)
        _log(job_id, f"啟動 {ALGORITHMS[algorithm]['label']} 台股訓練工作")
        processed = prepare_finrl_data(
            config["tickers"],
            config["train_start"],
            config["trade_end"],
            log=lambda message: _log(job_id, message),
        )
        train, trade = split_finrl_data(processed, config["trade_start"])
        save_finrl_data(train, trade, run_dir)
        _log(
            job_id,
            f"訓練資料 {train['date'].nunique()} 日；回測資料 {trade['date'].nunique()} 日",
            "ok",
        )

        _update(job_id, stage="訓練模型", progress=35)
        _log(job_id, f"開始訓練，共 {config['total_timesteps']:,} timesteps")
        if algorithm in {"ppo_lstm", "ppo_mamba"}:
            account, actions = _train_sequence_model(
                algorithm,
                train,
                trade,
                config,
                run_dir,
            )
        else:
            account, actions = _train_standard(
                algorithm,
                train,
                trade,
                config,
                run_dir,
            )

        _update(job_id, stage="整理回測結果", progress=90)
        result = _result_payload(
            account,
            actions,
            trade,
            config["tickers"],
            initial_amount=config["initial_amount"],
            transaction_cost=config["transaction_cost"],
        )
        account.to_csv(run_dir / "account_value.csv", index=False)
        actions.to_csv(run_dir / "actions.csv")
        _log(job_id, "訓練與回測完成", "ok")
        _update(job_id, status="completed", stage="完成", progress=100, result=result)
    except Exception as exc:
        _log(job_id, str(exc), "error")
        _update(
            job_id,
            status="failed",
            stage="失敗",
            error=str(exc),
            traceback=traceback.format_exc(),
        )
