"""Train, backtest, and plot all dashboard algorithms on the same data.

The script caches each algorithm's account-value CSV. Re-running it only trains
missing algorithms unless ``--force`` is supplied.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from algorithm_defaults import get_algorithm_config
from rl_trainer import (
    ALGORITHMS,
    _buy_and_hold_curve,
    _train_sequence_model,
    _train_standard,
)


ALGORITHM_ORDER = [
    "ppo",
    "a2c",
    "td3",
    "sac",
    "ddpg",
    "ppo_lstm",
    "ppo_mamba",
]

COLORS = {
    "ppo": "#1f77b4",
    "a2c": "#ff7f0e",
    "td3": "#2ca02c",
    "sac": "#d62728",
    "ddpg": "#9467bd",
    "ppo_lstm": "#8c564b",
    "ppo_mamba": "#e377c2",
    "buy_hold": "#111111",
}


def load_finrl_data(path: Path) -> pd.DataFrame:
    """Load a FinRL CSV while preserving its integer trading-day index."""
    data = pd.read_csv(path)
    data = data.set_index(data.columns[0])
    data.index.names = [""]
    return data


def normalize_account(account: pd.DataFrame) -> pd.Series:
    """Convert an account-value frame into a date-indexed value series."""
    frame = account.copy()
    frame["date"] = frame["date"].astype(str)
    return frame.drop_duplicates("date").set_index("date")["account_value"].astype(float)


def metrics(values: pd.Series) -> dict[str, float]:
    """Calculate common portfolio metrics."""
    daily_return = values.pct_change().dropna()
    drawdown = values / values.cummax() - 1
    return {
        "final_value": float(values.iloc[-1]),
        "total_return": float(values.iloc[-1] / values.iloc[0] - 1),
        "sharpe": float(
            (252**0.5) * daily_return.mean() / daily_return.std()
            if not daily_return.empty and daily_return.std()
            else 0
        ),
        "max_drawdown": float(drawdown.min()),
    }


def train_or_load(
    algorithm: str,
    train: pd.DataFrame,
    trade: pd.DataFrame,
    config: dict,
    output_dir: Path,
    *,
    force: bool,
) -> pd.DataFrame:
    """Train one algorithm or load its cached account-value output."""
    algorithm_dir = output_dir / algorithm
    account_path = algorithm_dir / "account_value.csv"
    actions_path = algorithm_dir / "actions.csv"
    if account_path.exists() and not force:
        print(f"[cache] {ALGORITHMS[algorithm]['label']}: {account_path}")
        return pd.read_csv(account_path)

    print(f"[train] {ALGORITHMS[algorithm]['label']} ({config['total_timesteps']:,} timesteps)")
    algorithm_dir.mkdir(parents=True, exist_ok=True)
    if algorithm in {"ppo_lstm", "ppo_mamba"}:
        account, actions = _train_sequence_model(
            algorithm,
            train,
            trade,
            config,
            algorithm_dir,
        )
    else:
        account, actions = _train_standard(
            algorithm,
            train,
            trade,
            config,
            algorithm_dir,
        )
    account.to_csv(account_path, index=False)
    actions.to_csv(actions_path)
    return account


def plot_comparison(
    comparison: pd.DataFrame,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Save portfolio-value and normalized-return comparison charts."""
    labels = {key: value["label"] for key, value in ALGORITHMS.items()}
    labels["buy_hold"] = "Equal-Weight Buy & Hold"

    value_path = output_dir / "all_algorithms_portfolio_value.png"
    return_path = output_dir / "all_algorithms_cumulative_return.png"

    plt.figure(figsize=(16, 9))
    for column in comparison.columns:
        plt.plot(
            comparison.index,
            comparison[column],
            label=labels[column],
            color=COLORS[column],
            linewidth=2.4 if column == "buy_hold" else 1.5,
            linestyle="--" if column == "buy_hold" else "-",
        )
    plt.title("Taiwan Stock RL Algorithms - Portfolio Value")
    plt.xlabel("Backtest Date")
    plt.ylabel("Portfolio Value (TWD)")
    plt.legend(ncol=3)
    plt.grid(alpha=0.2)
    plt.xticks(rotation=35)
    plt.tight_layout()
    plt.savefig(value_path, dpi=180, bbox_inches="tight")
    plt.close()

    normalized = comparison.div(comparison.iloc[0]).sub(1).mul(100)
    plt.figure(figsize=(16, 9))
    for column in normalized.columns:
        plt.plot(
            normalized.index,
            normalized[column],
            label=labels[column],
            color=COLORS[column],
            linewidth=2.4 if column == "buy_hold" else 1.5,
            linestyle="--" if column == "buy_hold" else "-",
        )
    plt.axhline(0, color="#777777", linewidth=0.8)
    plt.title("Taiwan Stock RL Algorithms - Cumulative Return")
    plt.xlabel("Backtest Date")
    plt.ylabel("Cumulative Return (%)")
    plt.legend(ncol=3)
    plt.grid(alpha=0.2)
    plt.xticks(rotation=35)
    plt.tight_layout()
    plt.savefig(return_path, dpi=180, bbox_inches="tight")
    plt.close()
    return value_path, return_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare PPO, A2C, TD3, SAC, DDPG, PPO+LSTM, PPO+Mamba, and buy-and-hold."
    )
    parser.add_argument("--train-data", type=Path, default=Path("train_data.csv"))
    parser.add_argument("--trade-data", type=Path, default=Path("trade_data.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/all_algorithms"))
    parser.add_argument("--total-timesteps", type=int, default=20_000)
    parser.add_argument("--n-steps", type=int, default=None, help="Override n_steps for applicable algorithms.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size for applicable algorithms.")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override learning rate for every algorithm.")
    parser.add_argument("--initial-amount", type=float, default=1_000_000)
    parser.add_argument("--transaction-cost", type=float, default=0.001)
    parser.add_argument("--hmax", type=int, default=100)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force", action="store_true", help="Retrain even when cached CSVs exist.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train = load_finrl_data(args.train_data)
    trade = load_finrl_data(args.trade_data)
    tickers = sorted(trade["tic"].unique().tolist())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shared_config = {
        "total_timesteps": args.total_timesteps,
        "hmax": args.hmax,
        "initial_amount": args.initial_amount,
        "transaction_cost": args.transaction_cost,
        "seed": args.seed,
        "device": args.device,
    }

    series = {}
    for algorithm in ALGORITHM_ORDER:
        config = get_algorithm_config(
            algorithm,
            {
                "learning_rate": args.learning_rate,
                "n_steps": args.n_steps,
                "batch_size": args.batch_size,
                "sequence_length": args.sequence_length if algorithm == "ppo_mamba" else None,
                **shared_config,
            },
        )
        if algorithm in {"ppo", "ppo_lstm", "ppo_mamba"}:
            if config["n_steps"] % config["batch_size"]:
                raise ValueError(
                    f"{algorithm}: batch_size must divide n_steps "
                    f"({config['batch_size']} does not divide {config['n_steps']})."
                )
        account = train_or_load(
            algorithm,
            train,
            trade,
            config,
            args.output_dir,
            force=args.force,
        )
        series[algorithm] = normalize_account(account)

    prices = trade.pivot_table(index="date", columns="tic", values="close").sort_index()
    prices = prices.reindex(columns=tickers)
    series["buy_hold"] = _buy_and_hold_curve(
        prices,
        initial_amount=args.initial_amount,
        transaction_cost=args.transaction_cost,
    )
    series["buy_hold"].index = series["buy_hold"].index.astype(str)

    comparison = pd.concat(series, axis=1).sort_index().ffill().dropna()
    comparison.index.name = "date"
    comparison.to_csv(args.output_dir / "all_algorithms_portfolio_value.csv")

    summary = pd.DataFrame(
        {name: metrics(comparison[name]) for name in comparison.columns}
    ).T
    summary.index.name = "algorithm"
    summary.to_csv(args.output_dir / "all_algorithms_metrics.csv")

    value_path, return_path = plot_comparison(comparison, args.output_dir)
    print("\nComparison metrics:")
    print(summary.to_string(float_format=lambda value: f"{value:,.4f}"))
    print(f"\nSaved portfolio-value chart: {value_path}")
    print(f"Saved cumulative-return chart: {return_path}")


if __name__ == "__main__":
    main()
