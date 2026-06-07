"""Train and backtest RecurrentPPO with FinRL's stock trading environment.

This file intentionally lives outside the ``finrl`` package. It uses FinRL as a
library and keeps the RecurrentPPO-specific integration local to this script.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecNormalize

from finrl.config import INDICATORS
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv


DEFAULT_MODEL_PATH = Path("trained_models/agent_recurrent_ppo")
DEFAULT_OUTPUT_DIR = Path("results/recurrent_ppo")


def get_normalization_path(model_path: Path) -> Path:
    """Return the observation-normalization statistics path for a model."""
    return model_path.parent / f"{model_path.name}_vecnormalize.pkl"


def load_data(path: Path) -> pd.DataFrame:
    """Load FinRL's processed CSV and restore its integer day index."""
    data = pd.read_csv(path)
    data = data.set_index(data.columns[0])
    data.index.names = [""]

    required_columns = {"date", "tic", "close", *INDICATORS}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{path} is missing required columns: {missing}")

    return data


def build_environment(
    data: pd.DataFrame,
    *,
    turbulence_threshold: float | None = None,
) -> StockTradingEnv:
    """Create the standard FinRL stock trading environment."""
    stock_dimension = data["tic"].nunique()
    state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension

    return StockTradingEnv(
        df=data,
        hmax=100,
        initial_amount=1_000_000,
        num_stock_shares=[0] * stock_dimension,
        buy_cost_pct=[0.001] * stock_dimension,
        sell_cost_pct=[0.001] * stock_dimension,
        state_space=state_space,
        stock_dim=stock_dimension,
        tech_indicator_list=INDICATORS,
        action_space=stock_dimension,
        reward_scaling=1e-4,
        turbulence_threshold=turbulence_threshold,
        risk_indicator_col="vix",
    )


def train(
    train_data: pd.DataFrame,
    *,
    model_path: Path,
    output_dir: Path,
    total_timesteps: int,
    n_steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: str,
) -> RecurrentPPO:
    """Train RecurrentPPO using a FinRL StockTradingEnv."""
    if (n_steps % batch_size) != 0:
        raise ValueError(
            "With one training environment, batch_size must divide n_steps exactly."
        )

    environment = build_environment(train_data)
    vec_env, _ = environment.get_sb_env()
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    model = RecurrentPPO(
        policy="MlpLstmPolicy",
        env=vec_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        ent_coef=0.01,
        policy_kwargs={
            "lstm_hidden_size": 256,
            "n_lstm_layers": 1,
            "shared_lstm": False,
            "enable_critic_lstm": True,
            "net_arch": {"pi": [128, 128], "vf": [128, 128]},
        },
        verbose=1,
        seed=seed,
        device=device,
    )
    model.set_logger(configure(str(output_dir / "training_log"), ["stdout", "csv"]))
    model.learn(total_timesteps=total_timesteps)
    model.save(model_path)
    normalization_path = get_normalization_path(model_path)
    vec_env.save(normalization_path)
    vec_env.close()

    print(f"Saved RecurrentPPO model to {model_path}.zip")
    print(f"Saved observation normalization statistics to {normalization_path}")
    return model


def backtest(
    trade_data: pd.DataFrame,
    *,
    model_path: Path,
    output_dir: Path,
    turbulence_threshold: float,
    device: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Backtest while preserving the recurrent LSTM state between trading days."""
    environment = build_environment(
        trade_data, turbulence_threshold=turbulence_threshold
    )
    vec_env, _ = environment.get_sb_env()
    normalization_path = get_normalization_path(model_path)
    if not normalization_path.exists():
        raise FileNotFoundError(
            f"Missing observation normalization statistics: {normalization_path}. "
            "Train the normalized model before backtesting it."
        )

    vec_env = VecNormalize.load(normalization_path, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False
    observation = vec_env.reset()
    model = RecurrentPPO.load(model_path, env=vec_env, device=device)

    lstm_states = None
    episode_starts = np.ones((vec_env.num_envs,), dtype=bool)
    max_steps = len(trade_data.index.unique()) - 1

    for _ in range(max_steps):
        action, lstm_states = model.predict(
            observation,
            state=lstm_states,
            episode_start=episode_starts,
            deterministic=True,
        )
        observation, _, dones, _ = vec_env.step(action)
        episode_starts = dones

    account_value = environment.save_asset_memory()
    actions = environment.save_action_memory()
    vec_env.close()

    output_dir.mkdir(parents=True, exist_ok=True)
    account_path = output_dir / "account_value.csv"
    actions_path = output_dir / "actions.csv"
    plot_path = output_dir / "account_value.png"
    account_value.to_csv(account_path, index=False)
    actions.to_csv(actions_path)

    plt.figure(figsize=(15, 5))
    plt.plot(account_value["date"], account_value["account_value"])
    plt.title("RecurrentPPO Portfolio Value")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    daily_return = account_value["account_value"].pct_change().dropna()
    total_return = account_value["account_value"].iloc[-1] / account_value[
        "account_value"
    ].iloc[0] - 1
    sharpe = (
        np.sqrt(252) * daily_return.mean() / daily_return.std()
        if not daily_return.empty and daily_return.std() != 0
        else float("nan")
    )

    print(f"Final portfolio value: {account_value['account_value'].iloc[-1]:,.2f}")
    print(f"Total return: {total_return:.2%}")
    print(f"Annualized Sharpe ratio: {sharpe:.3f}")
    print(f"Saved backtest outputs to {output_dir}")
    return account_value, actions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use sb3-contrib RecurrentPPO with FinRL without modifying finrl/."
    )
    parser.add_argument(
        "mode",
        nargs="?",
        choices=("train", "backtest", "all"),
        default="all",
        help="Operation to run. Default: all",
    )
    parser.add_argument("--train-data", type=Path, default=Path("train_data.csv"))
    parser.add_argument("--trade-data", type=Path, default=Path("trade_data.csv"))
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--total-timesteps", type=int, default=300_000)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--turbulence-threshold", type=float, default=70.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode in {"train", "all"}:
        train_data = load_data(args.train_data)
        print(
            f"Training with {train_data['tic'].nunique()} stocks and "
            f"{len(train_data.index.unique())} trading days."
        )
        train(
            train_data,
            model_path=args.model_path,
            output_dir=args.output_dir,
            total_timesteps=args.total_timesteps,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            device=args.device,
        )

    if args.mode in {"backtest", "all"}:
        trade_data = load_data(args.trade_data)
        print(
            f"Backtesting with {trade_data['tic'].nunique()} stocks and "
            f"{len(trade_data.index.unique())} trading days."
        )
        backtest(
            trade_data,
            model_path=args.model_path,
            output_dir=args.output_dir,
            turbulence_threshold=args.turbulence_threshold,
            device=args.device,
        )


if __name__ == "__main__":
    main()
