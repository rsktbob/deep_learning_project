"""Train and backtest PPO with a lightweight Mamba-style encoder for FinRL.

The official ``mamba-ssm`` package is not required. This script implements the
selective state-space scan in pure PyTorch and uses ``VecFrameStack`` to turn
the latest observations into a short time series for the encoder.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.vec_env import VecNormalize
from torch import nn
from torch.nn import functional as F

from finrl.config import INDICATORS
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv


DEFAULT_MODEL_PATH = Path("trained_models/agent_ppo_mamba")
DEFAULT_OUTPUT_DIR = Path("results/ppo_mamba")


class MambaBlock(nn.Module):
    """A compact selective state-space block implemented in pure PyTorch."""

    def __init__(
        self,
        d_model: int,
        d_state: int,
        expand: int,
        conv_kernel: int,
    ) -> None:
        super().__init__()
        inner_dim = d_model * expand
        self.inner_dim = inner_dim
        self.d_state = d_state

        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * inner_dim)
        self.depthwise_conv = nn.Conv1d(
            inner_dim,
            inner_dim,
            kernel_size=conv_kernel,
            padding=conv_kernel - 1,
            groups=inner_dim,
        )
        self.delta_proj = nn.Linear(inner_dim, inner_dim)
        self.b_proj = nn.Linear(inner_dim, d_state)
        self.c_proj = nn.Linear(inner_dim, d_state)

        initial_a = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.a_log = nn.Parameter(initial_a.log().repeat(inner_dim, 1))
        self.skip = nn.Parameter(torch.ones(inner_dim))
        self.out_proj = nn.Linear(inner_dim, d_model)

    def selective_scan(self, u: torch.Tensor) -> torch.Tensor:
        """Run the input-dependent diagonal state-space recurrence."""
        batch_size, sequence_length, _ = u.shape
        delta = F.softplus(self.delta_proj(u)) + 1e-4
        b = self.b_proj(u)
        c = self.c_proj(u)
        a = -torch.exp(self.a_log.float()).to(dtype=u.dtype)

        state = torch.zeros(
            batch_size,
            self.inner_dim,
            self.d_state,
            dtype=u.dtype,
            device=u.device,
        )
        outputs = []

        for step in range(sequence_length):
            delta_t = delta[:, step].unsqueeze(-1)
            u_t = u[:, step].unsqueeze(-1)
            b_t = b[:, step].unsqueeze(1)
            c_t = c[:, step].unsqueeze(1)

            state = (
                torch.exp(delta_t * a.unsqueeze(0)) * state
                + delta_t * b_t * u_t
            )
            output_t = (state * c_t).sum(dim=-1) + self.skip * u[:, step]
            outputs.append(output_t)

        return torch.stack(outputs, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        projected = self.in_proj(self.norm(x))
        u, gate = projected.chunk(2, dim=-1)

        sequence_length = u.shape[1]
        u = self.depthwise_conv(u.transpose(1, 2))[..., :sequence_length]
        u = F.silu(u.transpose(1, 2))

        y = self.selective_scan(u)
        y = y * F.silu(gate)
        return residual + self.out_proj(y)


class MambaFeaturesExtractor(BaseFeaturesExtractor):
    """Encode a flattened observation history with selective SSM blocks."""

    def __init__(
        self,
        observation_space: spaces.Box,
        state_dim: int,
        sequence_length: int = 16,
        d_model: int = 128,
        d_state: int = 16,
        n_layers: int = 2,
        expand: int = 2,
        conv_kernel: int = 4,
    ) -> None:
        super().__init__(observation_space, features_dim=d_model)

        expected_dim = state_dim * sequence_length
        actual_dim = int(np.prod(observation_space.shape))
        if actual_dim != expected_dim:
            raise ValueError(
                "Stacked observation size does not match state_dim * "
                f"sequence_length: {actual_dim} != {expected_dim}."
            )

        self.state_dim = state_dim
        self.sequence_length = sequence_length
        self.input_projection = nn.Linear(state_dim, d_model)
        self.blocks = nn.ModuleList(
            [
                MambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    expand=expand,
                    conv_kernel=conv_kernel,
                )
                for _ in range(n_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        sequence = observations.reshape(
            observations.shape[0],
            self.sequence_length,
            self.state_dim,
        )
        hidden = self.input_projection(sequence)
        for block in self.blocks:
            hidden = block(hidden)
        return self.output_norm(hidden[:, -1])


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


def validate_model_parameters(
    *,
    n_steps: int,
    batch_size: int,
    sequence_length: int,
    d_model: int,
    d_state: int,
    n_layers: int,
    expand: int,
    conv_kernel: int,
) -> None:
    """Reject invalid PPO and Mamba hyperparameters before training."""
    if n_steps % batch_size != 0:
        raise ValueError(
            "With one training environment, batch_size must divide n_steps exactly."
        )

    positive_parameters = {
        "sequence_length": sequence_length,
        "d_model": d_model,
        "d_state": d_state,
        "n_layers": n_layers,
        "expand": expand,
        "conv_kernel": conv_kernel,
    }
    invalid = [name for name, value in positive_parameters.items() if value < 1]
    if invalid:
        raise ValueError(f"These parameters must be positive: {', '.join(invalid)}")


def train(
    train_data: pd.DataFrame,
    *,
    model_path: Path,
    output_dir: Path,
    total_timesteps: int,
    n_steps: int,
    batch_size: int,
    learning_rate: float,
    sequence_length: int,
    d_model: int,
    d_state: int,
    n_layers: int,
    expand: int,
    conv_kernel: int,
    seed: int,
    device: str,
) -> PPO:
    """Train PPO using a Mamba-style encoder over recent observations."""
    validate_model_parameters(
        n_steps=n_steps,
        batch_size=batch_size,
        sequence_length=sequence_length,
        d_model=d_model,
        d_state=d_state,
        n_layers=n_layers,
        expand=expand,
        conv_kernel=conv_kernel,
    )

    environment = build_environment(train_data)
    state_dim = environment.observation_space.shape[0]
    base_vec_env, _ = environment.get_sb_env()
    normalizer = VecNormalize(
        base_vec_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
    )
    vec_env = VecFrameStack(normalizer, n_stack=sequence_length)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        ent_coef=0.01,
        policy_kwargs={
            "features_extractor_class": MambaFeaturesExtractor,
            "features_extractor_kwargs": {
                "state_dim": state_dim,
                "sequence_length": sequence_length,
                "d_model": d_model,
                "d_state": d_state,
                "n_layers": n_layers,
                "expand": expand,
                "conv_kernel": conv_kernel,
            },
            "net_arch": {"pi": [128, 128], "vf": [128, 128]},
            "activation_fn": nn.SiLU,
        },
        verbose=1,
        seed=seed,
        device=device,
    )
    parameter_count = sum(parameter.numel() for parameter in model.policy.parameters())
    print(
        f"Mamba sequence length: {sequence_length}, state dimension: {state_dim}, "
        f"policy parameters: {parameter_count:,}"
    )

    model.set_logger(configure(str(output_dir / "training_log"), ["stdout", "csv"]))
    model.learn(total_timesteps=total_timesteps)
    model.save(model_path)
    normalization_path = get_normalization_path(model_path)
    normalizer.save(normalization_path)
    vec_env.close()

    print(f"Saved PPO + Mamba model to {model_path}.zip")
    print(f"Saved observation normalization statistics to {normalization_path}")
    return model


def infer_sequence_length(model: PPO, state_dim: int) -> int:
    """Infer the frame-stack length stored in a trained model."""
    observation_shape = model.observation_space.shape
    if len(observation_shape) != 1 or observation_shape[0] % state_dim != 0:
        raise ValueError(
            "The model observation shape is incompatible with the trading "
            f"environment: {observation_shape} versus state dimension {state_dim}."
        )
    return observation_shape[0] // state_dim


def backtest(
    trade_data: pd.DataFrame,
    *,
    model_path: Path,
    output_dir: Path,
    turbulence_threshold: float,
    device: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Backtest PPO + Mamba while preserving the rolling observation history."""
    environment = build_environment(
        trade_data, turbulence_threshold=turbulence_threshold
    )
    state_dim = environment.observation_space.shape[0]
    base_vec_env, _ = environment.get_sb_env()

    normalization_path = get_normalization_path(model_path)
    if not normalization_path.exists():
        raise FileNotFoundError(
            f"Missing observation normalization statistics: {normalization_path}. "
            "Train the normalized model before backtesting it."
        )

    normalizer = VecNormalize.load(normalization_path, base_vec_env)
    normalizer.training = False
    normalizer.norm_reward = False

    model = PPO.load(model_path, device=device)
    sequence_length = infer_sequence_length(model, state_dim)
    vec_env = VecFrameStack(normalizer, n_stack=sequence_length)
    model.set_env(vec_env)
    observation = vec_env.reset()
    max_steps = len(trade_data.index.unique()) - 1

    for _ in range(max_steps):
        action, _ = model.predict(observation, deterministic=True)
        observation, _, dones, _ = vec_env.step(action)
        if dones[0]:
            break

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
    plt.title("PPO + Mamba Portfolio Value")
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

    print(f"Mamba sequence length: {sequence_length}")
    print(f"Final portfolio value: {account_value['account_value'].iloc[-1]:,.2f}")
    print(f"Total return: {total_return:.2%}")
    print(f"Annualized Sharpe ratio: {sharpe:.3f}")
    print(f"Saved backtest outputs to {output_dir}")
    return account_value, actions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use PPO with a pure-PyTorch Mamba-style encoder in FinRL."
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
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--d-state", type=int, default=16)
    parser.add_argument("--mamba-layers", type=int, default=2)
    parser.add_argument("--expand", type=int, default=2)
    parser.add_argument("--conv-kernel", type=int, default=4)
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
            sequence_length=args.sequence_length,
            d_model=args.d_model,
            d_state=args.d_state,
            n_layers=args.mamba_layers,
            expand=args.expand,
            conv_kernel=args.conv_kernel,
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
