"""Default hyperparameters shared by the dashboard and comparison scripts."""

from __future__ import annotations


# PPO, TD3, and SAC follow FinRL_StockTrading_2026_2_train.py.
# A2C and DDPG follow FinRL's native stable-baselines3 defaults.
# Recurrent and Mamba values are conservative sequence-model defaults.
ALGORITHM_DEFAULTS = {
    "ppo": {
        "learning_rate": 0.00025,
        "n_steps": 2048,
        "batch_size": 128,
        "ent_coef": 0.01,
    },
    "a2c": {
        "learning_rate": 0.0007,
        "n_steps": 5,
        "ent_coef": 0.01,
    },
    "td3": {
        "learning_rate": 0.001,
        "batch_size": 100,
        "buffer_size": 1_000_000,
    },
    "sac": {
        "learning_rate": 0.0001,
        "batch_size": 128,
        "buffer_size": 100_000,
        "learning_starts": 100,
        "ent_coef": "auto_0.1",
    },
    "ddpg": {
        "learning_rate": 0.001,
        "batch_size": 128,
        "buffer_size": 50_000,
    },
    "ppo_lstm": {
        "learning_rate": 0.00025,
        "n_steps": 512,
        "batch_size": 128,
        "ent_coef": 0.01,
        "lstm_hidden_size": 256,
        "n_lstm_layers": 1,
    },
    "ppo_mamba": {
        "learning_rate": 0.00025,
        "n_steps": 512,
        "batch_size": 128,
        "ent_coef": 0.01,
        "sequence_length": 16,
        "d_model": 128,
        "d_state": 16,
        "mamba_layers": 2,
        "expand": 2,
        "conv_kernel": 4,
    },
}


def get_algorithm_config(algorithm: str, overrides: dict | None = None) -> dict:
    """Return one algorithm's defaults with non-None overrides applied."""
    config = dict(ALGORITHM_DEFAULTS[algorithm])
    if overrides:
        config.update({key: value for key, value in overrides.items() if value is not None})
    return config
