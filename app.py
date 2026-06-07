"""FastAPI entry point for the Taiwan FinRL dashboard."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

from rl_trainer import ALGORITHMS, create_job, dependency_status, get_job, list_jobs
from taiwan_stocks import DEFAULT_TICKERS, TAIWAN_STOCKS, VALID_TICKERS


ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "web"

app = FastAPI(title="Taiwan FinRL Lab", version="1.0.0")
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


class TrainRequest(BaseModel):
    algorithm: Literal[
        "ppo",
        "a2c",
        "td3",
        "sac",
        "ddpg",
        "ppo_lstm",
        "ppo_mamba",
    ] = "ppo"
    tickers: list[str] = Field(
        default_factory=lambda: DEFAULT_TICKERS.copy(),
        min_length=1,
        max_length=30,
    )
    train_start: date = date(2018, 1, 1)
    trade_start: date = date(2024, 1, 1)
    trade_end: date = date(2026, 6, 1)
    total_timesteps: int = Field(20_000, ge=1_000, le=2_000_000)
    learning_rate: float = Field(0.00025, gt=0, le=0.1)
    batch_size: int = Field(128, ge=8, le=2048)
    n_steps: int = Field(2048, ge=5, le=8192)
    initial_amount: float = Field(1_000_000, ge=10_000)
    hmax: int = Field(100, ge=1, le=10_000)
    transaction_cost: float = Field(0.001, ge=0, le=0.1)
    sequence_length: int = Field(16, ge=2, le=128)
    seed: int = Field(42, ge=0)
    device: Literal["auto", "cpu", "cuda"] = "auto"

    @model_validator(mode="after")
    def validate_request(self):
        invalid = sorted(set(self.tickers).difference(VALID_TICKERS))
        if invalid:
            raise ValueError(f"不支援的股票代碼：{', '.join(invalid)}")
        if not self.train_start < self.trade_start < self.trade_end:
            raise ValueError("日期必須符合：訓練開始 < 回測開始 < 回測結束")
        if (
            self.algorithm in {"ppo", "ppo_lstm", "ppo_mamba"}
            and self.n_steps % self.batch_size
        ):
            raise ValueError("PPO 系列的 batch_size 必須整除 n_steps")
        return self


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok", "dependencies": dependency_status()}


@app.get("/api/config")
def config():
    return {
        "stocks": TAIWAN_STOCKS,
        "default_tickers": DEFAULT_TICKERS,
        "algorithms": ALGORITHMS,
        "dependencies": dependency_status(),
    }


@app.get("/api/jobs")
def jobs():
    return list_jobs()


@app.post("/api/jobs", status_code=202)
def start_job(request: TrainRequest):
    return create_job(request.model_dump(mode="json"))


@app.get("/api/jobs/{job_id}")
def job(job_id: str):
    result = get_job(job_id)
    if not result:
        raise HTTPException(status_code=404, detail="找不到這個訓練工作")
    return result
