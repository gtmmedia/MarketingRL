from __future__ import annotations

from fastapi import FastAPI

from .baseline import run_baseline
from .env import MarketingCampaignEnv

app = FastAPI(title="Marketing OpenEnv", version="1.0.0")


@app.get("/")
def root() -> dict:
    env = MarketingCampaignEnv()
    obs = env.reset()
    return {
        "name": "Marketing Campaign OpenEnv",
        "tasks": env.tasks,
        "sample_observation": obs.model_dump(),
        "message": "Use /baseline to run baseline scoring across tasks.",
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/baseline")
def baseline(model: str = "gpt-4.1-mini", seed: int = 11) -> dict:
    results = run_baseline(model=model, seed=seed)
    avg = sum(item.score for item in results) / max(1, len(results))
    return {
        "model": model,
        "seed": seed,
        "avg_score": round(avg, 4),
        "tasks": [
            {
                "task_id": r.task_id,
                "score": round(r.score, 4),
                "total_reward": r.total_reward,
                "steps": r.steps,
                "grade": r.grade,
            }
            for r in results
        ],
    }
