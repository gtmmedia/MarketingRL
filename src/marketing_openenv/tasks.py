from __future__ import annotations

from .models import Difficulty, TaskSpec


TASKS = {
    "easy_ctr_recovery": TaskSpec(
        task_id="easy_ctr_recovery",
        title="Recover CTR for underperforming launch",
        difficulty=Difficulty.EASY,
        objective=(
            "Increase click-through rate while keeping spend within budget and"
            " generating at least 20 conversions."
        ),
        max_steps=12,
        total_budget=700.0,
        target_ctr=0.028,
        target_roas=1.6,
        target_cpa=30.0,
        target_conversions=20,
    ),
    "medium_conversion_push": TaskSpec(
        task_id="medium_conversion_push",
        title="Drive efficient conversion growth",
        difficulty=Difficulty.MEDIUM,
        objective=(
            "Maximize conversions while keeping ROAS healthy and CPA controlled"
            " as auction pressure rises."
        ),
        max_steps=16,
        total_budget=1400.0,
        target_ctr=0.032,
        target_roas=2.1,
        target_cpa=28.0,
        target_conversions=58,
    ),
    "hard_multi_segment_stability": TaskSpec(
        task_id="hard_multi_segment_stability",
        title="Stabilize multi-segment campaign over time",
        difficulty=Difficulty.HARD,
        objective=(
            "Balance spend across channels, avoid fatigue, and maintain robust"
            " returns across volatile user segments."
        ),
        max_steps=22,
        total_budget=2400.0,
        target_ctr=0.034,
        target_roas=2.4,
        target_cpa=26.0,
        target_conversions=105,
    ),
}


def get_task(task_id: str) -> TaskSpec:
    if task_id not in TASKS:
        valid = ", ".join(sorted(TASKS.keys()))
        raise ValueError(f"Unknown task_id '{task_id}'. Valid task ids: {valid}")
    return TASKS[task_id]
