from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List

from openai import OpenAI

from .env import MarketingCampaignEnv
from .models import Action
from .tasks import TASKS

SYSTEM_PROMPT = (
    "You are an ad operations agent. Return only valid JSON with one action for"
    " the current campaign state."
)


@dataclass
class TaskRunResult:
    task_id: str
    score: float
    total_reward: float
    steps: int
    grade: Dict[str, float]


def _serialize_observation(obs) -> Dict:
    return obs.model_dump()


def _model_action(client: OpenAI, model: str, observation: Dict, seed: int) -> Action:
    completion = client.chat.completions.create(
        model=model,
        temperature=0,
        seed=seed,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Current campaign observation JSON:\n"
                    f"{json.dumps(observation)}\n\n"
                    "Pick one action. Valid schema:\n"
                    "{\"action_type\": <adjust_bid|shift_budget|pause_channel|resume_channel|create_variant|wait>,"
                    "\"channel\": string|null, \"delta\": float|null, \"from_channel\": string|null,"
                    "\"to_channel\": string|null, \"amount\": float|null}."
                ),
            },
        ],
    )
    raw = completion.choices[0].message.content or "{}"
    payload = json.loads(raw)
    return Action.model_validate(payload)


def run_baseline(model: str, seed: int) -> List[TaskRunResult]:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")

    client = OpenAI()
    results: List[TaskRunResult] = []

    for idx, task_id in enumerate(sorted(TASKS.keys())):
        env = MarketingCampaignEnv(task_id=task_id, seed=seed + idx)
        obs = env.reset(task_id=task_id, seed=seed + idx)

        done = False
        total_reward = 0.0
        grade = {"score": 0.0}
        steps = 0

        while not done:
            obs_json = _serialize_observation(obs)
            try:
                action = _model_action(client, model, obs_json, seed + idx)
            except Exception:
                action = Action(action_type="wait")

            obs, reward, done, info = env.step(action)
            total_reward += reward.value
            steps += 1
            if done and "grade" in info:
                grade = info["grade"]

        results.append(
            TaskRunResult(
                task_id=task_id,
                score=float(grade.get("score", 0.0)),
                total_reward=round(total_reward, 4),
                steps=steps,
                grade=grade,
            )
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OpenAI baseline on all tasks.")
    parser.add_argument("--model", default="gpt-4.1-mini", help="OpenAI model name")
    parser.add_argument("--seed", type=int, default=11, help="Random seed")
    parser.add_argument(
        "--out",
        default=".baseline_results.json",
        help="Output JSON file for reproducible baseline scores",
    )
    args = parser.parse_args()

    results = run_baseline(model=args.model, seed=args.seed)
    avg_score = sum(r.score for r in results) / max(1, len(results))

    payload = {
        "model": args.model,
        "seed": args.seed,
        "avg_score": round(avg_score, 4),
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

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
