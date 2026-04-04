from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List

from dotenv import load_dotenv
from groq import Groq

from .agent_policy import extract_json_object, heuristic_action
from .env import MarketingCampaignEnv
from .models import Action

TASK_ORDER = [
    "easy_ctr_recovery",
    "medium_conversion_push",
    "hard_multi_segment_stability",
]

SYSTEM_PROMPT = (
    "You are an ad operations agent. Return only valid JSON with one action for"
    " the current campaign state."
)


# Load environment variables from a local .env file if present.
load_dotenv()


@dataclass
class TaskRunResult:
    task_id: str
    score: float
    total_reward: float
    steps: int
    grade: Dict[str, float]


def _serialize_observation(obs) -> Dict:
    return obs.model_dump()


def _model_action(client: Groq, model: str, observation: Dict) -> Action:
    completion = client.chat.completions.create(
        model=model,
        temperature=0,
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
    payload = extract_json_object(raw)
    return Action.model_validate(payload)


def run_baseline(model: str, seed: int) -> List[TaskRunResult]:
    use_heuristic = model.lower() == "heuristic"
    client = None
    if not use_heuristic:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Set it with:\n"
                "  $env:GROQ_API_KEY = 'your_key_here'\n"
                "Or create a .env file with GROQ_API_KEY=your_key_here"
            )
        client = Groq(api_key=api_key)
    results: List[TaskRunResult] = []

    for idx, task_id in enumerate(TASK_ORDER):
        env = MarketingCampaignEnv(task_id=task_id, seed=seed + idx)
        obs = env.reset(task_id=task_id, seed=seed + idx)

        done = False
        total_reward = 0.0
        grade = {"score": 0.0}
        steps = 0

        while not done:
            obs_json = _serialize_observation(obs)
            try:
                if use_heuristic:
                    action = heuristic_action(obs_json)
                else:
                    action = _model_action(client, model, obs_json)
            except Exception:
                action = heuristic_action(obs_json)

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
    parser = argparse.ArgumentParser(description="Run Groq baseline on all tasks.")
    parser.add_argument(
        "--model",
        default="heuristic",
        help="Groq model name (e.g. mixtral-8x7b-32768) or heuristic for offline baseline",
    )
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
