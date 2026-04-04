from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

from dotenv import load_dotenv
from openai import OpenAI

from src.marketing_openenv.agent_policy import extract_json_object, heuristic_action
from src.marketing_openenv.env import MarketingCampaignEnv
from src.marketing_openenv.models import Action

TASK_ORDER = [
    "easy_ctr_recovery",
    "medium_conversion_push",
    "hard_multi_segment_stability",
]


def _log(tag: str, payload: Dict) -> None:
    print(f"[{tag}] {json.dumps(payload, separators=(',', ':'), ensure_ascii=True)}")


def _compact_observation(obs: Dict) -> Dict:
    compact_channels = []
    for ch in obs.get("channels", []):
        compact_channels.append(
            {
                "channel": ch.get("channel"),
                "active": ch.get("active"),
                "budget": round(float(ch.get("budget", 0.0)), 4),
                "bid": round(float(ch.get("bid", 0.0)), 4),
                "fatigue": round(float(ch.get("fatigue", 0.0)), 4),
                "ctr": round(float(ch.get("clicks", 0)) / max(1, float(ch.get("impressions", 0))), 5),
                "cvr": round(float(ch.get("conversions", 0)) / max(1, float(ch.get("clicks", 0))), 5),
                "conversions": int(ch.get("conversions", 0)),
            }
        )
    return {
        "task_id": obs.get("task_id"),
        "step_index": obs.get("step_index"),
        "max_steps": obs.get("max_steps"),
        "remaining_budget": obs.get("remaining_budget"),
        "total_spend": obs.get("total_spend"),
        "total_revenue": obs.get("total_revenue"),
        "total_conversions": obs.get("total_conversions"),
        "avg_ctr": obs.get("avg_ctr"),
        "avg_cvr": obs.get("avg_cvr"),
        "avg_cpa": obs.get("avg_cpa"),
        "roas": obs.get("roas"),
        "channels": compact_channels,
    }


def _llm_action(client: OpenAI, model_name: str, observation: Dict) -> Action:
    completion = client.chat.completions.create(
        model=model_name,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an ad campaign optimization agent. Return exactly one JSON object "
                    "that follows the Action schema."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Observation JSON:\n"
                    + json.dumps(_compact_observation(observation), separators=(",", ":"))
                    + "\n\n"
                    "Return valid JSON only with keys: action_type, channel, delta, from_channel, to_channel, amount."
                ),
            },
        ],
    )
    raw = completion.choices[0].message.content or "{}"
    payload = extract_json_object(raw)
    return Action.model_validate(payload)


def _validate_required_env() -> Dict[str, str]:
    required = ["API_BASE_URL", "MODEL_NAME", "HF_TOKEN"]
    values = {key: os.getenv(key, "").strip() for key in required}
    missing = [k for k, v in values.items() if not v]
    if missing:
        raise RuntimeError(f"Missing required env vars: {', '.join(missing)}")
    return values


def run(seed: int) -> Dict:
    load_dotenv()
    env_cfg = _validate_required_env()

    api_base_url = env_cfg["API_BASE_URL"]
    model_name = env_cfg["MODEL_NAME"]
    hf_token = env_cfg["HF_TOKEN"]

    use_heuristic = model_name.lower() == "heuristic"
    client = OpenAI(base_url=api_base_url, api_key=hf_token, timeout=20.0)

    _log(
        "START",
        {
            "api_base_url": api_base_url,
            "model_name": model_name,
            "seed": seed,
            "tasks": TASK_ORDER,
        },
    )

    task_results: List[Dict] = []

    for idx, task_id in enumerate(TASK_ORDER):
        env = MarketingCampaignEnv(task_id=task_id, seed=seed + idx)
        obs = env.reset(task_id=task_id, seed=seed + idx)
        done = False
        total_reward = 0.0
        steps = 0
        info: Dict = {}

        while not done:
            obs_json = obs.model_dump()
            try:
                action = heuristic_action(obs_json) if use_heuristic else _llm_action(client, model_name, obs_json)
            except Exception:
                action = heuristic_action(obs_json)

            obs, reward, done, info = env.step(action)
            steps += 1
            total_reward += reward.value

            reward_0_1 = max(0.0, min(1.0, (reward.value + 1.0) / 2.0))
            _log(
                "STEP",
                {
                    "task_id": task_id,
                    "step": steps,
                    "action_type": action.action_type,
                    "reward_0_1": round(reward_0_1, 6),
                    "done": done,
                },
            )

        grade = info.get("grade", {})
        score = float(grade.get("score", 0.0))
        if score < 0.0 or score > 1.0:
            raise RuntimeError(f"Grader score out of range for {task_id}: {score}")

        result = {
            "task_id": task_id,
            "score": round(score, 6),
            "total_reward": round(total_reward, 6),
            "steps": steps,
            "termination_reason": info.get("termination_reason", "unknown"),
        }
        task_results.append(result)
        _log("END", result)

    avg_score = sum(t["score"] for t in task_results) / max(1, len(task_results))
    summary = {
        "avg_score": round(avg_score, 6),
        "task_count": len(task_results),
        "seed": seed,
    }
    _log("END", summary)
    return {"summary": summary, "tasks": task_results}


def main() -> None:
    parser = argparse.ArgumentParser(description="Submission inference runner")
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    output = run(seed=args.seed)
    with open(".inference_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()
