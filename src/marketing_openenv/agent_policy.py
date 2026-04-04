from __future__ import annotations

import json
from typing import Any, Dict

from .models import Action


def extract_json_object(raw: str) -> Dict[str, Any]:
    """Parse the first JSON object from model output."""
    raw = (raw or "").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(raw[start : end + 1])
    raise ValueError("Model output is not valid JSON")


def heuristic_action(observation: Dict[str, Any]) -> Action:
    """Cheap deterministic policy used for offline baseline and fallback."""
    channels = observation.get("channels", [])
    if not channels:
        return Action(action_type="wait")

    if observation.get("avg_ctr", 0.0) < 0.02:
        return Action(action_type="adjust_bid", channel="search", delta=0.15)

    high_fatigue = max(channels, key=lambda c: c.get("fatigue", 0.0))
    if high_fatigue.get("fatigue", 0.0) > 0.62:
        return Action(
            action_type="create_variant",
            channel=high_fatigue.get("channel", "search"),
        )

    by_efficiency = sorted(
        channels,
        key=lambda c: c.get("conversions", 0) / max(1, c.get("clicks", 0)),
    )
    donor = by_efficiency[0]
    receiver = by_efficiency[-1]
    if donor.get("channel") != receiver.get("channel") and donor.get("budget", 0.0) > 10.0:
        return Action(
            action_type="shift_budget",
            from_channel=donor.get("channel"),
            to_channel=receiver.get("channel"),
            amount=min(40.0, float(donor.get("budget", 0.0)) * 0.25),
        )

    return Action(action_type="wait")
