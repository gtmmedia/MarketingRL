from __future__ import annotations

from typing import Dict

from .models import EpisodeState, TaskSpec


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 1e-9:
        return 0.0
    return numerator / denominator


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _diversity_score(state: EpisodeState) -> float:
    spends = [ch.spend for ch in state.channels if ch.spend > 0.0]
    if not spends:
        return 0.0
    total = sum(spends)
    if total <= 1e-9:
        return 0.0
    proportions = [s / total for s in spends]
    # Normalized entropy-like measure.
    entropy = -sum(p * (0.0 if p <= 1e-9 else __import__("math").log(p)) for p in proportions)
    max_entropy = __import__("math").log(len(proportions)) if len(proportions) > 1 else 1.0
    return _clamp01(entropy / max_entropy)


def _stability_score(state: EpisodeState) -> float:
    if len(state.action_history) <= 2:
        return 1.0
    oscillations = 0
    for i in range(2, len(state.action_history)):
        a = state.action_history[i - 2]
        b = state.action_history[i - 1]
        c = state.action_history[i]
        if a == c and a != b:
            oscillations += 1
    penalty = oscillations / max(1, len(state.action_history) - 2)
    return _clamp01(1.0 - penalty)


def grade_episode(task: TaskSpec, state: EpisodeState) -> Dict[str, float]:
    ctr = _safe_ratio(state.total_clicks, max(1, sum(ch.impressions for ch in state.channels)))
    roas = _safe_ratio(state.total_revenue, max(1e-9, state.total_spend))
    cpa = _safe_ratio(state.total_spend, max(1, state.total_conversions))

    ctr_component = _clamp01(_safe_ratio(ctr, task.target_ctr))
    budget_component = _clamp01(
        1.0 - _safe_ratio(max(0.0, state.total_spend - task.total_budget), task.total_budget)
    )
    conv_component = _clamp01(_safe_ratio(state.total_conversions, task.target_conversions))
    roas_component = _clamp01(_safe_ratio(roas, task.target_roas))
    cpa_component = _clamp01(_safe_ratio(task.target_cpa, max(1e-9, cpa)))

    if task.difficulty.value == "easy":
        score = 0.40 * ctr_component + 0.30 * budget_component + 0.30 * conv_component
    elif task.difficulty.value == "medium":
        score = 0.35 * conv_component + 0.35 * roas_component + 0.30 * cpa_component
    else:
        avg_fatigue = sum(ch.fatigue for ch in state.channels) / max(1, len(state.channels))
        fatigue_component = _clamp01(1.0 - avg_fatigue)
        diversity_component = _diversity_score(state)
        stability_component = _stability_score(state)
        score = (
            0.30 * conv_component
            + 0.25 * fatigue_component
            + 0.20 * diversity_component
            + 0.25 * stability_component
        )

    score = _clamp01(score)

    return {
        "score": score,
        "ctr": ctr,
        "roas": roas,
        "cpa": cpa,
        "conversions": float(state.total_conversions),
    }
