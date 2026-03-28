from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .graders import grade_episode
from .models import Action, ChannelSnapshot, EpisodeState, Observation, Reward
from .tasks import TASKS, get_task


@dataclass(frozen=True)
class ChannelTemplate:
    channel: str
    base_ctr: float
    base_cvr: float
    base_cpc: float
    base_volume: float


CHANNEL_TEMPLATES: List[ChannelTemplate] = [
    ChannelTemplate("search", base_ctr=0.034, base_cvr=0.12, base_cpc=1.65, base_volume=2200),
    ChannelTemplate("social", base_ctr=0.020, base_cvr=0.06, base_cpc=1.05, base_volume=2600),
    ChannelTemplate("display", base_ctr=0.012, base_cvr=0.03, base_cpc=0.72, base_volume=3400),
    ChannelTemplate("video", base_ctr=0.015, base_cvr=0.04, base_cpc=1.18, base_volume=1900),
]


class MarketingCampaignEnv:
    """OpenEnv-style environment for real-world ad campaign optimization."""

    def __init__(self, task_id: str = "easy_ctr_recovery", seed: int = 7):
        self._base_seed = seed
        self._rng = np.random.default_rng(seed)
        self._task_id = task_id
        self._task = get_task(task_id)
        self._state: EpisodeState | None = None

    @property
    def tasks(self) -> List[str]:
        return sorted(TASKS.keys())

    def reset(self, task_id: str | None = None, seed: int | None = None) -> Observation:
        if task_id is not None:
            self._task_id = task_id
            self._task = get_task(task_id)
        if seed is not None:
            self._base_seed = seed
        self._rng = np.random.default_rng(self._base_seed)

        base_budget = self._task.total_budget
        per_channel_budget = base_budget / len(CHANNEL_TEMPLATES)
        channels = []
        for template in CHANNEL_TEMPLATES:
            quality = float(self._rng.uniform(0.85, 1.05))
            channels.append(
                ChannelSnapshot(
                    channel=template.channel,
                    active=True,
                    budget=per_channel_budget,
                    bid=1.0,
                    quality_score=quality,
                    fatigue=0.0,
                    impressions=0,
                    clicks=0,
                    conversions=0,
                    spend=0.0,
                    revenue=0.0,
                )
            )

        hidden_segment_response = {
            "price_sensitive": float(self._rng.uniform(0.8, 1.3)),
            "impulse_buyers": float(self._rng.uniform(0.85, 1.4)),
            "high_intent": float(self._rng.uniform(0.9, 1.35)),
        }

        self._state = EpisodeState(
            task_id=self._task_id,
            step_index=0,
            max_steps=self._task.max_steps,
            total_budget=self._task.total_budget,
            channels=channels,
            total_spend=0.0,
            total_revenue=0.0,
            total_clicks=0,
            total_conversions=0,
            done=False,
            invalid_action_count=0,
            action_history=[],
            reward_history=[],
            task_progress_metric=0.0,
            hidden_segment_response=hidden_segment_response,
        )
        return self._build_observation(last_feedback="Environment reset.")

    def state(self) -> EpisodeState:
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        return self._state.model_copy(deep=True)

    def step(self, action: Action | Dict) -> Tuple[Observation, Reward, bool, Dict]:
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        if self._state.done:
            obs = self._build_observation(last_feedback="Episode already done. Call reset().")
            reward = Reward(value=0.0, components={"post_done": 0.0}, rationale="No-op after done")
            return obs, reward, True, {"grade": grade_episode(self._task, self._state)}

        parsed_action = Action.model_validate(action)
        action_feedback, action_penalty = self._apply_action(parsed_action)
        sim_feedback = self._simulate_market_tick()

        previous_progress = self._state.task_progress_metric
        current_progress = self._progress_metric()
        self._state.task_progress_metric = current_progress

        progress_delta = current_progress - previous_progress
        spend_penalty = max(0.0, self._state.total_spend - self._state.total_budget) / self._state.total_budget
        invalid_penalty = 0.04 * self._state.invalid_action_count
        loop_penalty = self._loop_penalty()

        reward_value = (
            0.65 * progress_delta
            - 0.50 * spend_penalty
            - action_penalty
            - invalid_penalty
            - loop_penalty
        )

        self._state.step_index += 1
        done = self._state.step_index >= self._state.max_steps
        self._state.done = done

        info: Dict = {
            "task_id": self._task_id,
            "step_index": self._state.step_index,
        }

        if done:
            grade = grade_episode(self._task, self._state)
            terminal_bonus = 0.35 * grade["score"]
            reward_value += terminal_bonus
            info["grade"] = grade
            info["terminal_bonus"] = terminal_bonus

        reward_value = float(np.clip(reward_value, -1.0, 1.0))
        self._state.reward_history.append(reward_value)

        reward = Reward(
            value=reward_value,
            components={
                "progress_delta": float(progress_delta),
                "spend_penalty": float(-0.50 * spend_penalty),
                "action_penalty": float(-action_penalty),
                "invalid_penalty": float(-invalid_penalty),
                "loop_penalty": float(-loop_penalty),
            },
            rationale=(
                f"{action_feedback} {sim_feedback}"
                f" Progress shift={progress_delta:.4f}; overspend={spend_penalty:.4f}."
            ),
        )

        obs = self._build_observation(last_feedback=reward.rationale)
        return obs, reward, done, info

    def _build_observation(self, last_feedback: str) -> Observation:
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        total_impressions = max(1, sum(ch.impressions for ch in self._state.channels))
        total_clicks = self._state.total_clicks
        total_conversions = self._state.total_conversions

        avg_ctr = total_clicks / total_impressions
        avg_cvr = total_conversions / max(1, total_clicks)
        avg_cpa = self._state.total_spend / max(1, total_conversions)
        roas = self._state.total_revenue / max(1e-9, self._state.total_spend)

        return Observation(
            task_id=self._task_id,
            task_difficulty=self._task.difficulty,
            objective=self._task.objective,
            step_index=self._state.step_index,
            max_steps=self._state.max_steps,
            remaining_budget=max(0.0, self._state.total_budget - self._state.total_spend),
            total_spend=self._state.total_spend,
            total_revenue=self._state.total_revenue,
            total_clicks=total_clicks,
            total_conversions=total_conversions,
            avg_ctr=avg_ctr,
            avg_cvr=avg_cvr,
            avg_cpa=avg_cpa,
            roas=roas,
            channels=[ch.model_copy(deep=True) for ch in self._state.channels],
            last_action_feedback=last_feedback,
        )

    def _apply_action(self, action: Action) -> Tuple[str, float]:
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        channel_map = {ch.channel: ch for ch in self._state.channels}
        feedback = ""
        penalty = 0.0

        if action.action_type == "wait":
            feedback = "Agent waited this turn."

        elif action.action_type == "adjust_bid":
            if not action.channel or action.delta is None or action.channel not in channel_map:
                self._state.invalid_action_count += 1
                return "Invalid adjust_bid action.", 0.08
            ch = channel_map[action.channel]
            old = ch.bid
            ch.bid = float(np.clip(ch.bid * (1.0 + action.delta), 0.4, 2.5))
            feedback = f"Adjusted bid for {ch.channel} from {old:.2f} to {ch.bid:.2f}."

        elif action.action_type == "shift_budget":
            if (
                not action.from_channel
                or not action.to_channel
                or action.amount is None
                or action.from_channel not in channel_map
                or action.to_channel not in channel_map
                or action.from_channel == action.to_channel
            ):
                self._state.invalid_action_count += 1
                return "Invalid shift_budget action.", 0.1

            source = channel_map[action.from_channel]
            target = channel_map[action.to_channel]
            amount = min(action.amount, source.budget * 0.8)
            if amount <= 0.0:
                self._state.invalid_action_count += 1
                return "Budget shift amount too low.", 0.06
            source.budget -= amount
            target.budget += amount
            feedback = (
                f"Shifted ${amount:.2f} budget from {source.channel} to {target.channel}."
            )

        elif action.action_type == "pause_channel":
            if not action.channel or action.channel not in channel_map:
                self._state.invalid_action_count += 1
                return "Invalid pause_channel action.", 0.07
            ch = channel_map[action.channel]
            ch.active = False
            feedback = f"Paused channel {ch.channel}."

        elif action.action_type == "resume_channel":
            if not action.channel or action.channel not in channel_map:
                self._state.invalid_action_count += 1
                return "Invalid resume_channel action.", 0.07
            ch = channel_map[action.channel]
            ch.active = True
            feedback = f"Resumed channel {ch.channel}."

        elif action.action_type == "create_variant":
            if not action.channel or action.channel not in channel_map:
                self._state.invalid_action_count += 1
                return "Invalid create_variant action.", 0.08
            ch = channel_map[action.channel]
            ch.quality_score = float(np.clip(ch.quality_score + 0.06, 0.5, 1.6))
            ch.fatigue = float(np.clip(ch.fatigue - 0.10, 0.0, 1.0))
            feedback = f"Created new creative variant for {ch.channel}."
            penalty = 0.01

        else:
            self._state.invalid_action_count += 1
            return "Unknown action type.", 0.10

        self._state.action_history.append(action.model_dump_json())
        return feedback, penalty

    def _simulate_market_tick(self) -> str:
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        segment = self._state.hidden_segment_response
        total_step_spend = 0.0
        total_step_revenue = 0.0
        total_step_clicks = 0
        total_step_conversions = 0

        for template, channel_state in zip(CHANNEL_TEMPLATES, self._state.channels):
            if not channel_state.active:
                channel_state.fatigue = float(np.clip(channel_state.fatigue - 0.02, 0.0, 1.0))
                continue

            fatigue_multiplier = max(0.45, 1.0 - 0.75 * channel_state.fatigue)
            bid_multiplier = np.clip(channel_state.bid, 0.5, 2.3)
            quality = channel_state.quality_score

            impressions = int(
                template.base_volume
                * bid_multiplier
                * quality
                * fatigue_multiplier
                * self._rng.uniform(0.78, 1.22)
            )
            impressions = max(80, impressions)

            ctr = template.base_ctr * quality * fatigue_multiplier * self._rng.uniform(0.85, 1.18)
            ctr = float(np.clip(ctr, 0.003, 0.11))
            clicks = int(self._rng.binomial(impressions, ctr))

            segment_mix = 0.35 * segment["price_sensitive"] + 0.30 * segment["impulse_buyers"] + 0.35 * segment["high_intent"]
            cvr = template.base_cvr * quality * segment_mix * self._rng.uniform(0.80, 1.15)
            cvr = float(np.clip(cvr, 0.005, 0.35))
            conversions = int(self._rng.binomial(max(1, clicks), cvr))

            cpc = template.base_cpc * (0.85 + 0.4 * bid_multiplier) * self._rng.uniform(0.90, 1.15)
            spend = float(min(channel_state.budget, clicks * cpc))
            revenue_per_conversion = 38.0 * self._rng.uniform(0.92, 1.22)
            revenue = float(conversions * revenue_per_conversion)

            channel_state.impressions += impressions
            channel_state.clicks += clicks
            channel_state.conversions += conversions
            channel_state.spend += spend
            channel_state.revenue += revenue
            channel_state.budget = max(0.0, channel_state.budget - spend)

            fatigue_gain = 0.05 + 0.14 * (spend / max(1e-9, self._task.total_budget / len(CHANNEL_TEMPLATES)))
            channel_state.fatigue = float(np.clip(channel_state.fatigue + fatigue_gain, 0.0, 1.0))

            total_step_spend += spend
            total_step_revenue += revenue
            total_step_clicks += clicks
            total_step_conversions += conversions

        self._state.total_spend += total_step_spend
        self._state.total_revenue += total_step_revenue
        self._state.total_clicks += total_step_clicks
        self._state.total_conversions += total_step_conversions

        return (
            f"Tick results: spend=${total_step_spend:.2f}, "
            f"revenue=${total_step_revenue:.2f}, conversions={total_step_conversions}."
        )

    def _progress_metric(self) -> float:
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        total_impressions = max(1, sum(ch.impressions for ch in self._state.channels))
        ctr = self._state.total_clicks / total_impressions
        roas = self._state.total_revenue / max(1e-9, self._state.total_spend)
        cpa = self._state.total_spend / max(1, self._state.total_conversions)

        ctr_norm = np.clip(ctr / self._task.target_ctr, 0.0, 1.25)
        roas_norm = np.clip(roas / self._task.target_roas, 0.0, 1.25)
        cpa_norm = np.clip(self._task.target_cpa / max(1e-9, cpa), 0.0, 1.25)
        conv_norm = np.clip(self._state.total_conversions / self._task.target_conversions, 0.0, 1.25)

        if self._task.difficulty.value == "easy":
            progress = 0.50 * ctr_norm + 0.30 * conv_norm + 0.20 * roas_norm
        elif self._task.difficulty.value == "medium":
            progress = 0.45 * conv_norm + 0.30 * roas_norm + 0.25 * cpa_norm
        else:
            avg_fatigue = sum(ch.fatigue for ch in self._state.channels) / len(self._state.channels)
            fatigue_norm = np.clip(1.0 - avg_fatigue, 0.0, 1.25)
            progress = 0.35 * conv_norm + 0.25 * roas_norm + 0.20 * cpa_norm + 0.20 * fatigue_norm

        return float(progress)

    def _loop_penalty(self) -> float:
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        if len(self._state.action_history) < 3:
            return 0.0
        recent = self._state.action_history[-3:]
        if len(set(recent)) == 1:
            return 0.03
        a, b, c = recent
        if a == c and a != b:
            return 0.02
        return 0.0
