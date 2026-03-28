from __future__ import annotations

from enum import Enum
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class ChannelSnapshot(BaseModel):
    channel: str
    active: bool
    budget: float = Field(ge=0.0)
    bid: float = Field(ge=0.0)
    quality_score: float = Field(ge=0.0, le=2.0)
    fatigue: float = Field(ge=0.0, le=1.0)
    impressions: int = Field(ge=0)
    clicks: int = Field(ge=0)
    conversions: int = Field(ge=0)
    spend: float = Field(ge=0.0)
    revenue: float = Field(ge=0.0)


class Observation(BaseModel):
    task_id: str
    task_difficulty: Difficulty
    objective: str
    step_index: int = Field(ge=0)
    max_steps: int = Field(ge=1)
    remaining_budget: float = Field(ge=0.0)
    total_spend: float = Field(ge=0.0)
    total_revenue: float = Field(ge=0.0)
    total_clicks: int = Field(ge=0)
    total_conversions: int = Field(ge=0)
    avg_ctr: float = Field(ge=0.0, le=1.0)
    avg_cvr: float = Field(ge=0.0, le=1.0)
    avg_cpa: float = Field(ge=0.0)
    roas: float = Field(ge=0.0)
    channels: List[ChannelSnapshot]
    last_action_feedback: str


class Action(BaseModel):
    action_type: Literal[
        "adjust_bid",
        "shift_budget",
        "pause_channel",
        "resume_channel",
        "create_variant",
        "wait",
    ]
    channel: Optional[str] = None
    delta: Optional[float] = Field(default=None, ge=-0.7, le=0.7)
    from_channel: Optional[str] = None
    to_channel: Optional[str] = None
    amount: Optional[float] = Field(default=None, ge=0.0)


class Reward(BaseModel):
    value: float = Field(ge=-1.0, le=1.0)
    components: Dict[str, float]
    rationale: str


class TaskSpec(BaseModel):
    task_id: str
    title: str
    difficulty: Difficulty
    objective: str
    max_steps: int
    total_budget: float
    target_ctr: float
    target_roas: float
    target_cpa: float
    target_conversions: int


class EpisodeState(BaseModel):
    task_id: str
    step_index: int
    max_steps: int
    total_budget: float
    channels: List[ChannelSnapshot]
    total_spend: float
    total_revenue: float
    total_clicks: int
    total_conversions: int
    done: bool
    invalid_action_count: int
    action_history: List[str]
    reward_history: List[float]
    task_progress_metric: float
    hidden_segment_response: Dict[str, float]
