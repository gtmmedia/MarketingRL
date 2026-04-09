📈 Marketing Campaign Optimizer (OpenEnv)

A deterministic, OpenEnv-style marketing campaign simulation environment for testing policies, heuristics, and LLM-based decision agents in multi-channel ad systems.

This environment models real-world marketing dynamics such as budget constraints, bid tuning, creative fatigue, auction pressure, and hidden user segments, while remaining fast, reproducible, and cheap to run.

🚀 What This Is
A step-based decision environment
Deterministic & reproducible
Designed for inference-time agents
Compatible with OpenEnv benchmarks
Suitable for LLMs, rule-based systems, and classical control logic

This is not a toy simulator — it enforces realistic constraints and penalties.

🧠 What’s Modeled
Multi-channel campaigns:
search, social, display, video
Budget allocation & bid adjustments
Creative fatigue & recovery
Hidden user segments:
price-sensitive
impulse buyers
high-intent users
Core metrics:
CTR, CVR, CPA, ROAS
Overspend guardrails
Oscillation & instability penalties
🧱 Project Structure
.
├── Dockerfile
├── openenv.yaml
├── inference.py
├── README.md
├── requirements.txt
├── scripts/
│   └── run_baseline.py
└── src/
    └── marketing_openenv/
        ├── __init__.py
        ├── app.py
        ├── baseline.py
        ├── env.py          # Core environment
        ├── graders.py      # Deterministic scoring
        ├── models.py       # Typed state/action/reward models
        └── tasks.py        # Task definitions
🎯 Tasks
1️⃣ easy_ctr_recovery (Easy)

Objective: Recover CTR while staying budget disciplined
Signals:

CTR improvement
Budget control
Conversion progress
2️⃣ medium_conversion_push (Medium)

Objective: Increase conversions efficiently
Signals:

Conversion target
ROAS target
CPA control
3️⃣ hard_multi_segment_stability (Hard)

Objective: Manage volatility across segments without instability
Signals:

Conversion achievement
Fatigue control
Spend diversity
Policy stability

All graders return deterministic scores in [0.0, 1.0].

🎮 Action Space

Supported actions:

adjust_bid(channel, delta)
shift_budget(from_channel, to_channel, amount)
pause_channel(channel)
resume_channel(channel)
create_variant(channel)
wait

Invalid or destructive actions are penalized.

👀 Observation Space

Each step returns:

Episode progress (step_index, max_steps)
Spend, revenue, clicks, conversions
Efficiency metrics:
avg_ctr
avg_cvr
avg_cpa
roas
Per-channel snapshots:
budget
bid
quality score
fatigue
impressions, clicks, conversions
Natural-language feedback from the previous step
🏆 Reward Design

Reward is dense and shaped:

Positive signal for objective progress
Penalty for overspend
Penalty for invalid actions
Penalty for oscillation or loops
Terminal bonus based on final task grade

This encourages stable, disciplined decision-making.

⚡ Quick Start
Setup
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
export PYTHONPATH=src
Minimal Usage
'''python

from marketing_openenv.env import MarketingCampaignEnv
from marketing_openenv.models import Action

env = MarketingCampaignEnv(task_id="easy_ctr_recovery", seed=11)
obs = env.reset()

action = Action(
    action_type="adjust_bid",
    channel="search",
    delta=0.15
)

obs, reward, done, info = env.step(action)

print(reward.value, done, info)
'''
🧪 Inference Runner

A root inference script is provided:

python inference.py --seed 11
Output Contract
[START] task= env= model=
[STEP] step= action= reward= done= error=
[END] success= steps= score= rewards=

Results are saved to .inference_results.json.

🔑 Environment Variables

Used by inference.py:

API_BASE_URL
MODEL_NAME
HF_TOKEN
API_KEY
MY_ENV_V4_TASK
MY_ENV_V4_BENCHMARK
🤖 Baseline Agent

A baseline policy runner is included:

python -m marketing_openenv.baseline --seed 11
Uses deterministic heuristics or LLM-based policies
Outputs per-task and average scores
Saves results to .baseline_results.json
🐳 Docker
docker build -t marketing-openenv .
docker run -p 7860:7860 marketing-openenv

Endpoints:

/health
/
/baseline
🎯 Intended Use Cases
LLM policy evaluation
Heuristic decision systems
Control & optimization experiments
Marketing strategy simulations
OpenEnv benchmark submissions
