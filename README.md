# 📈 Marketing Campaign Optimizer (OpenEnv)

A **deterministic, OpenEnv-style marketing campaign simulation environment** for testing policies, heuristics, and **LLM-based decision agents** in multi-channel ad systems.

This environment models **real-world marketing dynamics** such as budget constraints, bid tuning, creative fatigue, auction pressure, and hidden user segments—while remaining **fast, reproducible, and cheap to run**.

---

## 🚀 What This Is

- Step-based decision environment  
- Deterministic & reproducible  
- Designed for inference-time agents  
- Compatible with OpenEnv benchmarks  
- Suitable for:
  - LLM agents
  - Rule-based systems
  - Classical control logic  

> ⚠️ This is **not** a toy simulator — it enforces realistic constraints and penalties.

---

## 🧠 What’s Modeled

### 📊 Multi-Channel Campaigns
- `search`
- `social`
- `display`
- `video`

### 🎯 Core Mechanics
- Budget allocation & bid adjustments  
- Creative fatigue & recovery  
- Auction pressure  
- Overspend guardrails  

### 👤 Hidden User Segments
- Price-sensitive users  
- Impulse buyers  
- High-intent users  

### 📈 Core Metrics
- CTR (Click-Through Rate)  
- CVR (Conversion Rate)  
- CPA (Cost Per Acquisition)  
- ROAS (Return on Ad Spend)  

### ⚠️ Penalties
- Overspend  
- Oscillation & instability  
- Invalid or destructive actions  

---

## 🧱 Project Structure

```text
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
```

---

## 🎯 Tasks

### 1️⃣ `easy_ctr_recovery` (Easy)

**Objective:** Recover CTR while staying budget disciplined

**Signals:**
- CTR improvement  
- Budget control  
- Conversion progress  

---

### 2️⃣ `medium_conversion_push` (Medium)

**Objective:** Increase conversions efficiently

**Signals:**
- Conversion target  
- ROAS target  
- CPA control  

---

### 3️⃣ `hard_multi_segment_stability` (Hard)

**Objective:** Manage volatility across segments without instability

**Signals:**
- Conversion achievement  
- Fatigue control  
- Spend diversity  
- Policy stability  

✅ All graders return **deterministic scores in `[0.0, 1.0]`**

---

## 🎮 Action Space

Supported actions:

- `adjust_bid(channel, delta)`
- `shift_budget(from_channel, to_channel, amount)`
- `pause_channel(channel)`
- `resume_channel(channel)`
- `create_variant(channel)`
- `wait`

❌ Invalid or destructive actions are **penalized**.

---

## 👀 Observation Space

Each step returns:

### Episode Info
- `step_index`
- `max_steps`

### Aggregate Metrics
- Spend  
- Revenue  
- Clicks  
- Conversions  

### Efficiency Metrics
- `avg_ctr`
- `avg_cvr`
- `avg_cpa`
- `roas`

### Per-Channel Snapshot
- Budget  
- Bid  
- Quality score  
- Fatigue  
- Impressions  
- Clicks  
- Conversions  

### Feedback
- Natural-language feedback from the **previous step**

---

## 🏆 Reward Design

Reward is **dense and shaped**:

- ✅ Positive signal for objective progress  
- ❌ Penalty for overspend  
- ❌ Penalty for invalid actions  
- ❌ Penalty for oscillation or loops  
- 🎁 Terminal bonus based on final task grade  

This encourages **stable, disciplined decision-making**.

---

## ⚡ Quick Start

### Setup

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
export PYTHONPATH=src
```

### Minimal Usage

```python
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
```

---

## 🧪 Inference Runner

Run:

```bash
python inference.py --seed 11
```

### Output Contract

```text
[START] task= env= model=
[STEP]  step= action= reward= done= error=
[END]   success= steps= score= rewards=
```

Results are saved to:

```text
.inference_results.json
```

---

## 🔑 Environment Variables

Used by `inference.py`:

```text
API_BASE_URL
MODEL_NAME
HF_TOKEN
API_KEY
MY_ENV_V4_TASK
MY_ENV_V4_BENCHMARK
```

---

## 🤖 Baseline Agent

Run baseline:

```bash
python -m marketing_openenv.baseline --seed 11
```

Outputs:
- Per-task and average scores  
- Saves results to:

```text
.baseline_results.json
```

---

## 🐳 Docker

Build & run:

```bash
docker build -t marketing-openenv .
docker run -p 7860:7860 marketing-openenv
```

Endpoints:
- `/health`
- `/`
- `/baseline`

---

## 🎯 Intended Use Cases

- LLM policy evaluation  
- Heuristic decision systems  
- Control & optimization experiments  
- Marketing strategy simulations  
- OpenEnv benchmark submissions  

---

## 📜 License

MIT License
