# DICT Action Usage Analysis

## Overview

This analysis evaluates the efficiency of the **DICT action** (oracle lookup) across all RL algorithms. The DICT action always returns the correct POS tag but comes with a steep cost (-0.9 penalty). 

The key question: **When agents choose DICT, was it truly necessary, or would KEEP have been sufficient?**

## Metrics

### DICT Necessity Rate
**When DICT was chosen, how often was the base tagger wrong?**

- **High (>70%)**: Smart use of expensive oracle action - agent only uses DICT when base tagger would fail
- **Medium (40-70%)**: Moderate efficiency - some unnecessary DICT usage
- **Low (<40%)**: Wasteful - agent frequently overrides correct base predictions

### DICT Waste Rate
**When DICT was chosen, how often was the base tagger already correct?**

- **High (>30%)**: Overusing DICT when KEEP would suffice - wasted cost
- **Low (<30%)**: Efficient - agent reserves DICT for difficult cases

## How to Run Analysis

### 1. During Training (Automatic)

All three algorithm scripts now automatically analyze DICT usage after training:

```bash
python Algorithms/BaseTaggerPOS/q-learning.py
python Algorithms/BaseTaggerPOS/dqn.py
python Algorithms/BaseTaggerPOS/reinforce.py
```

Output will include:
```
--- DICT Action Usage Analysis ---

Q-Learning DICT Analysis:
  Total DICT uses: 42
  Necessary (base was wrong): 38 (90.5%)
  Unnecessary (base was correct): 4 (9.5%)
  ✓ Efficient use of DICT action
```

### 2. Standalone Analysis Script

Run analysis on all trained models at once:

```bash
cd Algorithms/BaseTaggerPOS
python analyze_dict_usage.py
```

This will:
- Load all trained models (Q-Learning, SARSA, DQN, REINFORCE)
- Evaluate DICT usage for each
- Generate a summary comparison table

### 3. Comprehensive Analysis (Integrated)

The `comprehensive_analysis.py` script now includes DICT analysis:

```bash
python Algorithms/BaseTaggerPOS/comprehensive_analysis.py
```

DICT statistics are included in:
- Console output during evaluation
- Results tables (with "DICT Necessary %" and "DICT Waste %" columns)
- Saved JSON results files

## Code Integration

### New Function: `analyze_dict_usage()`

Located in `Algorithms/BaseTaggerPOSUtils/rl_utils.py`:

```python
def analyze_dict_usage(agent, env, obs_to_state_fn, num_episodes=100, 
                       is_dqn=False, is_policy=False):
    """
    Analyze DICT action usage: when DICT is chosen, check if KEEP would 
    have been correct.
    
    Returns:
        dict: Statistics including dict_total, dict_necessary, 
              dict_unnecessary, necessity_rate, waste_rate
    """
```

### Usage Examples

**For Tabular Methods (Q-Learning, SARSA):**
```python
from Algorithms.BaseTaggerPOSUtils.rl_utils import analyze_dict_usage, obs_to_discrete_state

dict_stats = analyze_dict_usage(
    agent=Q_table,
    env=test_env,
    obs_to_state_fn=obs_to_discrete_state,
    num_episodes=50,
    is_dqn=False,
    is_policy=False
)
```

**For DQN (Stable-Baselines3):**
```python
dict_stats = analyze_dict_usage(
    agent=dqn_model,
    env=test_env,
    obs_to_state_fn=obs_to_tensor,
    num_episodes=50,
    is_dqn=True,
    is_policy=False
)
```

**For REINFORCE (Policy Network):**
```python
dict_stats = analyze_dict_usage(
    agent=policy_network,
    env=test_env,
    obs_to_state_fn=obs_to_tensor,
    num_episodes=50,
    is_dqn=False,
    is_policy=True
)
```

## Interpretation Guide

### Scenario 1: High Necessity (>70%)
**Example:** DICT used 50 times, 45 necessary (90%)

**Interpretation:** Agent has learned to use DICT judiciously - only when the base tagger is likely wrong. This demonstrates:
- Good state representation
- Effective learning of when to trust base predictions
- Cost-benefit optimization (paying the -0.9 penalty only when necessary)

### Scenario 2: High Waste (>30%)
**Example:** DICT used 100 times, 40 necessary (40%), 60 unnecessary (60%)

**Interpretation:** Agent overuses DICT, often overriding correct base predictions. Possible causes:
- Insufficient penalty for DICT action
- Poor confidence feature representation
- Exploration artifacts (not fully converged)
- Model hasn't learned to trust the base tagger

### Scenario 3: Never Used
**Example:** DICT used 0 times

**Interpretation:** Agent learned to avoid the expensive DICT action entirely. This could indicate:
- Penalty is too high (agent prefers KEEP/SHIFT even when wrong)
- Base tagger + SHIFT are sufficient for most cases
- Limited exploration during training

## Expected Behavior

Based on the reward structure:
- **DICT correct**: +1.0 reward, -0.9 cost = **+0.1 net**
- **KEEP/SHIFT correct**: +1.0 reward = **+1.0 net**

Optimal agents should:
1. Prefer KEEP when base tagger is confident and likely correct
2. Use SHIFT to correct errors when second prediction is better
3. Reserve DICT for high-stakes cases where KEEP/SHIFT would fail

**Target Efficiency:** >70% necessity rate indicates intelligent DICT usage.

## Files Modified

- `Algorithms/BaseTaggerPOSUtils/rl_utils.py`: Added `analyze_dict_usage()` function
- `Algorithms/BaseTaggerPOS/q-learning.py`: Integrated DICT analysis after training
- `Algorithms/BaseTaggerPOS/dqn.py`: Integrated DICT analysis after training
- `Algorithms/BaseTaggerPOS/reinforce.py`: Integrated DICT analysis after training
- `Algorithms/BaseTaggerPOS/comprehensive_analysis.py`: Added DICT tracking to evaluation
- `Algorithms/BaseTaggerPOS/analyze_dict_usage.py`: Standalone analysis script

## Results Persistence

DICT analysis results are saved to JSON files:
- `tabular_results.json`: Q-Learning and SARSA DICT stats
- `dqn_results.json`: DQN DICT stats
- `reinforce_results.json`: REINFORCE DICT stats

Each includes a `dict_analysis` field with full statistics.

## Future Enhancements

Potential extensions:
1. **Per-Tag DICT Analysis**: Which POS tags trigger DICT usage?
2. **Confidence Correlation**: Do low-confidence states lead to more DICT usage?
3. **Temporal Patterns**: Does DICT usage change during training?
4. **Reward Sensitivity**: How does DICT penalty affect usage patterns?
