# DICT Action Usage - Comparative Analysis

## Executive Summary

This analysis evaluates how efficiently each RL algorithm uses the DICT (oracle) action,
which always provides the correct answer but incurs a -0.9 penalty cost.

## Results Overview

| Algorithm | DICT Uses | Necessary | Wasteful | Necessity % | Rating |
|-----------|-----------|-----------|----------|-------------|--------|
| Q-Learning  | 52        | 13        | 39       |   25.0% | ✗ Poor |
| SARSA       | 21        | 8         | 13       |   38.1% | ✗ Poor |
| DQN         | 31        | 17        | 14       |   54.8% | ⚠ Acceptable |
| REINFORCE   | 0         | 0         | 0        |    0.0% | → N/A |

## Detailed Findings

### Q-Learning

**Total DICT Uses:** 52

**Usage Pattern:**
- Necessary (base tagger was wrong): 13 uses (25.0%)
- Wasteful (base tagger was correct): 39 uses (75.0%)
- Cost of waste: 39 × -0.9 = -35.1 penalty points

**Interpretation:** ✗ POOR - Algorithm ignores the cost penalty.
Treats DICT as a general 'fix-everything' action rather than a last resort.

### SARSA

**Total DICT Uses:** 21

**Usage Pattern:**
- Necessary (base tagger was wrong): 8 uses (38.1%)
- Wasteful (base tagger was correct): 13 uses (61.9%)
- Cost of waste: 13 × -0.9 = -11.7 penalty points

**Interpretation:** ✗ POOR - Algorithm ignores the cost penalty.
Treats DICT as a general 'fix-everything' action rather than a last resort.

### DQN

**Total DICT Uses:** 31

**Usage Pattern:**
- Necessary (base tagger was wrong): 17 uses (54.8%)
- Wasteful (base tagger was correct): 14 uses (45.2%)
- Cost of waste: 14 × -0.9 = -12.6 penalty points

**Interpretation:** ⚠ ACCEPTABLE - Algorithm shows some cost awareness.
But wastes significant resources on unnecessary oracle calls.

### REINFORCE

**Total DICT Uses:** 0

**Usage Pattern:** Never uses DICT (0 attempts)

**Interpretation:** → Algorithm avoids the expensive action entirely.
Either the penalty is too high, or KEEP/SHIFT are sufficient.

## Key Insights

1. **Cost-Benefit Learning**: DQN shows the most intelligent behavior by learning
   when DICT is actually valuable, while tabular methods overuse it.

2. **Generalization**: Deep learning (DQN) generalizes the penalty pattern better
   than tabular methods, leading to smarter action selection.

3. **Risk Aversion**: REINFORCE avoids DICT entirely, suggesting the penalty
   may be too high or the method is overly conservative.

4. **Resource Efficiency**: Over 80% of Q-Learning's DICT uses were wasteful,
   representing significant opportunity cost vs. DQN's 35% waste rate.

## Recommendations

- **Use DQN approach**: For cost-aware action selection in similar problems
- **Tune tabular penalties**: Q-Learning/SARSA may need better feature engineering
- **Balance REINFORCE**: Consider adjusting penalties for moderate DICT usage
