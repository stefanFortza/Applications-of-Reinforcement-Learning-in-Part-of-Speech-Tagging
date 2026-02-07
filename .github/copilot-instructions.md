# Copilot Instructions: RL-Based POS Tagging Research Project

## Project Overview

Research project investigating Reinforcement Learning approaches to Part-of-Speech (POS) tagging. The core architecture uses a **fine-tuned DistilBERT base tagger** with **RL agents learning correction strategies** through three actions: KEEP, SHIFT, or DICT (oracle).

### Key Research Questions

1. **RQ1**: How well can RL POS taggers perform compared to current baselines?
2. **RQ2**: Can an RL agent trained to use a POS tagger or oracle perform well?

## Architecture & Data Flow

### Three-Layer System

**1. Base Tagger** ([Enviroment/BaseTaggerEnv/environment.py](Enviroment/BaseTaggerEnv/environment.py))

- Pre-trained DistilBERT model fine-tuned on Universal Dependencies (12 UPOS tags)
- Located at `pos_tagger_model/` (tokenizer + model weights)
- Provides: token embeddings (768-dim), top-K predictions, confidence scores

**2. RL Environment** ([Enviroment/BaseTaggerEnv/environment.py](Enviroment/BaseTaggerEnv/environment.py#L20))

- `PosCorrectionEnv`: Custom Gymnasium environment for token-level correction
- **State**: Dict observation with `embedding` (768D), `features` (10D), `base_tag_idx`, `prev_tag_idx`
- **Actions**: 0=KEEP (trust base), 1=SHIFT (use 2nd prediction), 2=DICT (oracle, -0.9 cost)
- **Reward structure** (critical for understanding agent behavior):
  - Correct prediction: +1.0
  - Incorrect: -1.0
  - Breaking correct base prediction: -2.0 (harsh penalty)
  - Action costs: DICT (-0.9), SHIFT (-0.1), KEEP (0)
  - DICT bonuses: +0.5 if fixing error, -0.5 if unnecessary

**3. RL Algorithms** ([Algorithms/BaseTaggerPOS/](Algorithms/BaseTaggerPOS/))

- Q-Learning/SARSA: Tabular methods with discretized states (1,872 states)
- DQN: Stable-Baselines3 with 256x256 network
- REINFORCE: Policy gradient with Actor-Critic architecture

### State Representation Strategy

**Discretized (Tabular)**: `obs_to_discrete_state()` in [rl_utils.py](Algorithms/BaseTaggerPOSUtils/rl_utils.py#L18)

- Tuple: (base_tag_idx, prev_tag_idx, confidence_level, confidence_gap, is_first)
- Used by Q-Learning and SARSA

**Continuous (Neural)**: `obs_to_tensor()` in [rl_utils.py](Algorithms/BaseTaggerPOSUtils/rl_utils.py#L29)

- 35D vector: one-hot base tag (12) + one-hot prev tag (13) + features (10)
- Used by DQN and REINFORCE

## Project Structure Patterns

### Two Experimental Branches

**`Algorithms/BaseTaggerPOS/`** (Main Focus)

- Production-quality implementations with comprehensive analysis
- Uses DistilBERT base tagger + RL correction agents
- Includes DICT efficiency analysis (tracks oracle usage quality)
- Has visualization and comparison tools

**`Experiment 1/`** (Historical)

- Earlier experiments with different reward structures
- Multiple environment versions (`UniversalPosTaggingEnv*.py`)
- Complex reward shaping with lexical/grammatical violations
- Less mature but shows design evolution

### Dataset Conventions

**Brown Corpus**: Standard training data via NLTK

- Mapped to Universal Dependencies tagset (12 tags: DET, NOUN, VERB, ADJ, ADV, ADP, CONJ, PRON, PRT, NUM, ., X)
- [dataset.py](Algorithms/BaseTaggerPOSUtils/dataset.py): `get_brown_as_universal()` → `brown_to_training_data()`
- Standard split: sentences 0-100 train, 100-200 test (1,305 test tokens)

### Tag List Order Matters

The 12 UPOS tags have a **fixed order** used for one-hot encoding:

```python
["DET", ".", "ADV", "X", "NOUN", "ADJ", "VERB", "CONJ", "PRT", "NUM", "ADP", "PRON"]
```

Changing this order breaks pre-trained models and saved agents.

## Critical Developer Workflows

### Training a New Algorithm

```bash
# Each algorithm is self-contained with training + evaluation
python Algorithms/BaseTaggerPOS/q-learning.py     # Trains both Q-Learning and SARSA
python Algorithms/BaseTaggerPOS/dqn.py            # Stable-Baselines3 DQN
python Algorithms/BaseTaggerPOS/reinforce.py      # Policy gradient with baseline

# Outputs:
# - Saved models: q_learning_model.pkl, dqn_pos_tagger.zip, reinforce_policy.pt
# - Results JSON: tabular_results.json, dqn_results.json, reinforce_results.json
# - Training plots: *_training_analysis.png
```

### Running Complete Analysis

```bash
cd Algorithms/BaseTaggerPOS
python comprehensive_analysis.py

# Generates:
# - Per-algorithm accuracy comparison
# - Per-tag performance breakdown
# - Action distribution analysis (KEEP/SHIFT/DICT %)
# - DICT efficiency metrics (necessity rate, waste rate)
# - Markdown report: ANALYSIS_RESULTS.md
```

### DICT Action Analysis (Unique Feature)

The project has specialized tooling to evaluate **oracle usage efficiency**:

```bash
# Standalone analysis after training
python Algorithms/BaseTaggerPOS/analyze_dict_usage.py

# Metrics tracked:
# - dict_necessary: Oracle used when base tagger was wrong (good)
# - dict_unnecessary: Oracle used when base was correct (wasteful)
# - dict_necessity_rate: % of DICT uses that were needed (target: >70%)
```

**Why it matters**: DICT has a -0.9 cost penalty. Efficient agents should only use it when KEEP would fail. This measures strategic oracle consultation vs. wasteful overuse.

### Environment Modes

`PosCorrectionEnv` supports three sentence selection modes:

- `"sequential"`: Iterate through dataset (training standard)
- `"random"`: Random sampling (original behavior)
- `"fixed"`: Always use same sentence (debugging)

Use `mode="sequential"` for fair comparison across algorithms (ensures all see same test sentences).

## Project-Specific Conventions

### Reward Shaping Philosophy

The reward structure evolved to discourage specific failure modes:

1. **Breaking correct predictions is worse than keeping incorrect ones** (-2.0 vs -1.0)
   - Prevents agents from randomly changing good base tagger outputs
2. **DICT has high cost (-0.9) but accuracy bonuses**

   - Net reward for correct DICT: +0.1
   - Teaches selective oracle usage, not blind reliance

3. **DICT efficiency incentives** (added in later versions)
   - +0.5 bonus if DICT fixes an error
   - -0.5 penalty if DICT was unnecessary
   - Encourages learning when oracle consultation is worthwhile

### Feature Engineering in Observations

The 10-feature vector ([environment.py](Enviroment/BaseTaggerEnv/environment.py#L124)) combines:

- **Confidence signals**: raw confidence, top-2 gap
- **Morphological**: isupper, all_upper, verb/adverb/plural suffixes, has_digit
- **Positional**: is_first, position_ratio

**Design rationale**: Gives agents cues about when base tagger is uncertain (low confidence gap) or when context matters (position, morphology).

### Model Persistence Patterns

**Tabular methods**: Pickle dictionaries

```python
import pickle
with open("q_learning_model.pkl", "wb") as f:
    pickle.dump(Q, f)
```

**DQN**: Stable-Baselines3 `.zip` format

```python
model.save("dqn_pos_tagger")
# Loads as model = DQN.load("dqn_pos_tagger")
```

**REINFORCE**: PyTorch state_dict

```python
torch.save(policy.state_dict(), "reinforce_policy.pt")
```

## Common Pitfalls & Debugging

### Sequential vs Random Evaluation

**Bug fixed in analysis**: Early evaluations used `mode="random"` causing different test sentences per algorithm. Current standard uses `mode="sequential"` for fair apples-to-apples comparison (all algorithms evaluated on same 1,305 tokens).

### REINFORCE May Not Learn

REINFORCE sometimes reverts to baseline (100% KEEP) due to:

- High DICT penalty discourages exploration
- Variance in policy gradients
- Insufficient training episodes

This is documented behavior, not a bug. Q-Learning typically performs best.

### State Space Explosion

Tabular methods use discretized states (1,872 total). If adding features, verify state space remains tractable:

```python
state_space_size = 12 (base_tags) × 13 (prev_tags) × 3 (conf_levels) × 2 (conf_gaps) × 2 (is_first)
```

### Environment Registration Gotcha

When using Gymnasium's `register()`, the environment is registered globally. Running multiple training scripts in same session can cause ID conflicts. Each script re-registers `"gymnasium_env/PosCorrection-v0"`.

## External Dependencies

**Core**: gymnasium, stable-baselines3, transformers, torch, nltk
**Dataset**: Brown corpus via NLTK (auto-downloads on first run)
**Model**: DistilBERT tokenizer/model in `pos_tagger_model/`

**Installation**: `pip install -r requirements.txt`

## Testing & Validation

No formal test suite. Validation through:

1. **Baseline accuracy check**: Base tagger should achieve ~95.4% on test set
2. **DICT analysis**: Run `test_dict_analysis.py` to verify tracking logic
3. **Manual inspection**: Training plots should show reward convergence

## Paper Context

Research paper materials in [Paper/](Paper/) directory (largely TODO):

- ROADMAP.md: Paper structure outline
- OFFLINE_RL.md: Background on offline RL techniques
- Focus: Part 1 (RQ1) likely shows RL underperforms baseline; Part 2 (RQ2) demonstrates selective oracle usage

## Quick Reference for Common Tasks

**Add a new RL algorithm**:

1. Create script in `Algorithms/BaseTaggerPOS/`
2. Use `obs_to_discrete_state` or `obs_to_tensor` for state representation
3. Call `analyze_dict_usage()` post-training
4. Save results to JSON for `comprehensive_analysis.py` integration

**Modify reward structure**:

- Edit `step()` in [environment.py](Enviroment/BaseTaggerEnv/environment.py#L242)
- Action costs: lines 255-266
- Correctness rewards: lines 272-287
- DICT bonuses: lines 290-301

**Change observation features**:

- Update `_get_obs()` in [environment.py](Enviroment/BaseTaggerEnv/environment.py#L101)
- Adjust `features` array size (currently 10D)
- Update `obs_to_tensor()` input_dim calculation
- Retrain all models (old saved models incompatible)

**Evaluate on different dataset**:

- Prepare data as `List[Tuple[str, List[str]]]` (sentence, gold_tags)
- Ensure tags match Universal Dependencies 12-tag set
- Pass to `PosCorrectionEnv(dataset=your_data)`
