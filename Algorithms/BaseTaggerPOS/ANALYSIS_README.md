# Comprehensive Analysis Tools

This directory contains tools for generating publication-ready plots and tables for your RL-based POS tagging paper.

## Quick Start

### Option 1: Complete Pipeline (Recommended)
Run everything in one go:
```bash
cd Algorithms/BaseTaggerPOS
python run_complete_analysis.py
```

This will:
1. Train all three algorithms (Q-Learning, DQN, REINFORCE)
2. Generate all plots and tables
3. Create interpretation report

**Time:** ~15-30 minutes on CPU

### Option 2: Manual Workflow
If you want more control:

1. **Train individual algorithms:**
```bash
python q-learning.py      # Trains Q-Learning & SARSA
python dqn.py             # Trains DQN
python reinforce.py       # Trains REINFORCE
```

2. **Generate analysis:**
```bash
python comprehensive_analysis.py
```

## Generated Outputs

### Training Plots (Individual Algorithms)
- `training_rewards.png` - Q-Learning vs SARSA rewards
- `dqn_training_analysis.png` - DQN rewards and Q-values
- `reinforce_training_rewards.png` - REINFORCE rewards

### Comprehensive Analysis (`analysis_results/`)

#### Tables
- **`results_table.txt`** - Main comparison table
  - Accuracy for each algorithm
  - Action distribution percentages
  - Improvement over baseline
  
- **`per_tag_results.txt`** - Per-tag accuracy breakdown
  - Accuracy for each UPOS tag
  - Tags with highest improvement

#### Plots
- **`accuracy_comparison.png`** - Bar chart comparing all algorithms
- **`action_distribution.png`** - Stacked bar chart of action usage
- **`per_tag_heatmap.png`** - Heatmap of per-tag accuracies
- **`improvement_analysis.png`** - Improvement over baseline

#### Interpretation
- **`interpretation_report.txt`** - Comprehensive analysis including:
  - Overall performance analysis
  - Key findings and insights
  - Action strategy analysis
  - Practical implications
  - Limitations and future work

## For Your Paper

### Grafice/tabele cu rezultatele obținute (1p)

Use these files:
1. **Main Results Table:** `analysis_results/results_table.txt`
   - Copy into paper as LaTeX table
   
2. **Accuracy Comparison Plot:** `analysis_results/accuracy_comparison.png`
   - Shows clear visual comparison
   
3. **Per-Tag Heatmap:** `analysis_results/per_tag_heatmap.png`
   - Shows which tags are difficult/easy
   
4. **Action Distribution:** `analysis_results/action_distribution.png`
   - Shows strategy differences between algorithms

### Interpretarea rezultatelor (1p)

Use: **`analysis_results/interpretation_report.txt`**

This report provides:
- Quantitative analysis of results
- Comparison of algorithm strategies
- Identification of difficult tags
- Practical implications
- Discussion of limitations

You can directly quote or paraphrase sections from this report.

## Saved Models

After training, these files are created:
- `q_learning_model.pkl` - Q-Learning Q-table
- `sarsa_model.pkl` - SARSA Q-table (from q-learning.py)
- `dqn_pos_tagger.zip` - DQN neural network
- `reinforce_policy.pt` - REINFORCE policy network

These can be reloaded for further analysis without retraining.

## Customization

### Change Training Size
Edit the training scripts:
```python
train_data = brown_to_training_data(brown_data[:100])  # Change 100 to desired size
```

### Modify Plots
Edit `comprehensive_analysis.py` to customize:
- Plot styles (colors, fonts)
- Figure sizes
- Additional metrics

### Add New Algorithms
To add a new algorithm to the analysis:

1. Train and save your model
2. Add evaluation code in `comprehensive_analysis.py`:
```python
your_results = evaluator.evaluate_agent(
    your_agent, test_env, 
    agent_type="tabular",  # or "dqn" or "policy"
    algorithm_name="Your Algorithm"
)
all_results.append(your_results)
```

## Troubleshooting

**Problem:** "Model not found" errors
- **Solution:** Run training scripts first: `python q-learning.py`, etc.

**Problem:** Import errors
- **Solution:** Make sure you're in the correct directory and have all dependencies:
```bash
pip install -r ../../requirements.txt
```

**Problem:** Plots don't show differences
- **Solution:** Models might not be well-trained. Try:
  - Increasing training episodes
  - Adjusting hyperparameters
  - Using more training data

## Tips for Best Results

1. **Train on more data** for better generalization:
   ```python
   train_data = brown_to_training_data(brown_data[:500])  # More sentences
   ```

2. **Run multiple seeds** and average results for robustness

3. **Compare to additional baselines** (e.g., random baseline, majority-class baseline)

4. **Analyze failure cases** - look at which specific sentences/tokens cause errors
