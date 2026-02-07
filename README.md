# Applications of Reinforcement Learning in Part-of-Speech Tagging

This repository explores the application of Reinforcement Learning (RL) techniques to the Natural Language Processing task of Part-of-Speech (POS) tagging. It investigates whether RL agents can learn to effectively tag sentences or strategically utilize external resources (like a dictionary/oracle) to improve accuracy.

## 📌 Project Overview

The primary goal of this project is to answer two main research questions:
1. **RL Performance:** How well does an RL agent perform on POS tagging compared to traditional baselines?
2. **Strategic Oracle Usage:** Can an RL agent learn *when* to use a costly "dictionary" or "oracle" to look up a tag, balancing accuracy gains against the "cost" of using the external resource?

We implement and compare multiple RL algorithms (Tabular and Deep RL) using custom OpenAI Gym (Gymnasium) environments.

## ✨ Key Features

*   **Multiple RL Algorithms:**
    *   **Tabular Methods:** Q-Learning, SARSA.
    *   **Deep RL:** Deep Q-Networks (DQN).
    *   **Policy Gradient:** REINFORCE.
*   **Custom Environments:** specialized `Gymnasium` environments designed for sequence labeling tasks.
*   **Comprehensive Analysis Pipeline:** Automated tools to generate:
    *   Training curves (Rewards, Loss).
    *   Comparative Accuracy Plots.
    *   Confusion Matrices & Heatmaps.
    *   Action Distribution Analysis.
    *   Detailed Interpretation Reports.
*   **Dataset:** Uses the **Brown Corpus** (via NLTK) with Universal POS tags.

## 📂 Repository Structure

*   **`Algorithms/BaseTaggerPOS/`**: The core implementation of RL agents and analysis tools.
    *   `dqn.py`, `reinforce.py`, `q-learning.py`: Training scripts for specific algorithms.
    *   `comprehensive_analysis.py`: Generates plots and reports.
    *   `run_complete_analysis.py`: One-click script for training and analysis.
*   **`Enviroment/`**: Custom Gym environments (`BaseTaggerEnv`).
*   **`Experiment 1/`**: Experimental scripts and iterations on Universal POS tagging.
*   **`analysis_results/`**: Generated plots, tables, and text reports from the experiments.
*   **`Paper/`** & **`latex/`**: Documentation, roadmaps, and LaTeX sources for the project report.

## 🚀 Getting Started

### Prerequisites

Ensure you have Python installed (3.8+ recommended).

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/Applications-of-Reinforcement-Learning-in-Part-of-Speech-Tagging.git
    cd Applications-of-Reinforcement-Learning-in-Part-of-Speech-Tagging
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Key libraries: `gymnasium`, `numpy`, `nltk`, `torch` (or `tensorflow`/`keras`), `matplotlib`, `stable-baselines3`.*

## 💻 Usage

The cleanest implementation and analysis tools are located in `Algorithms/BaseTaggerPOS`.

### Option 1: Run Complete Pipeline (Recommended)
To train all models and generate a full suite of analysis plots and reports:

```bash
cd Algorithms/BaseTaggerPOS
python run_complete_analysis.py
```
*This process takes ~15-30 minutes depending on your hardware.*

### Option 2: Individual Training & Analysis

1.  **Train specific agents:**
    ```bash
    cd Algorithms/BaseTaggerPOS
    python dqn.py          # Train Deep Q-Network
    python reinforce.py    # Train REINFORCE Agent
    python q-learning.py   # Train Q-Learning & SARSA
    ```

2.  **Generate Analysis:**
    After training, run the analysis script to visualize comparisons:
    ```bash
    python comprehensive_analysis.py
    ```

## 📊 Results & Visualization

The analysis pipeline generates several key outputs (check `analysis_results/`):

*   **`accuracy_comparison.png`**: Side-by-side comparison of all trained agents vs. baseline.
*   **`per_tag_heatmap.png`**: Heatmap showing which POS tags are hardest for the agents.
*   **`action_distribution.png`**: Shows how often agents choose specific tags vs. using the dictionary.
*   **`interpretation_report.txt`**: A generated text summary interpreting the results.

## 📄 References

*   **Dataset:** [Brown Corpus](https://en.wikipedia.org/wiki/Brown_Corpus) (Francis & Kucera, 1979).
*   **Paper:** See `Paper/` directory for drafts and roadmaps related to this research.

---
*Created for the "Applications of Reinforcement Learning in Part-of-Speech Tagging" research project.*
