"""
Complete Analysis Runner - Trains all algorithms and generates comprehensive results
Run this script to get all plots and tables for your paper
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import subprocess
import time
from pathlib import Path
import sys


def get_python_executable():
    """Get the Python executable path (handles venv)."""
    return sys.executable


def run_training():
    """Run all training scripts sequentially."""
    algorithms = [
        ("Q-Learning & SARSA", "q-learning.py"),
        ("DQN", "dqn.py"),
        ("REINFORCE", "reinforce.py"),
    ]

    print("=" * 80)
    print("TRAINING ALL ALGORITHMS")
    print("=" * 80)
    print("\nThis will train all three RL algorithms sequentially.")
    print("Expected time: ~15-30 minutes depending on your hardware\n")

    input("Press Enter to start training...")

    base_dir = Path(__file__).parent

    for name, script in algorithms:
        print(f"\n{'='*80}")
        print(f"Training: {name}")
        print(f"{'='*80}\n")

        script_path = base_dir / script
        start_time = time.time()

        try:
            result = subprocess.run(
                [get_python_executable(), str(script_path)],
                cwd=str(base_dir),
                check=True,
                capture_output=False,
                text=True,
            )

            elapsed = time.time() - start_time
            print(f"\n✓ {name} completed in {elapsed:.1f} seconds")

        except subprocess.CalledProcessError as e:
            print(f"\n✗ Error training {name}:")
            print(e)
            return False

    print("\n" + "=" * 80)
    print("ALL TRAINING COMPLETE!")
    print("=" * 80)
    return True


def run_analysis():
    """Run comprehensive analysis script."""
    print("\n" + "=" * 80)
    print("RUNNING COMPREHENSIVE ANALYSIS")
    print("=" * 80 + "\n")

    base_dir = Path(__file__).parent
    analysis_script = base_dir / "comprehensive_analysis.py"

    try:
        subprocess.run(
            [get_python_executable(), str(analysis_script)],
            cwd=str(base_dir),
            check=True,
            capture_output=False,
            text=True,
        )
        print("\n✓ Analysis complete!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error running analysis:")
        print(e)
        return False


def main():
    print(
        """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║         RL-BASED POS TAGGING - COMPLETE TRAINING & ANALYSIS PIPELINE        ║
║                                                                              ║
║  This script will:                                                           ║
║  1. Train Q-Learning & SARSA (tabular methods)                               ║
║  2. Train DQN (deep Q-network)                                               ║
║  3. Train REINFORCE (policy gradient)                                        ║
║  4. Generate comprehensive plots and tables                                  ║
║  5. Create interpretation report                                             ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    )

    # Check if models already exist
    base_dir = Path(__file__).parent
    models_exist = (
        (base_dir / "q_learning_model.pkl").exists()
        and (base_dir / "dqn_pos_tagger.zip").exists()
        and (base_dir / "reinforce_policy.pt").exists()
    )

    if models_exist:
        print("\n⚠ Existing trained models detected.")
        response = input("Do you want to retrain? (y/N): ").strip().lower()
        if response == "y":
            print("\nRetraining all models...")
            if not run_training():
                return
        else:
            print("\nUsing existing models...")
    else:
        print("\nNo existing models found. Training from scratch...")
        if not run_training():
            return

    # Run analysis
    if not run_analysis():
        return

    print(
        """
    
╔══════════════════════════════════════════════════════════════════════════════╗
║                              ANALYSIS COMPLETE!                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

Generated files:
├── Training plots (individual algorithms):
│   ├── training_rewards.png (Q-Learning vs SARSA)
│   ├── dqn_training_analysis.png
│   └── reinforce_training_rewards.png
│
└── Comprehensive analysis (analysis_results/):
    ├── results_table.txt             - Main results comparison
    ├── per_tag_results.txt            - Per-tag accuracy breakdown
    ├── accuracy_comparison.png        - Bar chart of accuracies
    ├── action_distribution.png        - Action usage by algorithm
    ├── per_tag_heatmap.png           - Heatmap of tag accuracies
    ├── improvement_analysis.png       - Improvement over baseline
    └── interpretation_report.txt      - Detailed interpretation

Use these files for your paper to satisfy:
✓ Grafice/tabele cu rezultatele obținute (1p)
✓ Interpretarea rezultatelor (1p)
    """
    )


if __name__ == "__main__":
    main()
