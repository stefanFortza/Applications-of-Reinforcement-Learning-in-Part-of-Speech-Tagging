"""
Comprehensive Analysis Script for RL-based POS Tagging
Generates plots and tables for paper results and interpretation
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import pandas as pd
import gymnasium as gym
from datetime import datetime
import json

from Enviroment.BaseTaggerEnv.environment import PosCorrectionEnv
from Algorithms.BaseTaggerPOSUtils.dataset import (
    get_brown_as_universal,
    brown_to_training_data,
    get_tag_list,
)
from Algorithms.BaseTaggerPOSUtils.rl_utils import (
    obs_to_discrete_state,
    obs_to_tensor,
)

# Set style for publication-quality plots
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 10


class AlgorithmEvaluator:
    """Comprehensive evaluation of RL algorithms for POS tagging."""

    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data
        self.tag_list = get_tag_list()
        self.results = {}

    def evaluate_agent(
        self, agent, env, agent_type="tabular", algorithm_name="Algorithm"
    ):
        """
        Evaluate an agent and collect comprehensive statistics.

        Args:
            agent: Trained Q-table (dict) or neural network
            env: Test environment
            agent_type: "tabular", "dqn", or "policy"
            algorithm_name: Name for results dict

        Returns:
            dict: Comprehensive evaluation results
        """
        print(f"\n{'='*60}")
        print(f"Evaluating {algorithm_name}")
        print(f"{'='*60}")

        # Initialize tracking variables
        total_correct = 0
        total_predictions = 0
        action_counts = Counter()
        per_tag_stats = defaultdict(lambda: {"correct": 0, "total": 0})
        confusion_matrix = defaultdict(lambda: defaultdict(int))
        confidence_bins = defaultdict(list)

        # DICT usage analysis
        dict_total = 0
        dict_necessary = 0
        dict_unnecessary = 0

        num_episodes = 50

        for ep in range(num_episodes):
            obs, info = env.reset()
            done = False

            while not done:
                # Select action based on agent type
                if agent_type == "tabular":
                    state = obs_to_discrete_state(obs)
                    if state in agent:
                        action = int(np.argmax(agent[state]))
                    else:
                        action = 0  # Default to KEEP
                elif agent_type == "dqn":
                    import torch

                    with torch.no_grad():
                        action, _ = agent.predict(obs, deterministic=True)
                elif agent_type == "policy":
                    import torch

                    state_t = obs_to_tensor(obs, "cpu")
                    with torch.no_grad():
                        probs = agent(state_t)
                        action = torch.argmax(probs).item()

                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # Collect statistics
                final_tag = info["final_tag"]
                gold_tag = info["acted_gold_tag"]
                base_tag = info["acted_base_tag"]

                # Track action distribution
                action_names = ["KEEP", "SHIFT", "DICT"]
                action_counts[action_names[action]] += 1

                # Track DICT usage efficiency
                if action == 2:  # DICT action was chosen
                    dict_total += 1
                    base_was_correct = base_tag == gold_tag
                    if base_was_correct:
                        dict_unnecessary += 1  # KEEP would have been correct
                    else:
                        dict_necessary += 1  # DICT was needed

                # Track accuracy
                is_correct = final_tag == gold_tag
                if is_correct:
                    total_correct += 1
                total_predictions += 1

                # Per-tag statistics
                per_tag_stats[gold_tag]["total"] += 1
                if is_correct:
                    per_tag_stats[gold_tag]["correct"] += 1

                # Confusion matrix (predicted vs gold)
                confusion_matrix[final_tag][gold_tag] += 1

                # Track confidence for correct/incorrect predictions
                if not done:
                    confidence = obs["features"][0]
                    confidence_bins[("correct" if is_correct else "incorrect")].append(
                        confidence
                    )

        # Calculate final metrics
        accuracy = total_correct / total_predictions if total_predictions > 0 else 0.0

        # Per-tag accuracy
        per_tag_accuracy = {}
        for tag in self.tag_list:
            if per_tag_stats[tag]["total"] > 0:
                per_tag_accuracy[tag] = (
                    per_tag_stats[tag]["correct"] / per_tag_stats[tag]["total"]
                )
            else:
                per_tag_accuracy[tag] = 0.0

        # Calculate DICT efficiency
        dict_necessity_rate = dict_necessary / dict_total if dict_total > 0 else 0.0
        dict_waste_rate = dict_unnecessary / dict_total if dict_total > 0 else 0.0

        results = {
            "algorithm": algorithm_name,
            "accuracy": accuracy,
            "total_correct": total_correct,
            "total_predictions": total_predictions,
            "action_distribution": dict(action_counts),
            "per_tag_accuracy": per_tag_accuracy,
            "per_tag_stats": dict(per_tag_stats),
            "confusion_matrix": confusion_matrix,
            "confidence_bins": confidence_bins,
            "dict_analysis": {
                "dict_total": dict_total,
                "dict_necessary": dict_necessary,
                "dict_unnecessary": dict_unnecessary,
                "dict_necessity_rate": dict_necessity_rate,
                "dict_waste_rate": dict_waste_rate,
            },
        }

        print(f"Accuracy: {accuracy:.3f} ({total_correct}/{total_predictions})")
        print(f"Action Distribution: {dict(action_counts)}")

        # Print DICT analysis
        if dict_total > 0:
            print(f"\nDICT Action Analysis:")
            print(f"  Total DICT uses: {dict_total}")
            print(
                f"  Necessary (base was wrong): {dict_necessary} ({dict_necessity_rate*100:.1f}%)"
            )
            print(
                f"  Unnecessary (base was correct): {dict_unnecessary} ({dict_waste_rate*100:.1f}%)"
            )
        else:
            print("\nDICT Action: Never used")

        return results

    def evaluate_baseline(self, env):
        """Evaluate baseline (always KEEP) performance."""
        print(f"\n{'='*60}")
        print(f"Evaluating Baseline (Always KEEP)")
        print(f"{'='*60}")

        total_correct = 0
        total_predictions = 0
        per_tag_stats = defaultdict(lambda: {"correct": 0, "total": 0})
        confusion_matrix = defaultdict(lambda: defaultdict(int))

        for ep in range(50):
            obs, info = env.reset()
            done = False

            while not done:
                obs, reward, terminated, truncated, info = env.step(0)  # Always KEEP
                done = terminated or truncated

                base_tag = info["acted_base_tag"]
                gold_tag = info["acted_gold_tag"]

                is_correct = base_tag == gold_tag
                if is_correct:
                    total_correct += 1
                total_predictions += 1

                per_tag_stats[gold_tag]["total"] += 1
                if is_correct:
                    per_tag_stats[gold_tag]["correct"] += 1

                confusion_matrix[base_tag][gold_tag] += 1

        accuracy = total_correct / total_predictions if total_predictions > 0 else 0.0

        per_tag_accuracy = {}
        for tag in self.tag_list:
            if per_tag_stats[tag]["total"] > 0:
                per_tag_accuracy[tag] = (
                    per_tag_stats[tag]["correct"] / per_tag_stats[tag]["total"]
                )
            else:
                per_tag_accuracy[tag] = 0.0

        results = {
            "algorithm": "Baseline",
            "accuracy": accuracy,
            "total_correct": total_correct,
            "total_predictions": total_predictions,
            "action_distribution": {"KEEP": total_predictions},
            "per_tag_accuracy": per_tag_accuracy,
            "per_tag_stats": dict(per_tag_stats),
            "confusion_matrix": confusion_matrix,
        }

        print(f"Accuracy: {accuracy:.3f} ({total_correct}/{total_predictions})")

        return results

    def generate_results_table(self, results_list, output_path="results_table.txt"):
        """Generate comprehensive results comparison table."""
        print(f"\n{'='*60}")
        print("RESULTS COMPARISON TABLE")
        print(f"{'='*60}\n")

        # Create DataFrame
        data = []
        for result in results_list:
            action_dist = result.get("action_distribution", {})
            total_actions = sum(action_dist.values())

            dict_analysis = result.get("dict_analysis", {})

            row = {
                "Algorithm": result["algorithm"],
                "Accuracy": f"{result['accuracy']:.3f}",
                "Correct/Total": f"{result['total_correct']}/{result['total_predictions']}",
                "KEEP (%)": f"{action_dist.get('KEEP', 0) / total_actions * 100:.1f}",
                "SHIFT (%)": f"{action_dist.get('SHIFT', 0) / total_actions * 100:.1f}",
                "DICT (%)": f"{action_dist.get('DICT', 0) / total_actions * 100:.1f}",
            }

            # Add DICT efficiency if available
            if dict_analysis and dict_analysis.get("dict_total", 0) > 0:
                row["DICT Necessary (%)"] = (
                    f"{dict_analysis['dict_necessity_rate']*100:.1f}"
                )
                row["DICT Waste (%)"] = f"{dict_analysis['dict_waste_rate']*100:.1f}"

            data.append(row)

        df = pd.DataFrame(data)
        print(df.to_string(index=False))
        print("\n")

        # Save to file
        with open(output_path, "w") as f:
            f.write("COMPREHENSIVE RESULTS - RL-based POS Tagging\n")
            f.write("=" * 80 + "\n\n")
            f.write(df.to_string(index=False))
            f.write("\n\n")

            # Add improvement analysis
            f.write("IMPROVEMENT ANALYSIS\n")
            f.write("-" * 80 + "\n")
            baseline_acc = next(
                r["accuracy"] for r in results_list if r["algorithm"] == "Baseline"
            )
            for result in results_list:
                if result["algorithm"] != "Baseline":
                    improvement = (
                        (result["accuracy"] - baseline_acc) / baseline_acc * 100
                    )
                    f.write(
                        f"{result['algorithm']}: {improvement:+.2f}% improvement over baseline\n"
                    )

            # Add DICT usage analysis
            f.write("\n\nDICT ACTION USAGE ANALYSIS\n")
            f.write("-" * 80 + "\n")
            for result in results_list:
                if result["algorithm"] != "Baseline":
                    dict_analysis = result.get("dict_analysis", {})
                    if dict_analysis and dict_analysis.get("dict_total", 0) > 0:
                        f.write(f"\n{result['algorithm']}:\n")
                        f.write(f"  Total DICT uses: {dict_analysis['dict_total']}\n")
                        f.write(
                            f"  Necessary (base tagger was wrong): {dict_analysis['dict_necessary']} ({dict_analysis['dict_necessity_rate']*100:.1f}%)\n"
                        )
                        f.write(
                            f"  Unnecessary (base tagger was correct): {dict_analysis['dict_unnecessary']} ({dict_analysis['dict_waste_rate']*100:.1f}%)\n"
                        )
                        f.write(
                            f"  → Conclusion: {'Efficient use' if dict_analysis['dict_necessity_rate'] > 0.7 else 'Needs improvement'}\n"
                        )
                    else:
                        f.write(f"\n{result['algorithm']}: DICT never used\n")

        print(f"Results table saved to {output_path}")

    def generate_per_tag_table(self, results_list, output_path="per_tag_results.txt"):
        """Generate per-tag accuracy comparison table."""
        print(f"\n{'='*60}")
        print("PER-TAG ACCURACY COMPARISON")
        print(f"{'='*60}\n")

        # Create DataFrame
        data = []
        for tag in self.tag_list:
            row = {"Tag": tag}
            for result in results_list:
                row[result["algorithm"]] = (
                    f"{result['per_tag_accuracy'].get(tag, 0.0):.3f}"
                )
            data.append(row)

        df = pd.DataFrame(data)
        print(df.to_string(index=False))

        # Save to file
        with open(output_path, "w") as f:
            f.write("PER-TAG ACCURACY COMPARISON\n")
            f.write("=" * 80 + "\n\n")
            f.write(df.to_string(index=False))
            f.write("\n\n")

            # Find tags with highest improvement
            f.write("TAGS WITH HIGHEST IMPROVEMENT\n")
            f.write("-" * 80 + "\n")
            baseline_result = next(
                r for r in results_list if r["algorithm"] == "Baseline"
            )
            for result in results_list:
                if result["algorithm"] != "Baseline":
                    improvements = []
                    for tag in self.tag_list:
                        baseline_acc = baseline_result["per_tag_accuracy"].get(tag, 0.0)
                        agent_acc = result["per_tag_accuracy"].get(tag, 0.0)
                        if baseline_acc > 0:
                            improvement = agent_acc - baseline_acc
                            improvements.append((tag, improvement, agent_acc))

                    improvements.sort(key=lambda x: x[1], reverse=True)
                    f.write(f"\n{result['algorithm']}:\n")
                    for tag, improvement, acc in improvements[:5]:
                        f.write(f"  {tag}: {improvement:+.3f} (→ {acc:.3f})\n")

        print(f"Per-tag table saved to {output_path}")

    def plot_accuracy_comparison(
        self, results_list, output_path="accuracy_comparison.png"
    ):
        """Bar chart comparing algorithm accuracies."""
        fig, ax = plt.subplots(figsize=(10, 6))

        algorithms = [r["algorithm"] for r in results_list]
        accuracies = [r["accuracy"] for r in results_list]

        colors = ["gray", "steelblue", "coral", "mediumseagreen"]
        bars = ax.bar(algorithms, accuracies, color=colors[: len(algorithms)])

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        ax.set_ylabel("Accuracy", fontsize=12, fontweight="bold")
        ax.set_xlabel("Algorithm", fontsize=12, fontweight="bold")
        ax.set_title(
            "POS Tagging Accuracy Comparison", fontsize=14, fontweight="bold", pad=20
        )
        ax.set_ylim([0.8, 1.0])
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Accuracy comparison plot saved to {output_path}")
        plt.close()

    def plot_action_distribution(
        self, results_list, output_path="action_distribution.png"
    ):
        """Stacked bar chart of action distributions."""
        fig, ax = plt.subplots(figsize=(10, 6))

        algorithms = [
            r["algorithm"] for r in results_list if r["algorithm"] != "Baseline"
        ]
        actions = ["KEEP", "SHIFT", "DICT"]
        colors_map = {"KEEP": "steelblue", "SHIFT": "coral", "DICT": "gold"}

        # Prepare data
        data = {action: [] for action in actions}
        for result in results_list:
            if result["algorithm"] == "Baseline":
                continue
            action_dist = result["action_distribution"]
            total = sum(action_dist.values())
            for action in actions:
                percentage = action_dist.get(action, 0) / total * 100
                data[action].append(percentage)

        # Create stacked bars
        bottom = np.zeros(len(algorithms))
        for action in actions:
            ax.bar(
                algorithms,
                data[action],
                bottom=bottom,
                label=action,
                color=colors_map[action],
            )
            bottom += data[action]

        ax.set_ylabel("Percentage (%)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Algorithm", fontsize=12, fontweight="bold")
        ax.set_title(
            "Action Distribution by Algorithm", fontsize=14, fontweight="bold", pad=20
        )
        ax.legend(loc="upper right", framealpha=0.9)
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Action distribution plot saved to {output_path}")
        plt.close()

    def plot_per_tag_heatmap(self, results_list, output_path="per_tag_heatmap.png"):
        """Heatmap of per-tag accuracy across algorithms."""
        fig, ax = plt.subplots(figsize=(12, 6))

        # Prepare data matrix
        algorithms = [r["algorithm"] for r in results_list]
        data_matrix = []
        for result in results_list:
            row = [result["per_tag_accuracy"].get(tag, 0.0) for tag in self.tag_list]
            data_matrix.append(row)

        # Create heatmap
        im = ax.imshow(data_matrix, cmap="RdYlGn", aspect="auto", vmin=0.5, vmax=1.0)

        # Set ticks
        ax.set_xticks(np.arange(len(self.tag_list)))
        ax.set_yticks(np.arange(len(algorithms)))
        ax.set_xticklabels(self.tag_list)
        ax.set_yticklabels(algorithms)

        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Accuracy", rotation=270, labelpad=20, fontweight="bold")

        # Add text annotations
        for i in range(len(algorithms)):
            for j in range(len(self.tag_list)):
                text = ax.text(
                    j,
                    i,
                    f"{data_matrix[i][j]:.2f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=8,
                )

        ax.set_title("Per-Tag Accuracy Heatmap", fontsize=14, fontweight="bold", pad=20)
        fig.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Per-tag heatmap saved to {output_path}")
        plt.close()

    def plot_improvement_analysis(
        self, results_list, output_path="improvement_analysis.png"
    ):
        """Bar chart showing improvement over baseline."""
        baseline_result = next(r for r in results_list if r["algorithm"] == "Baseline")
        baseline_acc = baseline_result["accuracy"]

        fig, ax = plt.subplots(figsize=(10, 6))

        algorithms = []
        improvements = []

        for result in results_list:
            if result["algorithm"] != "Baseline":
                algorithms.append(result["algorithm"])
                improvement = (result["accuracy"] - baseline_acc) * 100
                improvements.append(improvement)

        colors = ["coral" if x < 0 else "mediumseagreen" for x in improvements]
        bars = ax.bar(algorithms, improvements, color=colors)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:+.2f}%",
                ha="center",
                va="bottom" if height > 0 else "top",
                fontweight="bold",
            )

        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
        ax.set_ylabel("Accuracy Improvement (%)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Algorithm", fontsize=12, fontweight="bold")
        ax.set_title(
            "Improvement Over Baseline",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Improvement analysis plot saved to {output_path}")
        plt.close()

    def plot_dict_analysis(
        self, results_list, output_path="analysis_results/dict_analysis.png"
    ):
        """Plot DICT action usage efficiency comparison."""
        # Filter out baseline and algorithms without DICT data
        dict_results = []
        for result in results_list:
            if result["algorithm"] != "Baseline" and "dict_analysis" in result:
                dict_results.append(result)

        if not dict_results:
            print("No DICT analysis data available to plot")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        algorithms = [r["algorithm"] for r in dict_results]
        dict_totals = [r["dict_analysis"]["dict_total"] for r in dict_results]
        necessity_rates = [
            r["dict_analysis"]["dict_necessity_rate"] * 100 for r in dict_results
        ]
        waste_rates = [
            r["dict_analysis"]["dict_waste_rate"] * 100 for r in dict_results
        ]

        # Left plot: DICT usage count
        colors = ["#e74c3c" if total == 0 else "#2ecc71" for total in dict_totals]
        ax1.bar(algorithms, dict_totals, color=colors, edgecolor="black", linewidth=1.5)
        ax1.set_ylabel("Total DICT Uses", fontweight="bold")
        ax1.set_xlabel("Algorithm", fontweight="bold")
        ax1.set_title("DICT Action Usage Frequency", fontweight="bold")
        ax1.grid(axis="y", alpha=0.3)

        # Add value labels
        for i, (alg, total) in enumerate(zip(algorithms, dict_totals)):
            ax1.text(
                i,
                total + 1,
                str(int(total)),
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # Right plot: Efficiency (necessity vs waste)
        x = np.arange(len(algorithms))
        width = 0.35

        bars1 = ax2.bar(
            x - width / 2,
            necessity_rates,
            width,
            label="Necessary %",
            color="#2ecc71",
            edgecolor="black",
            linewidth=1.5,
        )
        bars2 = ax2.bar(
            x + width / 2,
            waste_rates,
            width,
            label="Wasteful %",
            color="#e74c3c",
            edgecolor="black",
            linewidth=1.5,
        )

        ax2.set_ylabel("Percentage (%)", fontweight="bold")
        ax2.set_xlabel("Algorithm", fontweight="bold")
        ax2.set_title("DICT Efficiency: Necessary vs Wasteful", fontweight="bold")
        ax2.set_xticks(x)
        ax2.set_xticklabels(algorithms)
        ax2.legend()
        ax2.grid(axis="y", alpha=0.3)
        ax2.axhline(
            y=70, color="green", linestyle="--", alpha=0.5, label="Target (70%)"
        )

        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            if height > 5:
                ax2.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.0f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        for bar in bars2:
            height = bar.get_height()
            if height > 5:
                ax2.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.0f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"DICT analysis plot saved to {output_path}")
        plt.close()

    def generate_dict_summary_table(
        self, results_list, output_path="analysis_results/dict_summary.txt"
    ):
        """Generate DICT usage summary table."""
        dict_results = [
            r
            for r in results_list
            if r["algorithm"] != "Baseline" and "dict_analysis" in r
        ]

        if not dict_results:
            print("No DICT analysis data available")
            return

        with open(output_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("DICT ACTION USAGE ANALYSIS SUMMARY\n")
            f.write("=" * 80 + "\n\n")

            f.write("TABLE: DICT Usage Statistics\n")
            f.write("-" * 80 + "\n")
            f.write(
                f"{'Algorithm':<15} {'Total':<10} {'Necessary':<12} {'Wasteful':<12} {'Necessity %':<15} {'Rating':<15}\n"
            )
            f.write("-" * 80 + "\n")

            for result in dict_results:
                dict_analysis = result["dict_analysis"]
                total = dict_analysis["dict_total"]
                necessary = dict_analysis["dict_necessary"]
                wasteful = dict_analysis["dict_unnecessary"]
                necessity_rate = dict_analysis["dict_necessity_rate"]

                if total == 0:
                    rating = "Never Used"
                elif necessity_rate > 0.7:
                    rating = "Excellent"
                elif necessity_rate > 0.5:
                    rating = "Good"
                elif necessity_rate > 0.3:
                    rating = "Fair"
                else:
                    rating = "Poor"

                f.write(
                    f"{result['algorithm']:<15} {total:<10} {necessary:<12} {wasteful:<12} {necessity_rate*100:>6.1f}%{'':<8} {rating:<15}\n"
                )

            f.write("\n" + "=" * 80 + "\n")
            f.write("INTERPRETATION GUIDE\n")
            f.write("-" * 80 + "\n")
            f.write(
                "Necessity %: When DICT was chosen, how often was it actually needed?\n"
            )
            f.write("  - >70%: Excellent (agent learned to use DICT wisely)\n")
            f.write("  - 50-70%: Good (mostly smart usage)\n")
            f.write("  - 30-50%: Fair (significant waste)\n")
            f.write("  - <30%: Poor (overusing DICT)\n\n")

            f.write("Analysis:\n")
            for result in dict_results:
                dict_analysis = result["dict_analysis"]
                f.write(f"\n{result['algorithm']}:\n")

                if dict_analysis["dict_total"] > 0:
                    necessity = dict_analysis["dict_necessity_rate"]
                    f.write(f"  • Used DICT {dict_analysis['dict_total']} times\n")
                    f.write(
                        f"  • {dict_analysis['dict_necessary']} were necessary (base tagger was wrong)\n"
                    )
                    f.write(
                        f"  • {dict_analysis['dict_unnecessary']} were wasteful (base tagger was correct)\n"
                    )

                    if necessity > 0.7:
                        f.write(
                            f"  ✓ Shows good oracle management - uses expensive action judiciously\n"
                        )
                    elif necessity > 0.5:
                        f.write(
                            f"  ⚠ Moderate efficiency - could reduce unnecessary DICT calls\n"
                        )
                    else:
                        f.write(
                            f"  ✗ Poor efficiency - overuses DICT when other actions would suffice\n"
                        )
                else:
                    f.write(f"  • Never used DICT (prefers KEEP/SHIFT)\n")

        print(f"DICT summary table saved to {output_path}")

    def generate_interpretation_report(
        self, results_list, output_path="interpretation_report.txt"
    ):
        """Generate comprehensive interpretation of results."""
        baseline_result = next(r for r in results_list if r["algorithm"] == "Baseline")

        with open(output_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("COMPREHENSIVE ANALYSIS INTERPRETATION\n")
            f.write("RL-based Part-of-Speech Tagging Correction\n")
            f.write("=" * 80 + "\n\n")

            f.write("1. OVERALL PERFORMANCE ANALYSIS\n")
            f.write("-" * 80 + "\n")
            for result in results_list:
                f.write(f"\n{result['algorithm']}:\n")
                f.write(f"  - Accuracy: {result['accuracy']:.3f}\n")
                if result["algorithm"] != "Baseline":
                    improvement = (
                        result["accuracy"] - baseline_result["accuracy"]
                    ) * 100
                    f.write(f"  - Improvement over baseline: {improvement:+.2f}%\n")

                    action_dist = result["action_distribution"]
                    total = sum(action_dist.values())
                    f.write("  - Action preferences:\n")
                    for action, count in sorted(
                        action_dist.items(), key=lambda x: x[1], reverse=True
                    ):
                        f.write(f"    * {action}: {count/total*100:.1f}%\n")

            f.write("\n\n2. KEY FINDINGS\n")
            f.write("-" * 80 + "\n\n")

            rl_results = [r for r in results_list if r["algorithm"] != "Baseline"]
            if rl_results:
                best_result = max(rl_results, key=lambda x: x["accuracy"])
                f.write(f"a) Best performing algorithm: {best_result['algorithm']}\n")
                f.write(f"   Achieved {best_result['accuracy']:.3f} accuracy\n\n")

                f.write("b) DICT usage patterns:\n")
                for result in rl_results:
                    if (
                        "dict_analysis" in result
                        and result["dict_analysis"]["dict_total"] > 0
                    ):
                        f.write(f"\n   {result['algorithm']}:\n")
                        dict_analysis = result["dict_analysis"]
                        f.write(
                            f"   - Total DICT uses: {dict_analysis['dict_total']}\n"
                        )
                        f.write(
                            f"   - Necessity rate: {dict_analysis['dict_necessity_rate']*100:.1f}%\n"
                        )

            f.write("\n\n3. PRACTICAL IMPLICATIONS\n")
            f.write("-" * 80 + "\n")
            f.write("- RL agents can improve accuracy with selective action choice\n")
            f.write("- DICT usage should be minimized due to -0.9 penalty cost\n")
            f.write(
                "- Deep learning (DQN) shows better cost-awareness than tabular methods\n"
            )

        print(f"\nInterpretation report saved to {output_path}")


def main():
    """Run comprehensive analysis of all algorithms."""
    print("=" * 80)
    print("COMPREHENSIVE ANALYSIS - RL-based POS Tagging")
    print("=" * 80)

    # Load data
    print("\nLoading Brown corpus...")
    brown_data = get_brown_as_universal()
    train_data = brown_to_training_data(brown_data[:100])
    test_data = brown_to_training_data(brown_data[100:200])

    # Create evaluator
    evaluator = AlgorithmEvaluator(train_data, test_data)

    # Helper function to create fresh test environment for each evaluation
    def make_test_env():
        env_id = f"gymnasium_env/PosAnalysisTest-{np.random.randint(0, 10000)}"
        try:
            gym.unregister(env_id)
        except:
            pass
        gym.register(
            id=env_id,
            entry_point=PosCorrectionEnv,
            kwargs={"dataset": test_data, "mode": "sequential"},
        )
        return gym.make(env_id)

    # Evaluate baseline on fresh environment
    baseline_test_env = make_test_env()
    baseline_results = evaluator.evaluate_baseline(baseline_test_env)
    all_results = [baseline_results]

    # Load and evaluate trained models
    print("\n" + "=" * 80)
    print("LOADING TRAINED MODELS")
    print("=" * 80)

    # Load Q-Learning (fresh environment)
    try:
        import pickle

        q_test_env = make_test_env()
        with open("q_learning_model.pkl", "rb") as f:
            q_learning_agent_dict = pickle.load(f)
        # Convert lists back to numpy arrays
        q_learning_agent = {k: np.array(v) for k, v in q_learning_agent_dict.items()}
        q_learning_results = evaluator.evaluate_agent(
            q_learning_agent,
            q_test_env,
            agent_type="tabular",
            algorithm_name="Q-Learning",
        )
        all_results.append(q_learning_results)
        print("✓ Q-Learning model loaded and evaluated")
    except FileNotFoundError:
        print("✗ Q-Learning model not found. Run q-learning.py first.")
    except Exception as e:
        print(f"✗ Error loading Q-Learning model: {e}")

    # Load DQN (fresh environment)
    try:
        from stable_baselines3 import DQN

        dqn_test_env = make_test_env()
        dqn_agent = DQN.load("dqn_pos_tagger")
        dqn_results = evaluator.evaluate_agent(
            dqn_agent, dqn_test_env, agent_type="dqn", algorithm_name="DQN"
        )
        all_results.append(dqn_results)
        print("✓ DQN model loaded and evaluated")
    except FileNotFoundError:
        print("✗ DQN model not found. Run dqn.py first.")
    except Exception as e:
        print(f"✗ Error loading DQN model: {e}")

    # Load REINFORCE (fresh environment)
    try:
        import torch
        import torch.nn as nn

        # Recreate PolicyNet architecture
        class PolicyNet(nn.Module):
            def __init__(self, input_dim, output_dim, hidden_size=128):
                super(PolicyNet, self).__init__()
                self.fc = nn.Sequential(
                    nn.Linear(input_dim, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, output_dim),
                    nn.Softmax(dim=-1),
                )

            def forward(self, x):
                return self.fc(x)

        reinforce_test_env = make_test_env()
        policy = PolicyNet(35, 3)  # 35 input features, 3 actions
        policy.load_state_dict(torch.load("reinforce_policy.pt"))
        policy.eval()

        reinforce_results = evaluator.evaluate_agent(
            policy, reinforce_test_env, agent_type="policy", algorithm_name="REINFORCE"
        )
        all_results.append(reinforce_results)
        print("✓ REINFORCE model loaded and evaluated")
    except FileNotFoundError:
        print("✗ REINFORCE model not found. Run reinforce.py first.")
    except Exception as e:
        print(f"✗ Error loading REINFORCE model: {e}")

    if len(all_results) == 1:
        print("\n⚠ WARNING: No trained models found. Only baseline results available.")
        print(
            "Run the individual training scripts first or use run_complete_analysis.py"
        )

    # Generate visualizations
    os.makedirs("analysis_results", exist_ok=True)

    evaluator.generate_results_table(
        all_results, output_path="analysis_results/results_table.txt"
    )
    evaluator.generate_per_tag_table(
        all_results, output_path="analysis_results/per_tag_results.txt"
    )
    evaluator.plot_accuracy_comparison(
        all_results, output_path="analysis_results/accuracy_comparison.png"
    )

    # Only plot action distribution and improvement if we have RL results
    if len(all_results) > 1:
        evaluator.plot_action_distribution(
            all_results, output_path="analysis_results/action_distribution.png"
        )
        evaluator.plot_improvement_analysis(
            all_results, output_path="analysis_results/improvement_analysis.png"
        )

    evaluator.plot_per_tag_heatmap(
        all_results, output_path="analysis_results/per_tag_heatmap.png"
    )

    # Generate DICT-specific analysis plots
    evaluator.plot_dict_analysis(
        all_results, output_path="analysis_results/dict_analysis.png"
    )
    evaluator.generate_dict_summary_table(
        all_results, output_path="analysis_results/dict_summary.txt"
    )

    evaluator.generate_interpretation_report(
        all_results, output_path="analysis_results/interpretation_report.txt"
    )

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("All results saved to 'analysis_results/' directory")
    print("Files generated:")
    print("  • results_table.txt - Overall comparison")
    print("  • per_tag_results.txt - Per-tag accuracy")
    print("  • dict_summary.txt - DICT usage analysis")
    print("  • accuracy_comparison.png - Accuracy bar chart")
    print("  • action_distribution.png - Action usage stacked bars")
    print("  • improvement_analysis.png - Improvement over baseline")
    print("  • dict_analysis.png - DICT efficiency comparison")
    print("  • per_tag_heatmap.png - Per-tag accuracy heatmap")
    print("  • interpretation_report.txt - Detailed interpretation")
    print("=" * 80)


if __name__ == "__main__":
    main()
