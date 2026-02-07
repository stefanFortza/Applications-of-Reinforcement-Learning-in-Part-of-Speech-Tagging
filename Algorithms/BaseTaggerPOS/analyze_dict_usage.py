"""
Standalone script to analyze DICT action usage for trained RL agents.
Evaluates when DICT (oracle) is chosen and whether KEEP would have been sufficient.
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import numpy as np
import pickle
import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3 import DQN

from Enviroment.BaseTaggerEnv.environment import PosCorrectionEnv
from Algorithms.BaseTaggerPOSUtils.dataset import (
    get_brown_as_universal,
    brown_to_training_data,
)
from Algorithms.BaseTaggerPOSUtils.rl_utils import (
    obs_to_discrete_state,
    obs_to_tensor,
    analyze_dict_usage,
)


class PolicyNet(nn.Module):
    """Policy network for REINFORCE (copied from reinforce.py)"""

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


def main():
    print("=" * 60)
    print("DICT ACTION USAGE ANALYSIS")
    print("=" * 60)

    # Load dataset
    brown_data = get_brown_as_universal()
    train_data = brown_to_training_data(brown_data[:100])
    test_data = brown_to_training_data(brown_data[100:200])

    # Create test environment
    test_env = PosCorrectionEnv(test_data, mode="sequential")

    # Results storage
    results = {}

    # ============================================================
    # 1. Analyze Q-Learning
    # ============================================================
    print("\n" + "=" * 60)
    print("ANALYZING Q-LEARNING")
    print("=" * 60)

    try:
        with open("q_learning_model.pkl", "rb") as f:
            Q = pickle.load(f)

        dict_stats = analyze_dict_usage(
            agent=Q,
            env=test_env,
            obs_to_state_fn=obs_to_discrete_state,
            num_episodes=50,
            is_dqn=False,
            is_policy=False,
        )

        results["Q-Learning"] = dict_stats

        print(f"\nQ-Learning DICT Analysis:")
        print(f"  Total DICT uses: {dict_stats['dict_total']}")
        if dict_stats["dict_total"] > 0:
            print(
                f"  Necessary (base was wrong): {dict_stats['dict_necessary']} ({dict_stats['dict_necessity_rate']*100:.1f}%)"
            )
            print(
                f"  Unnecessary (base was correct): {dict_stats['dict_unnecessary']} ({dict_stats['dict_waste_rate']*100:.1f}%)"
            )
            print(f"  Action distribution: {dict_stats['action_counts']}")

            if dict_stats["dict_necessity_rate"] > 0.7:
                print("  ✓ Efficient use of DICT action")
            else:
                print("  ✗ DICT often used unnecessarily")
        else:
            print("  → DICT never used (agent avoids expensive oracle action)")

    except FileNotFoundError:
        print("  ✗ Q-Learning model not found (q_learning_model.pkl)")

    # ============================================================
    # 2. Analyze SARSA
    # ============================================================
    print("\n" + "=" * 60)
    print("ANALYZING SARSA")
    print("=" * 60)

    try:
        with open("sarsa_model.pkl", "rb") as f:
            Q_sarsa = pickle.load(f)

        dict_stats = analyze_dict_usage(
            agent=Q_sarsa,
            env=test_env,
            obs_to_state_fn=obs_to_discrete_state,
            num_episodes=50,
            is_dqn=False,
            is_policy=False,
        )

        results["SARSA"] = dict_stats

        print(f"\nSARSA DICT Analysis:")
        print(f"  Total DICT uses: {dict_stats['dict_total']}")
        if dict_stats["dict_total"] > 0:
            print(
                f"  Necessary (base was wrong): {dict_stats['dict_necessary']} ({dict_stats['dict_necessity_rate']*100:.1f}%)"
            )
            print(
                f"  Unnecessary (base was correct): {dict_stats['dict_unnecessary']} ({dict_stats['dict_waste_rate']*100:.1f}%)"
            )
            print(f"  Action distribution: {dict_stats['action_counts']}")

            if dict_stats["dict_necessity_rate"] > 0.7:
                print("  ✓ Efficient use of DICT action")
            else:
                print("  ✗ DICT often used unnecessarily")
        else:
            print("  → DICT never used (agent avoids expensive oracle action)")

    except FileNotFoundError:
        print("  ✗ SARSA model not found (sarsa_model.pkl)")

    # ============================================================
    # 3. Analyze DQN
    # ============================================================
    print("\n" + "=" * 60)
    print("ANALYZING DQN")
    print("=" * 60)

    try:
        dqn_model = DQN.load("dqn_pos_tagger")

        dict_stats = analyze_dict_usage(
            agent=dqn_model,
            env=test_env,
            obs_to_state_fn=obs_to_tensor,  # DQN uses raw obs, not discretized
            num_episodes=50,
            is_dqn=True,
            is_policy=False,
        )

        results["DQN"] = dict_stats

        print(f"\nDQN DICT Analysis:")
        print(f"  Total DICT uses: {dict_stats['dict_total']}")
        if dict_stats["dict_total"] > 0:
            print(
                f"  Necessary (base was wrong): {dict_stats['dict_necessary']} ({dict_stats['dict_necessity_rate']*100:.1f}%)"
            )
            print(
                f"  Unnecessary (base was correct): {dict_stats['dict_unnecessary']} ({dict_stats['dict_waste_rate']*100:.1f}%)"
            )
            print(f"  Action distribution: {dict_stats['action_counts']}")

            if dict_stats["dict_necessity_rate"] > 0.7:
                print("  ✓ Efficient use of DICT action")
            else:
                print("  ✗ DICT often used unnecessarily")
        else:
            print("  → DICT never used (agent avoids expensive oracle action)")

    except Exception as e:
        print(f"  ✗ DQN model not found or error loading: {e}")

    # ============================================================
    # 4. Analyze REINFORCE
    # ============================================================
    print("\n" + "=" * 60)
    print("ANALYZING REINFORCE")
    print("=" * 60)

    try:
        policy = PolicyNet(input_dim=35, output_dim=3, hidden_size=128)
        policy.load_state_dict(torch.load("reinforce_policy.pt"))
        policy.eval()

        dict_stats = analyze_dict_usage(
            agent=policy,
            env=test_env,
            obs_to_state_fn=obs_to_tensor,
            num_episodes=50,
            is_dqn=False,
            is_policy=True,
        )

        results["REINFORCE"] = dict_stats

        print(f"\nREINFORCE DICT Analysis:")
        print(f"  Total DICT uses: {dict_stats['dict_total']}")
        if dict_stats["dict_total"] > 0:
            print(
                f"  Necessary (base was wrong): {dict_stats['dict_necessary']} ({dict_stats['dict_necessity_rate']*100:.1f}%)"
            )
            print(
                f"  Unnecessary (base was correct): {dict_stats['dict_unnecessary']} ({dict_stats['dict_waste_rate']*100:.1f}%)"
            )
            print(f"  Action distribution: {dict_stats['action_counts']}")

            if dict_stats["dict_necessity_rate"] > 0.7:
                print("  ✓ Efficient use of DICT action")
            else:
                print("  ✗ DICT often used unnecessarily")
        else:
            print("  → DICT never used (agent avoids expensive oracle action)")

    except Exception as e:
        print(f"  ✗ REINFORCE model not found or error loading: {e}")

    # ============================================================
    # Summary Table
    # ============================================================
    print("\n" + "=" * 60)
    print("SUMMARY: DICT ACTION EFFICIENCY")
    print("=" * 60)

    if results:
        print(
            f"\n{'Algorithm':<15} {'DICT Uses':<12} {'Necessary %':<15} {'Waste %':<10} {'Efficiency'}"
        )
        print("-" * 70)

        for algo_name, stats in results.items():
            if stats["dict_total"] > 0:
                efficiency = "✓" if stats["dict_necessity_rate"] > 0.7 else "✗"
                print(
                    f"{algo_name:<15} {stats['dict_total']:<12} "
                    f"{stats['dict_necessity_rate']*100:<14.1f}% "
                    f"{stats['dict_waste_rate']*100:<9.1f}% {efficiency}"
                )
            else:
                print(f"{algo_name:<15} {'0 (never used)'}")

        print("\n" + "=" * 60)
        print("INTERPRETATION:")
        print(
            "  • Necessary %: When DICT was used, how often was the base tagger wrong?"
        )
        print(
            "  • Waste %: When DICT was used, how often was the base tagger already correct?"
        )
        print("  • High necessity % (>70%) = Smart use of expensive oracle action")
        print("  • High waste % (>30%) = Overusing DICT when KEEP would suffice")
        print("=" * 60)
    else:
        print("No trained models found. Train algorithms first.")


if __name__ == "__main__":
    main()
