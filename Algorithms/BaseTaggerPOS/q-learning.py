import sys
import os

# --- Accuracy Evaluation ---
# Baseline (KEEP): 0.954 (1245/1305)
# Q-Learning:     0.981 (1119/1141)
# SARSA:          0.962 (1256/1305)


# Add the project root to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from collections import defaultdict

from Enviroment.BaseTaggerEnv.environment import PosCorrectionEnv
from Algorithms.BaseTaggerPOSUtils.dataset import (
    get_brown_as_universal,
    brown_to_training_data,
)
from Algorithms.BaseTaggerPOSUtils.rl_utils import (
    obs_to_discrete_state,
    evaluate_accuracy,
    evaluate_baseline_accuracy,
    plot_rewards,
    analyze_dict_usage,
)

# === Hyperparameters ===
alpha = 0.1  # learning rate
gamma = 0.99  # discount factor
epsilon = 1.0  # exploration rate
eps_min = 0.01
eps_decay = 0.995
episodes = 1050


# === Action Selection ===
def epsilon_greedy(Q, s, epsilon, n_actions=3):
    """Epsilon-greedy action selection."""
    if np.random.rand() < epsilon:
        return np.random.randint(n_actions)
    return int(np.argmax(Q[s]))


# === Training Algorithm ===
def run_agent(env, algorithm="qlearning", seed=42):
    """
    Train agent using Q-learning or SARSA.

    Args:
        env: Gymnasium environment
        algorithm: "qlearning" or "sarsa"
        seed: Random seed for reproducibility

    Returns:
        Q: Learned Q-table
        rewards: List of episode rewards
    """
    Q = defaultdict(lambda: np.zeros(3))  # 3 actions: KEEP, SHIFT, DICT
    rewards = []

    global epsilon
    epsilon = 1.0

    # Set seed for reproducibility
    np.random.seed(seed)
    env.reset(seed=seed)

    for ep in range(episodes):
        obs, info = env.reset(seed=seed)  # Different seed per episode
        s = obs_to_discrete_state(obs)
        a = epsilon_greedy(Q, s, epsilon)
        done = False
        total = 0
        steps = 0

        while not done:
            obs2, r, terminated, truncated, info = env.step(a)
            s2 = obs_to_discrete_state(obs2)
            a2 = epsilon_greedy(Q, s2, epsilon)
            done = terminated or truncated
            total += r
            steps += 1

            if algorithm == "sarsa":
                # SARSA: On-policy TD control
                Q[s][a] += alpha * (r + gamma * Q[s2][a2] * (not done) - Q[s][a])
            elif algorithm == "qlearning":
                # Q-learning: Off-policy TD control
                Q[s][a] += alpha * (r + gamma * np.max(Q[s2]) * (not done) - Q[s][a])

            s, a = s2, a2

        epsilon = max(eps_min, epsilon * eps_decay)

        # Normalize reward by sentence length (steps) for comparability
        avg_reward = total / steps if steps > 0 else 0.0
        rewards.append(avg_reward)

        if ep % 50 == 0:
            # Calculate moving average of the normalized rewards
            mov_avg = np.mean(rewards[-50:]) if ep > 0 else avg_reward
            print(
                f"Episode {ep:4d} | Avg Reward/Token: {avg_reward:6.3f} | Mov Avg: {mov_avg:6.3f} | Epsilon: {epsilon:.3f}"
            )

    return Q, rewards


def print_q_table(Q):
    """Print sample Q-values."""
    print("\n--- Q-table Summary ---")
    print(f"Total states discovered: {len(Q)}")
    print("\nSample Q-values (state -> [KEEP, SHIFT, DICT]):")

    for i, (state, q_vals) in enumerate(Q.items()):
        if i >= 10:
            break
        base_tag, prev_tag, conf, gap, is_first = state
        best_action = ["KEEP", "SHIFT", "DICT"][np.argmax(q_vals)]
        print(
            f"  Base={base_tag:2d}, Prev={prev_tag:2d}, Conf={conf}, Gap={gap}, First={is_first} -> "
            f"[{q_vals[0]:6.2f}, {q_vals[1]:6.2f}, {q_vals[2]:6.2f}] -> {best_action}"
        )


def main():
    # --- Load Data ---
    print("--- Loading Training Data ---")
    brown_data = get_brown_as_universal()
    training_data = brown_to_training_data(brown_data[:100])
    print(f"Loaded {len(training_data)} sentences")

    # --- Setup Environment ---
    print("\n--- Setting up Environment ---")
    gym.register(
        id="gymnasium_env/PosCorrection-v0",
        entry_point=PosCorrectionEnv,
        kwargs={
            "dataset": training_data,
            "mode": "sequential",
        },
    )
    env = gym.make("gymnasium_env/PosCorrection-v0")

    # --- Train Q-Learning ---
    print("\n" + "=" * 50)
    print("Training Q-Learning Agent")
    print("=" * 50)
    Q_qlearning, rewards_qlearning = run_agent(env, algorithm="qlearning", seed=42)
    print_q_table(Q_qlearning)

    # --- Train SARSA ---
    print("\n" + "=" * 50)
    print("Training SARSA Agent")
    print("=" * 50)
    Q_sarsa, rewards_sarsa = run_agent(env, algorithm="sarsa", seed=42)
    print_q_table(Q_sarsa)

    # --- Plot Results ---
    plot_rewards({"Q-Learning": rewards_qlearning, "SARSA": rewards_sarsa})

    # --- Evaluate Accuracy ---
    print("\n--- Accuracy Evaluation ---")

    # Create test environment with different data
    test_data = brown_to_training_data(brown_data[100:200])  # Different sentences
    gym.register(
        id="gymnasium_env/PosCorrectionTest-v0",
        entry_point=PosCorrectionEnv,
        kwargs={
            "dataset": test_data,
            "mode": "sequential",
        },
    )
    test_env = gym.make("gymnasium_env/PosCorrectionTest-v0")

    # Evaluate baseline (always KEEP)
    baseline_acc, baseline_correct, baseline_total = evaluate_baseline_accuracy(
        test_env, num_episodes=50
    )
    print(f"Baseline (KEEP): {baseline_acc:.3f} ({baseline_correct}/{baseline_total})")

    # Evaluate Q-Learning
    q_acc, q_correct, q_total = evaluate_accuracy(
        Q_qlearning, test_env, obs_to_discrete_state, num_episodes=50
    )
    print(f"Q-Learning:     {q_acc:.3f} ({q_correct}/{q_total})")

    # Evaluate SARSA
    sarsa_acc, sarsa_correct, sarsa_total = evaluate_accuracy(
        Q_sarsa, test_env, obs_to_discrete_state, num_episodes=50
    )
    print(f"SARSA:          {sarsa_acc:.3f} ({sarsa_correct}/{sarsa_total})")

    # --- Analyze DICT Usage ---
    print("\n--- DICT Action Usage Analysis ---")

    # Q-Learning DICT analysis
    q_dict_stats = analyze_dict_usage(
        agent=Q_qlearning,
        env=test_env,
        obs_to_state_fn=obs_to_discrete_state,
        num_episodes=50,
        is_dqn=False,
        is_policy=False,
    )

    print(f"\nQ-Learning DICT Analysis:")
    print(f"  Total DICT uses: {q_dict_stats['dict_total']}")
    if q_dict_stats["dict_total"] > 0:
        print(
            f"  Necessary (base was wrong): {q_dict_stats['dict_necessary']} ({q_dict_stats['dict_necessity_rate']*100:.1f}%)"
        )
        print(
            f"  Unnecessary (base was correct): {q_dict_stats['dict_unnecessary']} ({q_dict_stats['dict_waste_rate']*100:.1f}%)"
        )
        if q_dict_stats["dict_necessity_rate"] > 0.7:
            print("  ✓ Efficient use of DICT action")
        else:
            print("  ✗ DICT often used when not needed")

    # SARSA DICT analysis
    sarsa_dict_stats = analyze_dict_usage(
        agent=Q_sarsa,
        env=test_env,
        obs_to_state_fn=obs_to_discrete_state,
        num_episodes=50,
        is_dqn=False,
        is_policy=False,
    )

    print(f"\nSARSA DICT Analysis:")
    print(f"  Total DICT uses: {sarsa_dict_stats['dict_total']}")
    if sarsa_dict_stats["dict_total"] > 0:
        print(
            f"  Necessary (base was wrong): {sarsa_dict_stats['dict_necessary']} ({sarsa_dict_stats['dict_necessity_rate']*100:.1f}%)"
        )
        print(
            f"  Unnecessary (base was correct): {sarsa_dict_stats['dict_unnecessary']} ({sarsa_dict_stats['dict_waste_rate']*100:.1f}%)"
        )
        if sarsa_dict_stats["dict_necessity_rate"] > 0.7:
            print("  ✓ Efficient use of DICT action")
        else:
            print("  ✗ DICT often used when not needed")

    # --- Final Comparison ---
    print("\n--- Final Comparison (last 50 episodes) ---")
    print(
        f"Q-Learning: {np.mean(rewards_qlearning[-50:]):.3f} ± {np.std(rewards_qlearning[-50:]):.3f}"
    )
    print(
        f"SARSA:      {np.mean(rewards_sarsa[-50:]):.3f} ± {np.std(rewards_sarsa[-50:]):.3f}"
    )

    # --- Save Models ---
    print("\n--- Saving Models ---")
    import pickle

    # Convert defaultdicts to regular dicts for pickling
    Q_qlearning_dict = {k: v.tolist() for k, v in Q_qlearning.items()}
    Q_sarsa_dict = {k: v.tolist() for k, v in Q_sarsa.items()}

    with open("q_learning_model.pkl", "wb") as f:
        pickle.dump(Q_qlearning_dict, f)
    print("Q-Learning model saved to q_learning_model.pkl")

    with open("sarsa_model.pkl", "wb") as f:
        pickle.dump(Q_sarsa_dict, f)
    print("SARSA model saved to sarsa_model.pkl")

    # --- Save Results Summary ---
    results_summary = {
        "q_learning": {
            "accuracy": q_acc,
            "baseline_accuracy": baseline_acc,
            "final_avg_reward": np.mean(rewards_qlearning[-50:]),
            "final_std_reward": np.std(rewards_qlearning[-50:]),
            "dict_analysis": {
                "dict_total": q_dict_stats["dict_total"],
                "dict_necessary": q_dict_stats["dict_necessary"],
                "dict_unnecessary": q_dict_stats["dict_unnecessary"],
                "dict_necessity_rate": q_dict_stats["dict_necessity_rate"],
                "dict_waste_rate": q_dict_stats["dict_waste_rate"],
            },
        },
        "sarsa": {
            "accuracy": sarsa_acc,
            "baseline_accuracy": baseline_acc,
            "final_avg_reward": np.mean(rewards_sarsa[-50:]),
            "final_std_reward": np.std(rewards_sarsa[-50:]),
            "dict_analysis": {
                "dict_total": sarsa_dict_stats["dict_total"],
                "dict_necessary": sarsa_dict_stats["dict_necessary"],
                "dict_unnecessary": sarsa_dict_stats["dict_unnecessary"],
                "dict_necessity_rate": sarsa_dict_stats["dict_necessity_rate"],
                "dict_waste_rate": sarsa_dict_stats["dict_waste_rate"],
            },
        },
    }
    import json

    with open("tabular_results.json", "w") as f:
        json.dump(results_summary, f, indent=2)
    print("Results summary saved to tabular_results.json")


if __name__ == "__main__":
    main()
