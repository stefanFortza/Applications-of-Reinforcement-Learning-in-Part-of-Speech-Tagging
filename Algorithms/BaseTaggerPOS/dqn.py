import sys
import os
import gymnasium as gym
import numpy as np
import torch as th
import matplotlib.pyplot as plt

# Add the project root to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback

from Enviroment.BaseTaggerEnv.environment import PosCorrectionEnv
from Algorithms.BaseTaggerPOSUtils.dataset import (
    get_brown_as_universal,
    brown_to_training_data,
)
from Algorithms.BaseTaggerPOSUtils.rl_utils import (
    evaluate_baseline_accuracy,
    analyze_dict_usage,
    obs_to_tensor,
)


# === Reward and Q-Value Logger Callback ===
class MonitorDQNCallback(BaseCallback):
    """
    Callback to monitor the evolution of the q-value for initial states
    and track episode rewards.
    """

    def __init__(self, env, sample_interval: int = 500):
        super().__init__()
        self.episode_rewards = []
        self.max_q_values = []
        self.timesteps = []
        self.sample_interval = sample_interval

        # Sample some initial states to monitor Q-values
        self.monitor_obs = []
        for _ in range(10):
            obs, _ = env.reset()
            self.monitor_obs.append(obs)

    def _on_step(self) -> bool:
        # Log rewards
        if "episode" in self.locals["infos"][0]:
            self.episode_rewards.append(self.locals["infos"][0]["episode"]["r"])

        # Log Max Q-values for initial states
        if self.n_calls % self.sample_interval == 0:
            q_values_list = []
            for obs in self.monitor_obs:
                with th.no_grad():
                    # Convert dict obs to SB3 compatible format
                    obs_tensor, _ = self.model.policy.obs_to_tensor(obs)
                    # Move to device
                    for key in obs_tensor:
                        obs_tensor[key] = obs_tensor[key].to(self.model.device)

                    q_values = self.model.q_net(obs_tensor)
                    q_values_list.append(q_values.cpu().numpy().max())

            self.max_q_values.append(np.mean(q_values_list))
            self.timesteps.append(self.num_timesteps)

        return True


def main():
    # --- Load Data ---
    print("--- Loading Training Data ---")
    brown_data = get_brown_as_universal()
    training_data = brown_to_training_data(brown_data[:100])
    print(f"Loaded {len(training_data)} sentences")

    # --- Setup Environment ---
    print("\n--- Setting up Environment ---")

    def make_env(data):
        env = PosCorrectionEnv(dataset=data, mode="sequential")
        from stable_baselines3.common.monitor import Monitor

        return Monitor(env)

    train_env = make_env(training_data)

    # --- Create DQN Model ---
    print("\n--- Initializing DQN (Stable-Baselines3) ---")
    model = DQN(
        "MultiInputPolicy",
        train_env,
        verbose=1,
        learning_rate=4e-3,
        batch_size=128,
        buffer_size=10000,
        learning_starts=1000,
        gamma=0.98,
        target_update_interval=600,
        train_freq=16,
        gradient_steps=8,
        exploration_fraction=0.2,
        exploration_final_eps=0.07,
        policy_kwargs=dict(net_arch=[256, 256]),
        seed=42,
    )

    # --- Train Agent ---
    print("\n--- Training Agent ---")
    monitor_callback = MonitorDQNCallback(train_env)
    model.learn(total_timesteps=10000, callback=monitor_callback, log_interval=10)

    # --- Save Model ---
    model.save("dqn_pos_tagger")

    # --- Plot Results ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # Plot Rewards
    rewards = monitor_callback.episode_rewards
    if len(rewards) > 0:
        window = 20
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax1.plot(smoothed)
        ax1.set_title("DQN Training: Episode Rewards (Smoothed)")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Total Reward")
        ax1.grid(True, alpha=0.3)

    # Plot Max Q-values
    if len(monitor_callback.max_q_values) > 0:
        ax2.plot(
            monitor_callback.timesteps, monitor_callback.max_q_values, color="orange"
        )
        ax2.set_title("Evolution of Max Q-value for Initial States")
        ax2.set_xlabel("Timesteps")
        ax2.set_ylabel("Average Max Q-value")
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("dqn_training_analysis.png")
    print("Analysis plots saved to dqn_training_analysis.png")

    # --- Evaluate ---
    print("\n--- Final Evaluation ---")
    test_data = brown_to_training_data(brown_data[100:200])
    test_env = make_env(test_data)

    mean_reward, std_reward = evaluate_policy(
        model,
        test_env,
        deterministic=True,
        n_eval_episodes=50,
    )
    print(f"Post-train Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # Accuracy Evaluation
    baseline_acc, _, _ = evaluate_baseline_accuracy(test_env, num_episodes=50)

    total_correct = 0
    total_predictions = 0
    for _ in range(50):
        obs, _ = test_env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            done = terminated or truncated
            if info["final_tag"] == info["acted_gold_tag"]:
                total_correct += 1
            total_predictions += 1

    dqn_acc = total_correct / total_predictions if total_predictions > 0 else 0.0
    print(f"Baseline (KEEP): {baseline_acc:.3f}")
    print(f"DQN Accuracy:    {dqn_acc:.3f}")

    # --- Analyze DICT Usage ---
    print("\n--- DICT Action Usage Analysis ---")
    dict_stats = analyze_dict_usage(
        agent=model,
        env=test_env,
        obs_to_state_fn=obs_to_tensor,
        num_episodes=50,
        is_dqn=True,
        is_policy=False,
    )

    print(f"\nDQN DICT Analysis:")
    print(f"  Total DICT uses: {dict_stats['dict_total']}")
    if dict_stats["dict_total"] > 0:
        print(
            f"  Necessary (base was wrong): {dict_stats['dict_necessary']} ({dict_stats['dict_necessity_rate']*100:.1f}%)"
        )
        print(
            f"  Unnecessary (base was correct): {dict_stats['dict_unnecessary']} ({dict_stats['dict_waste_rate']*100:.1f}%)"
        )
        if dict_stats["dict_necessity_rate"] > 0.7:
            print("  ✓ Efficient use of DICT action")
        else:
            print("  ✗ DICT often used when not needed")

    # --- Save Results Summary ---
    print("\n--- Saving Results Summary ---")
    results_summary = {
        "dqn": {
            "accuracy": dqn_acc,
            "baseline_accuracy": baseline_acc,
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "dict_analysis": {
                "dict_total": dict_stats["dict_total"],
                "dict_necessary": dict_stats["dict_necessary"],
                "dict_unnecessary": dict_stats["dict_unnecessary"],
                "dict_necessity_rate": dict_stats["dict_necessity_rate"],
                "dict_waste_rate": dict_stats["dict_waste_rate"],
            },
        }
    }
    import json

    with open("dqn_results.json", "w") as f:
        json.dump(results_summary, f, indent=2)
    print("Results summary saved to dqn_results.json")


if __name__ == "__main__":
    main()
