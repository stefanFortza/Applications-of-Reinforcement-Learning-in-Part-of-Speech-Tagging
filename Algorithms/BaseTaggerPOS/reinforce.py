import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import matplotlib.pyplot as plt

# Add the project root to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from Enviroment.BaseTaggerEnv.environment import PosCorrectionEnv
from Algorithms.BaseTaggerPOSUtils.dataset import (
    get_brown_as_universal,
    brown_to_training_data,
)
from Algorithms.BaseTaggerPOSUtils.rl_utils import (
    obs_to_tensor,
    evaluate_accuracy,
    evaluate_baseline_accuracy,
    analyze_dict_usage,
)

# === Hyperparameters ===
LR = 0.001
GAMMA = 0.99
EPISODES = 1000
HIDDEN_SIZE = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# === Policy Network ===
class PolicyNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, output_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.fc(x)


# === Value Network (Baseline) ===
class ValueNet(nn.Module):
    def __init__(self, input_dim):
        super(ValueNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, 1),
        )

    def forward(self, x):
        return self.fc(x)


# === Helper Functions ===
def compute_returns(rewards, gamma=0.99):
    G = 0
    returns = []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns


def train_reinforce(env, seed=42):
    input_dim = 35  # 12 (base) + 13 (prev) + 10 (features)
    output_dim = 3  # KEEP, SHIFT, DICT

    policy = PolicyNet(input_dim, output_dim).to(device)
    value_fn = ValueNet(input_dim).to(device)

    optimizer = optim.Adam(
        list(policy.parameters()) + list(value_fn.parameters()), lr=LR
    )

    all_rewards = []

    torch.manual_seed(seed)
    np.random.seed(seed)

    for ep in range(EPISODES):
        obs, info = env.reset(seed=seed)
        log_probs = []
        values = []
        rewards = []
        done = False
        steps = 0

        while not done:
            state_t = obs_to_tensor(obs, device)

            # Policy forward
            probs = policy(state_t)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

            # Critic forward
            value = value_fn(state_t)

            next_obs, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated

            log_probs.append(dist.log_prob(action))
            values.append(value)
            rewards.append(reward)

            obs = next_obs
            steps += 1

        # Total reward for logging (normalized by steps)
        total_reward = sum(rewards)
        avg_reward = total_reward / steps if steps > 0 else 0.0
        all_rewards.append(avg_reward)

        # Compute discounted returns
        returns = torch.tensor(compute_returns(rewards, GAMMA), dtype=torch.float32).to(
            device
        )

        # Compute advantages
        values = torch.cat(values).squeeze()
        advantages = returns - values.detach()

        # Loss = Policy loss + Value loss
        policy_loss = 0
        for log_p, adv in zip(log_probs, advantages):
            policy_loss += -(log_p * adv)

        value_loss = F.mse_loss(values, returns)

        loss = policy_loss + value_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ep % 50 == 0:
            mov_avg = np.mean(all_rewards[-50:]) if ep > 0 else avg_reward
            print(
                f"Episode {ep:4d} | Avg Reward/Token: {avg_reward:6.3f} | Mov Avg: {mov_avg:6.3f}"
            )

    return policy, all_rewards


def main():
    # --- Load Data ---
    print("--- Loading Training Data ---")
    brown_data = get_brown_as_universal()
    training_data = brown_to_training_data(brown_data[:100])
    print(f"Loaded {len(training_data)} sentences")

    # --- Setup Environment ---
    gym.register(
        id="gymnasium_env/PosCorrectionReinforce-v0",
        entry_point=PosCorrectionEnv,
        kwargs={"dataset": training_data, "mode": "sequential"},
    )
    env = gym.make("gymnasium_env/PosCorrectionReinforce-v0")

    # --- Train REINFORCE ---
    print("\n--- Training REINFORCE Agent (with Baseline) ---")
    policy, rewards = train_reinforce(env)

    # --- Plot Results ---
    plt.figure(figsize=(10, 5))
    window = 20
    smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
    plt.plot(smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Avg Reward per Token (Smoothed)")
    plt.title("POS Correction: REINFORCE + Baseline Training")
    plt.grid(True, alpha=0.3)
    plt.savefig("reinforce_training_rewards.png")
    print("Training plot saved to reinforce_training_rewards.png")

    # --- Evaluate ---
    print("\n--- Accuracy Evaluation ---")
    test_data = brown_to_training_data(brown_data[100:200])
    gym.register(
        id="gymnasium_env/PosCorrectionReinforceTest-v0",
        entry_point=PosCorrectionEnv,
        kwargs={"dataset": test_data, "mode": "sequential"},
    )
    test_env = gym.make("gymnasium_env/PosCorrectionReinforceTest-v0")

    baseline_acc, _, _ = evaluate_baseline_accuracy(test_env, num_episodes=50)

    # Custom evaluation for PolicyNet
    total_correct = 0
    total_predictions = 0
    policy.eval()
    with torch.no_grad():
        for _ in range(50):
            obs, _ = test_env.reset()
            done = False
            while not done:
                state_t = obs_to_tensor(obs, device)
                probs = policy(state_t)
                action = torch.argmax(probs).item()

                obs, reward, terminated, truncated, info = test_env.step(action)
                done = terminated or truncated

                if info["final_tag"] == info["acted_gold_tag"]:
                    total_correct += 1
                total_predictions += 1

    reinforce_acc = total_correct / total_predictions if total_predictions > 0 else 0.0
    print(f"Baseline (KEEP): {baseline_acc:.3f}")
    print(f"REINFORCE Acc:   {reinforce_acc:.3f}")

    # --- Analyze DICT Usage ---
    print("\n--- DICT Action Usage Analysis ---")
    dict_stats = analyze_dict_usage(
        agent=policy,
        env=test_env,
        obs_to_state_fn=obs_to_tensor,
        num_episodes=50,
        is_dqn=False,
        is_policy=True,
    )

    print(f"\nREINFORCE DICT Analysis:")
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

    # --- Save Model ---
    print("\n--- Saving Model ---")
    torch.save(policy.state_dict(), "reinforce_policy.pt")
    print("REINFORCE policy saved to reinforce_policy.pt")

    # --- Save Results Summary ---
    results_summary = {
        "reinforce": {
            "accuracy": reinforce_acc,
            "baseline_accuracy": baseline_acc,
            "final_avg_reward": np.mean(rewards[-50:]),
            "final_std_reward": np.std(rewards[-50:]),
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

    with open("reinforce_results.json", "w") as f:
        json.dump(results_summary, f, indent=2)
    print("Results summary saved to reinforce_results.json")


if __name__ == "__main__":
    main()
