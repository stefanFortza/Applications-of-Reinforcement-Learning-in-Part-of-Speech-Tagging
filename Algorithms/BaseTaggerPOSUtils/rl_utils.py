import numpy as np
import matplotlib.pyplot as plt
import torch


def discretize_confidence(confidence):
    """Discretize confidence into bins: LOW, MEDIUM, HIGH."""
    if confidence < 0.5:
        return 0  # LOW
    elif confidence < 0.8:
        return 1  # MEDIUM
    else:
        return 2  # HIGH


def discretize_confidence_gap(gap):
    """Discretize confidence gap: SMALL (uncertain), LARGE (confident)."""
    if gap < 0.3:
        return 0  # Small gap - model is uncertain between top choices
    else:
        return 1  # Large gap - model is confident


def obs_to_discrete_state(obs):
    """
    Convert observation to discrete state tuple for tabular methods.
    Total: 12 × 13 × 3 × 2 × 2 = 1,872 states
    """
    base_tag_idx = obs["base_tag_idx"]
    prev_tag_idx = obs["prev_tag_idx"]
    features = obs["features"]

    confidence_level = discretize_confidence(features[0])
    confidence_gap = discretize_confidence_gap(features[1])
    is_first = int(features[8] > 0.5)

    return (base_tag_idx, prev_tag_idx, confidence_level, confidence_gap, is_first)


def obs_to_tensor(obs, device="cpu"):
    """
    Convert observation to a flat tensor for neural networks (DQN).
    Combines one-hot encodings of tags with continuous features.
    """
    base_tag_idx = obs["base_tag_idx"]
    prev_tag_idx = obs["prev_tag_idx"]
    features = obs["features"]  # 10 features

    # One-hot base tag (12 tags)
    base_one_hot = np.zeros(12)
    base_one_hot[base_tag_idx] = 1.0

    # One-hot prev tag (13 tags: 12 + START)
    prev_one_hot = np.zeros(13)
    prev_one_hot[prev_tag_idx] = 1.0

    # Concatenate: 12 + 13 + 10 = 35 dimensions
    state_vector = np.concatenate([base_one_hot, prev_one_hot, features])

    return torch.FloatTensor(state_vector).to(device)


def evaluate_accuracy(agent, env, obs_to_state_fn, num_episodes=100, is_dqn=False):
    """
    Evaluate the accuracy of a trained agent.
    """
    total_correct = 0
    total_predictions = 0

    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False

        while not done:
            state = obs_to_state_fn(obs)

            if is_dqn:
                with torch.no_grad():
                    # state is already a tensor from obs_to_tensor
                    q_values = agent(state.unsqueeze(0))
                    action = q_values.argmax().item()
            else:
                # agent is the Q-table
                if state in agent:
                    action = int(np.argmax(agent[state]))
                else:
                    action = 0  # Default to KEEP

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            final_tag = info["final_tag"]
            gold_tag = info["acted_gold_tag"]

            if final_tag == gold_tag:
                total_correct += 1
            total_predictions += 1

    accuracy = total_correct / total_predictions if total_predictions > 0 else 0.0
    return accuracy, total_correct, total_predictions


def evaluate_baseline_accuracy(env, num_episodes=100):
    """
    Evaluate the accuracy of the base model (always KEEP action).
    """
    total_correct = 0
    total_predictions = 0

    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False

        while not done:
            obs, reward, terminated, truncated, info = env.step(0)
            done = terminated or truncated

            base_tag = info["acted_base_tag"]
            gold_tag = info["acted_gold_tag"]

            if base_tag == gold_tag:
                total_correct += 1
            total_predictions += 1

    accuracy = total_correct / total_predictions if total_predictions > 0 else 0.0
    return accuracy, total_correct, total_predictions


def plot_rewards(
    rewards_dict, title="Training Rewards", save_path="training_rewards.png"
):
    """
    Plot training rewards for multiple algorithms.
    rewards_dict: { "Alg Name": [reward_list] }
    """
    plt.figure(figsize=(10, 5))
    window = 20

    for label, rewards in rewards_dict.items():
        if len(rewards) < window:
            plt.plot(rewards, label=label, alpha=0.3)
            continue
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        plt.plot(smoothed, label=label)

    plt.xlabel("Episode")
    plt.ylabel("Avg Reward per Token (Smoothed)")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.show()


def analyze_dict_usage(
    agent, env, obs_to_state_fn, num_episodes=100, is_dqn=False, is_policy=False
):
    """
    Analyze DICT action usage: when DICT is chosen, check if KEEP would have been correct.

    Args:
        agent: Trained agent (Q-table, DQN model, or policy network)
        env: Test environment
        obs_to_state_fn: Function to convert observation to state
        num_episodes: Number of episodes to evaluate
        is_dqn: Whether agent is a DQN (Stable-Baselines3)
        is_policy: Whether agent is a policy network (REINFORCE)

    Returns:
        dict: Statistics about DICT usage
    """
    dict_total = 0  # Total times DICT was chosen
    dict_necessary = 0  # Times DICT was necessary (base tagger was wrong)
    dict_unnecessary = 0  # Times DICT was unnecessary (base tagger was correct)

    # Track all actions for comparison
    action_counts = {"KEEP": 0, "SHIFT": 0, "DICT": 0}

    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False

        while not done:
            # Get base tag prediction
            base_tag = info["base_tag"]
            gold_tag = info["gold_tag"]

            # Select action based on agent type
            if is_dqn:
                with torch.no_grad():
                    action, _ = agent.predict(obs, deterministic=True)
            elif is_policy:
                state_t = obs_to_state_fn(obs)
                with torch.no_grad():
                    probs = agent(state_t)
                    action = torch.argmax(probs).item()
            else:
                # Tabular Q-learning/SARSA
                state = obs_to_state_fn(obs)
                if state in agent:
                    action = int(np.argmax(agent[state]))
                else:
                    action = 0  # Default to KEEP

            # Track action counts
            action_names = ["KEEP", "SHIFT", "DICT"]
            action_counts[action_names[action]] += 1

            # If DICT was chosen, analyze whether it was necessary
            if action == 2:  # DICT action
                dict_total += 1
                base_was_correct = base_tag == gold_tag

                if base_was_correct:
                    dict_unnecessary += 1
                else:
                    dict_necessary += 1

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

    results = {
        "dict_total": dict_total,
        "dict_necessary": dict_necessary,
        "dict_unnecessary": dict_unnecessary,
        "dict_necessity_rate": dict_necessary / dict_total if dict_total > 0 else 0.0,
        "dict_waste_rate": dict_unnecessary / dict_total if dict_total > 0 else 0.0,
        "action_counts": action_counts,
    }

    return results
