import sys
import os

from Algorithms.BaseTaggerPOSUtils.model_inference import PosTaggerInference
from Algorithms.BaseTaggerPOSUtils.model_training import train_pos_tagger

# Add the project root to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import torch
import gymnasium as gym
from Enviroment.BaseTaggerEnv.environment import PosCorrectionEnv
from Algorithms.BaseTaggerPOSUtils.dataset import (
    brown_to_training_data,
    get_brown_as_universal,
    get_tag_list,
)


def main():
    # --- 1. Dataset Setup ---
    print("Loading dataset...")
    dataset = get_brown_as_universal()
    tag_list = get_tag_list()
    print(f"Tag list: {tag_list}, Number of tags: {len(tag_list)}")

    # Format: (Sentence, [List of Universal Tags])
    # Using first 1000 sentences for demonstration
    training_data = brown_to_training_data([dataset[i] for i in range(10000)])

    # --- 2. Train and Save Model ---
    print("\n--- Training Model ---")
    # This function handles tokenization, dataset creation, training, and saving
    train_pos_tagger(training_data, tag_list, output_dir="./pos_tagger_model", epochs=5)

    # --- 3. Load and Test Model ---
    print("\n--- Testing Model Inference ---")
    saved_model_path = "./pos_tagger_model"
    inference = PosTaggerInference(saved_model_path, tag_list)

    text = "Reinforcement learning is fascinating."
    results = inference.predict(text)

    print(f"Input: {text}")
    for token, tag in results:
        print(f"{token:<15} -> {tag}")

    # # --- 4. RL Environment Setup ---
    # print("\n--- Setting up RL Environment ---")
    # gym.register(
    #     id="gymnasium_env/PosCorrection-v0",
    #     entry_point=PosCorrectionEnv,
    #     kwargs={
    #         "dataset": training_data,
    #         "nlp_model": "en_core_web_sm",
    #     },
    # )

    # env = gym.make("gymnasium_env/PosCorrection-v0")
    # obs, info = env.reset(seed=42)

    # print(f"Start Word: '{info['word']}' | Base Tag: {info['base_tag']}")
    # print(f"Observation Keys: {obs.keys()}")

    # done = False
    # total_reward = 0

    # print("\n--- Starting Episode ---")
    # while not done:
    #     # Random Agent for demonstration
    #     action = 0  # Always KEEP

    #     # Execute Step
    #     obs, reward, terminated, truncated, info = env.step(action)

    #     action_name = ["KEEP", "SHIFT", "DICT"][action]
    #     word = info.get("word", "END")

    #     print(f"Action: {action_name: <5} | Word: {word: <10} | Reward: {reward:.1f}")

    #     total_reward += reward
    #     done = terminated or truncated

    # print(f"Episode Finished. Total Reward: {total_reward}")


if __name__ == "__main__":
    main()
