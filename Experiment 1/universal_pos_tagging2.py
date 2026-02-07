import gymnasium
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from collections import deque
import random
import nltk
from nltk.corpus import brown
import pickle
import sys
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

# 1. Path Fix (Sibling Import)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# 2. Import the folder name to run the __init__.py code
import Enviroment

# --- DATA LOADING ---
nltk.download('brown')
nltk.download('universal_tagset')

def load_brown_data():
    print("Loading Brown Corpus...")
    tagged_sents = brown.tagged_sents(tagset='universal')
    all_tags = sorted(list({tag for sent in tagged_sents for word, tag in sent}))
    tag_to_idx = {tag: i for i, tag in enumerate(all_tags)}
    
    sentences_data = []
    tags_data = []
    for sent in tagged_sents:
        words = [word for word, tag in sent]
        tag_indices = [tag_to_idx[tag] for word, tag in sent]
        sentences_data.append(words)
        tags_data.append(tag_indices)
        
    return sentences_data, tags_data, tag_to_idx

# Load Data
train_sentences, train_tags, tag_map = load_brown_data()

# Instantiate Environment
env = gymnasium.make("UniversalPosTagging-v1", sentences=train_sentences, tags=train_tags)

INPUT_DIM = env.observation_space.shape[0] 
OUTPUT_DIM = env.action_space.n          

def create_q_model():
    model = keras.Sequential([
        layers.Input(shape=(INPUT_DIM,)),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(OUTPUT_DIM, activation='linear')
    ])
    return model

class DQNAgent:
    def __init__(self):
        self.gamma = 0.99           
        self.epsilon = 1.0          
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.999 
        self.learning_rate = 0.001
        self.batch_size = 64
        self.memory = deque(maxlen=20000) 

        self.model = create_q_model()
        self.target_model = create_q_model()
        self.target_model.set_weights(self.model.get_weights())
        
        self.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.loss_function = keras.losses.MeanSquaredError()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(OUTPUT_DIM) 
        
        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
        q_values = self.model(state_tensor)
        return np.argmax(q_values[0])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return 0  # Return 0 loss if not training

        minibatch = random.sample(self.memory, self.batch_size)
        
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        future_rewards = self.target_model.predict_on_batch(next_states)
        
        target_q_values = rewards + self.gamma * np.amax(future_rewards, axis=1) * (1 - dones)

        with tf.GradientTape() as tape:
            all_q_values = self.model(states)
            masks = tf.one_hot(actions, OUTPUT_DIM)
            current_q_values = tf.reduce_sum(tf.multiply(all_q_values, masks), axis=1)
            loss = self.loss_function(target_q_values, current_q_values)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        return loss.numpy() # Return loss for tracking

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

# --- STATS HELPERS ---
def calculate_moving_average(data, window_size=50):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_training_stats(history_dict, save_folder):
    """
    Generates and saves plots for Reward, Loss, and Accuracy.
    """
    episodes = range(1, len(history_dict['rewards']) + 1)
    
    # 1. Plot Rewards
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, history_dict['rewards'], label='Episode Reward', alpha=0.3, color='blue')
    # Add Moving Average
    if len(history_dict['rewards']) > 50:
        ma = calculate_moving_average(history_dict['rewards'])
        plt.plot(range(50, len(history_dict['rewards']) + 1), ma, label='50-Ep Avg', color='red', linewidth=2)
    plt.title('Training Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_folder, "plot_rewards.png"))
    plt.close()

    # 2. Plot Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, history_dict['accuracies'], label='Episode Accuracy', color='green')
    plt.title('Training Accuracy per Episode (Percentage of Correct Tags)')
    plt.xlabel('Episode')
    plt.ylabel('Accuracy (0-1)')
    plt.grid(True)
    plt.savefig(os.path.join(save_folder, "plot_accuracy.png"))
    plt.close()
    
    # 3. Plot Loss
    # Since loss is per batch, we average it per episode
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, history_dict['avg_losses'], label='Avg Loss', color='orange')
    plt.title('Average Training Loss per Episode')
    plt.xlabel('Episode')
    plt.ylabel('MSE Loss')
    plt.yscale('log') # Log scale helps see loss decay better
    plt.grid(True)
    plt.savefig(os.path.join(save_folder, "plot_loss.png"))
    plt.close()
    
    print(f"Graphs saved to {save_folder}")

# --- MAIN SETUP ---
agent = DQNAgent()
EPISODES = 5000
TARGET_UPDATE_FREQ = 10 

# --- METRICS STORAGE ---
history = {
    'rewards': [],
    'accuracies': [],
    'avg_losses': []
}

print(f"Starting Training on {len(train_sentences)} sentences...")

script_folder = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_folder, "pos_tagger_dqn.keras")
map_path = os.path.join(script_folder, "tag_map.pkl")
stats_path = os.path.join(script_folder, "training_stats.pkl")

# --- THE TRAINING LOOP ---
try:
    progress_bar = tqdm(range(EPISODES), desc="Training Episodes", unit="ep")
    
    for e in progress_bar:
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        # Stats for this episode
        correct_tags = 0
        total_tags = 0
        losses = []
        
        while not done:
            action = agent.get_action(state)
            next_state, reward, done, _, _ = env.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            
            # Track Accuracy (Reward > 0 means correct in our Env)
            # Note: Reward is 1.0 or 5.0 (OOV bonus) if correct, -1.0 if wrong
            if reward > 0:
                correct_tags += 1
            total_tags += 1
            
            # Train model
            loss = agent.replay()
            if loss != 0:
                losses.append(loss)
        
        # End of Episode Updates
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay    
        
        if e % TARGET_UPDATE_FREQ == 0:
            agent.update_target_model()
        
        # --- SAVE STATS ---
        ep_accuracy = correct_tags / total_tags if total_tags > 0 else 0
        ep_avg_loss = np.mean(losses) if losses else 0
        
        history['rewards'].append(total_reward)
        history['accuracies'].append(ep_accuracy)
        history['avg_losses'].append(ep_avg_loss)
        
        # Update progress bar
        if e % 10 == 0:
            progress_bar.set_postfix({
                "Score": f"{total_reward:.1f}", 
                "Acc": f"{ep_accuracy:.2f}",
                "Eps": f"{agent.epsilon:.2f}"
            })

except KeyboardInterrupt:
    print("\n\nTraining interrupted manually (Ctrl+C)!")

except Exception as error:
    print(f"\n\nCRITICAL ERROR: {error}")
    import traceback
    traceback.print_exc()

finally:
    print("\n--- SAVING MODEL & STATS ---")
    
    # 1. Save Keras Model
    agent.model.save(model_path)
    print(f"Neural Network saved to: {model_path}")

    # 2. Save Tag Map
    with open(map_path, "wb") as f:
        pickle.dump(tag_map, f)
    
    # 3. Save Raw Stats (so you can re-plot later if you want)
    with open(stats_path, "wb") as f:
        pickle.dump(history, f)
        
    # 4. Generate Graphs
    plot_training_stats(history, script_folder)
    
    print("Safe exit complete.")

def predict_sentence(sentence_str):
    print(f"\n--- Predicting: '{sentence_str}' ---")
    words = sentence_str.split()
    
    # Setup dummy env
    env.unwrapped.target_sentence = words
    env.unwrapped.target_tags = [0] * len(words) 
    env.unwrapped.current_word_idx = 0
    
    # Get initial state
    state = env.unwrapped._get_observation(0, -1)
    
    print(f"{'Word':<15} | {'Predicted Tag'}")
    print("-" * 35)

    for i in range(len(words)):
        # 1. Predict
        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
        q_values = agent.model(state_tensor)
        action_idx = np.argmax(q_values[0])
        
        # 2. Decode Tag
        tag_name = [k for k, v in tag_map.items() if v == action_idx][0]
        print(f"{words[i]:<15} | {tag_name}")
        
        # 3. Update State (ONLY if there is a next word)
        if i < len(words) - 1:
            state = env.unwrapped._get_observation(i+1, action_idx)

# Test
predict_sentence("The algorithm was processing quickly")