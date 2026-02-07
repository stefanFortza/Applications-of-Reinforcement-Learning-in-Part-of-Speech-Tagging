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
# NOTE: Ensure 'UniversalPosTagging-v6' is registered to your new class in Enviroment/__init__.py
# If not, change this string to match your registration ID.
env = gymnasium.make("UniversalPosTagging-v6", sentences=train_sentences, tags=train_tags)

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
        self.memory = deque(maxlen=50000) 

        self.model = create_q_model()
        self.target_model = create_q_model()
        self.target_model.set_weights(self.model.get_weights())
        
        self.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate, clipnorm=1.0) 
        self.loss_function = keras.losses.Huber() 

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # --- CHANGED: Added word_mask argument ---
    def get_action(self, state, word_mask):
        # 1. Random Exploration (Epsilon-Greedy)
        if np.random.rand() <= self.epsilon:
            # FILTER: Only sample from indices where mask == 1.0 (Valid tags)
            valid_indices = np.where(word_mask == 1.0)[0]
            if len(valid_indices) == 0: return random.randrange(OUTPUT_DIM) # Fallback
            return np.random.choice(valid_indices)
        
        # 2. Model Prediction
        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
        q_values = self.model(state_tensor)[0].numpy()
        
        # MASKING: Add huge penalty to forbidden actions so Argmax never picks them
        # If mask is 0 (forbidden), (1-0)*-1e9 becomes -1,000,000,000
        q_values = q_values + (1.0 - word_mask) * -1e9 
        
        return np.argmax(q_values)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return 0 

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
        
        return loss.numpy() 

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

# --- STATS HELPERS ---
def calculate_moving_average(data, window_size=50):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_training_stats(history_dict, save_folder):
    episodes = range(1, len(history_dict['rewards']) + 1)
    
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, history_dict['rewards'], label='Episode Reward', alpha=0.3, color='blue')
    if len(history_dict['rewards']) > 50:
        ma = calculate_moving_average(history_dict['rewards'])
        plt.plot(range(50, len(history_dict['rewards']) + 1), ma, label='50-Ep Avg', color='red', linewidth=2)
    plt.title('Training Reward per Episode')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_folder, "plot_rewards.png"))
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, history_dict['accuracies'], label='Episode Accuracy', color='green')
    plt.title('Training Accuracy per Episode')
    plt.grid(True)
    plt.savefig(os.path.join(save_folder, "plot_accuracy.png"))
    plt.close()
    
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, history_dict['avg_losses'], label='Avg Loss', color='orange')
    plt.title('Average Training Loss per Episode')
    plt.yscale('log')
    plt.grid(True)
    plt.savefig(os.path.join(save_folder, "plot_loss.png"))
    plt.close()

# --- MAIN SETUP ---
agent = DQNAgent()
EPISODES = 5000
TARGET_UPDATE_FREQ = 20 

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
        
        correct_tags = 0
        total_tags = 0
        losses = []
        
        while not done:
            # --- NEW LOGIC: Get Mask from Environment ---
            # 1. Get the word we are currently trying to tag
            current_word = env.unwrapped.target_sentence[env.unwrapped.current_word_idx]
            
            # 2. Ask environment for valid tags for this word
            valid_mask = env.unwrapped.get_action_mask(current_word)
            
            # 3. Pass mask to agent so it picks a valid action
            action = agent.get_action(state, valid_mask)
            
            next_state, reward, done, _, _ = env.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            
            if reward > 0:
                correct_tags += 1
            total_tags += 1
            
            loss = agent.replay()
            if loss != 0:
                losses.append(loss)
        
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay    
        
        if e % TARGET_UPDATE_FREQ == 0:
            agent.update_target_model()
        
        ep_accuracy = correct_tags / total_tags if total_tags > 0 else 0
        ep_avg_loss = np.mean(losses) if losses else 0
        
        history['rewards'].append(total_reward)
        history['accuracies'].append(ep_accuracy)
        history['avg_losses'].append(ep_avg_loss)
        
        if e % 10 == 0:
            progress_bar.set_postfix({
                "Score": f"{total_reward:.1f}", 
                "Acc": f"{ep_accuracy:.2f}",
                "Eps": f"{agent.epsilon:.2f}"
            })

except KeyboardInterrupt:
    print("\n\nTraining interrupted manually!")
except Exception as error:
    print(f"\n\nCRITICAL ERROR: {error}")
    import traceback
    traceback.print_exc()

finally:
    print("\n--- SAVING MODEL & STATS ---")
    agent.model.save(model_path)
    with open(map_path, "wb") as f:
        pickle.dump(tag_map, f)
    with open(stats_path, "wb") as f:
        pickle.dump(history, f)
    plot_training_stats(history, script_folder)
    print("Safe exit complete.")

# --- UPDATED TEST SUITE ---
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
        # --- NEW LOGIC FOR TEST: Apply Masking ---
        current_word = words[i]
        valid_mask = env.unwrapped.get_action_mask(current_word)
        
        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
        q_values = agent.model(state_tensor)[0].numpy()
        
        # Apply mask
        q_values = q_values + (1.0 - valid_mask) * -1e9 
        action_idx = np.argmax(q_values)
        
        tag_name = [k for k, v in tag_map.items() if v == action_idx][0]
        print(f"{words[i]:<15} | {tag_name}")
        
        if i < len(words) - 1:
            state = env.unwrapped._get_observation(i+1, action_idx)

def run_test_suite():
    print("\n" + "="*50)
    print("RUNNING DIAGNOSTIC TEST SUITE (WITH MASKING)")
    print("="*50)

    test_cases = [
        "The computer computes the result",
        "I will book the flight",
        "The book is on the table",
        "The run was long and tiring",
        "They run very fast",
        "The gloober is jumping lazily",
        "The big red quick fox jumped",
        "The cat slept under the warm blanket",
        "I like her cooking",
        "Give it to her"
    ]

    for sentence in test_cases:
        predict_sentence(sentence)
        print("\n")

run_test_suite()