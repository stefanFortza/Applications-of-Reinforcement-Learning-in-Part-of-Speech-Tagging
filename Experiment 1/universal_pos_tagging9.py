import gymnasium
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random
import nltk
from nltk.corpus import brown
import pickle
import sys
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

# 1. Path Fix
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

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

# --- PER HELPER CLASSES ---
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0: self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree): return idx
        if s <= self.tree[left]: return self._retrieve(left, s)
        else: return self._retrieve(right, s - self.tree[left])

    def total(self): return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write += 1
        if self.write >= self.capacity: self.write = 0
        if self.n_entries < self.capacity: self.n_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.epsilon = 0.01

    def sample(self, n, beta=0.4):
        batch, idxs, priorities = [], [], []
        segment = self.tree.total() / n
        for i in range(n):
            s = random.uniform(segment * i, segment * (i + 1))
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)
        prob = np.array(priorities) / self.tree.total()
        is_weights = np.power(self.tree.n_entries * prob, -beta)
        is_weights /= is_weights.max()
        return batch, idxs, is_weights

    def update(self, idx, error):
        p = (error + self.epsilon) ** self.alpha
        self.tree.update(idx, p)

# --- AGENT ARCHITECTURE ---
def create_q_model(input_dim, output_dim):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(output_dim, activation='linear')
    ])
    return model

class DQNAgent:
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gamma = 0.99           
        self.epsilon = 1.0          
        self.epsilon_min = 0.05
        # Adjusted for 50k steps: 0.9999 means slower decay
        self.epsilon_decay = 0.9999 
        self.learning_rate = 0.0005 # Slightly lower for stability in long runs
        self.batch_size = 64
        self.cql_alpha = 1.0
        self.beta = 0.4 

        self.memory = PrioritizedReplayBuffer(50000)
        self.model = create_q_model(input_dim, output_dim)
        self.target_model = create_q_model(input_dim, output_dim)
        self.target_model.set_weights(self.model.get_weights())
        
        self.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate, clipnorm=1.0) 
        self.loss_function = keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE)

    def remember(self, state, action, reward, next_state, done):
        max_p = np.max(self.memory.tree.tree[-self.memory.tree.capacity:])
        if max_p == 0: max_p = 1.0
        self.memory.tree.add(max_p, (state, action, reward, next_state, done))

    def get_action(self, state, word_mask):
        if np.random.rand() <= self.epsilon:
            valid_indices = np.where(word_mask == 1.0)[0]
            return np.random.choice(valid_indices)
        q_values = self.model(tf.convert_to_tensor([state]))[0].numpy()
        q_values = q_values + (1.0 - word_mask) * -1e9 
        return np.argmax(q_values)

    def replay(self):
        if self.memory.tree.n_entries < self.batch_size: return 0
        minibatch, idxs, is_weights = self.memory.sample(self.batch_size, self.beta)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        is_weights_tensor = tf.convert_to_tensor(is_weights, dtype=tf.float32)
        future_q = self.target_model.predict_on_batch(next_states)
        targets = rewards + self.gamma * np.amax(future_q, axis=1) * (1 - dones)

        with tf.GradientTape() as tape:
            all_q = self.model(states)
            q_data = tf.reduce_sum(all_q * tf.one_hot(actions, self.output_dim), axis=1)
            td_errors = tf.abs(targets - q_data)
            element_loss = self.loss_function(targets, q_data)
            cql_loss = tf.reduce_mean(tf.reduce_logsumexp(all_q, axis=1) - q_data)
            total_loss = tf.reduce_mean(element_loss * is_weights_tensor) + (self.cql_alpha * cql_loss)

        grads = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        for i in range(self.batch_size):
            self.memory.update(idxs[i], td_errors.numpy()[i])
        return total_loss.numpy()

# --- PLOTTING FUNCTION ---
def plot_results(history):
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    
    # 1. Rewards
    axs[0].plot(history['rewards'], alpha=0.3, color='blue', label='Raw Reward')
    axs[0].plot(np.convolve(history['rewards'], np.ones(100)/100, mode='valid'), color='darkblue', label='MA 100')
    axs[0].set_title('Episode Rewards')
    axs[0].legend()

    # 2. Accuracy
    axs[1].plot(history['accuracies'], alpha=0.3, color='green', label='Raw Accuracy')
    axs[1].plot(np.convolve(history['accuracies'], np.ones(100)/100, mode='valid'), color='darkgreen', label='MA 100')
    axs[1].set_title('Tagging Accuracy')
    axs[1].set_ylim([0, 1])
    axs[1].legend()

    # 3. Loss
    axs[2].plot(history['losses'], color='red')
    axs[2].set_title('Average Loss (CQL + Weighted Huber)')
    axs[2].set_yscale('log')

    plt.tight_layout()
    plt.savefig('training_performance.png')
    plt.show()

# --- MAIN LOOP ---
train_sentences, train_tags, tag_map = load_brown_data()
env = gymnasium.make("UniversalPosTagging-v6", sentences=train_sentences, tags=train_tags)

agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)

# MODIFY HERE: Set to 50000
EPISODES = 50000 
TARGET_UPDATE_FREQ = 100 # Adjusted for longer run

history = {'rewards': [], 'accuracies': [], 'losses': []}

try:
    pbar = tqdm(range(EPISODES), desc="Training")
    for e in pbar:
        state, _ = env.reset()
        total_reward, correct, total, ep_losses = 0, 0, 0, []
        done = False
        
        while not done:
            curr_word = env.unwrapped.target_sentence[env.unwrapped.current_word_idx]
            mask = env.unwrapped.get_action_mask(curr_word)
            
            action = agent.get_action(state, mask)
            next_state, reward, done, _, _ = env.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if reward > 0: correct += 1
            total += 1
            
            loss = agent.replay()
            if loss != 0: ep_losses.append(loss)

        if agent.epsilon > agent.epsilon_min: agent.epsilon *= agent.epsilon_decay
        if agent.beta < 1.0: agent.beta += (1.0 - 0.4) / EPISODES
        if e % TARGET_UPDATE_FREQ == 0: agent.target_model.set_weights(agent.model.get_weights())
            
        history['rewards'].append(total_reward)
        history['accuracies'].append(correct/total)
        history['losses'].append(np.mean(ep_losses) if ep_losses else 0)

        if e % 50 == 0:
            pbar.set_postfix({"Acc": f"{np.mean(history['accuracies'][-50:]):.2f}", "Eps": f"{agent.epsilon:.2f}"})

except KeyboardInterrupt:
    print("\nInterrupted.")

finally:
    print("\nSaving files and generating plots...")
    agent.model.save("pos_tagger_full.keras")
    with open("tag_map.pkl", "wb") as f:
        pickle.dump(tag_map, f)
    with open("training_stats.pkl", "wb") as f:
        pickle.dump(history, f)
    
    # Generate the graphics
    plot_results(history)
    print("Files saved and training_performance.png generated.")