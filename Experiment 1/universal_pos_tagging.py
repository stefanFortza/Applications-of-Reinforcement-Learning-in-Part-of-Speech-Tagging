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
env = gymnasium.make("UniversalPosTagging-v0", sentences=train_sentences, tags=train_tags)

# Get Dimensions automatically
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
        self.epsilon_decay = 0.999 # Slower decay for more exploration
        self.learning_rate = 0.001
        self.batch_size = 64
        self.memory = deque(maxlen=20000) # Increased memory

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
            return

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

        

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
# Initialize Agent
agent = DQNAgent()
EPISODES = 2000
TARGET_UPDATE_FREQ = 20 

print(f"Starting Training on {len(train_sentences)} sentences...")

# --- DEFINE PATHS FIRST ---
script_folder = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_folder, "pos_tagger_dqn.keras")
map_path = os.path.join(script_folder, "tag_map.pkl")

# --- THE SAFETY NET LOOP ---
try:
    # Use tqdm for a nice progress bar
    progress_bar = tqdm(range(EPISODES), desc="Training Episodes", unit="ep")
    
    for e in progress_bar:
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.get_action(state)
            next_state, reward, done, _, _ = env.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            
            # Train model
            agent.replay()
        
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay    
        
        # Update Target Network
        if e % TARGET_UPDATE_FREQ == 0:
            agent.update_target_model()
        
        # Update the progress bar description with the score so you know it's alive
        if e % 10 == 0:
            progress_bar.set_postfix({"Score": f"{total_reward:.1f}", "Epsilon": f"{agent.epsilon:.2f}"})


        
except KeyboardInterrupt:
    print("\n\nTraining interrupted manually (Ctrl+C)!")
    print("Saving what we have so far...")

except Exception as error:
    print(f"\n\nCRITICAL ERROR: {error}")
    import traceback
    traceback.print_exc()

finally:
    # THIS BLOCK ALWAYS RUNS - Even if it crashes or you stop it!
    print("\n--- SAVING MODEL ---")
    
    # 1. Save Keras Model
    agent.model.save(model_path)
    print(f"Neural Network saved to: {model_path}")

    # 2. Save Tag Map
    with open(map_path, "wb") as f:
        pickle.dump(tag_map, f)
    print(f"Tag Map saved to: {map_path}")
    
    print("Safe exit complete.")

def predict_sentence(sentence_str):
    print(f"\n--- Predicting: '{sentence_str}' ---")
    words = sentence_str.split()
    
    # CORRECTION 3: Use .unwrapped to access custom attributes
    env.unwrapped.target_sentence = words
    env.unwrapped.target_tags = [0] * len(words) 
    env.unwrapped.current_word_idx = 0
    
    # Access internal method via unwrapped
    state = env.unwrapped._get_observation(0, -1)
    
    print(f"{'Word':<15} | {'Predicted Tag'}")
    print("-" * 35)

    for i in range(len(words)):
        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
        action_idx = np.argmax(agent.model(state_tensor)[0])
        
        tag_name = [k for k, v in tag_map.items() if v == action_idx][0]
        
        print(f"{words[i]:<15} | {tag_name}")
        
        state = env.unwrapped._get_observation(i+1, action_idx)

# Test
predict_sentence("The algorithm was processing quickly")