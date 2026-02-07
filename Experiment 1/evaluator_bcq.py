import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle
import os
import sys
import nltk
from nltk.corpus import brown
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import gymnasium as gym

# 1. Path Fix
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# 2. Import Enviroment to trigger the 'register' calls
import Enviroment  

# --- CONFIGURATION ---
ENV_ID = "UniversalPosTagging-v6" 
MODEL_FILE = "stats_50000_eps_dqn_bcq_per_cql/pos_tagger_full.keras" # Point to your Step 3 model
TAG_MAP_FILE = "stats_50000_eps_dqn_bcq_per_cql/tag_map.pkl"
TEST_SIZE = 1000 
SAVE_PREFIX = "evaluation-v7"

# --- DATA PREP ---
print("Loading Test Data...")
nltk.download('brown')
nltk.download('universal_tagset')
all_sents = brown.tagged_sents(tagset='universal')
test_sents = all_sents[-TEST_SIZE:]

# --- LOAD RESOURCES ---
with open(TAG_MAP_FILE, "rb") as f:
    tag_map = pickle.load(f)

idx_to_tag = {v: k for k, v in tag_map.items()}
target_names = [idx_to_tag[i] for i in range(len(tag_map))]

print(f"Loading Model: {MODEL_FILE}")
model = keras.models.load_model(MODEL_FILE)

# 3. Initialize Environment via Registry
print(f"Initializing Environment: {ENV_ID}")
env = gym.make(ENV_ID, sentences=[[w for w,t in s] for s in test_sents], 
                       tags=[[tag_map[t] for w,t in s] for s in test_sents])

def run_full_evaluation():
    y_true = []
    y_pred = []
    error_log = [] # To track which words failed
    
    print(f"Evaluating {TEST_SIZE} sentences with Offline RL Constraints...")
    for sent in tqdm(test_sents):
        words = [w for w, t in sent]
        true_tags = [tag_map[t] for w, t in sent]
        
        # Reset env state
        env.unwrapped.target_sentence = words
        env.unwrapped.target_tags = true_tags
        env.unwrapped.current_word_idx = 0
        
        state = env.unwrapped._get_observation(0, -1)
        
        for i in range(len(words)):
            current_word = words[i]
            
            # --- OFFLINE RL TECHNIQUE: LEXICAL MASKING ---
            # Even if the model predicts a wrong tag, the mask ensures it's a VALID wrong tag
            valid_mask = env.unwrapped.get_action_mask(current_word)
            
            state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
            q_values = model(state_tensor)[0].numpy()
            
            # Apply Mask: Penalty to illegal actions
            masked_qs = q_values + (1.0 - valid_mask) * -1e9 
            action_idx = np.argmax(masked_qs)
            
            y_pred.append(action_idx)
            y_true.append(true_tags[i])
            
            # Error tracking for Discussion section
            if action_idx != true_tags[i]:
                error_log.append({
                    'word': current_word,
                    'true': idx_to_tag[true_tags[i]],
                    'pred': idx_to_tag[action_idx],
                    'context': " ".join(words[max(0, i-2):i+3]) # Window of 5 words
                })
            
            # Move to next word (Auto-regressive)
            if i < len(words) - 1:
                state = env.unwrapped._get_observation(i + 1, action_idx)

    return np.array(y_true), np.array(y_pred), error_log

# --- EXECUTION ---
y_true, y_pred, errors = run_full_evaluation()

# 1. Standard Metrics
acc = accuracy_score(y_true, y_pred)
print(f"\nOverall Test Accuracy: {acc * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=target_names))

# 2. Plot Confusion Matrix

cm = confusion_matrix(y_true, y_pred, normalize='true')
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='.2f', cmap='magma', 
            xticklabels=target_names, yticklabels=target_names)
plt.title(f'Normalized Confusion Matrix (Offline RL Agent)\nAccuracy: {acc*100:.2f}%')
plt.ylabel('True Tag')
plt.xlabel('Predicted Tag')
plt.savefig(f"{SAVE_PREFIX}_confusion_matrix.png")

# 3. Plot Per-Tag Performance

per_tag_acc = cm.diagonal()
plt.figure(figsize=(10, 6))
plt.bar(target_names, per_tag_acc, color='teal')
plt.axhline(y=acc, color='red', linestyle='--', label=f'Avg Acc: {acc:.2f}')
plt.title('Recall Per Tag (Model: DQN + BCQ)')
plt.ylabel('Accuracy')
plt.ylim(0, 1.1)
plt.legend()
plt.tight_layout()
plt.savefig(f"{SAVE_PREFIX}_per_tag.png")

# 4. ERROR ANALYSIS (For Paper Discussion)
print("\n" + "="*30)
print("TOP 10 MOST CONFUSING WORDS")
print("="*30)
from collections import Counter
word_counts = Counter([e['word'] for e in errors])
for word, count in word_counts.most_common(10):
    # Find one example of the error
    ex = next(e for e in errors if e['word'] == word)
    print(f"'{word}': {count} errors. Typical: {ex['true']} predicted as {ex['pred']}")
    print(f"   Context: [... {ex['context']} ...]\n")

print(f"\nEvaluation Finished. Reports saved to current directory.")