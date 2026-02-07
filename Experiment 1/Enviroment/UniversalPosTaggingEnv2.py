import gymnasium as gym
from gymnasium import spaces
import numpy as np
import spacy

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    print("Error: 'en_core_web_md' not found. Please run: python -m spacy download en_core_web_md")
    raise

# --- CONSTANTS ---
NUM_UNIVERSAL_TAGS = 12
NUM_MORPHOLOGICAL_RULES = 20
SIZE_WORD_EMBEDDING = nlp.vocab.vectors_length 

# --- HELPER FUNCTIONS ---
def get_embedded_vector(word):
    if word in nlp.vocab:
        return nlp.vocab[word].vector
    else:
        return nlp(word).vector

def get_morphological_vector(word):
    """ Creates binary vector for OOV features. """
    vec = []
    w_original = word
    w_lower = word.lower()
    
    # Orthographic (4)
    vec.append(1 if w_original[0].isupper() else 0)
    vec.append(1 if w_original.isupper() and len(w_original) > 1 else 0)
    vec.append(1 if any(char.isdigit() for char in w_original) else 0)
    vec.append(1 if '-' in w_original else 0)

    # Suffixes (16)
    vec.append(1 if w_lower.endswith('s') and not w_lower.endswith('ss') else 0)
    vec.append(1 if w_lower.endswith('ing') else 0)
    vec.append(1 if w_lower.endswith('ed') else 0)
    vec.append(1 if w_lower.endswith('ly') else 0)
    vec.append(1 if w_lower.endswith('tion') or w_lower.endswith('sion') else 0)
    vec.append(1 if w_lower.endswith('ment') else 0)
    vec.append(1 if w_lower.endswith('ness') else 0)
    vec.append(1 if w_lower.endswith('ity') else 0)
    vec.append(1 if w_lower.endswith('er') or w_lower.endswith('or') else 0)
    vec.append(1 if w_lower.endswith('able') or w_lower.endswith('ible') else 0)
    vec.append(1 if w_lower.endswith('ous') else 0)
    vec.append(1 if w_lower.endswith('ive') else 0)
    vec.append(1 if w_lower.endswith('al') else 0)
    vec.append(1 if w_lower.endswith('ful') else 0)
    vec.append(1 if w_lower.endswith('ize') or w_lower.endswith('ise') else 0)
    vec.append(1 if w_lower.endswith('est') else 0)
    
    while len(vec) < NUM_MORPHOLOGICAL_RULES:
        vec.append(0)
    
    return np.array(vec[:NUM_MORPHOLOGICAL_RULES], dtype=np.float32)

def get_morphological_cache(sentences):
    morph_dict = {}
    unique_words = set(word for sent in sentences for word in sent)
    print(f"Caching morphological features for {len(unique_words)} unique words...")
    for word in unique_words:
        morph_dict[word] = get_morphological_vector(word)
    return morph_dict

# --- THE ENVIRONMENT CLASS ---

class UniversalPosTaggingEnv(gym.Env):
    
    def __init__(self, sentences, tags):
        super().__init__()
        
        self.sentences = sentences
        self.tags = tags
        
        self.action_space = spaces.Discrete(NUM_UNIVERSAL_TAGS)
        self.embed_dim = SIZE_WORD_EMBEDDING
        self.obs_dim = NUM_UNIVERSAL_TAGS + NUM_MORPHOLOGICAL_RULES + (2 * self.embed_dim) 
        
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.obs_dim,), 
            dtype=np.float32
        )
        
        # 1. Pre-compute Morphology Cache
        self.morph_cache = get_morphological_cache(sentences)

        # 2. Build Grammar Transition Matrix for Reward Shaping
        print("Building Grammar Transition Matrix from data...")
        self.transition_probs = self._build_transition_matrix(tags)

        # Internal state variables
        self.sentence_idx = 0
        self.current_word_idx = 0
        self.target_sentence = []
        self.target_tags = []
        self.last_action_taken = None # Tracks agent's previous choice

    def _build_transition_matrix(self, all_tags):
        """
        Calculates P(Tag_B | Tag_A) for all tag pairs in the corpus.
        Returns a 12x12 matrix of probabilities.
        """
        counts = np.zeros((NUM_UNIVERSAL_TAGS, NUM_UNIVERSAL_TAGS), dtype=np.float32)
        
        for sent_tags in all_tags:
            for i in range(len(sent_tags) - 1):
                curr_t = sent_tags[i]
                next_t = sent_tags[i+1]
                counts[curr_t][next_t] += 1
        
        # Normalize rows to get probabilities
        # Add epsilon to avoid division by zero
        row_sums = counts.sum(axis=1, keepdims=True) + 1e-9
        probs = counts / row_sums
        return probs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.sentence_idx = np.random.randint(0, len(self.sentences))
        self.target_sentence = self.sentences[self.sentence_idx]
        self.target_tags = self.tags[self.sentence_idx]
        
        self.current_word_idx = 0
        self.last_action_taken = None # Reset history
        
        observation = self._get_observation(
            index=self.current_word_idx, 
            prev_tag_action=-1
        )
        
        return observation, {}

    def _get_observation(self, index, prev_tag_action):
        current_word = self.target_sentence[index]
        curr_embed = get_embedded_vector(current_word)
        
        if index + 1 < len(self.target_sentence):
            next_word = self.target_sentence[index+1]
            next_embed = get_embedded_vector(next_word)
        else:
            next_embed = np.zeros(self.embed_dim, dtype=np.float32)
            
        morph_vec = self.morph_cache.get(current_word, np.zeros(NUM_MORPHOLOGICAL_RULES, dtype=np.float32))
        
        prev_tag_vec = np.zeros(NUM_UNIVERSAL_TAGS, dtype=np.float32)
        if prev_tag_action >= 0:
            prev_tag_vec[prev_tag_action] = 1.0
            
        obs = np.concatenate([prev_tag_vec, morph_vec, curr_embed, next_embed], axis=0)
        return obs.astype(np.float32)

    def step(self, action):
        true_tag = self.target_tags[self.current_word_idx]
        is_correct = (action == true_tag)
        
        # --- NEW REWARD FUNCTION ---
        reward = 0.0
        
        if is_correct:
            # Base Reward
            reward = 1.0
            
            # OOV Bonus: If word is unknown to spaCy, big reward
            word = self.target_sentence[self.current_word_idx]
            if word not in nlp.vocab: 
                reward = 5.0
        else:
            # Base Penalty
            reward = -1.0
            
            # GRAMMAR PENALTY (The Hard Constraint)
            # If we are not at the start of a sentence, check if the transition 
            # from (Last Agent Action) -> (Current Agent Action) is linguistically valid.
            if self.last_action_taken is not None:
                trans_prob = self.transition_probs[self.last_action_taken][action]
                
                # If probability is extremely low (< 0.1%), it's a "Hard Mistake"
                if trans_prob < 0.001:
                    reward = -5.0  # Huge penalty for breaking grammar rules
        
        # ---------------------------

        # Update History
        self.last_action_taken = action

        self.current_word_idx += 1
        terminated = False
        if self.current_word_idx >= len(self.target_sentence):
            terminated = True
            final_obs = np.zeros(self.obs_dim, dtype=np.float32)
            return final_obs, reward, terminated, False, {}
        
        next_obs = self._get_observation(self.current_word_idx, prev_tag_action=action)
        
        return next_obs, reward, terminated, False, {}

    def render(self):
        if self.current_word_idx < len(self.target_sentence):
            word = self.target_sentence[self.current_word_idx]
            print(f"Step: {self.current_word_idx} | Word: {word}")
            
    def close(self):
        pass