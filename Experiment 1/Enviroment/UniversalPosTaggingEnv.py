import gymnasium as gym
from gymnasium import spaces
import numpy as np
import spacy

# Load spaCy model once at module level
try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    print("Error: 'en_core_web_md' not found. Please run: python -m spacy download en_core_web_md")
    raise

# --- CONSTANTS ---
NUM_UNIVERSAL_TAGS = 12
NUM_MORPHOLOGICAL_RULES = 20
SIZE_WORD_EMBEDDING = nlp.vocab.vectors_length  # Usually 300

# --- HELPER FUNCTIONS ---

def get_embedded_vector(word):
    """
    Returns the GloVe vector for a word. 
    Optimized: Direct vocabulary lookup is faster than running the full nlp() pipeline.
    """
    if word in nlp.vocab:
        return nlp.vocab[word].vector
    else:
        # Fallback for strict OOV if not found in vocab (rare in spaCy md/lg models)
        return nlp(word).vector

def get_morphological_vector(word):
    """
    Creates a binary vector (size 20) of manual features to help classify OOV words.
    """
    vec = []
    w_original = word
    w_lower = word.lower()
    
    # --- ORTHOGRAPHIC FEATURES (4) ---
    vec.append(1 if w_original[0].isupper() else 0)               # Is Capitalized?
    vec.append(1 if w_original.isupper() and len(w_original) > 1 else 0) # All Caps?
    vec.append(1 if any(char.isdigit() for char in w_original) else 0)   # Numeric?
    vec.append(1 if '-' in w_original else 0)                     # Hyphenated?

    # --- SUFFIX FEATURES (16) ---
    # Manual check for common suffixes
    vec.append(1 if w_lower.endswith('s') and not w_lower.endswith('ss') else 0) # Plural
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
    
    # Safety padding: ensure vector is exactly size 20
    while len(vec) < NUM_MORPHOLOGICAL_RULES:
        vec.append(0)
    
    return np.array(vec[:NUM_MORPHOLOGICAL_RULES], dtype=np.float32)

def get_morphological_cache(sentences):
    """
    Pre-computes morphological features for all unique words to speed up training.
    """
    morph_dict = {}  # FIX: Dictionary, not list
    
    # Flatten list of sentences to find unique words
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
        
        # Action Space: 12 Universal Tags
        self.action_space = spaces.Discrete(NUM_UNIVERSAL_TAGS)
        
        # Observation Space Dimensions
        self.embed_dim = SIZE_WORD_EMBEDDING
        self.obs_dim = NUM_UNIVERSAL_TAGS + NUM_MORPHOLOGICAL_RULES + (2 * self.embed_dim) 
        # Total = 12 + 20 + 300 + 300 = 632
        
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.obs_dim,), 
            dtype=np.float32
        )
        
        # Pre-compute cache to make step() faster
        self.morph_cache = get_morphological_cache(sentences)
        
        # Internal state variables
        self.sentence_idx = 0
        self.current_word_idx = 0
        self.target_sentence = []
        self.target_tags = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Select random sentence
        self.sentence_idx = np.random.randint(0, len(self.sentences))
        self.target_sentence = self.sentences[self.sentence_idx]
        self.target_tags = self.tags[self.sentence_idx]
        
        # Reset cursor
        self.current_word_idx = 0
        
        # Get first observation (Prev Tag = -1 because start of sentence)
        observation = self._get_observation(
            index=self.current_word_idx, 
            prev_tag_action=-1
        )
        
        return observation, {}

    def _get_observation(self, index, prev_tag_action):
        # 1. Current Word Embedding
        current_word = self.target_sentence[index]
        curr_embed = get_embedded_vector(current_word)
        
        # 2. Next Word Embedding (Lookahead)
        if index + 1 < len(self.target_sentence):
            next_word = self.target_sentence[index+1]
            next_embed = get_embedded_vector(next_word)
        else:
            # End of sentence -> Zero vector
            next_embed = np.zeros(self.embed_dim, dtype=np.float32)
            
        # 3. Morphology (Fetch from cache)
        # Fallback to zeros if word somehow isn't in cache
        morph_vec = self.morph_cache.get(current_word, np.zeros(NUM_MORPHOLOGICAL_RULES, dtype=np.float32))
        
        # 4. Prev Tag (One-Hot Encoded)
        prev_tag_vec = np.zeros(NUM_UNIVERSAL_TAGS, dtype=np.float32)
        if prev_tag_action >= 0:
            prev_tag_vec[prev_tag_action] = 1.0
            
        # 5. Concatenate
        obs = np.concatenate([prev_tag_vec, morph_vec, curr_embed, next_embed], axis=0)
        return obs.astype(np.float32)

    def step(self, action):
        # 1. Check Correctness
        true_tag = self.target_tags[self.current_word_idx]
        is_correct = (action == true_tag)
        
        # 2. Calculate Reward
        reward = 1.0 if is_correct else -1.0
        
        # --- RESEARCH BONUS: OOV Handling ---
        if is_correct:
            word = self.target_sentence[self.current_word_idx]
            # Check if word is OOV in spaCy (meaning it wasn't in the training data of the embedder)
            # If word is NOT in vocab, give bonus
            if word not in nlp.vocab: 
                reward = 5.0
        
        # 3. Move Cursor
        self.current_word_idx += 1
        
        # 4. Check Termination
        terminated = False
        if self.current_word_idx >= len(self.target_sentence):
            terminated = True
            # Return dummy observation if done (gym requirement)
            final_obs = np.zeros(self.obs_dim, dtype=np.float32)
            return final_obs, reward, terminated, False, {}
        
        # 5. Get Next State
        # Pass the action we just took as 'prev_tag_action'
        next_obs = self._get_observation(self.current_word_idx, prev_tag_action=action)
        
        return next_obs, reward, terminated, False, {}

    def render(self):
        # Only prints when explicitly called
        if self.current_word_idx < len(self.target_sentence):
            word = self.target_sentence[self.current_word_idx]
            print(f"Step: {self.current_word_idx} | Word: {word}")

    def close(self):
        pass