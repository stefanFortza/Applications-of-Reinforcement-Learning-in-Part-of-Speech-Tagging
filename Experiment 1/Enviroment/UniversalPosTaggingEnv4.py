import gymnasium as gym
from gymnasium import spaces
import numpy as np
import spacy
from collections import defaultdict

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
    vec = []
    w_original = word
    w_lower = word.lower()
    
    # Orthographic
    vec.append(1 if w_original[0].isupper() else 0)
    vec.append(1 if w_original.isupper() and len(w_original) > 1 else 0)
    vec.append(1 if any(char.isdigit() for char in w_original) else 0)
    vec.append(1 if '-' in w_original else 0)

    # Suffixes
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
        
        # Obs = [Prev Tag] + [Morph] + [Prev Word] + [Curr Word] + [Next Word]
        self.obs_dim = NUM_UNIVERSAL_TAGS + NUM_MORPHOLOGICAL_RULES + (3 * self.embed_dim) 
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        
        # 1. Pre-compute Caches
        self.morph_cache = get_morphological_cache(sentences)

        # 2. Build Grammar Matrix (Transition Probability)
        print("Building Grammar Matrix...")
        self.transition_probs = self._build_transition_matrix(tags)
        
        # 3. Build Lexical Dictionary (Emission Probability) <-- NEW
        print("Building Lexical Dictionary (Word -> Allowed Tags)...")
        self.lexical_dict = self._build_lexical_dict(sentences, tags)

        self.sentence_idx = 0
        self.current_word_idx = 0
        self.target_sentence = []
        self.target_tags = []
        self.last_action_taken = None 

    def _build_transition_matrix(self, all_tags):
        counts = np.zeros((NUM_UNIVERSAL_TAGS, NUM_UNIVERSAL_TAGS), dtype=np.float32)
        for sent_tags in all_tags:
            for i in range(len(sent_tags) - 1):
                counts[sent_tags[i]][sent_tags[i+1]] += 1
        row_sums = counts.sum(axis=1, keepdims=True) + 1e-9
        return counts / row_sums

    def _build_lexical_dict(self, sentences, tags):
        """
        Creates a map: "bank" -> {NOUN, VERB}
        Returns a Set of allowed tags for each word.
        """
        lex_dict = defaultdict(set)
        for sent, sent_tags in zip(sentences, tags):
            for word, tag in zip(sent, sent_tags):
                # Store allowed tags for this word
                lex_dict[word].add(tag)
        return lex_dict

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sentence_idx = np.random.randint(0, len(self.sentences))
        self.target_sentence = self.sentences[self.sentence_idx]
        self.target_tags = self.tags[self.sentence_idx]
        self.current_word_idx = 0
        self.last_action_taken = None 
        
        observation = self._get_observation(self.current_word_idx, -1)
        return observation, {}

    def _get_observation(self, index, prev_tag_action):
        # Prev Word
        if index > 0:
            prev_embed = get_embedded_vector(self.target_sentence[index - 1])
        else:
            prev_embed = np.zeros(self.embed_dim, dtype=np.float32)

        # Current + Next Word
        current_word = self.target_sentence[index]
        curr_embed = get_embedded_vector(current_word)
        
        if index + 1 < len(self.target_sentence):
            next_embed = get_embedded_vector(self.target_sentence[index+1])
        else:
            next_embed = np.zeros(self.embed_dim, dtype=np.float32)
            
        # Morphology
        morph_vec = self.morph_cache.get(current_word, np.zeros(NUM_MORPHOLOGICAL_RULES, dtype=np.float32))
        
        # Prev Tag
        prev_tag_vec = np.zeros(NUM_UNIVERSAL_TAGS, dtype=np.float32)
        if prev_tag_action >= 0:
            prev_tag_vec[prev_tag_action] = 1.0
            
        obs = np.concatenate([prev_tag_vec, morph_vec, prev_embed, curr_embed, next_embed], axis=0)
        return obs.astype(np.float32)

    def step(self, action):
        true_tag = self.target_tags[self.current_word_idx]
        is_correct = (action == true_tag)
        current_word = self.target_sentence[self.current_word_idx]
        
        reward = 0.0
        
        # --- 1. CORRECT ---
        if is_correct:
            reward = 1.0
            # OOV Bonus
            if current_word not in nlp.vocab: 
                reward = 10.0
                
        # --- 2. INCORRECT ---
        else:
            # Base Penalty for being wrong
            reward -= 1.0 
            
            # Check 1: LEXICAL VIOLATION (Stacking Penalty)
            # If word is known, but tag is strictly forbidden
            if current_word in self.lexical_dict:
                if action not in self.lexical_dict[current_word]:
                    reward -= 5.0  # Total becomes -6.0
            
            # Check 2: GRAMMAR VIOLATION (Stacking Penalty)
            # If transition is linguistically impossible
            if self.last_action_taken is not None:
                trans_prob = self.transition_probs[self.last_action_taken][action]
                if trans_prob < 0.001:
                     reward -= 5.0 # Total could become -11.0 (Double Penalty)
        
        # ---------------------------

        self.last_action_taken = action
        self.current_word_idx += 1
        
        terminated = (self.current_word_idx >= len(self.target_sentence))
        if terminated:
            return np.zeros(self.obs_dim, dtype=np.float32), reward, terminated, False, {}
        
        next_obs = self._get_observation(self.current_word_idx, action)
        return next_obs, reward, terminated, False, {}

    def render(self):
        if self.current_word_idx < len(self.target_sentence):
            print(f"Word: {self.target_sentence[self.current_word_idx]}")
    def close(self): pass