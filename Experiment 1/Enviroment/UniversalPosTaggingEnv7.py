import gymnasium as gym
from gymnasium import spaces
import numpy as np
import spacy
from collections import defaultdict

# Load spaCy
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
        
        # --- REVERTED TO SMALLER STATE (632) ---
        # Obs = [Prev Tag(12)] + [Morph(20)] + [Curr Word(300)] + [Next Word(300)]
        # We REMOVED [Prev Word] because you proved it added noise.
        self.obs_dim = NUM_UNIVERSAL_TAGS + NUM_MORPHOLOGICAL_RULES + (2 * self.embed_dim) 
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        
        # 1. Pre-compute Caches
        self.morph_cache = get_morphological_cache(sentences)

        # 2. Build Grammar Matrix
        print("Building Grammar Matrix...")
        self.transition_probs = self._build_transition_matrix(tags)
        
        # 3. Build Lexical Dictionary (The crucial addition)
        print("Building Lexical Dictionary...")
        self.lexical_dict = self._build_lexical_dict(sentences, tags)

        self.sentence_idx = 0
        self.current_word_idx = 0
        self.target_sentence = []
        self.target_tags = []
        self.last_action_taken = None 

    def get_action_mask(self, word):
        mask = np.zeros(NUM_UNIVERSAL_TAGS, dtype=np.float32)
        if word in self.lexical_dict:
            # Set allowed tags to 1.0, others remain 0.0
            for tag_idx in self.lexical_dict[word]:
                mask[tag_idx] = 1.0
        else:
            # For OOV words, we allow all actions (or use morphology rules)
            mask.fill(1.0)
        return mask
    def _build_transition_matrix(self, all_tags):
        counts = np.zeros((NUM_UNIVERSAL_TAGS, NUM_UNIVERSAL_TAGS), dtype=np.float32)
        for sent_tags in all_tags:
            for i in range(len(sent_tags) - 1):
                counts[sent_tags[i]][sent_tags[i+1]] += 1
        row_sums = counts.sum(axis=1, keepdims=True) + 1e-9
        return counts / row_sums

    def _build_lexical_dict(self, sentences, tags):
        """ Maps 'bank' -> {NOUN, VERB} """
        lex_dict = defaultdict(set)
        for sent, sent_tags in zip(sentences, tags):
            for word, tag in zip(sent, sent_tags):
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
        # NOTE: No Previous Word Embedding here!
        
        # Current Word
        current_word = self.target_sentence[index]
        curr_embed = get_embedded_vector(current_word)
        
        # Next Word
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
            
        # Concatenate: [Prev Tag, Morph, Curr Word, Next Word]
        obs = np.concatenate([prev_tag_vec, morph_vec, curr_embed, next_embed], axis=0)
        return obs.astype(np.float32)

    def step(self, action):
        true_tag = self.target_tags[self.current_word_idx]
        is_correct = (action == true_tag)
        current_word = self.target_sentence[self.current_word_idx]
        
        reward = 0.0
        
        if is_correct:
            # --- CORRECT (+1.0) ---
            reward = 1.0
            if current_word not in nlp.vocab: 
                reward = 2.0 # OOV Bonus
        else:
            # --- INCORRECT (Base -1.0) ---
            reward -= 1.0
            
            # 1. LEXICAL CHECK (Stacking Penalty)
            if current_word in self.lexical_dict:
                if action not in self.lexical_dict[current_word]:
                    reward -= 1.0 

            # 2. GRAMMAR CHECK (Stacking Penalty)
            if self.last_action_taken is not None:
                trans_prob = self.transition_probs[self.last_action_taken][action]
                if trans_prob < 0.001:
                     reward -= 1.0 
        
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