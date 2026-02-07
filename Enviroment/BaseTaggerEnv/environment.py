import sys
import os

# Add the project root to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import torch
from typing import Optional, List, Tuple
from Algorithms.BaseTaggerPOSUtils.model_inference import PosTaggerInference
from Algorithms.BaseTaggerPOSUtils.dataset import get_tag_list


class PosCorrectionEnv(gym.Env):
    """
    A Custom Gymnasium Environment for POS Tagging Correction.

    Architecture:
    - Base Tagger: DistilBERT (Fine-tuned)
    - Tagset: Universal Dependencies (UPOS)
    - Action Space: KEEP, SHIFT, DICT
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        dataset: List[Tuple[str, List[str]]],
        model_path: str = "./pos_tagger_model",
        mode: str = "sequential",  # "sequential", "random", or "fixed"
    ):
        """
        Args:
            dataset: List of tuples (text_sentence, list_of_gold_tags)
            model_path: Path to the saved fine-tuned model
            mode: How to select sentences on reset:
                  - "sequential": Iterate through sentences in order (wraps around)
                  - "random": Randomly sample sentences (original behavior)
                  - "fixed": Always use the same sentence (index 0, or set via options)
        """
        self.dataset = dataset
        self.mode = mode

        # Resolve model path to absolute path
        if not os.path.isabs(model_path):
            # Try to resolve relative to project root
            project_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "../../")
            )
            model_path = os.path.join(project_root, model_path)

        # Load the fine-tuned model
        self.tag_list = get_tag_list()
        self.inference = PosTaggerInference(model_path, self.tag_list)

        # --- Define Action Space ---
        # 0: KEEP (Trust Base Tagger)
        # 1: SHIFT (Switch to 2nd probability)
        # 2: DICT (External Dictionary Lookup - Simulated Oracle)
        self.action_space = spaces.Discrete(3)

        # --- Define Observation Space ---
        # 1. 'embedding': 768-dim vector (DistilBERT hidden state)
        # 2. 'features': 4-dim vector [Confidence, IsUpper, Suffix_S, HasHyphen]
        # 3. 'base_tag_idx': The integer index of the tag proposed by the model

        self.tag2idx = {tag: i for i, tag in enumerate(self.tag_list)}

        self.observation_space = spaces.Dict(
            {
                "embedding": spaces.Box(
                    low=-1.0, high=1.0, shape=(768,), dtype=np.float32
                ),
                "features": spaces.Box(
                    low=0.0, high=1.0, shape=(10,), dtype=np.float32
                ),
                "base_tag_idx": spaces.Discrete(len(self.tag_list)),
                "prev_tag_idx": spaces.Discrete(
                    len(self.tag_list) + 1
                ),  # +1 for START token
            }
        )

        # Internal State
        self._current_doc_idx = 0
        self._current_token_idx = 0
        self._tokens = []
        self._gold_tags = []
        self._prev_final_tag_idx = len(self.tag_list)  # START token index
        self._model_outputs = (
            None  # Store logits/hidden states for the current sentence
        )

    def _get_obs(self):
        """Helper to construct the observation from current state."""
        current_token_data = self._model_outputs[self._current_token_idx]

        # 1. Embedding (DistilBERT hidden state)
        emb = current_token_data["embedding"]
        # Normalize embedding to help RL agent convergence
        emb = torch.nn.functional.normalize(emb, p=2, dim=0)

        # 2. Base Tagger Prediction
        base_tag = current_token_data["predicted_tag"]
        base_tag_idx = self.tag2idx.get(base_tag, 0)

        # 3. Features & Confidence
        probs = current_token_data["probs"]
        token_text = current_token_data["token"]

        # Get top-2 probabilities for confidence gap
        top2_probs, top2_indices = torch.topk(probs, k=min(2, len(probs)))
        confidence = top2_probs[0].item()
        confidence_gap = (
            (top2_probs[0] - top2_probs[1]).item() if len(top2_probs) > 1 else 1.0
        )

        # Position features
        is_first = 1.0 if self._current_token_idx == 0 else 0.0
        is_last = (
            1.0 if self._current_token_idx == len(self._model_outputs) - 1 else 0.0
        )
        position_ratio = self._current_token_idx / max(1, len(self._model_outputs) - 1)

        # Morphological features
        lower_token = token_text.lower()

        features = np.array(
            [
                confidence,  # 0: Model confidence
                confidence_gap,  # 1: Gap between top-2 predictions
                1.0 if token_text[0].isupper() else 0.0,  # 2: Starts with uppercase
                1.0 if token_text.isupper() else 0.0,  # 3: All uppercase
                1.0 if lower_token.endswith(("ing", "ed")) else 0.0,  # 4: Verb suffix
                1.0 if lower_token.endswith("ly") else 0.0,  # 5: Adverb suffix
                1.0 if lower_token.endswith("s") else 0.0,  # 6: Plural/verb suffix
                (
                    1.0 if any(c.isdigit() for c in token_text) else 0.0
                ),  # 7: Contains digit
                is_first,  # 8: Is first token
                position_ratio,  # 9: Relative position in sentence
            ],
            dtype=np.float32,
        )

        return {
            "embedding": emb.numpy().astype(np.float32),
            "features": features,
            "base_tag_idx": base_tag_idx,
            "prev_tag_idx": self._prev_final_tag_idx,
        }

    def _get_info(self):
        """Helper to return debug info."""
        current_token_data = self._model_outputs[self._current_token_idx]
        return {
            "word": current_token_data["token"],
            "base_tag": current_token_data["predicted_tag"],
            "gold_tag": self._gold_tags[self._current_token_idx],
            "probs": current_token_data["probs"],
            "sentence_idx": self._current_doc_idx,
            "total_sentences": len(self.dataset),
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Resets the environment to a new sentence.

        Args:
            seed: Random seed for reproducibility
            options: Optional dict with:
                - "sentence_idx": Force a specific sentence index (overrides mode)
        """
        super().reset(seed=seed)

        # Determine which sentence to use based on mode
        if options and "sentence_idx" in options:
            # Override: use specific sentence
            self._current_doc_idx = options["sentence_idx"] % len(self.dataset)
        elif self.mode == "sequential":
            # Sequential: move to next sentence (wrap around)
            self._current_doc_idx = (self._current_doc_idx + 1) % len(self.dataset)
        elif self.mode == "fixed":
            # Fixed: always use sentence 0 (or keep current if set via options before)
            pass  # Keep _current_doc_idx as is (initialized to 0)
        else:
            # Random: original behavior
            self._current_doc_idx = self.np_random.integers(0, len(self.dataset))

        text, gold_tags = self.dataset[self._current_doc_idx]

        # Run Model Inference on the whole sentence
        # Use the extended method from PosTaggerInference
        details = self.inference.get_full_details(text)

        tokens = details["tokens"]
        hidden_states = details["embeddings"]
        predicted_tags = details["predicted_tags"]
        probs = details["probs"]
        word_ids = details["word_ids"]

        self._model_outputs = []
        self._gold_tags = []  # We need to align gold tags too!

        # Alignment Logic (Simplified: Skip special tokens, assume gold tags align with words)
        # Note: This is tricky because BERT uses subwords.
        # We will map subwords to the corresponding gold tag of the word they belong to.

        for i, word_idx in enumerate(word_ids):
            if word_idx is None:
                continue  # Skip special tokens

            if word_idx < len(gold_tags):
                gold_tag = gold_tags[word_idx]

                self._model_outputs.append(
                    {
                        "token": tokens[i],
                        "embedding": hidden_states[i],
                        "predicted_tag": predicted_tags[i],
                        "probs": probs[i],
                        "word_idx": word_idx,
                    }
                )
                self._gold_tags.append(gold_tag)

        self._current_token_idx = 0
        self._prev_final_tag_idx = len(self.tag_list)  # START token index

        # Handle empty sentences or alignment failures
        if not self._model_outputs:
            return self.reset(seed=seed)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        """
        Executes the logic:
        Action 0 (KEEP): Accept Model tag
        Action 1 (SHIFT): Switch to 2nd highest probability tag
        Action 2 (DICT): Look up Gold Tag (Costly)
        """
        current_data = self._model_outputs[self._current_token_idx]
        base_tag = current_data["predicted_tag"]
        gold_tag = self._gold_tags[self._current_token_idx]
        probs = current_data["probs"]

        final_tag = base_tag
        cost = 0.0

        # --- Action Logic ---
        if action == 0:  # KEEP
            final_tag = base_tag
            cost = 0.0

        elif action == 1:  # SHIFT
            # Switch to the 2nd most probable tag
            top2_probs, top2_indices = torch.topk(probs, k=2)
            # If the top prediction is the base tag, take the second one
            # Otherwise (shouldn't happen if base_tag is argmax), take the second one anyway
            second_best_idx = top2_indices[1].item()
            final_tag = self.tag_list[second_best_idx]
            cost = -0.1  # Small penalty for shifting

        elif action == 2:  # DICT
            # Simulate looking up in Wiktionary (guarantees correctness)
            # INCREASED COST: Makes DICT a "break glass in emergency" option
            cost = -0.9
            final_tag = gold_tag

        # --- Reward Calculation ---
        reward = 0.0
        is_correct = final_tag == gold_tag
        base_was_correct = base_tag == gold_tag

        if is_correct:
            reward = 1.0
        else:
            # Critical Penalty: Changing a correct tag to an incorrect one
            if base_was_correct and not is_correct:
                # REDUCED PENALTY: Harsh, but not learning-destroying
                reward = -2.0
            else:
                reward = -1.0  # Simple failure

        # --- DICT-Specific Rewards/Penalties ---
        # Encourage intelligent DICT usage: reward when fixing errors, penalize unnecessary use
        if action == 2:  # DICT was chosen
            if base_was_correct:
                # PENALTY: DICT was unnecessary (KEEP would have been correct)
                # This is wasteful oracle usage
                reward -= 0.5  # Additional penalty for unnecessary DICT
            else:
                # REWARD: DICT fixed an error (KEEP would have been wrong)
                # This is smart oracle usage
                reward += 0.5  # Bonus reward for fixing errors with DICT

        total_reward = reward + cost

        # --- Update Previous Tag ---
        final_tag_idx = self.tag2idx.get(final_tag, 0)
        self._prev_final_tag_idx = final_tag_idx

        # --- Advance State ---
        self._current_token_idx += 1
        terminated = self._current_token_idx >= len(self._model_outputs)
        truncated = False

        # Get next observation (if not terminated)
        if not terminated:
            observation = self._get_obs()
        else:
            # Return dummy observation at terminal state
            observation = {
                "embedding": np.zeros(768, dtype=np.float32),
                "features": np.zeros(10, dtype=np.float32),
                "base_tag_idx": 0,
                "prev_tag_idx": self._prev_final_tag_idx,
            }

        # Build info - note: include gold_tag for the token we just acted on (not the next one)
        if not terminated:
            info = self._get_info() | {
                "final_tag": final_tag,
                "acted_gold_tag": gold_tag,  # Gold tag for the token we just predicted
                "acted_base_tag": base_tag,  # Base tag for the token we just predicted
            }
        else:
            info = {
                "final_status": "Sentence Complete",
                "final_tag": final_tag,
                "acted_gold_tag": gold_tag,
                "acted_base_tag": base_tag,
            }

        return observation, total_reward, terminated, truncated, info

    def render(self):
        """Simple console render."""
        if self._current_token_idx < len(self._model_outputs):
            token_data = self._model_outputs[self._current_token_idx]
            print(
                f"Token: {token_data['token']:<15} | Pred: {token_data['predicted_tag']:<5} | Gold: {self._gold_tags[self._current_token_idx]}"
            )
        else:
            print("END")
