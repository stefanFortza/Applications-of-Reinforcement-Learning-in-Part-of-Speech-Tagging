import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification


class PosTaggerInference:
    def __init__(self, model_path, tag_list):
        """
        Initialize the inference class with a saved model and a list of tags.

        Args:
            model_path (str): Path to the directory containing the saved model and tokenizer.
            tag_list (list): List of tag names corresponding to the model's labels.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path).to(
            self.device
        )
        self.tag_list = tag_list

    def predict(self, text):
        """
        Predict POS tags for a given text.

        Args:
            text (str): The input sentence or word.

        Returns:
            list: A list of tuples (token, tag).
        """
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits

        predictions = torch.argmax(logits, dim=-1)
        predicted_tags = [self.tag_list[p.item()] for p in predictions[0]]
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        # Filter out special tokens and align
        result = []
        for token, tag in zip(tokens, predicted_tags):
            if token not in ["[CLS]", "[SEP]", "[PAD]"]:
                result.append((token, tag))

        return result

    def get_full_details(self, text):
        """
        Get full details including embeddings, logits, and probabilities for a given text.

        Args:
            text (str): The input sentence.

        Returns:
            dict: A dictionary containing tokens, embeddings, predicted_tags, probs, and word_ids.
        """
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            logits = outputs.logits
            # Last hidden state is usually the embedding we want (batch, seq_len, hidden_dim)
            hidden_states = outputs.hidden_states[-1]

        # Process outputs
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        predictions = torch.argmax(logits, dim=-1)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        word_ids = inputs.word_ids()

        return {
            "tokens": tokens,
            "embeddings": hidden_states[0].cpu(),
            "predicted_tags": [self.tag_list[p.item()] for p in predictions[0]],
            "probs": probs[0].cpu(),
            "word_ids": word_ids,
        }
