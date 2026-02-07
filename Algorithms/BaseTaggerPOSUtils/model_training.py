import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
)


class PosDataset(Dataset):
    def __init__(self, data, tokenizer, tag_list):
        self.data = data
        self.tokenizer = tokenizer
        self.tag_list = tag_list
        self.tag_to_id = {tag: i for i, tag in enumerate(tag_list)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence, tags = self.data[idx]
        tokens = self.tokenizer(
            sentence,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt",
        )
        labels = [-100] * len(tokens["input_ids"][0])  # -100 for ignored in loss
        word_ids = tokens.word_ids()
        for i, word_id in enumerate(word_ids):
            if word_id is not None:
                # Ensure word_id is within bounds of tags list
                if word_id < len(tags):
                    labels[i] = self.tag_to_id.get(tags[word_id], -100)
                else:
                    # This can happen if tokenizer splits words differently than expected
                    # or if there's a mismatch in data. We ignore this token.
                    labels[i] = -100
        tokens["labels"] = torch.tensor(labels)
        return {k: v.squeeze() for k, v in tokens.items()}


def train_pos_tagger(
    training_data,
    tag_list,
    model_name="distilbert-base-uncased",
    output_dir="./pos_tagger_model",
    epochs=3,
    batch_size=16,
):
    """
    Trains a POS tagger model and saves it to the specified directory.
    """
    print(f"Loading tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=len(tag_list)
    )

    print("Preparing dataset...")
    train_dataset = PosDataset(training_data, tokenizer, tag_list)

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        save_steps=500,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Training complete and model saved.")
