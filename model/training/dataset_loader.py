import json
import random
from datasets import Dataset, DatasetDict
from transformers import MarianTokenizer

class BantuTranslationDataset:
    def __init__(self, tokenizer_path: str, data_path: str, src_lang: str, tgt_lang: str, seed: int = 42):
        self.tokenizer = MarianTokenizer.from_pretrained(tokenizer_path)
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seed = seed
        self.data = self.load_data(data_path)
        self.dataset = self.split_dataset(self.data)

    def load_data(self, file_path):
        """Loads the dataset from JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    def split_dataset(self, data):
        """Splits dataset into 80% train, 10% validation, 10% test."""
        random.seed(self.seed)
        random.shuffle(data)

        total_samples = len(data)
        train_size = int(0.8 * total_samples)
        val_size = int(0.1 * total_samples)

        train_data = data[:train_size]
        val_data = data[train_size:train_size + val_size]
        test_data = data[train_size + val_size:]

        return DatasetDict({
            "train": Dataset.from_dict({"src_text": [d[self.src_lang] for d in train_data], 
                                        "tgt_text": [d[self.tgt_lang] for d in train_data]}),
            "validation": Dataset.from_dict({"src_text": [d[self.src_lang] for d in val_data], 
                                             "tgt_text": [d[self.tgt_lang] for d in val_data]}),
            "test": Dataset.from_dict({"src_text": [d[self.src_lang] for d in test_data], 
                                       "tgt_text": [d[self.tgt_lang] for d in test_data]})
        })

    def preprocess_function(self, examples):
        """Tokenizes the text inputs."""
        model_inputs = self.tokenizer(examples["src_text"], max_length=128, truncation=True, padding="max_length")
        labels = self.tokenizer(examples["tgt_text"], max_length=128, truncation=True, padding="max_length")
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def get_tokenized_datasets(self):
        """Tokenizes train, validation, and test datasets."""
        tokenized_datasets = self.dataset.map(self.preprocess_function, batched=True)
        return tokenized_datasets["train"], tokenized_datasets["validation"], tokenized_datasets["test"]
