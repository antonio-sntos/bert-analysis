"""tokenizer.py
Small wrapper around the HuggingFace BERT tokenizer used in the notebooks.

It keeps a few helper methods for quick analysis and a small PyTorch
Dataset class used to build DataLoaders.
"""

from transformers import BertTokenizer
import torch
from torch.utils.data import Dataset


class BERTTokenizer:

    def __init__(self, model_name='bert-base-uncased'):

        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.vocab_size = len(self.tokenizer)
        print(f"Vocabulary size: {self.vocab_size}")

    def tokenize(self, text):
        """Tokenize text without special tokens."""
        return self.tokenizer.tokenize(text)

    def encode(self, text, max_length=512, padding='max_length', truncation=True):
        """Encode a single string into input IDs and attention mask.

        Returns a dict with `input_ids` and `attention_mask` as PyTorch tensors.
        """
        return self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_attention_mask=True,
            return_tensors='pt'
        )

    def decode(self, token_ids, skip_special_tokens=True):
        """Decode token IDs back to text."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def get_special_tokens(self):
        """Return special token IDs."""
        return {
            'pad_token': self.tokenizer.pad_token,
            'pad_token_id': self.tokenizer.pad_token_id,
            'cls_token': self.tokenizer.cls_token,
            'cls_token_id': self.tokenizer.cls_token_id,
            'sep_token': self.tokenizer.sep_token,
            'sep_token_id': self.tokenizer.sep_token_id,
            'unk_token': self.tokenizer.unk_token,
            'unk_token_id': self.tokenizer.unk_token_id,
        }

    def analyze_token_length(self, texts, max_samples=1000):
        """
        Analyze token lengths for a set of texts.

        Args:
            texts: List of text strings
            max_samples: Maximum number of texts to analyze

        Returns:
            Dictionary with statistics
        """
        import numpy as np

        if len(texts) > max_samples:
            import random
            texts = random.sample(texts, max_samples)

        token_lengths = []
        for text in texts:
            tokens = self.tokenize(text)
            token_lengths.append(len(tokens))

        return {
            'mean': np.mean(token_lengths),
            'median': np.median(token_lengths),
            'min': np.min(token_lengths),
            'max': np.max(token_lengths),
            '95th_percentile': np.percentile(token_lengths, 95),
            '99th_percentile': np.percentile(token_lengths, 99),
            'lengths': token_lengths
        }


class SentimentDataset(Dataset):
    """PyTorch Dataset for sentiment analysis with BERT."""

    def __init__(self, texts, labels, tokenizer, max_length=512):
        """
        Args:
            texts: List of text strings
            labels: List of integer labels
            tokenizer: BERTTokenizer instance
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer.encode(text, max_length=self.max_length)

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


def create_dataloaders(train_texts, train_labels, val_texts, val_labels,
                      test_texts, test_labels, tokenizer,
                      batch_size=16, max_length=512, num_workers=0):
    """
    Create PyTorch DataLoaders for train/val/test splits.

    Args:
        train_texts: Training texts
        train_labels: Training labels
        val_texts: Validation texts
        val_labels: Validation labels
        test_texts: Test texts
        test_labels: Test labels
        tokenizer: BERTTokenizer instance
        batch_size: Batch size
        max_length: Maximum sequence length
        num_workers: Number of data loading workers

    Returns:
        train_loader, val_loader, test_loader
    """
    from torch.utils.data import DataLoader

    # Create datasets
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = SentimentDataset(val_texts, val_labels, tokenizer, max_length)
    test_dataset = SentimentDataset(test_texts, test_labels, tokenizer, max_length)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"DataLoaders created:")
    print(f"  Training samples: {len(train_dataset)} ({len(train_loader)} batches)")
    print(f"  Validation samples: {len(val_dataset)} ({len(val_loader)} batches)")
    print(f"  Test samples: {len(test_dataset)} ({len(test_loader)} batches)")
    print(f"  Batch size: {batch_size}")
    print(f"  Max sequence length: {max_length}")

    return train_loader, val_loader, test_loader
