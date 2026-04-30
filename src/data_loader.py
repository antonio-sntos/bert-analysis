"""data_loader.py
Small funtions to load the IMDb dataset and prepare PyTorch DataLoaders.
"""

import numpy as np
import pandas as pd
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter


class IMDbDataset:

    def __init__(self):
        """Load the IMDb dataset from HuggingFace."""
        self.dataset = load_dataset("imdb")
        self.train_data = self.dataset['train']
        self.test_data = self.dataset['test']

    def get_info(self):
        """Print basic info about the dataset."""
        print("IMDb Dataset Info")

        print(f"Number of training examples: {len(self.train_data)}")
        print(f"Number of test examples: {len(self.test_data)}")

        train_labels = self.train_data['label']
        test_labels = self.test_data['label']

        print(f"\nTraining set class distribution:")
        print(Counter(train_labels))
        print(f"\nTest set class distribution:")
        print(Counter(test_labels))

    def get_sample_texts(self, n=3):
        """Print a few sample reviews from the training set."""
        print(f"Sample Reviews (n={n})")

        for i in range(n):
            text = self.train_data['text'][i]
            label = self.train_data['label'][i]
            sentiment = "Positive" if label == 1 else "Negative"
            print(f"\nReview {i+1} ({sentiment}):\n{text[:500]}...")  # Print first 500 chars

    def analyze_text_length(self):
        """Analyze text length distribution."""
        print("Text Length Analysis")

        train_lengths = [len(text.split()) for text in self.train_data['text']]

        print(f"Average words per review: {np.mean(train_lengths):.1f}")
        print(f"Median words per review: {np.median(train_lengths):.1f}")
        print(f"Min words: {np.min(train_lengths)}")
        print(f"Max words: {np.max(train_lengths)}")
        print(f"95th percentile: {np.percentile(train_lengths, 95):.0f}")

        return train_lengths

    def plot_class_distribution(self, save_path='figures/class_distribution.png'):
        """Plot class distribution."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Training set
        train_labels = self.train_data['label']
        train_counts = [train_labels.count(0), train_labels.count(1)]
        axes[0].bar(['Negative', 'Positive'], train_counts, color=['red', 'green'], alpha=0.7)
        axes[0].set_title('Training Set Distribution')
        axes[0].set_ylabel('Count')

        # Test set
        test_labels = self.test_data['label']
        test_counts = [test_labels.count(0), test_labels.count(1)]
        axes[1].bar(['Negative', 'Positive'], test_counts, color=['red', 'green'], alpha=0.7)
        axes[1].set_title('Test Set Distribution')
        axes[1].set_ylabel('Count')

        plt.tight_layout()
        plt.savefig(save_path)
        print(f"\nClass distribution plot saved")
        plt.close()

    def plot_text_length_distribution(self, save_path='figures/text_length_distribution.png'):
        """Plot text length distribution."""
        train_lengths = [len(text.split()) for text in self.train_data['text']]

        plt.figure(figsize=(10, 6))
        plt.hist(train_lengths, bins=50, alpha=0.7, edgecolor='black')
        plt.axvline(np.percentile(train_lengths, 95), color='r', linestyle='--',
                   label=f'95th percentile: {np.percentile(train_lengths, 95):.0f}')
        plt.axvline(np.median(train_lengths), color='g', linestyle='--',
                   label=f'Median: {np.median(train_lengths):.0f}')
        plt.xlabel('Number of Words')
        plt.ylabel('Frequency')
        plt.title('Distribution of Review Lengths (Training Set)')
        plt.legend()
        plt.xlim(0, 1000)
        plt.savefig(save_path)
        print(f"Text length distribution plot saved to {save_path}")
        plt.close()


class BERTDataset(Dataset):
    """PyTorch Dataset for BERT fine-tuning."""

    def __init__(self, texts, labels, tokenizer, max_length=512):
        """
        Args:
            texts: List of text strings
            labels: List of labels (0 or 1)
            tokenizer: HuggingFace tokenizer
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
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


def create_data_loaders(train_data, test_data, tokenizer, batch_size=16,
                       max_length=512, val_split=0.1):
    """Return train/val/test DataLoaders.

    train_data and test_data are HuggingFace Dataset splits. The function
    splits off a small validation set from the training split and wraps
    the texts with `BERTDataset` so they are ready for PyTorch.
    """
    # Split training data into train and validation
    train_size = int((1 - val_split) * len(train_data))
    val_size = len(train_data) - train_size

    # Create datasets
    train_texts = train_data['text'][:train_size]
    train_labels = train_data['label'][:train_size]
    val_texts = train_data['text'][train_size:]
    val_labels = train_data['label'][train_size:]

    train_dataset = BERTDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = BERTDataset(val_texts, val_labels, tokenizer, max_length)
    test_dataset = BERTDataset(test_data['text'], test_data['label'], tokenizer, max_length)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"\nDataLoaders created:")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Example usage
    imdb = IMDbDataset()
    imdb.get_info()
    imdb.get_sample_texts()
    imdb.analyze_text_length()

    # Create plots
    imdb.plot_class_distribution()
    imdb.plot_text_length_distribution()
