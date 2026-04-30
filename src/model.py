"""Model definitions used in the project.

This file contains a small BERT classification wrapper plus two
baseline models (LSTM and an MLP). The implementations are kept simple
for teaching and comparison purposes.
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertForSequenceClassification


class BERTSentimentClassifier(nn.Module):
    """BERT encoder plus a small linear head for classification."""

    def __init__(self, n_classes=2, model_name='bert-base-uncased',
                 dropout=0.1, freeze_bert=False):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)

    def unfreeze_layers(self, n_layers):
        """Unfreeze last n transformer layers."""
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        for i, layer in enumerate(self.bert.encoder.layer):
            if i >= len(self.bert.encoder.layer) - n_layers:
                for param in layer.parameters():
                    param.requires_grad = True
            else:
                for param in layer.parameters():
                    param.requires_grad = False


class BaselineLSTM(nn.Module):
    """A small bidirectional LSTM baseline using trainable embeddings."""

    def __init__(self, vocab_size, embed_dim=300, hidden_dim=256,
                 n_classes=2, n_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, n_layers,
                           batch_first=True, dropout=dropout, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, n_classes)

    def forward(self, input_ids, attention_mask=None):
        embedded = self.embedding(input_ids)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        hidden = self.dropout(hidden)
        return self.fc(hidden)


class BaselineMLP(nn.Module):
    """Simple mean-pooled embedding + MLP baseline.

    Useful as a cheap comparison against BERT fine-tuning.
    """

    def __init__(self, vocab_size, embed_dim=300, hidden_dim=256,
                 n_classes=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, n_classes)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask=None):
        embedded = self.embedding(input_ids)
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            embedded = embedded * mask_expanded
            summed = torch.sum(embedded, dim=1)
            mean_pooled = summed / (torch.sum(mask_expanded, dim=1) + 1e-10)
        else:
            mean_pooled = torch.mean(embedded, dim=1)

        x = self.fc1(mean_pooled)
        x = self.relu(x)
        x = self.dropout(x)
        return self.fc2(x)
