"""Lightweight training helpers used by the notebooks.

Trainer wraps a model, dataloaders and a simple training loop. The
implementation focuses on clarity rather than performance.
"""

import torch
import torch.nn as nn
from tqdm import tqdm
import os
import json
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class Trainer:
    """Simple trainer: train loop, eval, and a tiny checkpoint helper.

    The class keeps training history and saves light artifacts (plots,
    history JSON). It is intentionally small so it's easy to read in
    the notebooks.
    """

    def __init__(self, model, train_loader, val_loader, test_loader,
                 device, learning_rate=2e-5, output_dir='results', save_artifacts=True):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.output_dir = output_dir
        self.save_artifacts = save_artifacts

        if self.save_artifacts:
            os.makedirs(output_dir, exist_ok=True)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.history = {'train_loss': [], 'val_loss': [],
                       'train_acc': [], 'val_acc': []}

    def train_epoch(self):
        self.model.train()
        losses = []
        all_preds = []
        all_labels = []

        for batch in tqdm(self.train_loader, desc='Training'):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(input_ids, attention_mask)
            loss = self.criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            losses.append(loss.item())
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        avg_loss = sum(losses) / len(losses)
        accuracy = accuracy_score(all_labels, all_preds)
        return avg_loss, accuracy

    def evaluate(self, dataloader):
        self.model.eval()
        losses = []
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs, labels)
                losses.append(loss.item())

                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = sum(losses) / len(losses)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        return avg_loss, accuracy, f1, all_preds, all_labels

    def train(self, epochs, save_best=False):
        best_val_acc = 0

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc, val_f1, _, _ = self.evaluate(self.val_loader)

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")

            if save_best and val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_checkpoint(f"{self.output_dir}/best_model.pt")

        if self.save_artifacts:
            self.save_history()
            self.plot_training_curves()

    def save_checkpoint(self, path):
        if not self.save_artifacts:
            return
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
        print(f"Saved checkpoint to {path}")

    def save_history(self):
        if not self.save_artifacts:
            return
        with open(f"{self.output_dir}/training_history.json", 'w') as f:
            json.dump(self.history, f)

    def plot_training_curves(self):
        if not self.save_artifacts:
            return
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(self.history['train_loss'], label='Train')
        axes[0].plot(self.history['val_loss'], label='Val')
        axes[0].set_title('Loss'); axes[0].legend()
        axes[1].plot(self.history['train_acc'], label='Train')
        axes[1].plot(self.history['val_acc'], label='Val')
        axes[1].set_title('Accuracy'); axes[1].legend()
        plt.savefig(f"{self.output_dir}/training_curves.png")
        plt.close()

    def evaluate_test(self, class_names=['Negative', 'Positive'], plot_confusion_matrix=True):
        _, accuracy, f1, preds, labels = self.evaluate(self.test_loader)
        print(f"\nTest Accuracy: {accuracy:.4f}")
        print(f"Test F1 Score: {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(labels, preds, target_names=class_names))

        if plot_confusion_matrix:
            cm = confusion_matrix(labels, preds)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names)
            plt.title('Confusion Matrix')
            plt.ylabel('True')
            plt.xlabel('Predicted')
            if self.save_artifacts:
                plt.savefig(f"{self.output_dir}/confusion_matrix.png")
                plt.close()
            else:
                plt.show()

        return {'accuracy': accuracy, 'f1': f1, 'predictions': preds, 'labels': labels}
