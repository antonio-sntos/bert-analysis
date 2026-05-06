# Fine-tuning BERT for IMDb Sentiment Analysis

## Problem and Dataset


Group members :

António Santos anms@itu.dk
Francisco Oliveira faol@itu.dk
Henrique Aleixo hale@itu.dk


This project studies binary sentiment analysis on the IMDb movie review dataset. The task is to classify each review as positive or negative.

We use the full labeled IMDb dataset:

- 25,000 training reviews
- 25,000 test reviews
- balanced positive/negative labels
- long-form reviews, several hundred words

The training split is divided into 22,500 training examples and 2,500 validation examples using a stratified random split.

## Methods

We compare classical models trained from scratch and BERT-based approaches:

| Model | Purpose |
| ---- | ---- |
| TF-IDF + Logistic Regression | Classical bag-of-words baseline |
| MLP | Neural baseline trained from scratch |
| LSTM | Sequential neural baseline trained from scratch |
| Frozen BERT | Pretrained BERT as fixed feature extractor |
| Partial BERT | Fine-tune only the last 2 BERT layers, pooler, and classifier |
| Fine-tuned BERT | Fine-tune all BERT layers and classifier |

The BERT models use `bert-base-uncased`, maximum sequence length 256, batch size 16, and validation-F1

## Results

Saved test-set results:

| Model | Accuracy | F1 |
| ---- | ---- | ---- | 
| BERT Fine-tuned | 0.91652 | 0.91651 |
| BERT Partial | 0.91440 | 0.91439 |
| TF-IDF + LogReg | 0.89372 | 0.89372 |
| LSTM | 0.86328 | 0.86324 |
| MLP | 0.84672 | 0.84662 |
| BERT Frozen | 0.80164 | 0.80109 |

Main observations:

- Full BERT fine-tuning performs best.
- Partial fine-tuning is very close to full fine-tuning while training far fewer parameters.
- TF-IDF + Logistic Regression is a strong classical baseline.
- Frozen BERT performs much worse than fine-tuned BERT, showing that task-specific adaptation matters.
- MLP and LSTM baselines overfit more easily because they are trained from scratch.

## Qualitative Analysis

The notebook includes qualitative error analysis for the fine-tuned BERT model. It inspects misclassified reviews and highlights likely difficult cases:

- very long reviews where 256-token truncation may hide decisive sentiment
- mixed or ambiguous sentiment
- negation-heavy reviews
- HTML/noisy formatting
- dataset label ambiguity from forcing nuanced opinions into binary classes

Example outputs are saved to results/bert_finetuned/error_analysis_examples.csv so we can analyse them beter.

## Repository Structure

notebooks/main.ipynb       Main notebook
src/model.py               BERT, LSTM, and MLP model definitions
src/tokenizer.py           BERT tokenizer wrapper and PyTorch DataLoaders
src/trainer.py             Training, evaluation, checkpointing, and plots
results/                   Saved metrics, histories, plots, and error examples
figures/                   Dataset exploration figures


## Reproducibility

Install dependencies:

```bash
pip install -r requirements.txt
```

Then run the notebook:

```text
notebooks/main.ipynb
```
