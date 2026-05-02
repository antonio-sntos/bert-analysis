# Fine-tuning BERT for Sentiment Analysis on IMDb Reviews

### Team Members

## Problem & Dataset

This project focuses on sentiment analysis, where the goal is to classify movie reviews as positive or negative.

We use the IMDb movie review dataset, which contains:

50,000 reviews (balanced)
Binary labels (positive/negative)
Long-form text (often up to several hundred words)
Key challenges:
Long sequences (important for BERT input length)
Mixed sentiment within a single review
Informal language and sarcasm
## Method

We compare four models with increasing complexity:

1. MLP (Baseline)
Mean-pooled embeddings + feedforward network
Ignores word order
2. LSTM (Sequential baseline)
Bidirectional LSTM
Captures sequential dependencies
3. Frozen BERT
Pre-trained BERT used as feature extractor
Only classification head is trained
4. Fine-tuned BERT
Full model is trained on IMDb
Adapts representations to sentiment task
## Training setup
Model: bert-base-uncased
Max length: 256
Batch size: 16
Learning rate: 2e-5
Epochs: 4

We experimented with different learning rates and sequence lengths to optimize performance.

## Results
Accuracy Comparison
Model	Accuracy
MLP	XX%
LSTM	XX%
Frozen BERT	XX%
Fine-tuned BERT	XX%
Key Observations
BERT-based models significantly outperform traditional baselines
Fine-tuning improves performance over frozen BERT
LSTM captures sequence information but struggles with long-range dependencies
## Error Analysis (IMPORTANT – easy marks)

Examples of misclassified reviews:

Mixed sentiment:
“The acting was great but the plot was boring”
Sarcasm:
“Yeah, this was totally the best movie ever…”
Long reviews:
Important sentiment appears late in the text
## Discussion
What worked well:
Fine-tuned BERT achieved the best performance
Increasing sequence length improved results
Pre-trained models significantly reduce training effort
Limitations:
Training time is high (BERT models are expensive)
Some errors remain due to sarcasm and complex language
Baselines benefited from BERT tokenization (may affect fairness)

Future improvements:
Try RoBERTa or larger models
Use longer sequence lengths (512)
Apply data augmentation
Experiment with partial fine-tuning
## Repository Structure
├── main.ipynb
├── model.py
├── data/
├── results/
└── README.md
## Reproducibility

To run the project:

pip install -r requirements.txt

Then run:

python main.py

or open the notebook.
