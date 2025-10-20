# SWEBERT: Software Industry Text Classification

SWEBERT is a specialized text classifier designed to categorize software industry related article summaries into predefined technical categories. It leverages a fine-tuned variant of ModernBERT to provide accurate classification of technical content.

## Overview

SWEBERT is fine-tuned from the [ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base) model and specialized for classifying software engineering article summaries into categories such as:

_WIP LIST_
- networking

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/SWEBERT.git
cd SWEBERT

# Install dependencies
uv sync

uv run main.py
```

## Usage

### Classification Example

```python
from transformers import pipeline

# Load the classifier
classifier = pipeline("text-classification", model="./SWEBERT")

# Classify a software-related article
article = ("A SQL query is used to fetch data from a relational database.")

result = classifier(article)

print(f"Prediction: {result[0]['label']} (Score: {result[0]['score']:.4f})")
```

## Dataset (WIP)

The model is trained on a curated dataset of software engineering article summaries categorized into technical domains. The training data follows this format:

```
text,label
"A SQL query is used to fetch data from a relational database.",database
"Network latency and bandwidth are key performance metrics.",networking
"The model was trained using a support vector machine algorithm.",machine-learning
```

## Model Architecture

SWEBERT is based on the ModernBERT transformer architecture with:
- A sequence classification head for multi-class prediction
- Fine-tuning on software engineering specific content
- Optimized for technical text understanding

## Performance

TBD

## License

[Add your license information here]
