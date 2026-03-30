<div align="center">

# Tweet Sentiment Analysis

### Multinomial Naive Bayes · Bag-of-Words · Built from Scratch

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-Sentiment140-blue?style=for-the-badge)](https://www.kaggle.com/datasets/kazanova/sentiment140)

<br/>

| Test Accuracy | AUC-ROC | Training Samples | Vocabulary Size |
|:---:|:---:|:---:|:---:|
| **75.30%** | **0.8255** | **168,000** | **15,000** |

</div>

---

## Overview

Binary sentiment classification on the [Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140) dataset — 1.6 million real tweets labeled as positive or negative. Every component of the pipeline is implemented **from scratch**: text preprocessing, vocabulary construction, Bag-of-Words vectorization, Naive Bayes training, and inference. No high-level ML abstractions; just NumPy, math, and first principles.

---

## Pipeline

```
Raw Tweets (1.6M)
      │
      ▼
 Data Balancing ──── 120k per class → 240k total
      │
      ▼
 Train / Val / Test Split ──── 70% / 15% / 15%
      │
      ▼
 TweetPreprocessor
   ├── Lowercase
   ├── Remove URLs, mentions, punctuation
   ├── Stopword removal (NLTK)
   └── Porter Stemming
      │
      ▼
 VocabularyBuilder ──── min_freq=5 · max_features=15,000
      │
      ▼
 BagOfWordsVectorizer ──── (n_samples × 15,000) count matrix
      │
      ▼
 MultinomialNaiveBayes
   ├── Class priors: P(class)
   ├── Word likelihoods: P(word | class)
   ├── Laplace smoothing (α = 1.0)
   └── Log-probability inference
      │
      ▼
 Evaluation ──── Accuracy · AUC-ROC · Classification Report
```

---

## Dataset

The [Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140) dataset contains 1.6 million tweets automatically labeled via emoticons. For this project, a balanced subset of **240,000 tweets** (120k per class) is used for efficient training and evaluation.

| Split | Size | Negative | Positive |
|---|---|---|---|
| Train | 168,000 | 84,000 | 84,000 |
| Validation | 36,000 | 18,000 | 18,000 |
| Test | 36,000 | 18,000 | 18,000 |

---

## Implementation Details

### `TweetPreprocessor`
Handles all text normalization before vectorization:
- Strips URLs, `@mentions`, hashtag symbols, punctuation, and numbers
- Removes NLTK English stopwords
- Applies **Porter Stemming** — chosen over lemmatization for speed at scale

### `VocabularyBuilder`
Constructs the token index from training data only (no leakage):
- Words appearing fewer than **5 times** are discarded
- Top **15,000** words by frequency are retained

### `BagOfWordsVectorizer`
Converts tokenized tweets into a sparse word-count matrix of shape `(n_samples, 15000)`. The same vocabulary is applied to all splits.

### `MultinomialNaiveBayes`
A complete from-scratch probabilistic classifier:
- Computes **class priors** P(class) from label frequencies
- Computes **conditional word probabilities** P(word | class) from count matrices
- Applies **Laplace (add-1) smoothing** to handle words unseen during training
- Uses **log-probabilities** to avoid floating-point underflow
- Exposes both `predict()` and `predict_proba()` for hard and soft outputs

> **scikit-learn is used exclusively for evaluation utilities** (`accuracy_score`, `roc_auc_score`, `classification_report`) and `train_test_split`. All model logic is original.

---

## Results

### Test Set Performance

| Metric | Score |
|---|---|
| Accuracy | **75.30%** |
| AUC-ROC | **0.8255** |

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Negative (0) | 0.75 | 0.76 | 0.75 | 18,000 |
| Positive (1) | 0.75 | 0.75 | 0.75 | 18,000 |
| **Macro Avg** | **0.75** | **0.75** | **0.75** | **36,000** |

The model achieves balanced precision and recall across both classes, with an AUC-ROC of **0.83** — indicating strong discriminative ability well beyond random chance.

---

## Inference

A `predict_new_tweet()` utility is included for single-tweet classification:

```python
predict_new_tweet("I absolutely love this product! Best purchase ever!")
# Sentiment: Positive (94.3% confident)
# Probabilities: Negative=0.057, Positive=0.943

predict_new_tweet("This is terrible. Waste of money.")
# Sentiment: Negative
```

---

## Getting Started

**1. Clone the repository**
```bash
git clone https://github.com/sobanali256/Tweet-Sentiment-Analysis.git
cd Tweet-Sentiment-Analysis
```

**2. Install dependencies**
```bash
pip install numpy pandas nltk scikit-learn
```

**3. Download the dataset**

Download [`training.1600000.processed.noemoticon.csv`](https://www.kaggle.com/datasets/kazanova/sentiment140) from Kaggle and update the `data_path` variable in the notebook.

**4. Run the notebook**

Open `Tweet Analysis.ipynb` in Jupyter or Google Colab and run all cells top to bottom.

---

## Tech Stack

| | Tool |
|---|---|
| Language | Python 3 |
| Core Computation | NumPy, Pandas |
| NLP Utilities | NLTK |
| Evaluation | Scikit-learn |
| Environment | Google Colab / Jupyter Notebook |

---

## Key Concepts

`Bayes' Theorem` &nbsp;·&nbsp; `Laplace Smoothing` &nbsp;·&nbsp; `Log-Probability Inference` &nbsp;·&nbsp; `Bag-of-Words` &nbsp;·&nbsp; `Vocabulary Frequency Filtering` &nbsp;·&nbsp; `Stratified Splitting` &nbsp;·&nbsp; `AUC-ROC`

---

## License

Distributed under the [MIT License](LICENSE). See `LICENSE` for details.
