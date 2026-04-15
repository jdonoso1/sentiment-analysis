# Sentiment Analysis on Movie Reviews

Predict whether a movie review is **positive** or **negative** using classical NLP techniques.

**Dataset:** IMDB 50k movie reviews (25k train / 25k test, perfectly balanced)

---

## What this project covers

- Text preprocessing (lowercase, HTML stripping, non-alpha removal)
- Feature extraction: Bag of Words and TF-IDF (unigrams + bigrams)
- Models: Logistic Regression and Naive Bayes
- Evaluation: accuracy, precision, recall, F1, confusion matrices
- Error analysis: what kinds of reviews get misclassified

---

## Results

| Model | Accuracy | F1 |
|---|---|---|
| Logistic Regression (BoW) | ~0.87 | ~0.87 |
| Naive Bayes (BoW) | ~0.83 | ~0.83 |
| Logistic Regression (TF-IDF) | ~0.90 | ~0.90 |
| Naive Bayes (TF-IDF) | ~0.86 | ~0.86 |

**Winner: Logistic Regression + TF-IDF.** TF-IDF consistently beats raw counts because it down-weights common words. Naive Bayes is faster but less accurate here.

---

## Run it

```bash
pip install -r requirements.txt
python sentiment_analysis.py
```

All outputs (plots, metrics CSV, confusion matrices) are saved to `results/`.

---

## Project structure

```
sentiment-analysis/
├── sentiment_analysis.py   # full pipeline
├── data_notes.md           # dataset observations
├── preprocessing_notes.md  # cleaning decisions
├── requirements.txt
└── results/
    ├── eda.png
    ├── confusion_matrices.png
    ├── model_comparison.png
    └── metrics.csv
```
