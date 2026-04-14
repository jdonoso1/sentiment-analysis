# Sentiment Analysis on Movie Reviews
# Goal: predict whether a review is positive (1) or negative (0)
# Dataset: IMDB 50k movie reviews (HuggingFace datasets)

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
)
from datasets import load_dataset

os.makedirs("results", exist_ok=True)


# ─────────────────────────────────────────────
# Step 1: Load Dataset
# ─────────────────────────────────────────────
# IMDB 50k movie reviews. label: 0 = negative, 1 = positive

print("Loading IMDB dataset...")
dataset = load_dataset("imdb")

train_df = pd.DataFrame(dataset["train"])  # 25,000 reviews
test_df  = pd.DataFrame(dataset["test"])   # 25,000 reviews

print(f"Train: {len(train_df)} | Test: {len(test_df)}")


# ─────────────────────────────────────────────
# Step 2: Explore the Data
# ─────────────────────────────────────────────

print("\n--- Class Balance ---")
print(train_df["label"].value_counts())
print("0 = negative  |  1 = positive")

train_df["length"] = train_df["text"].apply(lambda x: len(x.split()))
print(f"\nAvg review length : {train_df['length'].mean():.0f} words")
print(f"Min / Max          : {train_df['length'].min()} / {train_df['length'].max()}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

train_df["label"].value_counts().plot(kind="bar", ax=axes[0], color=["#e74c3c", "#2ecc71"])
axes[0].set_title("Class Balance (Train)")
axes[0].set_xticklabels(["Negative (0)", "Positive (1)"], rotation=0)
axes[0].set_ylabel("Count")

train_df["length"].plot(kind="hist", bins=50, ax=axes[1], color="#3498db", edgecolor="white")
axes[1].set_title("Review Length Distribution")
axes[1].set_xlabel("Words per review")

plt.tight_layout()
plt.savefig("results/eda.png", dpi=150)
print("Saved → results/eda.png")


# ─────────────────────────────────────────────
# Step 3: Clean the Text
# ─────────────────────────────────────────────
# IMDB reviews contain HTML tags, so we strip those first

def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)        # strip HTML (e.g. <br/>)
    text = re.sub(r"[^a-z\s]", "", text)     # keep only letters
    text = re.sub(r"\s+", " ", text).strip()
    return text


train_df["clean"] = train_df["text"].apply(clean_text)
test_df["clean"]  = test_df["text"].apply(clean_text)

print("\nSample cleaned review:")
print(train_df["clean"].iloc[0][:200], "...")


# ─────────────────────────────────────────────
# Step 4: Convert Text into Features
# ─────────────────────────────────────────────

X_train = train_df["clean"].values
X_test  = test_df["clean"].values
y_train = train_df["label"].values
y_test  = test_df["label"].values

# Bag of Words — simple word counts
bow = CountVectorizer(max_features=20000)
X_train_bow = bow.fit_transform(X_train)
X_test_bow  = bow.transform(X_test)

# TF-IDF with unigrams + bigrams
tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)

print(f"\nBoW matrix   : {X_train_bow.shape}")
print(f"TF-IDF matrix: {X_train_tfidf.shape}")


# ─────────────────────────────────────────────
# Step 5: Train Baseline Models
# ─────────────────────────────────────────────

models = {
    "Logistic Regression (BoW)":    (LogisticRegression(max_iter=1000), X_train_bow,   X_test_bow),
    "Naive Bayes (BoW)":            (MultinomialNB(),                    X_train_bow,   X_test_bow),
    "Logistic Regression (TF-IDF)": (LogisticRegression(max_iter=1000), X_train_tfidf, X_test_tfidf),
    "Naive Bayes (TF-IDF)":         (MultinomialNB(),                    X_train_tfidf, X_test_tfidf),
}

results = {}

for name, (model, X_tr, X_te) in models.items():
    print(f"\nTraining: {name}")
    model.fit(X_tr, y_train)
    preds = model.predict(X_te)
    results[name] = {
        "accuracy":  accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall":    recall_score(y_test, preds),
        "f1":        f1_score(y_test, preds),
        "preds":     preds,
    }
    print(f"  Accuracy : {results[name]['accuracy']:.4f}")
    print(f"  F1 Score : {results[name]['f1']:.4f}")


# ─────────────────────────────────────────────
# Step 6: Evaluate — Confusion Matrices
# ─────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for ax, (name, res) in zip(axes.flatten(), results.items()):
    cm = confusion_matrix(y_test, res["preds"])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Negative", "Positive"])
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(name)

plt.tight_layout()
plt.savefig("results/confusion_matrices.png", dpi=150)
print("\nSaved → results/confusion_matrices.png")


# ─────────────────────────────────────────────
# Step 7: Compare Results
# ─────────────────────────────────────────────

summary = pd.DataFrame(
    {name: {k: v for k, v in res.items() if k != "preds"} for name, res in results.items()}
).T.round(4)

print("\n--- Results Summary ---")
print(summary.to_string())

summary.to_csv("results/metrics.csv")
print("Saved → results/metrics.csv")

# bar chart
summary[["accuracy", "precision", "recall", "f1"]].plot(
    kind="bar", figsize=(12, 5), ylim=(0.80, 0.95), rot=20
)
plt.title("Model Comparison — Sentiment Analysis")
plt.ylabel("Score")
plt.tight_layout()
plt.savefig("results/model_comparison.png", dpi=150)
print("Saved → results/model_comparison.png")

# pick best model for error analysis (LR + TF-IDF typically wins)
best_name  = max(results, key=lambda k: results[k]["f1"])
best_preds = results[best_name]["preds"]
print(f"\nBest model: {best_name}")


# ─────────────────────────────────────────────
# Step 8: Error Analysis — What Did We Get Wrong?
# ─────────────────────────────────────────────
# Understanding misclassifications reveals where the model is confused

wrong_idx = np.where(best_preds != y_test)[0]
print(f"\nMisclassified: {len(wrong_idx)} / {len(y_test)}  "
      f"({100 * len(wrong_idx) / len(y_test):.1f}%)")

print("\nSample misclassified reviews:")
for i in wrong_idx[:3]:
    true_lbl = "Positive" if y_test[i] == 1 else "Negative"
    pred_lbl = "Positive" if best_preds[i] == 1 else "Negative"
    snippet  = test_df["text"].iloc[i][:180].replace("\n", " ")
    print(f"\n  True: {true_lbl} → Predicted: {pred_lbl}")
    print(f"  \"{snippet}...\"")

print("\nDone. All outputs saved in results/")
