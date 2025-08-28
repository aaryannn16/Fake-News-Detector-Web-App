# train_and_predict.py
# Baseline: TF–IDF (1–2 grams) + Logistic Regression, no extra cleaning.

import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
from joblib import dump, load


# 1) Load combined CSV
# Expected columns: at least one text-bearing field (text or title+text) and a label.
# Common schemas:
# - Kaggle train.csv: ['id','title','author','text','label']
# - Custom merges: ['title','text','label'] or ['content','label']
df = pd.read_csv("C:/Users/Aryan/Documents/Programs/Fake News Detector/News _dataset/train.csv")

print("Columns:", list(df.columns))
print("Shape:", df.shape)

# 2) Identify text and label columns
# Prefer 'content' if provided, else concat title+text, else use 'text' directly.
if "content" in df.columns:
    df["content"] = df["content"].astype(str)
elif {"title", "text"}.issubset(df.columns):
    df["content"] = (df["title"].astype(str) + ". " + df["text"].astype(str)).str.strip()
elif "text" in df.columns:
    df["content"] = df["text"].astype(str)
else:
    raise ValueError("No suitable text columns found. Provide 'content', or 'title'+'text', or 'text'.")

# Label handling:
# - If label is already 0/1, use as is.
# - If label is strings like 'FAKE'/'REAL' or 'fake'/'real', map to {FAKE:1, REAL:0}.
if df["label"].dtype == object:
    label_map = {
        "FAKE": 1, "fake": 1, "Fake": 1,
        "REAL": 0, "real": 0, "Real": 0,
        "TRUE": 0, "true": 0, "True": 0,  # sometimes 'True' is used instead of 'Real'
        "FALSE": 1, "false": 1, "False": 1
    }
    if set(df["label"].unique()) - set(label_map.keys()):
        raise ValueError(f"Unrecognized label values: {set(df['label'].unique())}")
    df["label"] = df["label"].map(label_map)

df.dropna(subset=["content", "label"], inplace=True)
df = df.drop_duplicates(subset=["content"])
df["label"] = df["label"].astype(int)

print("Label counts:\n", df["label"].value_counts())

# 3) Train/validation split (stratified)
X = df["content"].tolist()
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4) TF–IDF vectorization (no extra cleaning)
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),      # unigrams + bigrams
    max_features=10_000,     # modest cap for speed
    min_df=2,                # ignore very rare terms
    strip_accents="unicode",
    sublinear_tf=True        # log-scale term frequency
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

# 5) Train Logistic Regression
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_vec, y_train)

# 6) Evaluate
y_pred = clf.predict(X_test_vec)

print("F1:", f1_score(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification report:\n", classification_report(y_test, y_pred, digits=4))

# 7) Save artifacts for reuse (API/inference)
Path("model").mkdir(parents=True, exist_ok=True)
dump(vectorizer, "model/vectorizer.joblib")
dump(clf,        "model/fake_news_model.joblib")
print("Saved artifacts to model/")

# 8) Reusable prediction function
def predict(text: str):
    """
    Returns:
      {"label": "FAKE"|"REAL", "score": probability_of_fake (float)}
    """
    vec = load("model/vectorizer.joblib")
    mdl = load("model/fake_news_model.joblib")
    x = vec.transform([text])
    proba_fake = float(mdl.predict_proba(x)[0][1])  # class 1 = FAKE
    label = "FAKE" if proba_fake >= 0.5 else "REAL"
    return label, proba_fake

# Example quick check after training:
# print(predict("Breaking: Major policy announced today to cut taxes."))
