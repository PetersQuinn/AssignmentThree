from pathlib import Path
import json
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score
)

DATA_DIR = Path("data")

# I load the training data that was already created in the Week 09 pipeline.
with open(DATA_DIR / "train_core_vs_neg.json", "r", encoding="utf-8") as f:
    train_data = json.load(f)

# I load the test data so I can evaluate the model on the same held-out set.
with open(DATA_DIR / "test_core_vs_neg.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)

# I separate the text and labels for the training set.
X_train_texts = [t for (t, y) in train_data]
y_train = [y for (t, y) in train_data]

# I separate the text and labels for the test set.
X_test_texts = [t for (t, y) in test_data]
y_test = [y for (t, y) in test_data]

# I use the exact same TF-IDF settings from the tutorial so this stays apples-to-apples.
vectorizer = TfidfVectorizer(
    lowercase=True,
    min_df=5,
    max_df=0.9
)

# I fit the TF-IDF vectorizer on the training texts and transform them into features.
X_train = vectorizer.fit_transform(X_train_texts)

# I transform the test texts using the already-fitted vectorizer.
X_test = vectorizer.transform(X_test_texts)

# I train the baseline logistic regression model with the default L2 regularization.
clf = LogisticRegression(
    max_iter=1000,
    n_jobs=1
)

clf.fit(X_train, y_train)

# I generate hard class predictions for the test set.
y_pred = clf.predict(X_test)

# I generate predicted probabilities for the positive class so I can compute ROC AUC.
y_prob = clf.predict_proba(X_test)[:, 1]

# I print the confusion matrix to see correct and incorrect predictions by class.
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:")
print(cm)

# I print the classification report to inspect precision, recall, and F1-score.
print("\nClassification report:")
print(classification_report(y_test, y_pred))

# I compute ROC AUC to measure ranking quality across all thresholds.
auc = roc_auc_score(y_test, y_prob)
print("ROC AUC:", round(auc, 3))

# I count how many coefficients are non-zero as a sparsity diagnostic.
coefs = clf.coef_[0]
nonzero_count = np.count_nonzero(coefs)
print("Non-zero coefficients:", nonzero_count)

# I grab the vocabulary terms so I can inspect which words got the biggest weights.
feature_names = vectorizer.get_feature_names_out()

# I sort coefficients from smallest to largest.
sorted_idx = np.argsort(coefs)

# I print the 15 most negative words, which are most predictive of NEG = 0.
print("\nTop 15 negative-weight words (predictive of NEG):")
for idx in sorted_idx[:15]:
    print(f"{feature_names[idx]}: {coefs[idx]:.4f}")

# I print the 15 most positive words, which are most predictive of CORE = 1.
print("\nTop 15 positive-weight words (predictive of CORE):")
for idx in sorted_idx[-15:][::-1]:
    print(f"{feature_names[idx]}: {coefs[idx]:.4f}")