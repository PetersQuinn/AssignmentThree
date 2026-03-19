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

# I load the same training data as the baseline model so the comparison is fair.
with open(DATA_DIR / "train_core_vs_neg.json", "r", encoding="utf-8") as f:
    train_data = json.load(f)

# I load the same test data as the baseline model.
with open(DATA_DIR / "test_core_vs_neg.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)

# I separate the text and labels for training.
X_train_texts = [t for (t, y) in train_data]
y_train = [y for (t, y) in train_data]

# I separate the text and labels for testing.
X_test_texts = [t for (t, y) in test_data]
y_test = [y for (t, y) in test_data]

# I keep the TF-IDF settings identical to the baseline so this is a controlled comparison.
vectorizer = TfidfVectorizer(
    lowercase=True,
    min_df=5,
    max_df=0.9
)

# I fit the vectorizer on the training set.
X_train = vectorizer.fit_transform(X_train_texts)

# I transform the test set using the same fitted vectorizer.
X_test = vectorizer.transform(X_test_texts)

# I train logistic regression with L1 regularization.
# I use liblinear because it supports the L1 penalty for this binary classification task.
clf = LogisticRegression(
    penalty="l1",
    solver="liblinear",
    max_iter=2000
)

clf.fit(X_train, y_train)

# I generate test-set predictions.
y_pred = clf.predict(X_test)

# I generate predicted probabilities for the positive class for ROC AUC.
y_prob = clf.predict_proba(X_test)[:, 1]

# I print the confusion matrix.
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:")
print(cm)

# I print the classification report.
print("\nClassification report:")
print(classification_report(y_test, y_pred))

# I compute ROC AUC.
auc = roc_auc_score(y_test, y_prob)
print("ROC AUC:", round(auc, 3))

# I count non-zero coefficients to measure how sparse the L1 model is.
coefs = clf.coef_[0]
nonzero_count = np.count_nonzero(coefs)
print("Non-zero coefficients:", nonzero_count)

# I get the vocabulary terms so I can inspect the learned weights.
feature_names = vectorizer.get_feature_names_out()

# I sort coefficients from smallest to largest.
sorted_idx = np.argsort(coefs)

# I print the most negative-weight words.
print("\nTop 15 negative-weight words (predictive of NEG):")
for idx in sorted_idx[:15]:
    print(f"{feature_names[idx]}: {coefs[idx]:.4f}")

# I print the most positive-weight words.
print("\nTop 15 positive-weight words (predictive of CORE):")
for idx in sorted_idx[-15:][::-1]:
    print(f"{feature_names[idx]}: {coefs[idx]:.4f}")