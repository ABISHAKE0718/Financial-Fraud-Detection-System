
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Load prepared data
train_df = pd.read_csv("data/processed/train_data.csv")
test_df = pd.read_csv("data/processed/test_data.csv")

X = train_df.drop("Class", axis=1)
y = train_df["Class"]

X_test = test_df.drop("Class", axis=1)
y_test = test_df["Class"]

# Create a validation set from training data
X_fit, X_val, y_fit, y_val = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Train temporary model
validation_model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

validation_model.fit(X_fit, y_fit)

# Get validation probabilities
val_prob = validation_model.predict_proba(X_val)[:, 1]

# Find threshold with at least 90% precision
candidates = []

for threshold in np.arange(0.05, 0.96, 0.01):

    predictions = (val_prob >= threshold).astype(int)

    precision = precision_score(
        y_val, predictions, zero_division=0
    )

    recall = recall_score(
        y_val, predictions, zero_division=0
    )

    f1 = f1_score(
        y_val, predictions, zero_division=0
    )

    if precision >= 0.90:
        candidates.append((threshold, precision, recall, f1))

if not candidates:
    raise ValueError(
        "No threshold achieved the required 90% precision. "
        "Review the precision target."
    )

# Select maximum recall among qualifying thresholds
best = max(candidates, key=lambda item: (item[2], item[3]))

best_threshold, val_precision, val_recall, val_f1 = best

print("Selected threshold:", best_threshold)
print("Validation precision:", val_precision)
print("Validation recall:", val_recall)
print("Validation F1:", val_f1)

# Retrain final model on complete training data
final_model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

final_model.fit(X, y)

# Evaluate once on untouched test data
test_prob = final_model.predict_proba(X_test)[:, 1]
test_predictions = (test_prob >= best_threshold).astype(int)

print("\nFINAL TEST RESULTS")
print(classification_report(
    y_test, test_predictions, zero_division=0
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, test_predictions))

# Save model and threshold
joblib.dump(final_model, "models/random_forest_tuned.pkl")
joblib.dump(best_threshold, "models/fraud_threshold.pkl")

print("\nTuned model and threshold saved successfully!")
