import pandas as pd
import joblib
from pathlib import Path

# Load test dataset
test_df = pd.read_csv("data/processed/test_data.csv")

X_test = test_df.drop("Class", axis=1)
y_test = test_df["Class"]

# Load tuned model and threshold
model = joblib.load("models/random_forest_tuned.pkl")
threshold = joblib.load("models/fraud_threshold.pkl")

# Predict fraud probabilities
fraud_probabilities = model.predict_proba(X_test)[:, 1]

# Apply selected threshold
predictions = (fraud_probabilities >= threshold).astype(int)

# Create risk categories
def assign_risk(probability):
    if probability >= threshold:
        return "High"
    elif probability >= 0.10:
        return "Medium"
    else:
        return "Low"

# Create output DataFrame
results = pd.DataFrame({
    "Transaction_ID": [
        f"TEST_{i:06d}" for i in range(1, len(test_df) + 1)
    ],
    "Actual_Class": y_test.values,
    "Fraud_Probability": fraud_probabilities,
    "Predicted_Class": predictions,
    "Risk_Level": [
        assign_risk(probability)
        for probability in fraud_probabilities
    ]
})

# Save predictions
output_path = Path("outputs/transaction_predictions.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)

results.to_csv(output_path, index=False)

# Display results
print("========== PREDICTION EXPORT ==========")
print("\nTotal transactions:", len(results))

print("\nPredicted transaction classes:")
print(results["Predicted_Class"].value_counts())

print("\nRisk level distribution:")
print(results["Risk_Level"].value_counts())

print("\nFirst 10 predictions:")
print(results.head(10))

print("\nPrediction file saved successfully!")
print("Location:", output_path)