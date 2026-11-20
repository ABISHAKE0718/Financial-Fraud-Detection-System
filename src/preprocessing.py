
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Load cleaned dataset
df = pd.read_csv("data/processed/creditcard_cleaned.csv")

print("Dataset loaded:", df.shape)

# Separate features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)

# Scale Time and Amount
scaler = StandardScaler()

X_train = X_train.copy()
X_test = X_test.copy()

X_train[["Time", "Amount"]] = scaler.fit_transform(
    X_train[["Time", "Amount"]]
)

X_test[["Time", "Amount"]] = scaler.transform(
    X_test[["Time", "Amount"]]
)

# Save preprocessing outputs
output_dir = Path("data/processed")
model_dir = Path("models")

output_dir.mkdir(parents=True, exist_ok=True)
model_dir.mkdir(parents=True, exist_ok=True)

train_data = X_train.copy()
train_data["Class"] = y_train

test_data = X_test.copy()
test_data["Class"] = y_test

train_data.to_csv(output_dir / "train_data.csv", index=False)
test_data.to_csv(output_dir / "test_data.csv", index=False)

joblib.dump(scaler, model_dir / "scaler.pkl")

print("\nTraining fraud distribution:")
print(y_train.value_counts())

print("\nTesting fraud distribution:")
print(y_test.value_counts())

print("\nPreprocessing completed successfully!")
