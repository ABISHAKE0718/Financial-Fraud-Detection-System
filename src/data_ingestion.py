
import pandas as pd
from pathlib import Path


def load_dataset(file_path):
    """Load a CSV dataset and return a DataFrame."""

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if path.suffix.lower() != ".csv":
        raise ValueError("Only CSV files are supported.")

    df = pd.read_csv(path)

    print("Dataset loaded successfully!")
    print("Dataset shape:", df.shape)
    print("Column names:", df.columns.tolist())

    return df


if __name__ == "__main__":
    df = load_dataset("data/raw/creditcard.csv")

    print("\nFirst five records:")
    print(df.head())

    print("\nMissing values:", df.isnull().sum().sum())
