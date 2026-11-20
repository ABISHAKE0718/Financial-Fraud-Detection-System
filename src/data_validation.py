
import pandas as pd
from pathlib import Path


def validate_dataset(df):

    print("\n========== DATA QUALITY REPORT ==========")

    # Dataset dimensions
    print("\n1. Dataset Shape:")
    print(df.shape)

    # Missing values
    print("\n2. Missing Values:")
    print(df.isnull().sum().sum())

    # Duplicate records
    print("\n3. Duplicate Records:")
    print(df.duplicated().sum())

    # Data types
    print("\n4. Data Types:")
    print(df.dtypes)

    # Fraud distribution
    print("\n5. Fraud Distribution:")

    if "Class" in df.columns:
        print(df["Class"].value_counts())
    else:
        print("Fraud target column not found.")

    print("\n========== VALIDATION COMPLETE ==========")


if __name__ == "__main__":

    file_path = Path("data/raw/creditcard.csv")

    df = pd.read_csv(file_path)

    validate_dataset(df)
