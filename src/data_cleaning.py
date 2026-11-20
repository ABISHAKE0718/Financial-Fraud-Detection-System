
import pandas as pd
from pathlib import Path


def clean_dataset(df):

    print("========== DATA CLEANING ==========")

    original_shape = df.shape

    print("\nOriginal dataset shape:", original_shape)

    # Check missing values
    print("\nMissing values:", df.isnull().sum().sum())

    # Count duplicate rows that will be removed
    duplicate_count = df.duplicated().sum()

    print("Duplicate rows to remove:", duplicate_count)

    # Count duplicates removed by fraud class
    duplicate_classes = df.loc[
        df.duplicated(keep="first"), "Class"
    ].value_counts()

    print("\nDuplicate rows removed by class:")
    print(duplicate_classes)

    # Remove exact duplicate rows
    cleaned_df = df.drop_duplicates().copy()

    # Compare dataset sizes
    print("\nCleaned dataset shape:", cleaned_df.shape)

    print(
        "\nRows removed:",
        original_shape[0] - cleaned_df.shape[0]
    )

    # Fraud distribution after cleaning
    print("\nFraud distribution after cleaning:")
    print(cleaned_df["Class"].value_counts())

    return cleaned_df


if __name__ == "__main__":

    input_path = Path("data/raw/creditcard.csv")
    output_path = Path("data/processed/creditcard_cleaned.csv")

    df = pd.read_csv(input_path)

    cleaned_df = clean_dataset(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cleaned_df.to_csv(output_path, index=False)

    print("\nCleaned dataset saved successfully!")
    print("Location:", output_path)
