
import pandas as pd

# Load original dataset
df = pd.read_csv("data/raw/creditcard.csv")

# Identify all rows belonging to duplicate groups
duplicate_rows = df[df.duplicated(keep=False)]

print("Total duplicate rows including original copies:")
print(len(duplicate_rows))

print("\nDuplicate records by fraud class:")
print(duplicate_rows["Class"].value_counts())

print("\nFraud distribution among duplicate rows:")
print(
    duplicate_rows["Class"]
    .value_counts(normalize=True) * 100
)

print("\nNumber of duplicate groups:")
print(df.duplicated().sum())

print("\nSample duplicate records:")
print(duplicate_rows.head(10))
