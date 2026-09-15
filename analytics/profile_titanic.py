import pandas as pd
from pathlib import Path


# Load the saved Titanic dataset
file_path = Path("analytics/data/titanic.csv")
df = pd.read_csv(file_path)


print("=" * 60)
print("STEP 5 — TITANIC DATA PROFILING")
print("=" * 60)


# 1. Dataset shape
print("\n1. DATASET SHAPE")
print("-" * 40)
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])
print("Shape:", df.shape)


# 2. Dataset information
print("\n2. DATASET INFO")
print("-" * 40)
df.info()


# 3. Descriptive statistics
print("\n3. DESCRIPTIVE STATISTICS")
print("-" * 40)
print(df.describe())


# 4. Missing-value counts
print("\n4. MISSING-VALUE COUNTS")
print("-" * 40)
missing_counts = df.isnull().sum()
print(missing_counts)


# 5. Missing-value percentages
print("\n5. MISSING-VALUE PERCENTAGES")
print("-" * 40)

missing_percentage = (df.isnull().sum() / len(df)) * 100

missing_table = pd.DataFrame({
    "missing_count": missing_counts,
    "missing_percentage": missing_percentage.round(2)
})

print(missing_table)


print("\n" + "=" * 60)
print("STEP 5 COMPLETE")
print("=" * 60)