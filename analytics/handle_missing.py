import pandas as pd
from pathlib import Path


# Load the saved Titanic dataset
input_path = Path("analytics/data/titanic.csv")
output_path = Path("analytics/data/titanic_cleaned.csv")

df = pd.read_csv(input_path)

print("=" * 60)
print("STEP 6 — MISSING-VALUE HANDLING")
print("=" * 60)

print("\nOriginal shape:")
print(df.shape)


# ---------------------------------------------------------
# 1. Handle age: 5%–30% missing → median imputation
# ---------------------------------------------------------
age_median = df["age"].median()
df["age"] = df["age"].fillna(age_median)

print("\n1. AGE")
print("-" * 40)
print("Missing percentage: 19.87%")
print("Action: Median imputation")
print("Median used:", age_median)
print("Remaining missing:", df["age"].isnull().sum())


# ---------------------------------------------------------
# 2. Handle embarked: <5% missing → drop affected rows
# ---------------------------------------------------------
before_embarked = len(df)

df = df.dropna(subset=["embarked"])

after_embarked = len(df)

print("\n2. EMBARKED")
print("-" * 40)
print("Missing percentage: 0.22%")
print("Action: Drop rows with missing embarked")
print("Rows removed:", before_embarked - after_embarked)


# ---------------------------------------------------------
# 3. Handle embark_town: <5% missing → drop affected rows
# ---------------------------------------------------------
before_embark_town = len(df)

df = df.dropna(subset=["embark_town"])

after_embark_town = len(df)

print("\n3. EMBARK_TOWN")
print("-" * 40)
print("Missing percentage: 0.22%")
print("Action: Drop rows with missing embark_town")
print("Rows removed:", before_embark_town - after_embark_town)


# ---------------------------------------------------------
# 4. Handle deck: >30% missing → drop column
# ---------------------------------------------------------
df = df.drop(columns=["deck"])

print("\n4. DECK")
print("-" * 40)
print("Missing percentage: 77.22%")
print("Action: Drop column")
print("Reason: Very high missing percentage")


# ---------------------------------------------------------
# 5. Verify remaining missing values
# ---------------------------------------------------------
print("\n5. REMAINING MISSING VALUES")
print("-" * 40)

missing_values = df.isnull().sum()

print(missing_values)


# ---------------------------------------------------------
# 6. Save cleaned dataset
# ---------------------------------------------------------
df.to_csv(output_path, index=False)

print("\nFinal shape:")
print(df.shape)

print("\nOutput file:")
print(output_path)

print("\nFile exists:", output_path.exists())

print("\n" + "=" * 60)
print("STEP 6 COMPLETE")
print("=" * 60)