import seaborn as sns
from pathlib import Path


# Step 3: Load Titanic dataset exactly once
df = sns.load_dataset("titanic")


# Step 4: Save the dataset immediately as CSV
output_path = Path("analytics/data/titanic.csv")
df.to_csv(output_path, index=False)


# Verification
print("=" * 50)
print("TITANIC DATASET SAVED SUCCESSFULLY")
print("=" * 50)

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])
print("CSV file:", output_path)
print("File exists:", output_path.exists())