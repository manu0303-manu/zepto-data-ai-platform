import pandas as pd
import os


# Input file
input_file = "data_pipeline/data/cleaned_books.csv"

# Output file
output_file = "data_pipeline/data/cleaned_books.csv"


# Fixed exchange rate required by the assignment
GBP_TO_INR = 105.50


# Load cleaned data
df = pd.read_csv(input_file)

print("Cleaned data loaded successfully")
print("Rows:", len(df))


# Make sure price_gbp is numeric
df["price_gbp"] = pd.to_numeric(
    df["price_gbp"],
    errors="coerce"
)


# Calculate INR price using the fixed rate
df["price_inr"] = (
    df["price_gbp"] * GBP_TO_INR
)


# Round INR price to 2 decimal places
df["price_inr"] = df["price_inr"].round(2)


# Display exchange rate
print("\nFixed exchange rate:")
print("1 GBP =", GBP_TO_INR, "INR")


# Display first 5 prices
print("\nFirst 5 price conversions:")

print(
    df[
        ["price_gbp", "price_inr"]
    ].head()
)


# Check missing INR prices
missing_inr = df["price_inr"].isna().sum()

print("\nMissing price_inr values:", missing_inr)


# Save updated cleaned data
os.makedirs(
    "data_pipeline/data",
    exist_ok=True
)

df.to_csv(
    output_file,
    index=False
)


print("\n==============================")
print("GBP TO INR CONVERSION COMPLETE")
print("==============================")

print("Rows:", len(df))
print("Exchange rate: 1 GBP = 105.50 INR")
print("Output file:", output_file)