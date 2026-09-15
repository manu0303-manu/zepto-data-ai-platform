import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# =========================================================
# PATHS
# =========================================================

input_path = Path("analytics/data/titanic_cleaned.csv")
output_dir = Path("analytics/output")

output_dir.mkdir(parents=True, exist_ok=True)


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(input_path)


print("=" * 70)
print("STEP 7 — EXPLORATORY DATA ANALYSIS")
print("=" * 70)

print("\nDataset shape:")
print(df.shape)


# =========================================================
# PART 1 — AGE HISTOGRAM
# =========================================================

plt.figure(figsize=(8, 5))

plt.hist(df["age"], bins=20, edgecolor="black")

plt.title("Titanic Passenger Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")

plt.tight_layout()

age_histogram_path = output_dir / "age_histogram.png"
plt.savefig(age_histogram_path, dpi=150)
plt.close()

print("\n1. Age histogram saved:")
print(age_histogram_path)


# =========================================================
# PART 2 — FARE HISTOGRAM
# =========================================================

plt.figure(figsize=(8, 5))

plt.hist(df["fare"], bins=20, edgecolor="black")

plt.title("Titanic Passenger Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Number of Passengers")

plt.tight_layout()

fare_histogram_path = output_dir / "fare_histogram.png"
plt.savefig(fare_histogram_path, dpi=150)
plt.close()

print("\n2. Fare histogram saved:")
print(fare_histogram_path)


# =========================================================
# PART 3 — AGE BOXPLOT
# =========================================================

plt.figure(figsize=(8, 5))

plt.boxplot(df["age"], vert=True)

plt.title("Titanic Passenger Age Boxplot")
plt.ylabel("Age")

plt.tight_layout()

age_boxplot_path = output_dir / "age_boxplot.png"
plt.savefig(age_boxplot_path, dpi=150)
plt.close()

print("\n3. Age boxplot saved:")
print(age_boxplot_path)


# =========================================================
# PART 4 — FARE BOXPLOT
# =========================================================

plt.figure(figsize=(8, 5))

plt.boxplot(df["fare"], vert=True)

plt.title("Titanic Passenger Fare Boxplot")
plt.ylabel("Fare")

plt.tight_layout()

fare_boxplot_path = output_dir / "fare_boxplot.png"
plt.savefig(fare_boxplot_path, dpi=150)
plt.close()

print("\n4. Fare boxplot saved:")
print(fare_boxplot_path)


# =========================================================
# PART 5 — IQR OUTLIER ANALYSIS
# =========================================================

def calculate_iqr_outliers(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)

    outliers = series[
        (series < lower_bound) |
        (series > upper_bound)
    ]

    return q1, q3, iqr, lower_bound, upper_bound, len(outliers)


age_q1, age_q3, age_iqr, age_lower, age_upper, age_outliers = (
    calculate_iqr_outliers(df["age"])
)

fare_q1, fare_q3, fare_iqr, fare_lower, fare_upper, fare_outliers = (
    calculate_iqr_outliers(df["fare"])
)


print("\n5. IQR OUTLIER ANALYSIS")
print("-" * 50)

print("\nAGE")
print("Q1:", round(age_q1, 2))
print("Q3:", round(age_q3, 2))
print("IQR:", round(age_iqr, 2))
print("Lower bound:", round(age_lower, 2))
print("Upper bound:", round(age_upper, 2))
print("Number of outliers:", age_outliers)

print("\nFARE")
print("Q1:", round(fare_q1, 2))
print("Q3:", round(fare_q3, 2))
print("IQR:", round(fare_iqr, 2))
print("Lower bound:", round(fare_lower, 2))
print("Upper bound:", round(fare_upper, 2))
print("Number of outliers:", fare_outliers)


# =========================================================
# PART 6 — FARE STATISTICS
# =========================================================

fare_mean = df["fare"].mean()
fare_median = df["fare"].median()
fare_mode = df["fare"].mode().iloc[0]
fare_skew = df["fare"].skew()


print("\n6. FARE STATISTICS")
print("-" * 50)

print("Mean:", round(fare_mean, 4))
print("Median:", round(fare_median, 4))
print("Mode:", round(fare_mode, 4))
print("Skewness:", round(fare_skew, 4))


# =========================================================
# PART 7 — SURVIVAL RATE BY SEX
# =========================================================

print("\n7. SURVIVAL RATE BY SEX")
print("-" * 50)

female_passengers = df["sex"] == "female"
male_passengers = df["sex"] == "male"

female_survival_rate = df.loc[female_passengers, "survived"].mean() * 100
male_survival_rate = df.loc[male_passengers, "survived"].mean() * 100

print("Female survival rate:",
      round(female_survival_rate, 2), "%")

print("Male survival rate:",
      round(male_survival_rate, 2), "%")


# =========================================================
# PART 8 — SURVIVAL RATE BY PCLASS
# =========================================================

print("\n8. SURVIVAL RATE BY PCLASS")
print("-" * 50)

first_class = df["pclass"] == 1
second_class = df["pclass"] == 2
third_class = df["pclass"] == 3

first_class_survival = df.loc[first_class, "survived"].mean() * 100
second_class_survival = df.loc[second_class, "survived"].mean() * 100
third_class_survival = df.loc[third_class, "survived"].mean() * 100

print("1st class survival rate:",
      round(first_class_survival, 2), "%")

print("2nd class survival rate:",
      round(second_class_survival, 2), "%")

print("3rd class survival rate:",
      round(third_class_survival, 2), "%")


# =========================================================
# PART 9 — SURVIVAL RATE BY SEX + PCLASS
# =========================================================

print("\n9. SURVIVAL RATE BY SEX + PCLASS")
print("-" * 50)


female_first = (df["sex"] == "female") & (df["pclass"] == 1)
female_second = (df["sex"] == "female") & (df["pclass"] == 2)
female_third = (df["sex"] == "female") & (df["pclass"] == 3)

male_first = (df["sex"] == "male") & (df["pclass"] == 1)
male_second = (df["sex"] == "male") & (df["pclass"] == 2)
male_third = (df["sex"] == "male") & (df["pclass"] == 3)


female_first_rate = df.loc[female_first, "survived"].mean() * 100
female_second_rate = df.loc[female_second, "survived"].mean() * 100
female_third_rate = df.loc[female_third, "survived"].mean() * 100

male_first_rate = df.loc[male_first, "survived"].mean() * 100
male_second_rate = df.loc[male_second, "survived"].mean() * 100
male_third_rate = df.loc[male_third, "survived"].mean() * 100


print("Female + 1st class:",
      round(female_first_rate, 2), "%")

print("Female + 2nd class:",
      round(female_second_rate, 2), "%")

print("Female + 3rd class:",
      round(female_third_rate, 2), "%")

print("Male + 1st class:",
      round(male_first_rate, 2), "%")

print("Male + 2nd class:",
      round(male_second_rate, 2), "%")

print("Male + 3rd class:",
      round(male_third_rate, 2), "%")


# =========================================================
# PART 10 — SAVE SURVIVAL ANALYSIS
# =========================================================

survival_results = pd.DataFrame({
    "group": [
        "Female",
        "Male",
        "1st Class",
        "2nd Class",
        "3rd Class",
        "Female + 1st Class",
        "Female + 2nd Class",
        "Female + 3rd Class",
        "Male + 1st Class",
        "Male + 2nd Class",
        "Male + 3rd Class"
    ],
    "survival_rate_percent": [
        female_survival_rate,
        male_survival_rate,
        first_class_survival,
        second_class_survival,
        third_class_survival,
        female_first_rate,
        female_second_rate,
        female_third_rate,
        male_first_rate,
        male_second_rate,
        male_third_rate
    ]
})

survival_output_path = output_dir / "survival_analysis.csv"

survival_results.to_csv(
    survival_output_path,
    index=False
)

print("\n10. Survival analysis saved:")
print(survival_output_path)


# =========================================================
# FINAL
# =========================================================

print("\n" + "=" * 70)
print("STEP 7 COMPLETE")
print("=" * 70)