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
print("STEP 9 — MULTIVARIATE ANALYSIS")
print("=" * 70)

print("\nDataset shape:")
print(df.shape)


# =========================================================
# CHART 1
# SEX + PCLASS + SURVIVAL
# =========================================================

survival_by_sex_class = (
    df.groupby(["sex", "pclass"])["survived"]
    .mean()
    .reset_index()
)

survival_by_sex_class["survival_rate"] = (
    survival_by_sex_class["survived"] * 100
)


plt.figure(figsize=(9, 6))

for sex in ["female", "male"]:
    subset = survival_by_sex_class[
        survival_by_sex_class["sex"] == sex
    ]

    plt.plot(
        subset["pclass"],
        subset["survival_rate"],
        marker="o",
        label=sex
    )

plt.title("Survival Rate by Sex and Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate (%)")
plt.xticks([1, 2, 3])
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

chart1_path = output_dir / "multivariate_sex_pclass_survival.png"
plt.savefig(chart1_path, dpi=150)
plt.close()

print("\nChart 1 saved:")
print(chart1_path)


# =========================================================
# CHART 2
# PCLASS + AGE + SURVIVAL
# =========================================================

plt.figure(figsize=(9, 6))

for survival_status in [0, 1]:

    subset = df[df["survived"] == survival_status]

    plt.scatter(
        subset["pclass"],
        subset["age"],
        alpha=0.5,
        label="Survived" if survival_status == 1 else "Did Not Survive"
    )

plt.title("Age, Passenger Class and Survival")
plt.xlabel("Passenger Class")
plt.ylabel("Age")
plt.xticks([1, 2, 3])
plt.legend()

plt.tight_layout()

chart2_path = output_dir / "multivariate_pclass_age_survival.png"
plt.savefig(chart2_path, dpi=150)
plt.close()

print("\nChart 2 saved:")
print(chart2_path)


# =========================================================
# CHART 3
# SEX + FARE
# =========================================================

plt.figure(figsize=(9, 6))

female_fares = df.loc[df["sex"] == "female", "fare"]
male_fares = df.loc[df["sex"] == "male", "fare"]

plt.boxplot(
    [female_fares, male_fares],
    tick_labels=["Female", "Male"]
)

plt.title("Fare Distribution by Sex")
plt.xlabel("Sex")
plt.ylabel("Fare")

plt.tight_layout()

chart3_path = output_dir / "multivariate_sex_fare.png"
plt.savefig(chart3_path, dpi=150)
plt.close()

print("\nChart 3 saved:")
print(chart3_path)


# =========================================================
# CHART 4
# PCLASS + FARE + SURVIVAL
# =========================================================

plt.figure(figsize=(9, 6))

survived_data = df[df["survived"] == 1]
not_survived_data = df[df["survived"] == 0]

plt.scatter(
    survived_data["pclass"],
    survived_data["fare"],
    alpha=0.5,
    label="Survived"
)

plt.scatter(
    not_survived_data["pclass"],
    not_survived_data["fare"],
    alpha=0.5,
    label="Did Not Survive"
)

plt.title("Passenger Class, Fare and Survival")
plt.xlabel("Passenger Class")
plt.ylabel("Fare")
plt.xticks([1, 2, 3])
plt.legend()

plt.tight_layout()

chart4_path = output_dir / "multivariate_pclass_fare_survival.png"
plt.savefig(chart4_path, dpi=150)
plt.close()

print("\nChart 4 saved:")
print(chart4_path)


# =========================================================
# INTERPRETATIONS
# =========================================================

interpretation_path = (
    output_dir / "multivariate_interpretations.txt"
)


with open(
    interpretation_path,
    "w",
    encoding="utf-8"
) as file:

    file.write("MULTIVARIATE ANALYSIS INTERPRETATIONS\n")
    file.write("=" * 70 + "\n\n")

    file.write("CHART 1 — SEX + PCLASS + SURVIVAL\n")
    file.write("-" * 50 + "\n")
    file.write(
        "Survival rates vary substantially by both sex and passenger "
        "class. Female passengers generally show higher survival "
        "rates than male passengers across passenger classes. "
        "The combination of sex and class provides useful information "
        "for understanding survival differences.\n\n"
    )

    file.write("CHART 2 — PCLASS + AGE + SURVIVAL\n")
    file.write("-" * 50 + "\n")
    file.write(
        "The scatter plot shows how passenger age is distributed "
        "across the three passenger classes while separating "
        "survivors from non-survivors. Younger passengers appear "
        "across all classes, while higher classes contain a broader "
        "range of ages. Survival patterns can be examined together "
        "with age and passenger class.\n\n"
    )

    file.write("CHART 3 — SEX + FARE\n")
    file.write("-" * 50 + "\n")
    file.write(
        "Fare distributions differ between female and male passengers. "
        "The boxplot shows the median and spread of fares for each sex "
        "and highlights potential high-fare observations. This indicates "
        "that fare levels were not distributed equally between the two groups.\n\n"
    )

    file.write("CHART 4 — PCLASS + FARE + SURVIVAL\n")
    file.write("-" * 50 + "\n")
    file.write(
        "Passenger class and fare are related because higher classes "
        "generally had higher ticket prices. The chart separates "
        "survivors from non-survivors, allowing survival patterns to "
        "be examined together with class and fare. Higher fares and "
        "higher passenger classes show different survival patterns "
        "in the dataset.\n\n"
    )

    file.write(
        "Note: These visualizations show associations in the Titanic "
        "dataset and do not establish causal relationships.\n"
    )


print("\nInterpretations saved:")
print(interpretation_path)


# =========================================================
# FINAL
# =========================================================

print("\n" + "=" * 70)
print("STEP 9 COMPLETE")
print("=" * 70)