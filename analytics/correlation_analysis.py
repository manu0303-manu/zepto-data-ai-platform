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
# LOAD CLEANED DATA
# =========================================================

df = pd.read_csv(input_path)


print("=" * 70)
print("STEP 8 — CORRELATION ANALYSIS")
print("=" * 70)

print("\nDataset shape:")
print(df.shape)


# =========================================================
# SELECT EXACTLY 6 REQUIRED COLUMNS
# =========================================================

correlation_columns = [
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

correlation_df = df[correlation_columns]


print("\nColumns used for correlation:")
print(correlation_columns)


# =========================================================
# CREATE CORRELATION MATRIX
# =========================================================

correlation_matrix = correlation_df.corr()


print("\n" + "=" * 70)
print("CORRELATION MATRIX")
print("=" * 70)

print(correlation_matrix.round(4))


# =========================================================
# SAVE CORRELATION MATRIX
# =========================================================

matrix_output_path = output_dir / "correlation_matrix.csv"

correlation_matrix.to_csv(matrix_output_path)

print("\nCorrelation matrix saved:")
print(matrix_output_path)


# =========================================================
# CREATE HEATMAP
# =========================================================

plt.figure(figsize=(9, 7))

plt.imshow(
    correlation_matrix,
    interpolation="nearest"
)

plt.title("Titanic Correlation Heatmap")

plt.xticks(
    range(len(correlation_columns)),
    correlation_columns,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(correlation_columns)),
    correlation_columns
)

plt.colorbar(label="Correlation")

# Add correlation values inside cells
for i in range(len(correlation_columns)):
    for j in range(len(correlation_columns)):
        value = correlation_matrix.iloc[i, j]

        plt.text(
            j,
            i,
            f"{value:.2f}",
            ha="center",
            va="center"
        )

plt.tight_layout()


heatmap_output_path = output_dir / "correlation_heatmap.png"

plt.savefig(
    heatmap_output_path,
    dpi=150
)

plt.close()


print("\nHeatmap saved:")
print(heatmap_output_path)


# =========================================================
# FIND TOP 2 ABSOLUTE OFF-DIAGONAL CORRELATIONS
# =========================================================

pairs = []

for i in range(len(correlation_columns)):
    for j in range(i + 1, len(correlation_columns)):

        column_1 = correlation_columns[i]
        column_2 = correlation_columns[j]

        correlation_value = correlation_matrix.loc[
            column_1,
            column_2
        ]

        pairs.append({
            "feature_1": column_1,
            "feature_2": column_2,
            "correlation": correlation_value,
            "absolute_correlation": abs(correlation_value)
        })


pairs_df = pd.DataFrame(pairs)

pairs_df = pairs_df.sort_values(
    by="absolute_correlation",
    ascending=False
).reset_index(drop=True)


top_2 = pairs_df.head(2)


# =========================================================
# PRINT TOP 2
# =========================================================

print("\n" + "=" * 70)
print("TOP 2 ABSOLUTE OFF-DIAGONAL CORRELATIONS")
print("=" * 70)

for index, row in top_2.iterrows():

    print(
        f"{index + 1}. "
        f"{row['feature_1']} ↔ {row['feature_2']} "
        f"= {row['correlation']:.4f}"
    )


# =========================================================
# INTERPRETATION
# =========================================================

print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)

first = top_2.iloc[0]
second = top_2.iloc[1]


def describe_correlation(value):
    absolute_value = abs(value)

    if absolute_value >= 0.70:
        strength = "strong"
    elif absolute_value >= 0.40:
        strength = "moderate"
    elif absolute_value >= 0.20:
        strength = "weak"
    else:
        strength = "very weak"

    direction = "positive" if value > 0 else "negative"

    return strength, direction


first_strength, first_direction = describe_correlation(
    first["correlation"]
)

second_strength, second_direction = describe_correlation(
    second["correlation"]
)


print(
    f"\n1. {first['feature_1']} and {first['feature_2']} "
    f"show a {first_strength} {first_direction} correlation "
    f"({first['correlation']:.4f})."
)

print(
    "This means that changes in these two variables "
    "tend to move together in the observed dataset."
)


print(
    f"\n2. {second['feature_1']} and {second['feature_2']} "
    f"show a {second_strength} {second_direction} correlation "
    f"({second['correlation']:.4f})."
)

print(
    "The relationship is measured using Pearson correlation "
    "and does not by itself imply causation."
)


# =========================================================
# SAVE TOP 2 RESULTS
# =========================================================

top_2_output_path = output_dir / "top_2_correlations.csv"

top_2.to_csv(
    top_2_output_path,
    index=False
)

print("\nTop 2 correlation results saved:")
print(top_2_output_path)


# =========================================================
# SAVE TEXT INTERPRETATION
# =========================================================

interpretation_path = output_dir / "correlation_interpretation.txt"

with open(
    interpretation_path,
    "w",
    encoding="utf-8"
) as file:

    file.write("TOP 2 ABSOLUTE OFF-DIAGONAL CORRELATIONS\n")
    file.write("=" * 60 + "\n\n")

    file.write(
        f"1. {first['feature_1']} and {first['feature_2']}: "
        f"{first['correlation']:.4f}\n"
    )

    file.write(
        f"Interpretation: {first_strength} "
        f"{first_direction} correlation.\n\n"
    )

    file.write(
        f"2. {second['feature_1']} and {second['feature_2']}: "
        f"{second['correlation']:.4f}\n"
    )

    file.write(
        f"Interpretation: {second_strength} "
        f"{second_direction} correlation.\n\n"
    )

    file.write(
        "Note: Correlation measures association and does not "
        "establish causation.\n"
    )


print("\nInterpretation saved:")
print(interpretation_path)


# =========================================================
# FINAL
# =========================================================

print("\n" + "=" * 70)
print("STEP 8 COMPLETE")
print("=" * 70)