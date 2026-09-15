import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


# ============================================================
# STEP 11 — CLASS IMBALANCE HANDLING
# ============================================================

print("=" * 70)
print("STEP 11 — CLASS IMBALANCE HANDLING")
print("=" * 70)


# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

file_path = Path("analytics/data/titanic.csv")

df = pd.read_csv(file_path)

print("\nDataset shape:")
print(df.shape)


# ------------------------------------------------------------
# 2. SELECT FEATURES AND TARGET
# ------------------------------------------------------------

features = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "fare",
    "embarked"
]

target = "survived"

X = df[features]
y = df[target]


# ------------------------------------------------------------
# 3. CLASS BALANCE
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("CLASS BALANCE")
print("=" * 70)

class_counts = y.value_counts().sort_index()
class_percentages = y.value_counts(normalize=True).sort_index() * 100

print("\nClass counts:")
print(class_counts)

print("\nClass percentages:")
print(class_percentages.round(2))

print("\nClass meaning:")
print("0 = Not Survived")
print("1 = Survived")


# ------------------------------------------------------------
# 4. TRAIN / TEST SPLIT
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n" + "=" * 70)
print("TRAIN / TEST SPLIT")
print("=" * 70)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

print("\nTraining class distribution:")
print(y_train.value_counts(normalize=True).sort_index().round(4))

print("\nTesting class distribution:")
print(y_test.value_counts(normalize=True).sort_index().round(4))


# ------------------------------------------------------------
# 5. PREPROCESSING
# ------------------------------------------------------------

numeric_features = [
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

categorical_features = [
    "sex",
    "embarked"
]


numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# ------------------------------------------------------------
# 6. BASELINE MODEL
# ------------------------------------------------------------

baseline_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ]
)


print("\n" + "=" * 70)
print("1. BASELINE MODEL")
print("=" * 70)

baseline_model.fit(X_train, y_train)

baseline_prediction = baseline_model.predict(X_test)

baseline_accuracy = accuracy_score(
    y_test,
    baseline_prediction
)

baseline_precision = precision_score(
    y_test,
    baseline_prediction,
    zero_division=0
)

baseline_recall = recall_score(
    y_test,
    baseline_prediction,
    zero_division=0
)

baseline_f1 = f1_score(
    y_test,
    baseline_prediction,
    zero_division=0
)

print("Accuracy :", round(baseline_accuracy, 4))
print("Precision:", round(baseline_precision, 4))
print("Recall   :", round(baseline_recall, 4))
print("F1       :", round(baseline_f1, 4))


# ------------------------------------------------------------
# 7. CLASS WEIGHT BALANCED
# ------------------------------------------------------------

balanced_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42
            )
        )
    ]
)


print("\n" + "=" * 70)
print("2. CLASS WEIGHT = BALANCED")
print("=" * 70)

balanced_model.fit(X_train, y_train)

balanced_prediction = balanced_model.predict(X_test)

balanced_accuracy = accuracy_score(
    y_test,
    balanced_prediction
)

balanced_precision = precision_score(
    y_test,
    balanced_prediction,
    zero_division=0
)

balanced_recall = recall_score(
    y_test,
    balanced_prediction,
    zero_division=0
)

balanced_f1 = f1_score(
    y_test,
    balanced_prediction,
    zero_division=0
)

print("Accuracy :", round(balanced_accuracy, 4))
print("Precision:", round(balanced_precision, 4))
print("Recall   :", round(balanced_recall, 4))
print("F1       :", round(balanced_f1, 4))


# ------------------------------------------------------------
# 8. SMOTE MODEL
# ------------------------------------------------------------

smote_model = ImbPipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "smote",
            SMOTE(
                random_state=42
            )
        ),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ]
)


print("\n" + "=" * 70)
print("3. SMOTE")
print("=" * 70)

smote_model.fit(X_train, y_train)

smote_prediction = smote_model.predict(X_test)

smote_accuracy = accuracy_score(
    y_test,
    smote_prediction
)

smote_precision = precision_score(
    y_test,
    smote_prediction,
    zero_division=0
)

smote_recall = recall_score(
    y_test,
    smote_prediction,
    zero_division=0
)

smote_f1 = f1_score(
    y_test,
    smote_prediction,
    zero_division=0
)

print("Accuracy :", round(smote_accuracy, 4))
print("Precision:", round(smote_precision, 4))
print("Recall   :", round(smote_recall, 4))
print("F1       :", round(smote_f1, 4))


# ------------------------------------------------------------
# 9. CREATE COMPARISON TABLE
# ------------------------------------------------------------

comparison = pd.DataFrame(
    {
        "Method": [
            "Baseline",
            "Class Weight Balanced",
            "SMOTE"
        ],
        "Accuracy": [
            baseline_accuracy,
            balanced_accuracy,
            smote_accuracy
        ],
        "Precision": [
            baseline_precision,
            balanced_precision,
            smote_precision
        ],
        "Recall": [
            baseline_recall,
            balanced_recall,
            smote_recall
        ],
        "F1": [
            baseline_f1,
            balanced_f1,
            smote_f1
        ]
    }
)

comparison[
    [
        "Accuracy",
        "Precision",
        "Recall",
        "F1"
    ]
] = comparison[
    [
        "Accuracy",
        "Precision",
        "Recall",
        "F1"
    ]
].round(4)


print("\n" + "=" * 70)
print("IMBALANCE METHOD COMPARISON")
print("=" * 70)

print(comparison.to_string(index=False))


# ------------------------------------------------------------
# 10. SAVE COMPARISON
# ------------------------------------------------------------

output_dir = Path("analytics/output")
output_dir.mkdir(parents=True, exist_ok=True)

comparison_path = (
    output_dir / "imbalance_method_comparison.csv"
)

comparison.to_csv(
    comparison_path,
    index=False
)

print("\nComparison table saved:")
print(comparison_path)


# ------------------------------------------------------------
# 11. DETERMINE BEST METHOD BY F1
# ------------------------------------------------------------

best_row = comparison.loc[
    comparison["F1"].idxmax()
]

best_method = best_row["Method"]
best_f1 = best_row["F1"]


# ------------------------------------------------------------
# 12. CREATE CONCLUSION
# ------------------------------------------------------------

conclusion = f"""
Class imbalance was evaluated using three approaches:

1. Baseline Logistic Regression
2. Logistic Regression with class_weight="balanced"
3. Logistic Regression with SMOTE applied only to the training data

The best method based on F1-score was {best_method},
with an F1-score of {best_f1:.4f}.

SMOTE was applied only after the training/test split and
only to the training portion through the pipeline, preventing
test-data leakage.

The final method should be selected by considering the trade-off
between precision, recall and F1-score rather than accuracy alone.
"""


conclusion_path = (
    output_dir / "imbalance_conclusion.txt"
)

conclusion_path.write_text(
    conclusion.strip(),
    encoding="utf-8"
)

print("\nConclusion saved:")
print(conclusion_path)


# ------------------------------------------------------------
# 13. PLOT COMPARISON
# ------------------------------------------------------------

plot_data = comparison.set_index("Method")[
    [
        "Precision",
        "Recall",
        "F1"
    ]
]

ax = plot_data.plot(
    kind="bar",
    figsize=(10, 6)
)

ax.set_title(
    "Class Imbalance Handling Comparison"
)

ax.set_xlabel(
    "Method"
)

ax.set_ylabel(
    "Score"
)

ax.set_ylim(
    0,
    1
)

plt.xticks(
    rotation=0
)

plt.tight_layout()

plot_path = (
    output_dir / "imbalance_method_comparison.png"
)

plt.savefig(
    plot_path,
    dpi=150
)

plt.close()

print("\nComparison chart saved:")
print(plot_path)


# ------------------------------------------------------------
# COMPLETE
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 11 COMPLETE")
print("=" * 70)