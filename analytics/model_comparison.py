import pandas as pd
from pathlib import Path


# ============================================================
# STEP 14 — MODEL COMPARISON
# ============================================================

print("=" * 70)
print("STEP 14 — MODEL COMPARISON")
print("=" * 70)


# ------------------------------------------------------------
# 1. DEFINE OUTPUT DIRECTORY
# ------------------------------------------------------------

output_dir = Path("analytics/output")

output_dir.mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------------------------
# 2. LOAD CLASSIFICATION RESULTS
# ------------------------------------------------------------

classification_path = (
    output_dir /
    "classification_model_comparison.csv"
)

classification_df = pd.read_csv(
    classification_path
)

print("\n" + "=" * 70)
print("CLASSIFICATION RESULTS")
print("=" * 70)

print(
    classification_df.to_string(
        index=False
    )
)


# ------------------------------------------------------------
# 3. LOAD IMBALANCE RESULTS
# ------------------------------------------------------------

imbalance_path = (
    output_dir /
    "imbalance_method_comparison.csv"
)

imbalance_df = pd.read_csv(
    imbalance_path
)

print("\n" + "=" * 70)
print("IMBALANCE HANDLING RESULTS")
print("=" * 70)

print(
    imbalance_df.to_string(
        index=False
    )
)


# ------------------------------------------------------------
# 4. LOAD RANDOM FOREST GRIDSEARCH RESULTS
# ------------------------------------------------------------

rf_metrics_path = (
    output_dir /
    "random_forest_gridsearch_metrics.csv"
)

rf_metrics_df = pd.read_csv(
    rf_metrics_path
)

rf_metrics = dict(
    zip(
        rf_metrics_df["Metric"],
        rf_metrics_df["Value"]
    )
)

print("\n" + "=" * 70)
print("BEST RANDOM FOREST GRIDSEARCH RESULTS")
print("=" * 70)

print(
    rf_metrics_df.to_string(
        index=False
    )
)


# ------------------------------------------------------------
# 5. LOAD REGRESSION RESULTS
# ------------------------------------------------------------

regression_path = (
    output_dir /
    "fare_regression_metrics.csv"
)

regression_df = pd.read_csv(
    regression_path
)

regression_metrics = dict(
    zip(
        regression_df["Metric"],
        regression_df["Value"]
    )
)

print("\n" + "=" * 70)
print("REGRESSION RESULTS")
print("=" * 70)

print(
    regression_df.to_string(
        index=False
    )
)


# ------------------------------------------------------------
# 6. PREPARE CLASSIFICATION TABLE
# ------------------------------------------------------------

classification_comparison = (
    classification_df.copy()
)

classification_comparison.insert(
    0,
    "Model Type",
    "Classification"
)

classification_comparison = (
    classification_comparison.rename(
        columns={
            "Model": "Model"
        }
    )
)

classification_comparison[
    "MAE"
] = float("nan")

classification_comparison[
    "RMSE"
] = float("nan")

classification_comparison[
    "R2"
] = float("nan")

classification_comparison[
    "Adjusted_R2"
] = float("nan")


classification_comparison = (
    classification_comparison[
        [
            "Model Type",
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "ROC_AUC",
            "MAE",
            "RMSE",
            "R2",
            "Adjusted_R2"
        ]
    ]
)


# ------------------------------------------------------------
# 7. ADD BEST SMOTE RESULT
# ------------------------------------------------------------

smote_row = imbalance_df[
    imbalance_df["Method"] == "SMOTE"
]

if not smote_row.empty:

    smote_row = smote_row.iloc[0]

    smote_classification_row = pd.DataFrame(
        [
            {
                "Model Type": "Classification",
                "Model": "Logistic Regression + SMOTE",
                "Accuracy": smote_row["Accuracy"],
                "Precision": smote_row["Precision"],
                "Recall": smote_row["Recall"],
                "F1": smote_row["F1"],
                "ROC_AUC": float("nan"),
                "MAE": float("nan"),
                "RMSE": float("nan"),
                "R2": float("nan"),
                "Adjusted_R2": float("nan")
            }
        ]
    )

    classification_comparison = pd.concat(
        [
            classification_comparison,
            smote_classification_row
        ],
        ignore_index=True
    )


# ------------------------------------------------------------
# 8. ADD BEST GRIDSEARCH RANDOM FOREST
# ------------------------------------------------------------

rf_row = classification_df[
    classification_df["Model"] == "Random Forest"
]

if not rf_row.empty:

    rf_row = rf_row.iloc[0]

    gridsearch_row = pd.DataFrame(
        [
            {
                "Model Type": "Classification",
                "Model": "Random Forest GridSearchCV",
                "Accuracy": rf_metrics.get(
                    "Accuracy",
                    float("nan")
                ),
                "Precision": rf_metrics.get(
                    "Precision",
                    float("nan")
                ),
                "Recall": rf_metrics.get(
                    "Recall",
                    float("nan")
                ),
                "F1": rf_metrics.get(
                    "F1",
                    float("nan")
                ),
                "ROC_AUC": rf_metrics.get(
                    "ROC-AUC",
                    float("nan")
                ),
                "MAE": float("nan"),
                "RMSE": float("nan"),
                "R2": float("nan"),
                "Adjusted_R2": float("nan")
            }
        ]
    )

    classification_comparison = pd.concat(
        [
            classification_comparison,
            gridsearch_row
        ],
        ignore_index=True
    )


# ------------------------------------------------------------
# 9. CREATE REGRESSION ROW
# ------------------------------------------------------------

regression_row = pd.DataFrame(
    [
        {
            "Model Type": "Regression",
            "Model": "Linear Regression",
            "Accuracy": float("nan"),
            "Precision": float("nan"),
            "Recall": float("nan"),
            "F1": float("nan"),
            "ROC_AUC": float("nan"),
            "MAE": regression_metrics.get(
                "MAE",
                float("nan")
            ),
            "RMSE": regression_metrics.get(
                "RMSE",
                float("nan")
            ),
            "R2": regression_metrics.get(
                "R2",
                float("nan")
            ),
            "Adjusted_R2": regression_metrics.get(
                "Adjusted_R2",
                float("nan")
            )
        }
    ]
)


# ------------------------------------------------------------
# 10. FINAL COMPARISON TABLE
# ------------------------------------------------------------

final_comparison = pd.concat(
    [
        classification_comparison,
        regression_row
    ],
    ignore_index=True
)


numeric_columns = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "ROC_AUC",
    "MAE",
    "RMSE",
    "R2",
    "Adjusted_R2"
]

final_comparison[
    numeric_columns
] = final_comparison[
    numeric_columns
].round(4)


print("\n" + "=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

print(
    final_comparison.to_string(
        index=False
    )
)


# ------------------------------------------------------------
# 11. SAVE FINAL COMPARISON
# ------------------------------------------------------------

comparison_path = (
    output_dir /
    "final_model_comparison.csv"
)

final_comparison.to_csv(
    comparison_path,
    index=False
)

print("\nFinal comparison saved:")
print(comparison_path)


# ------------------------------------------------------------
# 12. FIND BEST CLASSIFICATION MODEL
# ------------------------------------------------------------

classification_only = (
    final_comparison[
        final_comparison["Model Type"]
        == "Classification"
    ]
)

best_classification = (
    classification_only.loc[
        classification_only["F1"].idxmax()
    ]
)


# ------------------------------------------------------------
# 13. REGRESSION SUMMARY
# ------------------------------------------------------------

regression_only = (
    final_comparison[
        final_comparison["Model Type"]
        == "Regression"
    ]
)

regression_result = (
    regression_only.iloc[0]
)


# ------------------------------------------------------------
# 14. FINAL RECOMMENDATION
# ------------------------------------------------------------

best_model_name = (
    best_classification["Model"]
)

best_f1 = (
    best_classification["F1"]
)

best_accuracy = (
    best_classification["Accuracy"]
)

regression_r2 = (
    regression_result["R2"]
)

regression_adjusted_r2 = (
    regression_result["Adjusted_R2"]
)

recommendation = f"""
Final Model Recommendation
==========================

For the Titanic survival classification task, the recommended
classification model based on the highest F1-score is
{best_model_name}, with an F1-score of {best_f1:.4f} and
accuracy of {best_accuracy:.4f}.

The comparison shows that handling class imbalance can improve
the balance between precision and recall, so F1-score is more
informative than accuracy alone when selecting the classification
approach.

For fare prediction, Linear Regression achieved an R² of
{regression_r2:.4f} and an Adjusted R² of
{regression_adjusted_r2:.4f}. The regression model should be
evaluated separately from classification because its metrics
measure continuous-value prediction rather than class prediction.

Overall, the classification and regression models should be
selected according to their respective task-specific metrics.
"""


print("\n" + "=" * 70)
print("FINAL RECOMMENDATION")
print("=" * 70)

print(
    recommendation.strip()
)


# ------------------------------------------------------------
# 15. SAVE RECOMMENDATION
# ------------------------------------------------------------

recommendation_path = (
    output_dir /
    "final_model_recommendation.txt"
)

recommendation_path.write_text(
    recommendation.strip(),
    encoding="utf-8"
)

print("\nRecommendation saved:")
print(recommendation_path)


# ------------------------------------------------------------
# 16. SAVE SEPARATE CLASSIFICATION TABLE
# ------------------------------------------------------------

classification_table_path = (
    output_dir /
    "classification_comparison_final.csv"
)

classification_only.to_csv(
    classification_table_path,
    index=False
)

print("\nClassification comparison saved:")
print(classification_table_path)


# ------------------------------------------------------------
# 17. SAVE SEPARATE REGRESSION TABLE
# ------------------------------------------------------------

regression_table_path = (
    output_dir /
    "regression_comparison_final.csv"
)

regression_only.to_csv(
    regression_table_path,
    index=False
)

print("\nRegression comparison saved:")
print(regression_table_path)


# ------------------------------------------------------------
# COMPLETE
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 14 COMPLETE")
print("=" * 70)