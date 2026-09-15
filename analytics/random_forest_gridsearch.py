import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# ============================================================
# STEP 12 — RANDOM FOREST GRIDSEARCH
# ============================================================

print("=" * 70)
print("STEP 12 — RANDOM FOREST GRIDSEARCH + OOB SCORE")
print("=" * 70)


# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

file_path = Path("analytics/data/titanic.csv")

df = pd.read_csv(file_path)

print("\nDataset shape:")
print(df.shape)


# ------------------------------------------------------------
# 2. FEATURES AND TARGET
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

print("\nFeatures:")
print(features)

print("\nTarget:")
print(target)


# ------------------------------------------------------------
# 3. TRAIN / TEST SPLIT
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

print("\nTraining target distribution:")
print(
    y_train.value_counts(
        normalize=True
    ).sort_index().round(4)
)

print("\nTesting target distribution:")
print(
    y_test.value_counts(
        normalize=True
    ).sort_index().round(4)
)


# ------------------------------------------------------------
# 4. DEFINE FEATURE TYPES
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


# ------------------------------------------------------------
# 5. NUMERIC PREPROCESSING
# ------------------------------------------------------------

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


# ------------------------------------------------------------
# 6. CATEGORICAL PREPROCESSING
# ------------------------------------------------------------

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
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


# ------------------------------------------------------------
# 7. COLUMN TRANSFORMER
# ------------------------------------------------------------

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
# 8. RANDOM FOREST
# ------------------------------------------------------------

random_forest = RandomForestClassifier(
    random_state=42,
    oob_score=True,
    bootstrap=True
)


# ------------------------------------------------------------
# 9. COMPLETE PIPELINE
# ------------------------------------------------------------

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            random_forest
        )
    ]
)


# ------------------------------------------------------------
# 10. GRID PARAMETERS
# ------------------------------------------------------------

param_grid = {
    "model__n_estimators": [
        100,
        200
    ],
    "model__max_depth": [
        5,
        10,
        None
    ],
    "model__max_features": [
        "sqrt",
        "log2"
    ]
}


print("\n" + "=" * 70)
print("GRID SEARCH PARAMETERS")
print("=" * 70)

print("\nn_estimators:")
print([100, 200])

print("\nmax_depth:")
print([5, 10, None])

print("\nmax_features:")
print(["sqrt", "log2"])


# ------------------------------------------------------------
# 11. GRIDSEARCHCV
# ------------------------------------------------------------

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="f1",
    cv=5,
    n_jobs=-1,
    refit=True
)


print("\n" + "=" * 70)
print("STARTING GRID SEARCH")
print("=" * 70)

grid_search.fit(
    X_train,
    y_train
)


print("\nGrid search completed successfully.")


# ------------------------------------------------------------
# 12. BEST PARAMETERS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("BEST PARAMETERS")
print("=" * 70)

print(
    "Best parameters:"
)

print(
    grid_search.best_params_
)

print(
    "\nBest cross-validation F1:"
)

print(
    round(
        grid_search.best_score_,
        4
    )
)


# ------------------------------------------------------------
# 13. BEST PIPELINE
# ------------------------------------------------------------

best_pipeline = grid_search.best_estimator_

best_random_forest = (
    best_pipeline.named_steps["model"]
)


# ------------------------------------------------------------
# 14. OOB SCORE
# ------------------------------------------------------------

oob_score = (
    best_random_forest.oob_score_
)

print("\n" + "=" * 70)
print("OUT-OF-BAG SCORE")
print("=" * 70)

print(
    "OOB Score:",
    round(oob_score, 4)
)


# ------------------------------------------------------------
# 15. TEST PREDICTIONS
# ------------------------------------------------------------

y_pred = best_pipeline.predict(
    X_test
)

y_probability = (
    best_pipeline.predict_proba(
        X_test
    )[:, 1]
)


# ------------------------------------------------------------
# 16. TEST METRICS
# ------------------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


print("\n" + "=" * 70)
print("TEST SET PERFORMANCE")
print("=" * 70)

print(
    "Accuracy :",
    round(accuracy, 4)
)

print(
    "Precision:",
    round(precision, 4)
)

print(
    "Recall   :",
    round(recall, 4)
)

print(
    "F1       :",
    round(f1, 4)
)

print(
    "ROC-AUC  :",
    round(roc_auc, 4)
)


# ------------------------------------------------------------
# 17. SAVE BEST PARAMETERS
# ------------------------------------------------------------

output_dir = Path(
    "analytics/output"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True
)


parameters_path = (
    output_dir /
    "random_forest_best_parameters.txt"
)


with open(
    parameters_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "Random Forest GridSearchCV Results\n"
    )

    file.write(
        "=" * 50 + "\n\n"
    )

    file.write(
        f"Best parameters:\n"
    )

    file.write(
        f"{grid_search.best_params_}\n\n"
    )

    file.write(
        f"Best CV F1: "
        f"{grid_search.best_score_:.4f}\n"
    )

    file.write(
        f"OOB Score: "
        f"{oob_score:.4f}\n\n"
    )

    file.write(
        "Test Metrics\n"
    )

    file.write(
        "-" * 30 + "\n"
    )

    file.write(
        f"Accuracy: {accuracy:.4f}\n"
    )

    file.write(
        f"Precision: {precision:.4f}\n"
    )

    file.write(
        f"Recall: {recall:.4f}\n"
    )

    file.write(
        f"F1: {f1:.4f}\n"
    )

    file.write(
        f"ROC-AUC: {roc_auc:.4f}\n"
    )


print("\nBest parameter results saved:")
print(parameters_path)


# ------------------------------------------------------------
# 18. SAVE ALL GRIDSEARCH RESULTS
# ------------------------------------------------------------

grid_results = pd.DataFrame(
    grid_search.cv_results_
)

grid_results_path = (
    output_dir /
    "random_forest_gridsearch_results.csv"
)

grid_results.to_csv(
    grid_results_path,
    index=False
)

print("\nAll GridSearch results saved:")
print(grid_results_path)


# ------------------------------------------------------------
# 19. SAVE TEST METRICS
# ------------------------------------------------------------

metrics = pd.DataFrame(
    {
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "ROC-AUC",
            "OOB Score",
            "Best CV F1"
        ],
        "Value": [
            accuracy,
            precision,
            recall,
            f1,
            roc_auc,
            oob_score,
            grid_search.best_score_
        ]
    }
)

metrics_path = (
    output_dir /
    "random_forest_gridsearch_metrics.csv"
)

metrics.to_csv(
    metrics_path,
    index=False
)

print("\nMetrics saved:")
print(metrics_path)


# ------------------------------------------------------------
# 20. COMPLETE
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 12 COMPLETE")
print("=" * 70)