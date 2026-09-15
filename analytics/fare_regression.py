import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# STEP 13 — FARE REGRESSION
# ============================================================

print("=" * 70)
print("STEP 13 — FARE REGRESSION")
print("=" * 70)


# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

file_path = Path(
    "analytics/data/titanic.csv"
)

df = pd.read_csv(
    file_path
)

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
    "embarked"
]

target = "fare"

X = df[features]
y = df[target]

print("\nFeatures used to predict fare:")
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
    random_state=42
)

print("\n" + "=" * 70)
print("TRAIN / TEST SPLIT")
print("=" * 70)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))


# ------------------------------------------------------------
# 4. FEATURE TYPES
# ------------------------------------------------------------

numeric_features = [
    "pclass",
    "age",
    "sibsp",
    "parch"
]

categorical_features = [
    "sex",
    "embarked"
]


# ------------------------------------------------------------
# 5. NUMERIC PIPELINE
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
# 6. CATEGORICAL PIPELINE
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
# 8. LINEAR REGRESSION PIPELINE
# ------------------------------------------------------------

regression_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            LinearRegression()
        )
    ]
)


# ------------------------------------------------------------
# 9. TRAIN MODEL
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("TRAINING LINEAR REGRESSION")
print("=" * 70)

regression_pipeline.fit(
    X_train,
    y_train
)

print("Model training completed.")


# ------------------------------------------------------------
# 10. PREDICTIONS
# ------------------------------------------------------------

y_pred = regression_pipeline.predict(
    X_test
)


# ------------------------------------------------------------
# 11. CALCULATE METRICS
# ------------------------------------------------------------

mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_test,
    y_pred
)


# ------------------------------------------------------------
# 12. NUMBER OF PREDICTORS
# ------------------------------------------------------------

preprocessed_X_train = (
    regression_pipeline
    .named_steps["preprocessor"]
    .transform(X_train)
)

number_of_predictors = (
    preprocessed_X_train.shape[1]
)

number_of_test_rows = len(
    y_test
)


# ------------------------------------------------------------
# 13. ADJUSTED R-SQUARED
# ------------------------------------------------------------

if (
    number_of_test_rows
    - number_of_predictors
    - 1
) > 0:

    adjusted_r2 = (
        1
        - (
            (1 - r2)
            * (number_of_test_rows - 1)
            / (
                number_of_test_rows
                - number_of_predictors
                - 1
            )
        )
    )

else:

    adjusted_r2 = np.nan


# ------------------------------------------------------------
# 14. DISPLAY METRICS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("REGRESSION PERFORMANCE")
print("=" * 70)

print(
    "MAE        :",
    round(mae, 4)
)

print(
    "RMSE       :",
    round(rmse, 4)
)

print(
    "R²         :",
    round(r2, 4)
)

print(
    "Adjusted R²:",
    round(adjusted_r2, 4)
)

print(
    "Predictors :",
    number_of_predictors
)


# ------------------------------------------------------------
# 15. RESIDUALS
# ------------------------------------------------------------

residuals = (
    y_test.to_numpy()
    - y_pred
)


# ------------------------------------------------------------
# 16. RESIDUAL PLOT
# ------------------------------------------------------------

output_dir = Path(
    "analytics/output"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True
)

plt.figure(
    figsize=(10, 6)
)

plt.scatter(
    y_pred,
    residuals,
    alpha=0.6
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.title(
    "Fare Regression Residual Plot"
)

plt.xlabel(
    "Predicted Fare"
)

plt.ylabel(
    "Residual"
)

plt.tight_layout()

residual_plot_path = (
    output_dir /
    "fare_regression_residual_plot.png"
)

plt.savefig(
    residual_plot_path,
    dpi=150
)

plt.close()

print("\nResidual plot saved:")
print(residual_plot_path)


# ------------------------------------------------------------
# 17. HETEROSCEDASTICITY CHECK
# ------------------------------------------------------------

absolute_residuals = np.abs(
    residuals
)

low_prediction_mask = (
    y_pred <= np.median(y_pred)
)

high_prediction_mask = (
    y_pred > np.median(y_pred)
)

low_residual_spread = (
    absolute_residuals[
        low_prediction_mask
    ].mean()
)

high_residual_spread = (
    absolute_residuals[
        high_prediction_mask
    ].mean()
)

spread_ratio = (
    high_residual_spread
    / max(low_residual_spread, 1e-9)
)


if spread_ratio >= 1.5:

    heteroscedasticity_conclusion = (
        "The residual plot suggests possible "
        "heteroscedasticity because residual spread "
        "is substantially larger at higher predicted fares."
    )

else:

    heteroscedasticity_conclusion = (
        "The residual plot does not show strong evidence "
        "of heteroscedasticity based on the residual spread "
        "comparison across lower and higher predicted fares."
    )


print("\n" + "=" * 70)
print("HETEROSCEDASTICITY CHECK")
print("=" * 70)

print(
    "Average absolute residual "
    "for lower predicted fares:",
    round(low_residual_spread, 4)
)

print(
    "Average absolute residual "
    "for higher predicted fares:",
    round(high_residual_spread, 4)
)

print(
    "Residual spread ratio:",
    round(spread_ratio, 4)
)

print(
    "\nConclusion:"
)

print(
    heteroscedasticity_conclusion
)


# ------------------------------------------------------------
# 18. SAVE REGRESSION METRICS
# ------------------------------------------------------------

metrics = pd.DataFrame(
    {
        "Metric": [
            "MAE",
            "RMSE",
            "R2",
            "Adjusted_R2"
        ],
        "Value": [
            mae,
            rmse,
            r2,
            adjusted_r2
        ]
    }
)

metrics_path = (
    output_dir /
    "fare_regression_metrics.csv"
)

metrics.to_csv(
    metrics_path,
    index=False
)

print("\nRegression metrics saved:")
print(metrics_path)


# ------------------------------------------------------------
# 19. SAVE RESIDUAL CONCLUSION
# ------------------------------------------------------------

conclusion_path = (
    output_dir /
    "fare_regression_conclusion.txt"
)

conclusion_text = f"""
Fare Regression Summary
=======================

Model:
Linear Regression

Target:
fare

Features:
{", ".join(features)}

Training rows:
{len(X_train)}

Testing rows:
{len(X_test)}

MAE:
{mae:.4f}

RMSE:
{rmse:.4f}

R²:
{r2:.4f}

Adjusted R²:
{adjusted_r2:.4f}

Number of predictors after preprocessing:
{number_of_predictors}

Heteroscedasticity conclusion:
{heteroscedasticity_conclusion}
"""

conclusion_path.write_text(
    conclusion_text.strip(),
    encoding="utf-8"
)

print("\nRegression conclusion saved:")
print(conclusion_path)


# ------------------------------------------------------------
# 20. SAVE ACTUAL VS PREDICTED VALUES
# ------------------------------------------------------------

prediction_results = pd.DataFrame(
    {
        "actual_fare": y_test.to_numpy(),
        "predicted_fare": y_pred,
        "residual": residuals
    }
)

prediction_path = (
    output_dir /
    "fare_regression_predictions.csv"
)

prediction_results.to_csv(
    prediction_path,
    index=False
)

print("\nPrediction results saved:")
print(prediction_path)


# ------------------------------------------------------------
# COMPLETE
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 13 COMPLETE")
print("=" * 70)