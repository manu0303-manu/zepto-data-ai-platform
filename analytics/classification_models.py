import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score
)


# =========================================================
# PATHS
# =========================================================

input_path = Path("analytics/data/titanic.csv")
output_dir = Path("analytics/output")

output_dir.mkdir(parents=True, exist_ok=True)


# =========================================================
# LOAD RAW DATA
# =========================================================
# We use the original Titanic CSV here.
# The ML pipeline performs preprocessing after the train/test split.
# This prevents test-data information from being used during fitting.

df = pd.read_csv(input_path)

print("=" * 70)
print("STEP 10 — CLASSIFICATION MODELS")
print("=" * 70)

print("\nDataset shape:")
print(df.shape)


# =========================================================
# SELECT FEATURES AND TARGET
# =========================================================

target = "survived"

features = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "fare",
    "embarked"
]

X = df[features]
y = df[target]


print("\nFeatures:")
print(features)

print("\nTarget:")
print(target)


# =========================================================
# REMOVE VERY HIGH-MISSING COLUMN
# =========================================================
# deck has approximately 77% missing values.
# It is not used as a modeling feature.

print("\nDropping high-missing column:")
print("deck → approximately 77% missing")


# =========================================================
# STRATIFIED TRAIN / TEST SPLIT
# =========================================================

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
print(y_train.value_counts(normalize=True).round(4))

print("\nTesting target distribution:")
print(y_test.value_counts(normalize=True).round(4))


# =========================================================
# DEFINE NUMERIC AND CATEGORICAL FEATURES
# =========================================================

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


# =========================================================
# NUMERIC PIPELINE
# =========================================================

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


# =========================================================
# CATEGORICAL PIPELINE
# =========================================================

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


# =========================================================
# COLUMN TRANSFORMER
# =========================================================

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


# =========================================================
# DEFINE CLASSIFICATION MODELS
# =========================================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )
}


# =========================================================
# TRAIN MODELS
# =========================================================

trained_pipelines = {}

results = []

predictions = {}

probabilities = {}


print("\n" + "=" * 70)
print("TRAINING CLASSIFICATION MODELS")
print("=" * 70)


for model_name, model in models.items():

    print("\nTraining:", model_name)

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_probability = pipeline.predict_proba(X_test)[:, 1]

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

    auc = roc_auc_score(
        y_test,
        y_probability
    )

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC_AUC": auc
    })

    trained_pipelines[model_name] = pipeline
    predictions[model_name] = y_pred
    probabilities[model_name] = y_probability

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1       :", round(f1, 4))
    print("ROC-AUC  :", round(auc, 4))


# =========================================================
# MODEL COMPARISON TABLE
# =========================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    results_df.round(4).to_string(index=False)
)


comparison_path = (
    output_dir / "classification_model_comparison.csv"
)

results_df.to_csv(
    comparison_path,
    index=False
)

print("\nComparison table saved:")
print(comparison_path)


# =========================================================
# CONFUSION MATRICES
# =========================================================

print("\n" + "=" * 70)
print("CONFUSION MATRICES")
print("=" * 70)


for model_name in models.keys():

    y_pred = predictions[model_name]

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print("\n", model_name)
    print(cm)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Did Not Survive", "Survived"]
    )

    display.plot()

    plt.title(
        f"Confusion Matrix — {model_name}"
    )

    plt.tight_layout()

    safe_name = model_name.lower().replace(
        " ",
        "_"
    )

    cm_path = (
        output_dir /
        f"confusion_matrix_{safe_name}.png"
    )

    plt.savefig(
        cm_path,
        dpi=150
    )

    plt.close()

    print("Saved:", cm_path)


# =========================================================
# ROC CURVES
# =========================================================

plt.figure(figsize=(9, 6))


for model_name in models.keys():

    y_probability = probabilities[model_name]

    false_positive_rate, true_positive_rate, _ = roc_curve(
        y_test,
        y_probability
    )

    auc = roc_auc_score(
        y_test,
        y_probability
    )

    plt.plot(
        false_positive_rate,
        true_positive_rate,
        label=f"{model_name} (AUC = {auc:.3f})"
    )


plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.title("ROC Curves — Classification Models")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()

plt.tight_layout()

roc_path = output_dir / "classification_roc_curves.png"

plt.savefig(
    roc_path,
    dpi=150
)

plt.close()

print("\nROC curves saved:")
print(roc_path)


# =========================================================
# DECISION TREE VISUALIZATION
# =========================================================

decision_tree_pipeline = trained_pipelines[
    "Decision Tree"
]

decision_tree_model = decision_tree_pipeline.named_steps[
    "model"
]

decision_tree_preprocessor = (
    decision_tree_pipeline.named_steps["preprocessor"]
)


feature_names = (
    decision_tree_preprocessor
    .get_feature_names_out()
)


plt.figure(figsize=(22, 12))

plot_tree(
    decision_tree_model,
    feature_names=feature_names,
    class_names=["Did Not Survive", "Survived"],
    filled=True,
    rounded=True,
    fontsize=8
)

plt.title("Decision Tree Classifier")

plt.tight_layout()

tree_path = output_dir / "decision_tree.png"

plt.savefig(
    tree_path,
    dpi=150
)

plt.close()

print("\nDecision tree saved:")
print(tree_path)


# =========================================================
# SAVE TRAIN / TEST INFORMATION
# =========================================================

split_information_path = (
    output_dir / "classification_split_information.txt"
)


with open(
    split_information_path,
    "w",
    encoding="utf-8"
) as file:

    file.write("CLASSIFICATION TRAIN/TEST SPLIT\n")
    file.write("=" * 60 + "\n\n")

    file.write(
        f"Original dataset rows: {len(df)}\n"
    )

    file.write(
        f"Training rows: {len(X_train)}\n"
    )

    file.write(
        f"Testing rows: {len(X_test)}\n\n"
    )

    file.write(
        "Split method: train_test_split\n"
    )

    file.write(
        "Test size: 20%\n"
    )

    file.write(
        "Random state: 42\n"
    )

    file.write(
        "Stratification: stratify=y\n\n"
    )

    file.write(
        "Preprocessing is fitted inside the Pipeline "
        "using training data only.\n"
    )


print("\nSplit information saved:")
print(split_information_path)


# =========================================================
# FINAL
# =========================================================

print("\n" + "=" * 70)
print("STEP 10 COMPLETE")
print("=" * 70)