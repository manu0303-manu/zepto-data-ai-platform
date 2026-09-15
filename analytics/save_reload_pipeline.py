import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier


# ============================================================
# STEP 1: LOAD DATA
# ============================================================

data_path = Path("analytics/data/titanic.csv")
output_dir = Path("analytics/output")

output_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(data_path)


# ============================================================
# STEP 2: DEFINE FEATURES AND TARGET
# ============================================================

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


# ============================================================
# STEP 3: STRATIFIED TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# STEP 4: DEFINE NUMERIC AND CATEGORICAL FEATURES
# ============================================================

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


# ============================================================
# STEP 5: NUMERIC PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)


# ============================================================
# STEP 6: CATEGORICAL PREPROCESSING
# ============================================================

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


# ============================================================
# STEP 7: COMBINE PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ]
)


# ============================================================
# STEP 8: USE ACTUAL GRIDSEARCH BEST PARAMETERS
# ============================================================

best_random_forest = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    max_features="sqrt",
    random_state=42,
    oob_score=True,
    bootstrap=True
)


# ============================================================
# STEP 9: CREATE COMPLETE PIPELINE
# ============================================================

full_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", best_random_forest)
    ]
)


# ============================================================
# STEP 10: TRAIN PIPELINE
# ============================================================

full_pipeline.fit(X_train, y_train)


# ============================================================
# STEP 11: SAVE COMPLETE PIPELINE
# ============================================================

pipeline_path = output_dir / "best_classification_pipeline.joblib"

joblib.dump(
    full_pipeline,
    pipeline_path
)


# ============================================================
# STEP 12: RELOAD PIPELINE
# ============================================================

loaded_pipeline = joblib.load(pipeline_path)


# ============================================================
# STEP 13: RAW INPUT PREDICTION
# ============================================================

raw_passenger = pd.DataFrame(
    [
        {
            "pclass": 1,
            "sex": "female",
            "age": 30,
            "sibsp": 0,
            "parch": 0,
            "fare": 80.0,
            "embarked": "C"
        }
    ]
)


prediction = loaded_pipeline.predict(raw_passenger)
probability = loaded_pipeline.predict_proba(raw_passenger)


# ============================================================
# STEP 14: SAVE PREDICTION
# ============================================================

prediction_label = "SURVIVED" if prediction[0] == 1 else "DID NOT SURVIVE"

prediction_output = pd.DataFrame(
    [
        {
            "pclass": 1,
            "sex": "female",
            "age": 30,
            "sibsp": 0,
            "parch": 0,
            "fare": 80.0,
            "embarked": "C",
            "prediction": int(prediction[0]),
            "prediction_label": prediction_label,
            "probability_not_survived": float(probability[0][0]),
            "probability_survived": float(probability[0][1])
        }
    ]
)

prediction_path = output_dir / "reloaded_pipeline_prediction.csv"

prediction_output.to_csv(
    prediction_path,
    index=False
)


# ============================================================
# STEP 15: SAVE PIPELINE INFORMATION
# ============================================================

information_path = output_dir / "best_pipeline_information.txt"

with open(information_path, "w", encoding="utf-8") as file:

    file.write("Best Classification Pipeline\n")
    file.write("=" * 50 + "\n\n")

    file.write("Model: Random Forest Classifier\n")
    file.write("Source: Random Forest GridSearchCV\n\n")

    file.write("Best Parameters:\n")
    file.write("n_estimators: 100\n")
    file.write("max_depth: 5\n")
    file.write("max_features: sqrt\n")
    file.write("random_state: 42\n")
    file.write("oob_score: True\n")
    file.write("bootstrap: True\n\n")

    file.write("Training rows: ")
    file.write(str(len(X_train)))
    file.write("\n")

    file.write("Testing rows: ")
    file.write(str(len(X_test)))
    file.write("\n\n")

    file.write("Pipeline components:\n")
    file.write("1. Numeric median imputation\n")
    file.write("2. StandardScaler\n")
    file.write("3. Categorical most-frequent imputation\n")
    file.write("4. OneHotEncoder\n")
    file.write("5. Random Forest Classifier\n\n")

    file.write("Reload verification:\n")
    file.write("Pipeline saved using joblib.dump()\n")
    file.write("Pipeline reloaded using joblib.load()\n")
    file.write("Raw passenger input successfully processed\n")
    file.write("Prediction successfully generated\n\n")

    file.write("Prediction: ")
    file.write(prediction_label)
    file.write("\n")

    file.write("Probability not survived: ")
    file.write(f"{probability[0][0]:.4f}")
    file.write("\n")

    file.write("Probability survived: ")
    file.write(f"{probability[0][1]:.4f}")
    file.write("\n")


# ============================================================
# FINAL OUTPUT
# ============================================================

print("=" * 60)
print("BEST CLASSIFICATION PIPELINE SAVED AND RELOADED")
print("=" * 60)

print("Best parameters:")
print("n_estimators: 100")
print("max_depth: 5")
print("max_features: sqrt")

print()
print("Pipeline file:", pipeline_path)
print("Pipeline exists:", pipeline_path.exists())

print()
print("Reloaded prediction:", prediction_label)

print()
print("Prediction file:", prediction_path)
print("Information file:", information_path)

print()
print("STEP 15 CORRECTED AND VERIFIED")