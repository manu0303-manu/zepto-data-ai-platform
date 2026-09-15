# Analytics & Machine Learning

This module implements an end-to-end analytics and machine learning workflow using the Titanic dataset.

## Overview

The Analytics module demonstrates the complete workflow:

```text
Titanic Dataset
      ↓
Data Loading
      ↓
Data Profiling
      ↓
Missing Value Handling
      ↓
Exploratory Data Analysis
      ↓
Correlation Analysis
      ↓
Multivariate Analysis
      ↓
Classification
      ↓
Class Imbalance Handling
      ↓
Random Forest GridSearch
      ↓
Fare Regression
      ↓
Model Comparison
      ↓
Pipeline Save & Reload
```

## Features

* Titanic dataset loading
* Data profiling
* Dataset shape and information analysis
* Missing-value analysis
* Missing-value handling
* Exploratory Data Analysis
* Correlation analysis
* Multivariate analysis
* Classification models
* Class imbalance handling
* SMOTE
* Random Forest hyperparameter tuning
* GridSearchCV
* Out-of-Bag evaluation
* Fare regression
* Model comparison
* ML pipeline persistence using Joblib
* Saved pipeline reload and prediction

## Project Structure

```text id="a7r3v2"
analytics/
│
├── README.md
│
├── data/
│   ├── titanic.csv
│   └── titanic_cleaned.csv
│
├── output/
│   ├── classification results
│   ├── EDA charts
│   ├── correlation analysis
│   ├── multivariate charts
│   ├── imbalance results
│   ├── Random Forest GridSearch results
│   ├── regression results
│   ├── model comparison
│   └── best_classification_pipeline.joblib
│
├── load_titanic.py
├── profile_titanic.py
├── handle_missing.py
├── eda_titanic.py
├── correlation_analysis.py
├── multivariate_analysis.py
├── classification_models.py
├── imbalance_handling.py
├── random_forest_gridsearch.py
├── fare_regression.py
├── model_comparison.py
└── save_reload_pipeline.py
```

## 1. Dataset Loading

The Titanic dataset is loaded using Seaborn:

```python id="f0c8k5"
sns.load_dataset("titanic")
```

The dataset is saved locally as:

```text id="z0l7f4"
analytics/data/titanic.csv
```

The dataset contains:

```text id="q6q0z8"
891 rows
15 columns
```

## 2. Data Profiling

The profiling step analyzes:

* Dataset shape
* Column names
* Data types
* Missing values
* Descriptive statistics
* Numerical distributions
* Categorical information

The profiling results help identify data-quality issues before analysis and modeling.

## 3. Missing Value Handling

Missing values were analyzed using percentages.

Important missing-value columns include:

```text id="m2o7k1"
age
embarked
deck
embark_town
```

The project applies appropriate handling strategies based on the missing-value percentage.

### Handling Strategy

* Low missing percentage → appropriate row handling
* Moderate missing percentage → median/mode imputation where applicable
* Very high missing percentage → column removal

For example:

```text id="7l5t8v"
age → median imputation
embarked → missing rows handled
embark_town → missing rows handled
deck → removed because of very high missing percentage
```

The cleaned dataset is saved as:

```text id="4n8v2s"
analytics/data/titanic_cleaned.csv
```

## 4. Exploratory Data Analysis

The EDA workflow includes:

* Dataset overview
* Descriptive statistics
* Missing-value analysis
* Univariate analysis
* Survival analysis
* Numerical distributions
* Categorical analysis
* Data visualization

The analysis focuses on understanding the relationships between passenger characteristics and survival.

## 5. Correlation Analysis

The required six numerical columns were analyzed:

```text id="8n3j6a"
survived
pclass
age
sibsp
parch
fare
```

Correlation analysis is used to understand linear relationships between these numerical variables.

The analysis excludes unrelated derived columns that were not part of the required six-column correlation matrix.

## 6. Multivariate Analysis

The project includes multiple multivariate visualizations to analyze relationships between:

* Passenger class
* Age
* Fare
* Sex
* Survival
* Family-related variables

These visualizations help identify patterns that may not be visible from individual-variable analysis.

## 7. Classification

The main machine learning task is Titanic survival classification.

Target variable:

```text id="q8q3w6"
survived
```

Three classification algorithms were evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest

### Evaluation Metrics

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC

### Classification Results

| Model               | Accuracy | F1 Score | ROC-AUC |
| ------------------- | -------: | -------: | ------: |
| Logistic Regression |   80.45% |   72.44% |  84.37% |
| Decision Tree       |   76.54% |   65.57% |  79.71% |
| Random Forest       |   81.56% |   74.42% |  83.00% |

Random Forest achieved the highest accuracy and F1-score among the initial classification models.

## 8. Class Imbalance

The target classes were analyzed before model training.

Class distribution:

```text id="h3z4xk"
Class 0 → 549 passengers
Class 1 → 342 passengers
```

The project evaluates multiple approaches:

* Baseline model
* Class-weight balancing
* SMOTE

### Results

```text id="w9q1s6"
Baseline F1      → 0.7244
Class-weight F1  → 0.7552
SMOTE F1         → 0.7606
```

SMOTE achieved the strongest F1-score among the tested approaches.

## 9. Machine Learning Pipeline

The classification workflow uses a preprocessing pipeline to avoid data leakage.

The pipeline includes:

* Numerical preprocessing
* Categorical preprocessing
* Encoding
* Imputation
* Model training

A stratified train-test split is used to preserve the target-class distribution.

## 10. Random Forest GridSearch

Random Forest hyperparameters were optimized using:

```python id="8j3m0v"
GridSearchCV
```

The best parameters were:

```text id="9j6f2r"
max_depth = 5
max_features = sqrt
n_estimators = 100
```

The optimized model was evaluated using:

* Cross-validation
* Test-set performance
* OOB score
* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC

### Best Model Configuration

```text id="2k4h8z"
Cross-validation F1 → 0.7459
OOB Score            → 0.8272
```

## 11. Fare Regression

A regression workflow was also implemented using Titanic passenger data.

The regression workflow demonstrates:

* Feature preparation
* Train-test splitting
* Regression model training
* Prediction
* Regression evaluation
* Performance analysis

This extends the project beyond classification into a regression use case.

## 12. Model Comparison

The project compares the implemented machine learning models using consistent evaluation metrics.

The comparison helps identify the most suitable model based on:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC

Model performance is stored in the output directory.

## 13. Model Persistence

The best classification pipeline is saved using Joblib:

```text id="6v4k1p"
analytics/output/best_classification_pipeline.joblib
```

The saved pipeline is then reloaded and tested to verify that the trained model can be reused for new predictions.

## 14. Prediction Test

After reloading the saved pipeline, a sample passenger record is passed through the model.

The workflow verifies:

```text id="q2m6v9"
Saved Pipeline
      ↓
Reload Pipeline
      ↓
New Passenger Data
      ↓
Prediction
```

This demonstrates practical model deployment readiness.

## 15. Technology Stack

### Programming

* Python

### Data Analysis

* Pandas
* NumPy

### Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn
* Logistic Regression
* Decision Tree
* Random Forest
* SMOTE
* GridSearchCV

### Model Persistence

* Joblib

## 16. Learning Outcomes

This module demonstrates practical knowledge of:

* Data loading
* Data profiling
* Data cleaning
* Missing-value handling
* Exploratory Data Analysis
* Correlation analysis
* Multivariate analysis
* Feature preprocessing
* Classification
* Imbalanced classification
* SMOTE
* Hyperparameter tuning
* Cross-validation
* Random Forest
* Regression
* Model evaluation
* ML pipeline development
* Model persistence

## Conclusion

The Analytics & Machine Learning module provides a complete machine learning workflow using the Titanic dataset.

It demonstrates how raw data can be transformed into meaningful insights and machine learning predictions through systematic data preparation, exploratory analysis, model development, evaluation, optimization, and pipeline persistence.
