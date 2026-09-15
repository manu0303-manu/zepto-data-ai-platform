# Zepto Data & AI Platform

A complete Data Engineering, Analytics, Machine Learning, and GenAI Support Assistant capstone project.

## Project Overview

The Zepto Data & AI Platform is organized as a single repository containing three modules:

1. Data Pipeline
2. Analytics Pipeline
3. Support Assistant

The project demonstrates an end-to-end workflow covering web scraping, data cleaning, relational database design, SQL analysis, exploratory data analysis, machine learning, model evaluation, regression, pipeline persistence, and GenAI support-assistant development.

---

# Project Structure

```text
zepto-data-ai-platform/
│
├── README.md
│
├── data_pipeline/
│   ├── data/
│   │   ├── raw_books.csv
│   │   ├── cleaned_books.csv
│   │   ├── zepto_books.db
│   │   ├── sql_results.txt
│   │   └── pandas_merge_results.txt
│   │
│   ├── requirements.txt
│   ├── scrape_books.py
│   ├── clean_data.py
│   ├── create_database.py
│   ├── create_tables.py
│   ├── load_and_query.py
│   ├── pandas_sql.py
│   └── pandas_merge.py
│
├── analytics/
│   ├── data/
│   │   ├── titanic.csv
│   │   └── titanic_cleaned.csv
│   │
│   ├── output/
│   │   ├── classification results
│   │   ├── EDA charts
│   │   ├── correlation analysis
│   │   ├── multivariate charts
│   │   ├── regression results
│   │   ├── Random Forest GridSearch results
│   │   └── saved ML pipeline
│   │
│   ├── load_titanic.py
│   ├── profile_titanic.py
│   ├── handle_missing.py
│   ├── eda_titanic.py
│   ├── correlation_analysis.py
│   ├── multivariate_analysis.py
│   ├── classification_models.py
│   ├── imbalance_handling.py
│   ├── random_forest_gridsearch.py
│   ├── fare_regression.py
│   ├── model_comparison.py
│   └── save_reload_pipeline.py
│
└── support_assistant/
    └── Support Assistant implementation