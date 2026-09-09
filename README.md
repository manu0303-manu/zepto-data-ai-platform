# Zepto Data & AI Platform

A complete Data Engineering, Analytics, Machine Learning, and GenAI Support Assistant capstone project.

## Project Modules

This project contains three modules:

1. Data Pipeline
2. Analytics Pipeline
3. Support Assistant

All modules are maintained inside one GitHub repository.

---

# Module 1 — Data Pipeline

## Objective

The Data Pipeline module demonstrates an end-to-end data engineering workflow:

Website → Web Scraping → Raw Data → Data Cleaning → Currency Conversion → SQLite Database → SQL Queries → Pandas Analysis

## Data Source

The project uses Books to Scrape:

https://books.toscrape.com/

The website is a public scraping practice website and does not require an API key or login.

## Scraped Data

The scraper collects:

- title
- price
- star rating
- availability
- category

The first five pages of the All Products listing were scraped.

Total books collected:

100

This exceeds the required minimum of 60 books.

---

# Data Cleaning

## Price Cleaning

The original price contains the GBP currency symbol.

The currency symbol is removed and the value is converted to a floating-point number.

New column:

price_gbp

Example:

£51.77 → 51.77

## Rating Cleaning

The website provides ratings as text.

The values are converted into integers:

One → 1
Two → 2
Three → 3
Four → 4
Five → 5

New column:

rating

## Availability Cleaning

The availability text is converted into a Boolean value.

Example:

In stock → True

New column:

in_stock

## Missing Values

The cleaning process checks for parsing failures and missing values.

For the current 100-book dataset:

- Missing price values: 0
- Missing rating values: 0
- Missing availability values: 0

Therefore, no rows required deletion or numeric median imputation for the current dataset.

---

# GBP to INR Conversion

A fixed exchange rate is used:

1 GBP = 105.50 INR

Formula:

price_inr = price_gbp * 105.50

Example:

51.77 GBP × 105.50 = 5461.74 INR

No external currency API is required.

---

# SQLite Database

The cleaned data is stored in:

data_pipeline/data/zepto_books.db

The database contains two normalized tables:

- categories
- books

## Categories Table

- category_id — Primary Key
- category_name — Unique

## Books Table

- book_id — Primary Key
- title
- price_gbp
- price_inr
- rating
- in_stock
- category_id — Foreign Key

The foreign key relationship is:

books.category_id → categories.category_id

---

# SQL Queries

The project demonstrates the following SQL operations:

- SELECT
- WHERE
- ORDER BY
- LIMIT
- DISTINCT
- BETWEEN
- IN
- JOIN

SQL queries and their outputs are saved in:

data_pipeline/data/sql_results.txt

The JOIN query connects the books and categories tables.

---

# Pandas SQL Analysis

The project uses pandas.read_sql() to read SQL query results into Pandas DataFrames.

Implementation:

data_pipeline/pandas_sql.py

Two SQL query results are loaded using Pandas.

---

# Pandas Merge

The SQL JOIN is reproduced using Pandas.

Implementation:

data_pipeline/pandas_merge.py

The merge uses category_id as the common key.

The SQL JOIN and Pandas merge results were compared.

Result:

SQL JOIN rows: 10
Pandas merge rows: 10
Results match: True

The comparison output is saved in:

data_pipeline/data/pandas_merge_results.txt

---

# Installation

## Requirements

- Python 3.x
- pip
- VS Code or another Python IDE

## Create Virtual Environment

From the project root:

python -m venv .venv

Activate on Windows:

.venv\Scripts\activate

## Install Packages

pip install -r requirements.txt

---

# Running Module 1

Run the following commands from the project root.

## 1. Scrape Books

python data_pipeline/scrape_books.py

Output:

data_pipeline/data/raw_books.csv

## 2. Clean Data

python data_pipeline/clean_data.py

Output:

data_pipeline/data/cleaned_books.csv

## 3. Create Database

python data_pipeline/create_database.py

Output:

data_pipeline/data/zepto_books.db

## 4. Create Tables

python data_pipeline/create_tables.py

## 5. Load Data and Run SQL Queries

python data_pipeline/load_and_query.py

Output:

data_pipeline/data/sql_results.txt

## 6. Run Pandas SQL

python data_pipeline/pandas_sql.py

## 7. Run Pandas Merge

python data_pipeline/pandas_merge.py

Output:

data_pipeline/data/pandas_merge_results.txt

---

# Module 1 Deliverables

- [x] Public website scraping
- [x] 100 books collected
- [x] Multiple categories collected
- [x] Raw CSV dataset
- [x] Clean GBP price
- [x] Integer rating
- [x] Boolean availability
- [x] Fixed GBP to INR conversion
- [x] SQLite database
- [x] Normalized database schema
- [x] Primary keys
- [x] Foreign key
- [x] Required SQL queries
- [x] Pandas read_sql
- [x] Pandas merge
- [x] SQL JOIN vs Pandas merge validation

---

# Module 2 — Analytics Pipeline

The Analytics Pipeline will contain:

- Dataset profiling
- Missing-value analysis
- Outlier analysis
- Statistical analysis
- Data visualization
- Correlation analysis
- Classification models
- Model evaluation
- Class imbalance handling
- Hyperparameter tuning
- Regression modeling
- Model comparison
- Saved machine learning pipeline

Implementation directory:

analytics/

---

# Module 3 — Support Assistant

The Support Assistant will contain:

- Document ingestion
- Local embeddings
- ChromaDB
- Retrieval
- Structured prompting
- LangGraph workflow
- Deterministic MOCK_LLM mode
- Pydantic structured output
- FastAPI API
- Docker support

Implementation directory:

support_assistant/

---

# Design Decisions

## Fixed Currency Rate

A fixed exchange rate of 1 GBP = 105.50 INR is used because the project specification requires a fixed conversion rate.

## Normalized Database

Books and categories are stored in separate tables to reduce repeated category information and demonstrate a relational database design.

## Scraping

The scraper uses requests and BeautifulSoup and processes multiple paginated listing pages.

Book detail pages are used to collect category information.

## Reproducibility

Raw and cleaned datasets are stored locally, and Python scripts are provided to recreate the database and query results.

---

# Git Workflow

The project uses Git for version control.

The required workflow is:

1. Create a feature branch.
2. Make project changes.
3. Create at least two commits.
4. Merge the feature branch back into main.

---

# Project Status

## Module 1 — Data Pipeline

Completed.

## Module 2 — Analytics Pipeline

In Progress.

## Module 3 — Support Assistant

In Progress.

---

# Author

B Manoj Achari

AI/ML Engineering Capstone Project
Module 1 Data Pipeline implementation completed.