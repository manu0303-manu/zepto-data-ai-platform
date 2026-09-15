# Data Pipeline

This module implements an end-to-end data pipeline for collecting, cleaning, storing, and analyzing book data from Books to Scrape.

## Overview

The Data Pipeline demonstrates the complete workflow:

```text
Web Scraping
     ↓
Raw CSV
     ↓
Data Cleaning
     ↓
Price Conversion
     ↓
SQLite Database
     ↓
SQL Queries
     ↓
Pandas SQL
     ↓
Pandas Merge
```

## Features

* Web scraping using Requests and BeautifulSoup
* Pagination handling
* Book title extraction
* Price extraction
* Rating extraction
* Availability extraction
* Category extraction
* Data cleaning
* GBP to INR conversion
* SQLite database creation
* Normalized relational database tables
* SQL analysis
* Pandas SQL integration
* Pandas merge validation

## Project Structure

```text
data_pipeline/
│
├── README.md
├── requirements.txt
│
├── scrape_books.py
├── clean_data.py
├── create_database.py
├── create_tables.py
├── load_and_query.py
├── pandas_sql.py
└── pandas_merge.py
│
└── data/
    ├── raw_books.csv
    ├── cleaned_books.csv
    ├── zepto_books.db
    ├── sql_results.txt
    └── pandas_merge_results.txt
```

## 1. Web Scraping

The pipeline collects book information from:

```text
Books to Scrape
```

The first five product pages were scraped.

### Data Collected

The scraper collects:

* Book title
* Price
* Rating
* Availability
* Category
* Product URL

The pipeline successfully collected:

```text
100 books
29 categories
```

The raw dataset is stored at:

```text
data_pipeline/data/raw_books.csv
```

## 2. Data Cleaning

The raw book data is cleaned before database loading.

The cleaning process includes:

* Converting price values into numeric format
* Converting ratings into numeric values
* Converting availability into Boolean values
* Removing unwanted formatting
* Creating a clean structured dataset

Cleaned data is stored at:

```text
data_pipeline/data/cleaned_books.csv
```

## 3. GBP to INR Conversion

Book prices were originally available in GBP.

The project uses the required fixed conversion rate:

```text
1 GBP = ₹105.50
```

The INR price is calculated using:

```text
price_inr = price_gbp × 105.50
```

## 4. SQLite Database

The cleaned data is stored in a SQLite database:

```text
data_pipeline/data/zepto_books.db
```

The database contains two normalized tables:

```text
categories
books
```

## Categories Table

```text
categories
-------------------------
category_id
category_name
```

The `category_id` is the primary key.

## Books Table

```text
books
-------------------------
book_id
title
price_gbp
price_inr
rating
in_stock
category_id
```

The `book_id` is the primary key.

The `category_id` column is a foreign key connecting each book to the `categories` table.

## 5. SQL Analysis

The project demonstrates the following SQL operations:

### SELECT and WHERE

Used to filter books based on conditions such as rating.

### ORDER BY and LIMIT

Used to identify the most expensive books.

### DISTINCT

Used to identify unique ratings.

### BETWEEN

Used to filter books within a price range.

### IN

Used to filter multiple rating values.

### JOIN

Used to combine book information with category information.

SQL results are stored at:

```text
data_pipeline/data/sql_results.txt
```

## 6. Pandas SQL Integration

The project uses Pandas `read_sql` to retrieve database results into DataFrames.

Example workflow:

```text
SQLite Database
       ↓
SQL Query
       ↓
pd.read_sql()
       ↓
Pandas DataFrame
```

The results are validated against SQL query results.

## 7. Pandas Merge

The project also demonstrates joining related datasets using:

```python
pd.merge()
```

The Pandas merge result is compared with the SQL JOIN result to verify that both approaches produce consistent data.

Results are stored at:

```text
data_pipeline/data/pandas_merge_results.txt
```

## 8. Requirements

Install the required Python packages using:

```powershell
pip install -r requirements.txt
```

## 9. Running the Pipeline

Run the scripts in the following general order:

### Step 1 — Scrape Books

```powershell
python data_pipeline/scrape_books.py
```

### Step 2 — Clean Data

```powershell
python data_pipeline/clean_data.py
```

### Step 3 — Create Database

```powershell
python data_pipeline/create_database.py
```

### Step 4 — Create Tables

```powershell
python data_pipeline/create_tables.py
```

### Step 5 — Load Data and Run SQL Queries

```powershell
python data_pipeline/load_and_query.py
```

### Step 6 — Pandas SQL

```powershell
python data_pipeline/pandas_sql.py
```

### Step 7 — Pandas Merge

```powershell
python data_pipeline/pandas_merge.py
```

## 10. Final Output

The completed pipeline produces:

```text
100 scraped books
29 categories
Cleaned book dataset
SQLite database
Normalized categories table
Normalized books table
SQL query results
Pandas SQL results
Pandas merge results
```

## Technology Stack

* Python
* Requests
* BeautifulSoup
* Pandas
* SQLite
* SQL

## Learning Outcomes

This module demonstrates practical knowledge of:

* Web scraping
* Data extraction
* Data cleaning
* Data transformation
* Relational database design
* Primary keys
* Foreign keys
* SQL querying
* Database joins
* Pandas DataFrames
* SQL and Pandas integration
* Data validation

## Conclusion

The Data Pipeline module provides an end-to-end example of transforming unstructured web data into a clean, structured, queryable dataset.

It combines web scraping, data cleaning, relational database design, SQL analysis, and Pandas processing into a reproducible data engineering workflow.
