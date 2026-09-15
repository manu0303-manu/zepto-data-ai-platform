# Zepto Data & AI Platform

A complete Data Engineering, Analytics, Machine Learning, and GenAI Support Assistant capstone project.

## Project Overview

The Zepto Data & AI Platform is a single-repository end-to-end project containing three major modules:

1. Data Pipeline
2. Analytics & Machine Learning
3. GenAI Support Assistant

The project demonstrates web scraping, data cleaning, relational database design, SQL analysis, exploratory data analysis, machine learning, model evaluation, regression, model persistence, embeddings, RAG, LangGraph, MOCK_LLM, and FastAPI API development.

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
    ├── README.md
    ├── api.py
    ├── faq_matcher.py
    ├── mock_llm.py
    ├── rag_retriever.py
    ├── support_graph.py
    │
    ├── data/
    │   └── faq.csv
    │
    └── output/
        ├── complete_support_results.txt
        ├── langgraph_rag_results.txt
        └── rag_retrieval_results.txt
```

---

# Module 1 — Data Pipeline

The Data Pipeline module collects book data from Books to Scrape and processes it into structured datasets and a relational SQLite database.

## Features

* Web scraping using Python
* Scraping across multiple pages
* Book title extraction
* Price extraction
* Rating extraction
* Availability extraction
* Category extraction
* Data cleaning
* GBP to INR conversion
* SQLite database creation
* Normalized relational tables
* SQL querying
* Pandas SQL integration
* Pandas merge validation

## Dataset

The pipeline successfully collected 100 books from the first five product pages.

The conversion used:

```text
1 GBP = ₹105.50
```

## Database

SQLite database:

```text
data_pipeline/data/zepto_books.db
```

Tables:

```text
categories
books
```

The `books` table uses a foreign key to connect each book with its category.

## SQL Analysis

The module demonstrates:

* SELECT and WHERE
* ORDER BY
* LIMIT
* DISTINCT
* IN
* BETWEEN
* JOIN

SQL results are stored in:

```text
data_pipeline/data/sql_results.txt
```

Pandas SQL and merge validation results are stored in:

```text
data_pipeline/data/pandas_merge_results.txt
```

---

# Module 2 — Analytics & Machine Learning

The Analytics module uses the Titanic dataset to demonstrate data profiling, data cleaning, exploratory data analysis, machine learning classification, class imbalance handling, hyperparameter tuning, regression, and model persistence.

## Dataset

The Titanic dataset is loaded using:

```python
sns.load_dataset("titanic")
```

The dataset is saved as:

```text
analytics/data/titanic.csv
```

## Data Cleaning

Missing values were analyzed and handled using appropriate strategies.

Examples include:

* Median imputation for `age`
* Handling missing `embarked`
* Handling missing `embark_town`
* Removing `deck` because of its high missing-value percentage

A cleaned dataset is saved as:

```text
analytics/data/titanic_cleaned.csv
```

## Exploratory Data Analysis

The project includes:

* Dataset profiling
* Descriptive statistics
* Missing-value analysis
* Univariate analysis
* Survival analysis
* Correlation analysis
* Multivariate analysis
* Data visualizations

## Correlation Analysis

The required six numerical columns were analyzed:

```text
survived
pclass
age
sibsp
parch
fare
```

## Classification Models

Three classification algorithms were evaluated:

* Logistic Regression
* Decision Tree
* Random Forest

Evaluation metrics include:

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

## Imbalance Handling

Class imbalance was analyzed using:

* Baseline model
* Class-weight balancing
* SMOTE

SMOTE achieved the strongest F1 score among the tested imbalance-handling approaches.

## Random Forest GridSearch

Random Forest hyperparameters were optimized using `GridSearchCV`.

Best parameters:

```text
max_depth = 5
max_features = sqrt
n_estimators = 100
```

The model was evaluated using cross-validation, test-set metrics, and out-of-bag scoring.

## Fare Regression

A regression workflow was implemented to predict Titanic passenger fare-related values and evaluate regression performance.

## Model Persistence

The best classification pipeline was saved using Joblib:

```text
analytics/output/best_classification_pipeline.joblib
```

The saved pipeline was reloaded and tested successfully.

---

# Module 3 — GenAI Support Assistant

The Support Assistant module provides an offline customer-support question-answering workflow using a FAQ knowledge base, embeddings, RAG retrieval, LangGraph, MOCK_LLM, and FastAPI.

## Features

* FAQ knowledge base
* TF-IDF FAQ matching
* Semantic embeddings
* Cosine similarity
* Retrieval-Augmented Generation (RAG)
* Top-3 FAQ retrieval
* LangGraph workflow
* Offline MOCK_LLM
* FastAPI REST API
* Swagger API documentation
* End-to-end testing

## Support Assistant Structure

```text
support_assistant/
├── README.md
├── api.py
├── faq_matcher.py
├── mock_llm.py
├── rag_retriever.py
├── support_graph.py
│
├── data/
│   └── faq.csv
│
└── output/
    ├── complete_support_results.txt
    ├── langgraph_rag_results.txt
    └── rag_retrieval_results.txt
```

## FAQ Knowledge Base

The knowledge base contains 10 customer-support FAQs covering:

* Orders
* Delivery
* Refunds
* Products
* Support
* Payments

File:

```text
support_assistant/data/faq.csv
```

## RAG Retrieval

The RAG system uses FastEmbed with:

```text
BAAI/bge-small-en-v1.5
```

The embedding dimension is:

```text
384
```

The system:

1. Loads the FAQ knowledge base.
2. Creates embeddings for FAQ questions.
3. Converts the user question into an embedding.
4. Calculates cosine similarity.
5. Retrieves the top 3 relevant FAQs.
6. Passes the retrieved information to the support workflow.

## LangGraph

LangGraph is used to create the support-assistant workflow.

The workflow connects:

```text
User Question
      ↓
FAQ / RAG Retrieval
      ↓
Relevant Support Information
      ↓
MOCK_LLM Response
      ↓
Final Support Response
```

## MOCK_LLM

The project includes an offline deterministic response layer using:

```text
MOCK_LLM=1
```

This provides a free offline baseline without requiring a paid LLM API.

## FastAPI

The Support Assistant is exposed through a FastAPI application.

Run the API with:

```powershell
uvicorn support_assistant.api:app --reload
```

Default local address:

```text
http://127.0.0.1:8000
```

## API Endpoints

### Root

```text
GET /
```

Checks whether the Support Assistant API is running.

### Health

```text
GET /health
```

Checks the health of the RAG and MOCK_LLM components.

### Ask Support Question

```text
POST /ask
```

Example request:

```json
{
  "question": "I want my money back"
}
```

The API returns:

* User question
* Matched FAQ question
* Answer
* Category
* Similarity score
* Number of retrieved FAQs
* Final response

## Swagger Documentation

FastAPI automatically provides interactive API documentation at:

```text
http://127.0.0.1:8000/docs
```

## Testing

The Support Assistant was tested with questions related to:

* Order cancellation
* Order tracking
* Refund requests
* Payment failures

The tests verify:

* FAQ retrieval
* Similarity scores
* Top-3 retrieval
* LangGraph execution
* MOCK_LLM response generation
* FastAPI API responses

Output files:

```text
support_assistant/output/rag_retrieval_results.txt
support_assistant/output/langgraph_rag_results.txt
support_assistant/output/complete_support_results.txt
```

---

# Technology Stack

## Data Engineering

* Python
* Requests
* BeautifulSoup
* Pandas
* SQLite
* SQL

## Analytics & Machine Learning

* Python
* Pandas
* NumPy
* Seaborn
* Matplotlib
* Scikit-learn
* SMOTE
* GridSearchCV
* Joblib

## GenAI & Support Assistant

* Python
* Pandas
* TF-IDF
* FastEmbed
* Embeddings
* Cosine Similarity
* RAG
* LangGraph
* MOCK_LLM
* FastAPI
* Pydantic
* Uvicorn

---

# Git Workflow

The project uses Git for version control.

The repository includes:

* Feature branch development
* Multiple commits
* Merge commit
* Main branch
* GitHub remote repository

The Git history demonstrates the feature-branch workflow and merge back into `main`.

---

# Repository

GitHub repository:

```text
manu0303-manu/zepto-data-ai-platform
```

The repository contains all three modules in a single public repository.

---

# Project Outcomes

This capstone demonstrates an end-to-end technical workflow:

```text
Web Data
   ↓
Data Collection
   ↓
Data Cleaning
   ↓
SQLite Database
   ↓
SQL Analysis
   ↓
Exploratory Data Analysis
   ↓
Machine Learning
   ↓
Model Evaluation
   ↓
Model Persistence
   ↓
FAQ Knowledge Base
   ↓
Embeddings
   ↓
RAG Retrieval
   ↓
LangGraph
   ↓
MOCK_LLM
   ↓
FastAPI
```

The project combines Data Engineering, Analytics, Machine Learning, and Generative AI concepts into one integrated platform.

---

# Conclusion

The Zepto Data & AI Platform provides a complete end-to-end demonstration of modern data and AI workflows, from raw data collection and relational storage to machine learning and an API-based RAG customer-support assistant.

The project is designed as a practical AI/ML capstone demonstrating reproducible data processing, analytical modeling, machine learning evaluation, semantic retrieval, workflow orchestration, and API deployment.
