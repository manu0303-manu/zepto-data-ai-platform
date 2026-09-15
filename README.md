## Project Structure

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
