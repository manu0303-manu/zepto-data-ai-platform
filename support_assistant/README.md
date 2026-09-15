# Zepto Support Assistant

An offline RAG-based customer support assistant built using LangGraph, FastEmbed embeddings, TF-IDF retrieval, and a deterministic MOCK_LLM response layer.

## 1. Project Overview

The Zepto Support Assistant answers customer support questions by searching a structured FAQ knowledge base.

The workflow uses Retrieval-Augmented Generation (RAG) to:

1. Receive a customer question.
2. Convert FAQ questions and the user question into embeddings.
3. Retrieve the most relevant FAQ records.
4. Identify the best matching support topic.
5. Generate a deterministic response using MOCK_LLM.
6. Return the result through a FastAPI REST API.

The system is designed to work locally without requiring a paid LLM service.

## 2. Features

* FAQ knowledge base stored in CSV format.
* TF-IDF based FAQ matching.
* FastEmbed semantic embeddings.
* Cosine similarity based retrieval.
* Top-3 FAQ retrieval.
* LangGraph workflow orchestration.
* Offline MOCK_LLM response generation.
* FastAPI REST API.
* Swagger API documentation.
* Health-check endpoint.
* JSON API responses.
* No paid API required for the baseline workflow.

## 3. Project Structure

```text
support_assistant/
├── data/
│   └── faq.csv
├── output/
│   ├── rag_retrieval_results.txt
│   ├── langgraph_rag_results.txt
│   └── complete_support_results.txt
├── faq_matcher.py
├── rag_retriever.py
├── support_graph.py
├── mock_llm.py
├── api.py
└── README.md
```

## 4. Knowledge Base

The FAQ knowledge base contains customer support questions and answers covering:

* Orders
* Delivery
* Refunds
* Products
* Support
* Payments

The file is:

```text
data/faq.csv
```

The current knowledge base contains 10 FAQ records.

## 5. RAG Pipeline

The RAG pipeline uses FastEmbed with the following embedding model:

```text
BAAI/bge-small-en-v1.5
```

The workflow creates embeddings for the FAQ questions and compares them with the embedding of the user's question.

Cosine similarity is used to rank the FAQ records.

The system retrieves the top 3 relevant FAQ records.

## 6. LangGraph Workflow

The support assistant uses LangGraph to organize the workflow.

The main stages are:

```text
User Question
      ↓
RAG Retrieval
      ↓
Top-3 FAQ Results
      ↓
Best FAQ Selection
      ↓
MOCK_LLM Response
      ↓
Final Support Response
```

## 7. MOCK_LLM Mode

The baseline system uses an offline deterministic MOCK_LLM mode.

Environment variable:

```text
MOCK_LLM=1
```

This allows the complete support workflow to run without depending on an external paid LLM API.

The generated response includes:

* Support category
* Answer
* Reference FAQ
* Retrieval confidence

## 8. FastAPI

The support assistant is exposed through a FastAPI application.

Start the API from the project root:

```powershell
uvicorn support_assistant.api:app --reload
```

The server runs at:

```text
http://127.0.0.1:8000
```

## 9. API Endpoints

### GET /

Checks whether the API is running.

Example response:

```json
{
  "message": "Zepto Support Assistant API is running",
  "status": "online",
  "mode": "MOCK_LLM"
}
```

### GET /health

Checks the application health.

Example response:

```json
{
  "status": "healthy",
  "rag": "ready",
  "mock_llm": "enabled"
}
```

### POST /ask

Accepts a customer support question.

Request:

```json
{
  "question": "I want my money back"
}
```

Example response:

```json
{
  "question": "I want my money back",
  "matched_question": "How can I request a refund?",
  "answer": "Open the order details and select the refund option if the order is eligible for a refund.",
  "category": "Refunds",
  "similarity_score": 0.7911,
  "retrieved_count": 3,
  "final_response": "Based on your question, the relevant support topic is 'Refunds'."
}
```

## 10. Swagger Documentation

FastAPI automatically provides interactive API documentation.

Open:

```text
http://127.0.0.1:8000/docs
```

The `/docs` page can be used to test:

* GET `/`
* GET `/health`
* POST `/ask`

## 11. Testing

The system was tested with multiple support questions.

Example questions:

```text
How can I cancel my order?
How can I track my order?
I want my money back
My payment did not work
```

The RAG system successfully retrieved relevant FAQ records for these questions.

The API was also tested successfully using Swagger with HTTP `200 OK` responses.

## 12. Output Files

The workflow generates test and retrieval results in:

```text
support_assistant/output/
```

Important output files include:

```text
rag_retrieval_results.txt
langgraph_rag_results.txt
complete_support_results.txt
```

These files provide evidence of the RAG and support assistant workflow execution.

## 13. Technology Stack

* Python
* Pandas
* NumPy
* scikit-learn
* TF-IDF
* FastEmbed
* LangGraph
* FastAPI
* Uvicorn
* Pydantic
* MOCK_LLM
* SQLite/CSV-based local data handling

## 14. Offline Baseline

The baseline implementation does not require a paid LLM API.

The fol
