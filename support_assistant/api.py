from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from support_assistant.support_graph import build_support_graph


app = FastAPI(
    title="Zepto Support Assistant API",
    description="Offline RAG-based customer support API with MOCK_LLM.",
    version="1.0.0",
)


class SupportRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        description="Customer support question",
    )


class SupportResponse(BaseModel):
    question: str
    matched_question: str
    answer: str
    category: str
    similarity_score: float
    retrieved_count: int
    final_response: str


print("Loading Zepto Support Assistant...")
support_graph = build_support_graph()
print("Support graph loaded successfully.")


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "Zepto Support Assistant API is running",
        "status": "online",
        "mode": "MOCK_LLM",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "healthy",
        "rag": "ready",
        "mock_llm": "enabled",
    }


@app.post("/ask", response_model=SupportResponse)
def ask_support(request: SupportRequest) -> dict[str, Any]:
    result = support_graph.invoke(
        {
            "user_question": request.question
        }
    )

    return {
        "question": request.question,
        "matched_question": result["matched_question"],
        "answer": result["answer"],
        "category": result["category"],
        "similarity_score": result["similarity_score"],
        "retrieved_count": result["retrieved_count"],
        "final_response": result["final_response"],
    }