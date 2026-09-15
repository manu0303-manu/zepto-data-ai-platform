import os
from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from support_assistant.rag_retriever import (
    load_faq,
    load_embedding_model,
    create_faq_embeddings,
    retrieve_top_faqs,
)

from support_assistant.mock_llm import generate_mock_response


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

RESULT_FILE = os.path.join(
    OUTPUT_DIR,
    "complete_support_results.txt"
)

TOP_K = 3


class SupportState(TypedDict):
    user_question: str
    matched_question: str
    answer: str
    category: str
    similarity_score: float
    retrieved_count: int
    final_response: str


def load_rag_resources():
    """Load FAQ data, embedding model, and FAQ embeddings."""

    faq_df = load_faq()

    embedding_model = load_embedding_model()

    faq_embeddings = create_faq_embeddings(
    embedding_model,
    faq_df
        )
    

    return (
        faq_df,
        embedding_model,
        faq_embeddings
    )


def rag_retrieval_node(
    state: SupportState,
    faq_df,
    embedding_model,
    faq_embeddings
):
    """Retrieve the most relevant FAQ using embeddings."""

    results = retrieve_top_faqs(   
    user_question=state["user_question"],
    model=embedding_model,
    faq_df=faq_df,
    faq_embeddings=faq_embeddings,
    top_k=TOP_K
)

    best_result = results.iloc[0]

    return {
        "matched_question": best_result["question"],
        "answer": best_result["answer"],
        "category": best_result["category"],
        "similarity_score": float(
            best_result["similarity_score"]
        ),
        "retrieved_count": len(results)
    }


def mock_llm_node(state: SupportState):
    """Generate the final offline support response."""

    response = generate_mock_response(
        user_question=state["user_question"],
        matched_question=state["matched_question"],
        answer=state["answer"],
        category=state["category"],
        similarity_score=state["similarity_score"]
    )

    return {
        "final_response": response
    }


def build_support_graph(
    faq_df=None,
    embedding_model=None,
    faq_embeddings=None
):
    """Build the complete LangGraph RAG + MOCK_LLM workflow."""

    if (
        faq_df is None
        or embedding_model is None
        or faq_embeddings is None
    ):
        (
            faq_df,
            embedding_model,
            faq_embeddings
        ) = load_rag_resources()

    workflow = StateGraph(SupportState)

    def retrieval_node(state: SupportState):
        return rag_retrieval_node(
            state,
            faq_df,
            embedding_model,
            faq_embeddings
        )

    workflow.add_node(
        "rag_retrieval",
        retrieval_node
    )

    workflow.add_node(
        "mock_llm",
        mock_llm_node
    )

    workflow.add_edge(
        START,
        "rag_retrieval"
    )

    workflow.add_edge(
        "rag_retrieval",
        "mock_llm"
    )

    workflow.add_edge(
        "mock_llm",
        END
    )

    return workflow.compile()


def save_result(
    question,
    result
):
    """Save complete support workflow result."""

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    with open(
        RESULT_FILE,
        "a",
        encoding="utf-8"
    ) as file:

        file.write("=" * 70 + "\n")

        file.write(
            f"User question: {question}\n"
        )

        file.write(
            f"Matched FAQ: "
            f"{result['matched_question']}\n"
        )

        file.write(
            f"Category: "
            f"{result['category']}\n"
        )

        file.write(
            f"Similarity score: "
            f"{result['similarity_score']:.4f}\n"
        )

        file.write(
            f"Retrieved FAQs: "
            f"{result['retrieved_count']}\n"
        )

        file.write(
            "Final MOCK_LLM response:\n"
        )

        file.write(
            result["final_response"]
        )

        file.write("\n\n")


def main():

    print("=" * 70)

    print(
        "ZEpto Support Assistant - "
        "Complete Offline RAG + MOCK_LLM"
    )

    print("=" * 70)

    print()

    print(
        "MOCK_LLM mode:",
        os.getenv("MOCK_LLM", "1")
    )

    print()

    print(
        "Loading knowledge base and embeddings..."
    )

    (
        faq_df,
        embedding_model,
        faq_embeddings
    ) = load_rag_resources()

    print(
        f"Knowledge base records: "
        f"{len(faq_df)}"
    )

    print(
        f"Embedding matrix shape: "
        f"{faq_embeddings.shape}"
    )

    graph = build_support_graph(
        faq_df,
        embedding_model,
        faq_embeddings
    )

    test_questions = [
        "How do I cancel my order?",
        "Where can I track my delivery?",
        "I need to get my money back",
        "My payment did not work"
    ]

    if os.path.exists(RESULT_FILE):
        os.remove(RESULT_FILE)

    for question in test_questions:

        result = graph.invoke(
            {
                "user_question": question,
                "matched_question": "",
                "answer": "",
                "category": "",
                "similarity_score": 0.0,
                "retrieved_count": 0,
                "final_response": ""
            }
        )

        print()
        print("-" * 70)

        print(
            f"User question: {question}"
        )

        print()

        print(
            f"Retrieved FAQ: "
            f"{result['matched_question']}"
        )

        print(
            f"Category: "
            f"{result['category']}"
        )

        print(
            f"Similarity score: "
            f"{result['similarity_score']:.4f}"
        )

        print(
            f"Retrieved FAQs: "
            f"{result['retrieved_count']}"
        )

        print()

        print(
            "Final MOCK_LLM response:"
        )

        print(
            result["final_response"]
        )

        save_result(
            question,
            result
        )

    print()

    print("=" * 70)

    print(
        "RAG + MOCK_LLM COMPLETE WORKFLOW TEST COMPLETED"
    )

    print("=" * 70)

    print(
        f"Results saved to: "
        f"{RESULT_FILE}"
    )


if __name__ == "__main__":
    main()