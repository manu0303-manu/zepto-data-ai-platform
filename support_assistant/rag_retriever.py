import os
import numpy as np
import pandas as pd
from fastembed import TextEmbedding


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAQ_FILE = os.path.join(BASE_DIR, "data", "faq.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
RESULT_FILE = os.path.join(OUTPUT_DIR, "rag_retrieval_results.txt")


MODEL_NAME = "BAAI/bge-small-en-v1.5"
TOP_K = 3


def load_faq():
    """Load the FAQ knowledge base."""
    return pd.read_csv(FAQ_FILE)


def load_embedding_model():
    """Load the local FastEmbed model."""
    return TextEmbedding(MODEL_NAME)


def create_faq_embeddings(model, faq_df):
    """Create embeddings for all FAQ questions."""
    questions = faq_df["question"].astype(str).tolist()
    embeddings = list(model.embed(questions))
    return np.array(embeddings)


def cosine_similarity(query_vector, document_vectors):
    """Calculate cosine similarity between one query and FAQ embeddings."""
    query_norm = np.linalg.norm(query_vector)
    document_norms = np.linalg.norm(document_vectors, axis=1)

    if query_norm == 0:
        return np.zeros(len(document_vectors))

    safe_document_norms = np.where(document_norms == 0, 1, document_norms)

    scores = np.dot(document_vectors, query_vector)
    scores = scores / (safe_document_norms * query_norm)

    return scores


def retrieve_top_faqs(
    user_question,
    model,
    faq_df,
    faq_embeddings,
    top_k=TOP_K
):
    """Retrieve the most relevant FAQ records."""
    query_embedding = np.array(
        list(model.embed([user_question]))[0]
    )

    scores = cosine_similarity(
        query_embedding,
        faq_embeddings
    )

    ranked_indexes = np.argsort(scores)[::-1][:top_k]

    results = faq_df.iloc[ranked_indexes].copy()
    results["similarity_score"] = scores[ranked_indexes]

    return results.reset_index(drop=True)


def save_results(user_question, results):
    """Save retrieval results to a text file."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(RESULT_FILE, "a", encoding="utf-8") as file:
        file.write("=" * 70 + "\n")
        file.write(f"User question: {user_question}\n")
        file.write("-" * 70 + "\n")

        for index, row in results.iterrows():
            file.write(
                f"Rank: {index + 1}\n"
                f"FAQ: {row['question']}\n"
                f"Category: {row['category']}\n"
                f"Similarity: {row['similarity_score']:.4f}\n"
                f"Answer: {row['answer']}\n"
                + "-" * 70
                + "\n"
            )

        file.write("\n")


def main():
    print("=" * 70)
    print("ZEpto Support Assistant - Embedding RAG Retriever")
    print("=" * 70)

    faq_df = load_faq()

    print(f"Knowledge base records: {len(faq_df)}")
    print(f"Embedding model: {MODEL_NAME}")

    model = load_embedding_model()

    faq_embeddings = create_faq_embeddings(
        model,
        faq_df
    )

    print(f"Embedding matrix shape: {faq_embeddings.shape}")

    test_questions = [
        "How do I cancel my order?",
        "Where can I track my delivery?",
        "I need to get my money back",
        "My payment did not work"
    ]

    if os.path.exists(RESULT_FILE):
        os.remove(RESULT_FILE)

    for question in test_questions:
        print()
        print("-" * 70)
        print(f"User question: {question}")

        results = retrieve_top_faqs(
            question,
            model,
            faq_df,
            faq_embeddings,
            TOP_K
        )

        best_result = results.iloc[0]

        print()
        print("Top matching FAQ:")
        print(f"Question: {best_result['question']}")
        print(f"Category: {best_result['category']}")
        print(f"Similarity: {best_result['similarity_score']:.4f}")
        print(f"Answer: {best_result['answer']}")

        print()
        print("Top 3 retrieved FAQs:")

        for index, row in results.iterrows():
            print(
                f"{index + 1}. "
                f"{row['question']} "
                f"(score={row['similarity_score']:.4f})"
            )

        save_results(question, results)

    print()
    print("=" * 70)
    print("EMBEDDING RAG RETRIEVAL TEST COMPLETED")
    print("=" * 70)
    print(f"Results saved to: {RESULT_FILE}")


if __name__ == "__main__":
    main()