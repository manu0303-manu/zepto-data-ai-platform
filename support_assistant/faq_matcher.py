import os
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAQ_FILE = os.path.join(BASE_DIR, "data", "faq.csv")


def load_faq():
    """Load FAQ knowledge base."""
    return pd.read_csv(FAQ_FILE)


def build_matcher(faq_df):
    """Create TF-IDF vectors from FAQ questions."""
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english"
    )

    question_vectors = vectorizer.fit_transform(
        faq_df["question"].astype(str)
    )

    return vectorizer, question_vectors


def find_best_answer(user_question, faq_df, vectorizer, question_vectors):
    """Find the most similar FAQ question."""

    user_vector = vectorizer.transform([user_question])

    similarities = cosine_similarity(
        user_vector,
        question_vectors
    ).flatten()

    best_index = similarities.argmax()
    best_score = similarities[best_index]

    matched_question = faq_df.iloc[best_index]["question"]
    answer = faq_df.iloc[best_index]["answer"]
    category = faq_df.iloc[best_index]["category"]

    return {
        "question": matched_question,
        "answer": answer,
        "category": category,
        "score": float(best_score)
    }


def main():
    print("=" * 60)
    print("ZEpto Support Assistant - Offline FAQ Matcher")
    print("=" * 60)

    faq_df = load_faq()

    vectorizer, question_vectors = build_matcher(faq_df)

    print(f"Knowledge base records: {len(faq_df)}")
    print()

    test_questions = [
        "How do I cancel my order?",
        "Where can I see my delivery?",
        "I want a refund",
        "My payment failed"
    ]

    for question in test_questions:

        result = find_best_answer(
            question,
            faq_df,
            vectorizer,
            question_vectors
        )

        print("-" * 60)
        print(f"User question: {question}")
        print(f"Matched FAQ: {result['question']}")
        print(f"Category: {result['category']}")
        print(f"Similarity score: {result['score']:.4f}")
        print(f"Answer: {result['answer']}")

    print()
    print("=" * 60)
    print("OFFLINE FAQ MATCHER TEST COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()