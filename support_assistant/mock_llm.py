import os


MOCK_LLM = os.getenv("MOCK_LLM", "1")


def generate_mock_response(
    user_question,
    matched_question,
    answer,
    category,
    similarity_score
):
    """
    Generate a deterministic offline support response.

    This function is used when MOCK_LLM=1.
    No external API or internet service is required.
    """

    if MOCK_LLM != "1":
        raise RuntimeError(
            "MOCK_LLM is not enabled. "
            "Set MOCK_LLM=1 for offline mode."
        )

    response = (
        f"Based on your question, the relevant support topic "
        f"is '{category}'.\n\n"
        f"Answer: {answer}\n\n"
        f"Reference FAQ: {matched_question}\n"
        f"Retrieval confidence: {similarity_score:.4f}"
    )

    return response


def main():
    print("=" * 70)
    print("ZEpto Support Assistant - MOCK_LLM")
    print("=" * 70)

    print()
    print(f"MOCK_LLM mode: {MOCK_LLM}")

    user_question = "How can I cancel my order?"

    matched_question = "How can I cancel my order?"

    answer = (
        "You can cancel your order before it is "
        "dispatched from the order section."
    )

    category = "Orders"

    similarity_score = 0.9940

    response = generate_mock_response(
        user_question=user_question,
        matched_question=matched_question,
        answer=answer,
        category=category,
        similarity_score=similarity_score
    )

    print()
    print("User question:")
    print(user_question)

    print()
    print("MOCK_LLM response:")
    print(response)

    print()
    print("=" * 70)
    print("MOCK_LLM TEST COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()