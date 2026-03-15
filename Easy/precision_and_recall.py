# Precision and Recall
# Topic: Information Retrieval

def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    if k <= 0:
        raise ValueError("k must be positive")

    relevant_set = set(relevant)
    top_k = recommended[:k]

    hits = sum(1 for item in top_k if item in relevant_set)

    precision = hits / k
    recall = hits / len(relevant_set) if len(relevant_set) > 0 else 0.0

    return [precision, recall]
