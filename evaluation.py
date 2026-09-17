import numpy as np
from main import get_recommendations

EVALUATION_DATA = [
    {
        "competency": "Survey Methodology",
        "expected_courses": ["SUR-001", "SUR-002"],
    },
    {
        "competency": "Statistics",
        "expected_courses": ["STAT-001", "STAT-002"],
    },
    {
        "competency": "Machine Learning",
        "expected_courses": ["ML-001", "ML-004"],
    },
    {
        "competency": "Project Management",
        "expected_courses": ["PM-001", "PM-003"],
    },
    {
        "competency": "Data Analytics",
        "expected_courses": ["DATA-001", "DATA-009"],
    },
]

def precision_at_k(recommended, expected, k):
    recommended_k = recommended[:k]

    if not recommended_k:
        return 0.0

    relevant = sum(
        1 for course_id in recommended_k
        if course_id in expected
    )

    return relevant / len(recommended_k)


def recall_at_k(recommended, expected, k):
    recommended_k = recommended[:k]

    if not expected:
        return 0.0

    relevant = sum(
        1 for course_id in recommended_k
        if course_id in expected
    )

    return relevant / len(expected)


def ndcg_at_k(recommended, expected, k):
    recommended_k = recommended[:k]

    if not expected:
        return 0.0

    dcg = 0.0

    for rank, course_id in enumerate(recommended_k, start=1):
        if course_id in expected:
            dcg += 1 / np.log2(rank + 1)

    ideal_relevant = min(len(expected), k)

    idcg = sum(
        1 / np.log2(rank + 1)
        for rank in range(1, ideal_relevant + 1)
    )

    if idcg == 0:
        return 0.0

    return dcg / idcg

def evaluate_recommendations(recommendation_results, k=3):
    precision_scores = []
    recall_scores = []
    ndcg_scores = []

    for result, evaluation_case in zip(
        recommendation_results,
        EVALUATION_DATA,
    ):
        recommended_ids = [
            course["id"]
            for course in result
        ]

        expected_ids = evaluation_case["expected_courses"]

        precision_scores.append(
            precision_at_k(
                recommended_ids,
                expected_ids,
                k,
            )
        )

        recall_scores.append(
            recall_at_k(
                recommended_ids,
                expected_ids,
                k,
            )
        )

        ndcg_scores.append(
            ndcg_at_k(
                recommended_ids,
                expected_ids,
                k,
            )
        )

    return {
        "precision_at_k": np.mean(precision_scores),
        "recall_at_k": np.mean(recall_scores),
        "ndcg_at_k": np.mean(ndcg_scores),
    }

def generate_evaluation_results():
    results = []

    for case in EVALUATION_DATA:
        result = get_recommendations(
            gap_description=f"{case['competency']} competency gap",
            competency=case["competency"],
            official_current_level=0,
            required_level=3,
            top_k=3,
        )

        results.append(result)

    return results

if __name__ == "__main__":
    recommendation_results = generate_evaluation_results()

    metrics = evaluate_recommendations(
        recommendation_results,
        k=3,
    )

    print("\n========================================")
    print("       RECOMMENDATION EVALUATION")
    print("========================================")

    print(
        f"Precision@3: {metrics['precision_at_k']:.3f}"
    )

    print(
        f"Recall@3: {metrics['recall_at_k']:.3f}"
    )

    print(
        f"NDCG@3: {metrics['ndcg_at_k']:.3f}"
    )