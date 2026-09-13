import numpy as np
from sentence_transformers import SentenceTransformer

from catalogue import COURSES
from recommend import recommend_courses
from competency import build_profile_gaps

print("[Pipeline A] Loading embedding model (all-MiniLM-L6-v2)...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print(f"[Pipeline A] Embedding {len(COURSES)} courses from the catalogue...")

_catalogue_texts = [
    f"{c['title']}. {c['description']}"
    for c in COURSES
]

CATALOGUE_VECTORS = np.array(
    model.encode(_catalogue_texts)
)

print(
    "[Pipeline A] Catalogue embeddings ready. Vector dim:",
    CATALOGUE_VECTORS.shape[1],
)

def get_recommendations(
    gap_description: str,
    competency: str,
    official_current_level: int,
    required_level: int,
    top_k: int = 3,
) -> list[dict]:
    """
    Generate course recommendations for one competency gap.

    Task 3 ranking considers:
        - Semantic similarity
        - Competency/domain relevance
        - Level appropriateness
        - Difficulty match
        - Diversity through MMR

    The actual ranking logic is implemented in recommend.py.
    """

    results = recommend_courses(
        gap_description=gap_description,
        competency=competency,
        official_current_level=official_current_level,
        required_level=required_level,
        model=model,
        catalogue_courses=COURSES,
        catalogue_vectors=CATALOGUE_VECTORS,
        top_k=top_k,
    )

    return [
        {
            "id": r["id"],
            "title": r["title"],
            "level": r["level"],
            "difficulty": r.get("difficulty", "Unknown"),
            "domain": r.get("domain", "Unknown"),
            "description": r["description"],

            # Original semantic similarity
            "similarity": round(r["similarity"], 4),

            # Task 3 ranking components
            "competency_score": round(
                r.get("competency_score", 0.0),
                4,
            ),

            "level_score": round(
                r.get("level_score", 0.0),
                4,
            ),

            "difficulty_score": round(
                r.get("difficulty_score", 0.0),
                4,
            ),

            # Final ranking score
            "ranking_score": round(
                r["ranking_score"],
                4,
            ),
        }
        for r in results
    ]
def get_profile_recommendations(
    official_profile: dict,
    top_k: int = 3,
) -> dict:
    """
    Complete MVP recommendation pipeline.

    Official Profile
            ↓
    Competency Gap Engine
            ↓
    Gap Description
            ↓
    Recommendation Engine
            ↓
    Task 3 Ranking
            ↓
    Top-K Recommendations
    """

    # Step 1:
    # Convert official profile into competency gaps.
    gaps = build_profile_gaps(official_profile)

    results = []

    # Step 2:
    # Generate recommendations for every identified gap.
    for gap in gaps:

        recommendations = get_recommendations(
            gap_description=gap["description"],
            competency=gap["competency"],
            official_current_level=gap["current_level"],
            required_level=gap["required_level"],
            top_k=top_k,
        )

        results.append(
            {
                "competency": gap["competency"],
                "required_level": gap["required_level"],
                "current_level": gap["current_level"],
                "gap": gap["gap"],
                "confidence": gap["confidence"],
                "evidence": gap["evidence"],
                "recommendations": recommendations,
            }
        )

    return {
        "official_id": official_profile["official_id"],
        "role": official_profile["role"],
        "gaps": results,
    }

# MVP Test

if __name__ == "__main__":
    official_profile = {

        "official_id": "OFF-001",

        "role": "Statistical Officer",

        "competencies": [

            {
                "name": "Machine Learning",

                "required_level": 3,

                "current_level": 1,

                "confidence": 0.8,

                "evidence": (
                    "Completed basic statistics training "
                    "but has no machine learning project experience"
                ),
            },

            {
                "name": "Survey Methodology",

                "required_level": 3,

                "current_level": 1,

                "confidence": 0.9,

                "evidence": (
                    "Completed basic survey training"
                ),
            },

            {
                "name": "Data Analytics",

                "required_level": 3,

                "current_level": 1,

                "confidence": 0.85,

                "evidence": (
                    "Has basic statistics knowledge but limited "
                    "experience applying data analytics"
                ),
            },

            {
                "name": "Project Management",

                "required_level": 3,

                "current_level": 1,

                "confidence": 0.8,

                "evidence": (
                    "Has participated in projects but has limited "
                    "formal project management training"
                ),
            },
        ],
    }

    profile_result = get_profile_recommendations(
        official_profile,
        top_k=3,
    )
    print("\n")
    print("========================================")
    print("        OFFICIAL TRAINING PROFILE")
    print("========================================")

    print(
        "Official ID:",
        profile_result["official_id"],
    )

    print(
        "Role:",
        profile_result["role"],
    )

    for gap in profile_result["gaps"]:

        print("\n")
        print("----------------------------------------")

        print(
            "Competency:",
            gap["competency"],
        )

        print(
            "Required level:",
            gap["required_level"],
        )

        print(
            "Current level:",
            gap["current_level"],
        )

        print(
            "Gap:",
            gap["gap"],
        )

        print(
            "Confidence:",
            gap["confidence"],
        )

        print(
            "Evidence:",
            gap["evidence"],
        )


        print("\nRecommended courses:")
        for rank, course in enumerate(
            gap["recommendations"],
            start=1,
        ):

            print(
                f"  {rank}. "
                f"Score: {course['ranking_score']:.3f} | "
                f"Similarity: {course['similarity']:.3f} | "
                f"Competency: {course['competency_score']:.3f} | "
                f"Level: {course['level_score']:.3f} | "
                f"Difficulty: {course['difficulty_score']:.3f}"
            )

            print(
                f"     {course['id']} | "
                f"L{course['level']} | "
                f"{course['difficulty']} | "
                f"{course['title']}"
            )

            print(
                f"     Domain: {course['domain']}"
            )

            print(
                f"     {course['description']}"
            )