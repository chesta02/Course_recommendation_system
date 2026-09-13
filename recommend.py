import re
import numpy as np

# Similarity

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""

    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    denom = np.linalg.norm(a) * np.linalg.norm(b)

    if denom == 0:
        return 0.0

    return float(np.dot(a, b) / denom)


def normalize_similarity(score: float) -> float:
    """Convert cosine similarity from [-1, 1] to [0, 1]."""

    return (score + 1.0) / 2.0


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> set[str]:
    """Convert text into lowercase words."""

    text = text.lower()

    words = re.findall(r"[a-z0-9]+", text)

    return set(words)

# Competency aliases

COMPETENCY_ALIASES = {
    "machine learning": {
        "machine learning",
        "artificial intelligence",
        "ai",
        "ml",
        "deep learning",
        "predictive modeling",
    },

    "data analytics": {
        "data analytics",
        "data analysis",
        "analytics",
        "data visualization",
        "data interpretation",
        "big data",
    },

    "survey methodology": {
        "survey methodology",
        "survey design",
        "survey",
        "sampling",
        "questionnaire",
    },

    "project management": {
        "project management",
        "project work",
        "project planning",
        "project monitoring",
        "ict projects",
    },

    "statistics": {
        "statistics",
        "statistical analysis",
        "statistical methods",
        "data analysis",
    },

    "leadership": {
        "leadership",
        "management",
        "team leadership",
        "team building",
    },

    "communication": {
        "communication",
        "presentation",
        "presentation skills",
        "public speaking",
    },
}


def get_competency_terms(competency: str) -> set[str]:
    """
    Return useful terms for a competency.

    If the competency has predefined aliases, use them.
    Otherwise use the competency itself.
    """

    normalized = competency.lower().strip()

    if normalized in COMPETENCY_ALIASES:
        return COMPETENCY_ALIASES[normalized]

    return {normalized}

# Competency / domain relevance
def competency_relevance(
    competency: str,
    course: dict,
) -> float:
    """
    Measure how strongly a course is related to the target competency.

    Priority:
        1. Exact competency/alias match
        2. Strong word overlap
        3. Otherwise low relevance
    """

    domain = str(course.get("domain", "")).lower()
    title = str(course.get("title", "")).lower()

    title_words = normalize_text(title)
    domain_words = normalize_text(domain)

    competency_terms = get_competency_terms(competency)

    # 1. Strong signal: competency/alias appears as
    #    complete words in title or domain


    for term in competency_terms:

        term_words = normalize_text(term)

        if not term_words:
            continue

        if term_words.issubset(domain_words):
            return 1.0

        if term_words.issubset(title_words):
            return 1.0

    # 2. Fallback: original competency word overlap


    competency_words = normalize_text(competency)

    course_words = title_words | domain_words

    if not competency_words:
        return 0.0

    overlap = (
        len(competency_words & course_words)
        / len(competency_words)
    )

    if overlap >= 0.75:
        return 0.8

    if overlap >= 0.5:
        return 0.6

    if overlap > 0:
        return 0.3

    return 0.0
# Level matching
def level_match(
    course_level: int,
    current_level: int,
    required_level: int,
) -> float:
    """
    Prefer courses that represent the next reasonable learning step.

    Example:

        current = 1
        required = 3

    Ideal next level = 2

        level 1 -> good
        level 2 -> best
        level 3 -> acceptable
    """

    target_level = min(
        current_level + 1,
        required_level,
    )

    difference = abs(
        course_level - target_level
    )

    if difference == 0:
        return 1.0

    if difference == 1:
        return 0.7

    if difference == 2:
        return 0.4

    return 0.1

# Difficulty matching

def difficulty_match(
    course_difficulty: str,
    current_level: int,
) -> float:
    """Match course difficulty with the official's current level."""

    difficulty = str(
        course_difficulty
    ).lower().strip()

    if current_level <= 1:

        preferred = {
            "beginner": 1.0,
            "intermediate": 0.7,
            "advanced": 0.3,
        }

    elif current_level == 2:

        preferred = {
            "beginner": 0.5,
            "intermediate": 1.0,
            "advanced": 0.7,
        }

    else:

        preferred = {
            "beginner": 0.3,
            "intermediate": 0.7,
            "advanced": 1.0,
        }

    return preferred.get(
        difficulty,
        0.5,
    )

# Level filtering

def filter_by_level(
    courses: list[dict],
    current_level: int,
    required_level: int,
) -> list[dict]:
    """
    Keep courses up to the required competency level.

    Example:

        current level = 1
        required level = 3

    Level 3 courses remain available.
    """

    filtered = [
        course
        for course in courses
        if course.get("level", 0) <= required_level
    ]

    # Safety fallback
    if not filtered:
        return courses

    return filtered

# Combined ranking


def calculate_ranking_score(
    similarity: float,
    competency_score: float,
    level_score: float,
    difficulty_score: float,
) -> float:
    """

        Calculate final recommendation score.

        Weights:

            45% semantic similarity
            40% competency/domain relevance
            10% level appropriateness
            5% difficulty match
    """

    similarity_score = normalize_similarity(
        similarity
    )

    score = (
        0.45 * similarity_score
        + 0.40 * competency_score
        + 0.10 * level_score
        + 0.05 * difficulty_score
    )

    return float(score)

# Course ranking


def rank_courses(
    query_vector: np.ndarray,
    competency: str,
    courses: list[dict],
    course_vectors: np.ndarray,
    current_level: int,
    required_level: int,
) -> list[dict]:
    """Rank courses using multiple recommendation signals."""

    scored = []

    for course, vector in zip(
        courses,
        course_vectors,
    ):

        # Semantic similarity
        similarity = cosine_similarity(
            query_vector,
            vector,
        )

        # Competency/domain relevance
        competency_score = competency_relevance(
            competency,
            course,
        )

        # Level match
        level_score = level_match(
            course.get("level", 1),
            current_level,
            required_level,
        )

        # Difficulty match
        difficulty_score = difficulty_match(
            course.get("difficulty", ""),
            current_level,
        )

        # Final score
        ranking_score = calculate_ranking_score(
            similarity=similarity,
            competency_score=competency_score,
            level_score=level_score,
            difficulty_score=difficulty_score,
        )

        scored.append(
            {
                **course,
                "similarity": similarity,
                "competency_score": competency_score,
                "level_score": level_score,
                "difficulty_score": difficulty_score,
                "ranking_score": ranking_score,
            }
        )

    scored.sort(
        key=lambda course: course["ranking_score"],
        reverse=True,
    )

    return scored

#MMR (Maximal Marginal Relevance) for diversity
def mmr_rerank(
    candidates: list[dict],
    candidate_vectors: np.ndarray,
    top_k: int = 3,
    lambda_param: float = 0.85,
) -> list[dict]:
    """
    Maximal Marginal Relevance.

    Keeps recommendations relevant while reducing
    near-duplicate courses.
    """

    if len(candidates) <= top_k:
        return candidates

    candidate_vectors = np.asarray(
        candidate_vectors,
        dtype=np.float32,
    )

    remaining = list(
        range(len(candidates))
    )

    selected = []

    while (
        remaining
        and len(selected) < top_k
    ):

        best_index = None
        best_score = -np.inf

        for index in remaining:

            # Relevance
            relevance = candidates[index][
                "ranking_score"
            ]

            # Redundancy
            if selected:

                redundancy = max(
                    cosine_similarity(
                        candidate_vectors[index],
                        candidate_vectors[selected_index],
                    )
                    for selected_index in selected
                )

            else:

                redundancy = 0.0

            # MMR score
            mmr_score = (
                lambda_param * relevance
                - (1 - lambda_param) * redundancy
            )

            if mmr_score > best_score:

                best_score = mmr_score
                best_index = index

        selected.append(best_index)
        remaining.remove(best_index)

    return [
        candidates[index]
        for index in selected
    ]

# Main recommendation function

def recommend_courses(
    gap_description: str,
    competency: str,
    official_current_level: int,
    required_level: int,
    model,
    catalogue_courses: list[dict],
    catalogue_vectors: np.ndarray,
    top_k: int = 3,
    mmr_lambda: float = 0.85,
    mmr_pool_size: int = 8,
) -> list[dict]:
    """
    Complete Task #3 recommendation pipeline.

    Gap
      ↓
    Embedding similarity
      ↓
    Competency relevance
      ↓
    Level matching
      ↓
    Difficulty matching
      ↓
    Combined ranking
      ↓
    MMR diversity
      ↓
    Top-K
    """
    # 1. Embed the competency gap


    query_vector = model.encode(
        gap_description
    )
    # 2. Filter courses by required level

    filtered_courses = filter_by_level(
        catalogue_courses,
        official_current_level,
        required_level,
    )
    # 3. Match vectors with filtered courses

    vector_by_id = {
        course["id"]: vector
        for course, vector in zip(
            catalogue_courses,
            catalogue_vectors,
        )
    }

    filtered_vectors = np.array(
        [
            vector_by_id[course["id"]]
            for course in filtered_courses
        ]
    )

    # 4. Multi-signal ranking

    ranked = rank_courses(
        query_vector=query_vector,
        competency=competency,
        courses=filtered_courses,
        course_vectors=filtered_vectors,
        current_level=official_current_level,
        required_level=required_level,
    )

    # 5. Take top candidates for MMR
    pool = ranked[:mmr_pool_size]

    pool_vectors = np.array(
        [
            vector_by_id[course["id"]]
            for course in pool
        ]
    )
    # 6. MMR diversity

    final_results = mmr_rerank(
        candidates=pool,
        candidate_vectors=pool_vectors,
        top_k=top_k,
        lambda_param=mmr_lambda,
    )
    # 7. Final ordering

    final_results.sort(
        key=lambda course: course["ranking_score"],
        reverse=True,
    )

    return final_results