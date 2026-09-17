def calculate_gap(required_level: int, current_level: int) -> int:
    """
    Calculate the competency gap.
    """

    if not 0 <= required_level <= 5:
        raise ValueError("required_level must be between 0 and 5")

    if not 0 <= current_level <= 5:
        raise ValueError("current_level must be between 0 and 5")

    return max(0, required_level - current_level)


def build_competency_gap(competency: dict) -> dict:
    """
    Convert one competency from an official profile
    into a structured competency gap.
    """

    required_level = competency["required_level"]
    current_level = competency["current_level"]

    gap = calculate_gap(
        required_level,
        current_level,
    )

    description = (
        f"{competency['name']} competency gap. "
        f"Required level: {required_level}. "
        f"Current level: {current_level}. "
        f"Gap: {gap}. "
        f"Evidence: {competency['evidence']}"
    )

    return {
        "competency": competency["name"],
        "required_level": required_level,
        "current_level": current_level,
        "gap": gap,
        "confidence": competency["confidence"],
        "evidence": competency["evidence"],
        "description": description,
    }


def build_profile_gaps(profile: dict) -> list[dict]:
    """
    Calculate gaps for every competency in an official profile.
    """

    gaps = []

    for competency in profile["competencies"]:
        gap = build_competency_gap(competency)

    
        if gap["gap"] > 0:
            gaps.append(gap)

    return gaps


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
                "evidence": "Completed basic survey training",
            },
        ],
    }

    gaps = build_profile_gaps(official_profile)

    print("\n=== Official Profile ===")
    print("ID:", official_profile["official_id"])
    print("Role:", official_profile["role"])

    print("\n=== Competency Gaps ===")

    for gap in gaps:
        print(
            f"{gap['competency']}: "
            f"current={gap['current_level']}, "
            f"required={gap['required_level']}, "
            f"gap={gap['gap']}"
        )
