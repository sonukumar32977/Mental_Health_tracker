"""
aggregator.py - Wellness score computation logic.
Combines emotion scores and risk level into a unified 0-100 wellness score.
"""


# Positive and negative emotion weights for emotion_score calculation
POSITIVE_EMOTIONS = {"joy": 1.0, "surprise": 0.5, "neutral": 0.3}
NEGATIVE_EMOTIONS = {"sadness": 1.0, "fear": 0.8, "anger": 0.7}

# Risk penalty values subtracted from the 100-point base
RISK_PENALTY = {
    "normal": 0,
    "anxiety": 20,
    "depression": 45,
    "high-risk": 75,
    "high_risk": 75,  # Alias for consistency
}

# Alert level thresholds
ALERT_LEVELS = {
    "good":     (70, 100),
    "moderate": (45, 69),
    "low":      (0,  44),
}

HELPLINE_MESSAGE = (
    "⚠️ **You are not alone.** If you're struggling, please reach out:\n\n"
    "- **iCall (India):** 9152987821\n"
    "- **Vandrevala Foundation:** 1860-2662-345 (24/7)\n"
    "- **NIMHANS Helpline:** 080-46110007\n"
    "- **International Association for Suicide Prevention:** https://www.iasp.info/resources/Crisis_Centres/"
)


def compute_emotion_score(emotions: dict) -> float:
    """
    Compute a 0–100 emotion score by weighting positive vs negative emotions.

    Formula:
        positive_weighted = sum(score * weight for positive emotions)
        negative_weighted = sum(score * weight for negative emotions)
        emotion_score = (positive_weighted / (positive_weighted + negative_weighted + 1e-9)) * 100

    Args:
        emotions (dict): Mapping of emotion label (str) to confidence score (float, 0-1).

    Returns:
        float: Emotion score between 0 and 100.
    """
    positive_weighted = sum(
        emotions.get(emo, 0.0) * weight
        for emo, weight in POSITIVE_EMOTIONS.items()
    )
    negative_weighted = sum(
        emotions.get(emo, 0.0) * weight
        for emo, weight in NEGATIVE_EMOTIONS.items()
    )
    total = positive_weighted + negative_weighted + 1e-9  # Avoid division by zero
    emotion_score = (positive_weighted / total) * 100
    return round(emotion_score, 2)


def compute_wellness_score(emotions: dict, risk_level: str) -> float:
    """
    Compute the final wellness score (0–100).

    Formula:
        wellness = (emotion_score * 0.6) + ((100 - risk_penalty) * 0.4)

    Args:
        emotions (dict): Emotion label to confidence score mapping.
        risk_level (str): One of 'normal', 'anxiety', 'depression', 'high-risk'.

    Returns:
        float: Wellness score clamped to the range [0, 100].
    """
    emotion_score = compute_emotion_score(emotions)
    risk_penalty = RISK_PENALTY.get(risk_level.lower().replace(" ", "_"), 0)

    wellness = (emotion_score * 0.6) + ((100 - risk_penalty) * 0.4)
    wellness = max(0.0, min(100.0, wellness))  # Clamp to [0, 100]
    return round(wellness, 2)


def get_alert_level(wellness_score: float) -> dict:
    """
    Determine the alert level and color based on the wellness score.

    Args:
        wellness_score (float): Score between 0 and 100.

    Returns:
        dict: Contains keys: 'level' (str), 'color' (str), 'show_helpline' (bool), 'message' (str).
    """
    if wellness_score >= 70:
        return {
            "level": "Good",
            "color": "#2ECC71",
            "emoji": "🟢",
            "show_helpline": False,
            "message": "You're doing well! Keep journaling and maintaining your mental wellness."
        }
    elif wellness_score >= 45:
        return {
            "level": "Moderate",
            "color": "#F39C12",
            "emoji": "🟡",
            "show_helpline": False,
            "message": "Your wellness is moderate. Consider self-care activities and speaking with someone you trust."
        }
    else:
        return {
            "level": "Low",
            "color": "#E74C3C",
            "emoji": "🔴",
            "show_helpline": True,
            "message": HELPLINE_MESSAGE
        }
