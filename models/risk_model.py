"""
risk_model.py - Mental health risk level classifier.
Uses a rule-based + keyword approach combined with a zero-shot HuggingFace classifier
to map journal text to: normal | anxiety | depression | high-risk
"""

from transformers import pipeline
import streamlit as st

# Risk level labels (ordered by severity)
RISK_LABELS = ["normal", "anxiety", "depression", "high-risk"]

# Zero-shot classification hypothesis template
HYPOTHESIS_TEMPLATE = "This text expresses {} feelings related to mental health."

# Candidate labels for zero-shot
CANDIDATE_LABELS = [
    "normal mental health",
    "anxiety and worry",
    "depression and hopelessness",
    "high risk of self-harm or crisis"
]

# Mapping from candidate labels to internal risk levels
LABEL_MAP = {
    "normal mental health": "normal",
    "anxiety and worry": "anxiety",
    "depression and hopelessness": "depression",
    "high risk of self-harm or crisis": "high-risk",
}

# Keyword fallback patterns (high-priority override)
CRISIS_KEYWORDS = [
    "suicide", "kill myself", "end my life", "don't want to live",
    "want to die", "self-harm", "hurt myself", "not worth living",
    "can't go on", "no reason to live"
]
DEPRESSION_KEYWORDS = [
    "hopeless", "worthless", "empty", "no joy", "can't get up",
    "no energy", "everything is pointless", "numb", "miserable",
    "crying all the time", "no motivation"
]
ANXIETY_KEYWORDS = [
    "anxious", "panic", "worried", "can't stop thinking",
    "heart racing", "nervous", "overthinking", "dread", "scared", "fear"
]


@st.cache_resource(show_spinner=False)
def load_risk_model():
    """
    Load and cache the zero-shot classification pipeline.

    Returns:
        transformers.Pipeline: Zero-shot classifier pipeline, or None on failure.
    """
    try:
        classifier = pipeline(
            task="zero-shot-classification",
            model="facebook/bart-large-mnli",
            truncation=True,
        )
        return classifier
    except Exception as e:
        print(f"[RiskModel Error] Failed to load zero-shot model: {e}")
        return None


def _keyword_override(text: str) -> str | None:
    """
    Check for high-priority keywords that should override model prediction.

    Args:
        text (str): Lowercased journal entry text.

    Returns:
        str | None: Risk level string, or None if no keywords found.
    """
    text_lower = text.lower()
    if any(kw in text_lower for kw in CRISIS_KEYWORDS):
        return "high-risk"
    if any(kw in text_lower for kw in DEPRESSION_KEYWORDS):
        return "depression"
    if any(kw in text_lower for kw in ANXIETY_KEYWORDS):
        return "anxiety"
    return None


def predict_risk(text: str) -> str:
    """
    Predict the mental health risk level for the given text.

    Priority order:
        1. Keyword override (for safety-critical phrases)
        2. Zero-shot classifier
        3. Fallback: 'normal'

    Args:
        text (str): Preprocessed journal entry text.

    Returns:
        str: One of 'normal', 'anxiety', 'depression', 'high-risk'.
    """
    if not text:
        return "normal"

    # Step 1: Keyword safety check
    keyword_result = _keyword_override(text)
    if keyword_result:
        return keyword_result

    # Step 2: Zero-shot model prediction
    classifier = load_risk_model()
    if classifier is None:
        return "normal"

    try:
        result = classifier(
            text,
            candidate_labels=CANDIDATE_LABELS,
            hypothesis_template=HYPOTHESIS_TEMPLATE,
            multi_label=False,
        )
        # result['labels'][0] is the highest-scoring candidate
        top_label = result["labels"][0]
        return LABEL_MAP.get(top_label, "normal")

    except Exception as e:
        print(f"[RiskModel Error] Inference failed: {e}")
        return "normal"


def get_risk_display(risk_level: str) -> dict:
    """
    Return display metadata for a given risk level.

    Args:
        risk_level (str): Risk level string.

    Returns:
        dict: Keys: 'label', 'color', 'emoji', 'description'.
    """
    display_map = {
        "normal": {
            "label": "Normal",
            "color": "#2ECC71",
            "emoji": "✅",
            "description": "No significant mental health concerns detected."
        },
        "anxiety": {
            "label": "Anxiety",
            "color": "#F39C12",
            "emoji": "😟",
            "description": "Signs of anxiety or worry detected. Consider relaxation techniques."
        },
        "depression": {
            "label": "Depression",
            "color": "#E67E22",
            "emoji": "😔",
            "description": "Signs of depressive feelings detected. Please consider speaking with a professional."
        },
        "high-risk": {
            "label": "High Risk",
            "color": "#E74C3C",
            "emoji": "🚨",
            "description": "High-risk indicators detected. Please reach out to a mental health professional immediately."
        },
    }
    return display_map.get(risk_level, display_map["normal"])
