"""
emotion_model.py - Emotion classifier wrapper using Hugging Face Transformers.
Model: j-hartmann/emotion-english-distilroberta-base
Detects: joy, sadness, fear, anger, neutral, surprise, disgust
"""

from transformers import pipeline
import streamlit as st

# Canonical emotion labels we expose to the rest of the app
EMOTION_LABELS = ["joy", "sadness", "fear", "anger", "neutral", "surprise", "disgust"]


@st.cache_resource(show_spinner=False)
def load_emotion_model():
    """
    Load and cache the emotion classifier pipeline.
    Using st.cache_resource ensures the model is loaded only once per session.

    Returns:
        transformers.Pipeline: The loaded emotion classification pipeline.
    """
    try:
        classifier = pipeline(
            task="text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None,          # Return scores for all labels
            truncation=True,
            max_length=512,
        )
        return classifier
    except Exception as e:
        print(f"[EmotionModel Error] Failed to load model: {e}")
        return None


def predict_emotions(text: str) -> dict:
    """
    Run the emotion classifier on the given text.

    Args:
        text (str): Preprocessed journal entry text.

    Returns:
        dict: Mapping of emotion label (str) -> confidence score (float, 0-1).
              Returns a neutral baseline dict if model loading fails.
    """
    classifier = load_emotion_model()

    if classifier is None or not text:
        # Fallback: return a neutral emotion distribution
        return {label: (1.0 / len(EMOTION_LABELS)) for label in EMOTION_LABELS}

    try:
        results = classifier(text)
        # results is a list of lists: [[{'label': 'joy', 'score': 0.9}, ...]]
        if isinstance(results[0], list):
            results = results[0]

        # Build a dict from model output, normalizing labels to lowercase
        emotion_scores = {item["label"].lower(): round(item["score"], 4) for item in results}

        # Ensure all standard labels are present (fill missing with 0)
        for label in EMOTION_LABELS:
            if label not in emotion_scores:
                emotion_scores[label] = 0.0

        return emotion_scores

    except Exception as e:
        print(f"[EmotionModel Error] Inference failed: {e}")
        return {label: (1.0 / len(EMOTION_LABELS)) for label in EMOTION_LABELS}


def get_dominant_emotion(emotions: dict) -> str:
    """
    Return the emotion label with the highest confidence score.

    Args:
        emotions (dict): Emotion label to score mapping.

    Returns:
        str: Label of the dominant emotion.
    """
    if not emotions:
        return "neutral"
    return max(emotions, key=emotions.get)
