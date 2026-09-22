"""Compatibility API for the Streamlit app backed by the real model files."""

import numpy as np

from doctor_recommendation import get_specialist
from predict_from_text import clf, extract_symptoms as _extract_symptoms, le, vocab


VOCAB_INDEX = {symptom: index for index, symptom in enumerate(vocab)}


def extract_symptoms(user_text: str) -> list[str]:
    """Extract canonical symptoms with the trained model's text processor."""
    return sorted(_extract_symptoms(user_text))


def get_probable_symptoms(known_symptoms: list[str]) -> list[str]:
    """Suggest high-importance model symptoms not already reported."""
    known = set(known_symptoms)
    ranked_indices = np.argsort(clf.feature_importances_)[::-1]
    return [
        vocab[index]
        for index in ranked_indices
        if vocab[index] not in known
    ][:4]


def _predict_disease(symptoms: list[str]) -> str:
    features = np.zeros((1, len(vocab)), dtype=np.int8)
    for symptom in symptoms:
        index = VOCAB_INDEX.get(symptom)
        if index is not None:
            features[0, index] = 1
    probabilities = clf.predict_proba(features)[0]
    return le.inverse_transform([int(np.argmax(probabilities))])[0]


def predict_department(symptoms: list[str]) -> str:
    """Predict a disease, then return its mapped medical department."""
    if not symptoms:
        return "General Physician"
    disease = _predict_disease(symptoms)
    return get_specialist(disease) or "General Physician"
