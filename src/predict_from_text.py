"""
predict_from_text.py
---------------------
Loads the trained Random Forest model and lets you predict a disease
from a free-text description of symptoms, e.g.:

    "I've had a high fever, chills, and a bad headache for two days,
     and I feel really nauseous."

How it works:
1. The raw text is cleaned and split into words / n-grams.
2. Each n-gram is compared against the known symptom vocabulary using
   both exact phrase matching and fuzzy matching (to tolerate typos /
   slightly different wording), via rapidfuzz.
3. The matched symptoms are converted into the same multi-hot vector
   format the model was trained on.
4. The Random Forest predicts the disease (+ shows top-5 probabilities).
"""

import re
import json
from pathlib import Path

import joblib
import numpy as np
from rapidfuzz import fuzz

# <repo root>/models, resolved from this file so it works from any working directory.
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
FUZZY_THRESHOLD = 90  # 0-100, higher = stricter matching (fuzz.ratio scale)

STOPWORDS = {
    "i", "ive", "i've", "im", "i'm", "a", "an", "the", "and", "or", "of", "in",
    "on", "for", "to", "with", "have", "had", "has", "having", "been", "be",
    "am", "is", "are", "was", "were", "my", "me", "feel", "feeling", "felt",
    "really", "very", "quite", "bit", "little", "lot", "also", "some", "since",
    "day", "days", "week", "weeks", "past", "last", "two", "three", "four",
    "about", "bad", "severe", "mild", "moderate", "this", "that", "it", "its",
    "along", "getting", "get", "got", "experiencing", "experience", "suffering",
    "suffer", "from", "at", "night", "morning", "constant", "constantly",
}

# ---------------------------------------------------------------------
# Load trained artifacts
# ---------------------------------------------------------------------
clf = joblib.load(MODEL_DIR / "random_forest_disease_model.joblib")
le = joblib.load(MODEL_DIR / "label_encoder.joblib")
with open(MODEL_DIR / "symptom_vocab.json") as f:
    vocab = json.load(f)

# Human-readable phrase -> canonical symptom token, e.g. "skin rash" -> "skin_rash"
phrase_to_symptom = {s.replace("_", " "): s for s in vocab}
# Longest phrases (most words) first, so multi-word symptoms are matched
# before any of their individual words are considered on their own.
sorted_phrases = sorted(phrase_to_symptom.keys(), key=lambda p: -len(p.split()))

# Pre-compiled word-boundary regex for each phrase (flexible on whitespace)
phrase_patterns = [
    (phrase, phrase_to_symptom[phrase],
     re.compile(r"\b" + r"\s+".join(re.escape(w) for w in phrase.split()) + r"\b"))
    for phrase in sorted_phrases
]

single_word_phrases = [p for p in phrase_to_symptom if " " not in p]

# Common colloquial phrasing -> canonical vocabulary wording. This closes the
# gap between everyday language and the dataset's clinical symptom names,
# without relying on fragile fuzzy-matching alone.
SYNONYMS = {
    "itchy": "itching", "itch": "itching", "itches": "itching",
    "throwing up": "vomiting", "throw up": "vomiting", "puking": "vomiting",
    "puke": "vomiting", "vomit": "vomiting", "threw up": "vomiting",
    "tired": "fatigue", "exhausted": "fatigue", "exhaustion": "fatigue",
    "stomach ache": "stomach pain", "stomachache": "stomach pain",
    "tummy ache": "stomach pain", "belly ache": "belly pain",
    "sore throat": "throat irritation", "throat pain": "throat irritation",
    "scratchy throat": "throat irritation",
    "stuffy nose": "congestion", "blocked nose": "congestion",
    "cant sleep": "restlessness", "cannot sleep": "restlessness",
    "rash": "skin rash", "rashes": "skin rash",
    "red spots": "red spots over body",
    "yellow skin": "yellowish skin", "yellow eyes": "yellowing of eyes",
    "dizzy": "dizziness", "weak": "muscle weakness", "weakness": "muscle weakness",
    "shaky": "shivering", "shaking": "shivering", "sweaty": "sweating",
    "breathless": "breathlessness", "cant breathe": "breathlessness",
    "short of breath": "breathlessness", "trouble breathing": "breathlessness",
    "swollen joints": "swelling joints", "achy joints": "joint pain",
    "loss of weight": "weight loss", "losing weight": "weight loss",
    "gaining weight": "weight gain", "putting on weight": "weight gain",
    "no appetite": "loss of appetite", "not hungry": "loss of appetite",
    "always hungry": "excessive hunger", "very hungry": "excessive hunger",
    "dark colored urine": "dark urine",
    "blood in urine": "bloody stool",
    "cough with blood": "blood in sputum",
    "chills and shivering": "chills",
    "feavr": "high fever", "fever": "high fever", "high temperature": "high fever",
    "running a temperature": "high fever",
    # Common cold / flu phrasing
    "sneezing": "continuous sneezing", "sneeze": "continuous sneezing",
    "sneezes": "continuous sneezing", "sneezing a lot": "continuous sneezing",
    "nose is running": "runny nose", "my nose is running": "runny nose",
    "nose running": "runny nose",
    # Urinary / diabetes phrasing
    "urinating frequently": "polyuria", "urinating a lot": "polyuria",
    "peeing a lot": "polyuria", "peeing frequently": "polyuria",
    "frequent urination": "polyuria", "going to the bathroom a lot": "polyuria",
    # GERD / acid reflux phrasing
    "heartburn": "acidity", "acid reflux": "acidity",
    "acid comes back up": "acidity", "acid coming back up": "acidity",
    "food comes back up": "acidity", "food or acid comes back up": "acidity",
    "burning feeling in my chest": "acidity chest pain",
    "burning feeling in chest": "acidity chest pain",
    "burning sensation in my chest": "acidity chest pain",
    "burning sensation in chest": "acidity chest pain",
}

# Word-order-independent triggers: if ALL of these words appear anywhere in
# the text (not necessarily adjacent), add the mapped symptom phrase. This
# catches phrasing the literal SYNONYMS dict above would miss due to word
# order, e.g. "my nose is running" vs "runny nose".
BAG_SYNONYMS = [
    ({"nose", "running"}, "runny nose"),
    ({"nose", "runs"}, "runny nose"),
    ({"urinating", "frequently"}, "polyuria"),
    ({"urinating", "often"}, "polyuria"),
    ({"peeing", "often"}, "polyuria"),
    ({"peeing", "lot"}, "polyuria"),
    ({"burning", "chest"}, "acidity chest pain"),
    ({"acid", "throat"}, "acidity"),
    ({"acid", "back"}, "acidity"),
    ({"food", "back", "throat"}, "acidity"),
]

# All individual words used across the symptom vocabulary (for spelling
# correction of typos, e.g. "pian" -> "pain", "muscel" -> "muscle").
GLOBAL_WORDS = sorted({w for p in phrase_to_symptom for w in p.split()})


def clean_text(text: str) -> str:
    text = text.lower()
    text = text.replace("'", "")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def apply_synonyms(text: str) -> str:
    for phrase, replacement in SYNONYMS.items():
        text = re.sub(r"\b" + re.escape(phrase) + r"\b", replacement, text)
    return text


def apply_bag_synonyms(text: str) -> str:
    words = set(text.split())
    extra = []
    for triggers, replacement in BAG_SYNONYMS:
        if triggers <= words:
            extra.append(replacement)
    if extra:
        text = text + " " + " ".join(extra)
    return text


def correct_word(word: str) -> str:
    """Fix small typos by snapping a word to the closest known vocabulary word.

    Only corrects words that look like near-misspellings of a vocabulary
    word: same starting letter and similar length, so real English words
    (e.g. "eating") don't get mangled into an unrelated symptom word
    (e.g. "sweating") just because they share a lot of letters.
    """
    if word in GLOBAL_WORDS or word in STOPWORDS or len(word) < 4:
        return word
    best_word, best_score = word, 0
    for gw in GLOBAL_WORDS:
        if gw[0] != word[0] or abs(len(gw) - len(word)) > 2:
            continue
        score = fuzz.ratio(word, gw)
        if score > best_score:
            best_word, best_score = gw, score
    return best_word if best_score >= 75 else word


def extract_symptoms(text: str, verbose: bool = False):
    """Return the set of canonical symptom tokens found in free text."""
    cleaned = clean_text(text)
    cleaned = apply_synonyms(cleaned)
    cleaned = apply_bag_synonyms(cleaned)
    match_log = []
    matched_symptoms = set()
    remaining = cleaned

    # 1) Exact phrase matching (longest phrases first), removing matches
    #    from the text as we go so words aren't double-counted.
    for phrase, symptom, pattern in phrase_patterns:
        m = pattern.search(remaining)
        if m:
            matched_symptoms.add(symptom)
            match_log.append((phrase, symptom, 100))
            remaining = remaining[:m.start()] + " " + remaining[m.end():]

    # 2) Spelling-correct any leftover words, then retry exact phrase
    #    matching on the corrected text (catches typos like "joint pian").
    corrected = " ".join(correct_word(w) for w in remaining.split())
    if corrected != remaining:
        for phrase, symptom, pattern in phrase_patterns:
            if symptom in matched_symptoms:
                continue
            m = pattern.search(corrected)
            if m:
                matched_symptoms.add(symptom)
                match_log.append((phrase + " (corrected)", symptom, 90))
                corrected = corrected[:m.start()] + " " + corrected[m.end():]
        remaining = corrected

    # 3) Final fallback: fuzzy match any remaining content word directly
    #    against single-word symptom names (same first-letter safeguard).
    leftover_words = [w for w in remaining.split() if w not in STOPWORDS and len(w) >= 4]
    for word in leftover_words:
        best_phrase, best_score = None, 0
        for phrase in single_word_phrases:
            if phrase[0] != word[0] or abs(len(phrase) - len(word)) > 2:
                continue
            score = fuzz.ratio(word, phrase)
            if score > best_score:
                best_phrase, best_score = phrase, score
        if best_score >= FUZZY_THRESHOLD:
            symptom = phrase_to_symptom[best_phrase]
            if symptom not in matched_symptoms:
                matched_symptoms.add(symptom)
                match_log.append((word, symptom, round(best_score, 1)))

    if verbose:
        print("Matched phrases -> symptoms:")
        for phrase, symptom, score in match_log:
            print(f"  '{phrase}' -> {symptom} (score={score})")

    return matched_symptoms


def predict_disease(text: str, top_k: int = 1, verbose: bool = False):
    """Predict the disease from free text.

    By default this only prints the single most likely disease and its
    confidence percentage. Pass verbose=True (or top_k>1) if you want the
    symptom-matching debug info / a longer ranked list back, e.g. for
    troubleshooting.
    """
    symptoms = extract_symptoms(text, verbose=verbose)

    if not symptoms:
        print("Sorry, I couldn't pick out any recognizable symptoms from that. "
              "Try naming them more explicitly, e.g. 'fever, cough, headache'.")
        return None

    # Build feature vector
    x = np.zeros((1, len(vocab)), dtype=np.int8)
    for s in symptoms:
        x[0, vocab.index(s)] = 1

    proba = clf.predict_proba(x)[0]
    top_idx = np.argsort(proba)[::-1][:top_k]

    if verbose:
        print(f"\nDetected symptoms ({len(symptoms)}): {', '.join(sorted(symptoms))}\n")

    for i in top_idx:
        disease = le.inverse_transform([i])[0]
        print(f"{disease} ({proba[i]*100:.1f}%)")

    best_disease = le.inverse_transform([top_idx[0]])[0]
    return best_disease


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        user_text = " ".join(sys.argv[1:])
        predict_disease(user_text)
    else:
        print("Enter a free-text description of your symptoms (Ctrl+C to quit).\n")
        while True:
            try:
                user_text = input(">>> ")
                if not user_text.strip():
                    continue
                predict_disease(user_text)
                print()
            except KeyboardInterrupt:
                print("\nGoodbye.")
                break