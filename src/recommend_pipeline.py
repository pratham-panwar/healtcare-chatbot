"""
recommend_pipeline.py
----------------------
End-to-end demo: free-text symptoms -> predicted disease -> recommended
doctors (top 3, filtered by city/area if given).

Usage (run from the repository root):
    python -m src.recommend_pipeline "fever, cough and body ache" --city Ludhiana
    python -m src.recommend_pipeline "fever, cough and body ache" --city Ludhiana --area "Model Town"
    python -m src.recommend_pipeline   # interactive mode

This is a thin CLI wrapper around predict_from_text.extract_symptoms() and
doctor_recommender.recommend_doctors() -- see those modules if you want to
call this logic from a web backend, notebook, etc. instead of the CLI.
"""

import argparse
import numpy as np
import pandas as pd

# predict_disease() itself only prints results and returns a single disease
# name (or None) -- it doesn't hand back the ranked list. So instead of
# unpacking its return value, we reuse the building blocks it already
# exposes (extract_symptoms / clf / le / vocab) to compute our own ranked
# list here, without touching predict_from_text.py's logic at all.
from .predict_from_text import extract_symptoms, clf, le, vocab
from .doctor_recommender import recommend_doctors

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)


def run(symptom_text: str, city: str | None, area: str | None, top_n: int = 3):
    symptoms = extract_symptoms(symptom_text)

    if not symptoms:
        print("Sorry, I couldn't pick out any recognizable symptoms from that. "
              "Try naming them more explicitly, e.g. 'fever, cough, headache'.")
        return

    x = np.zeros((1, len(vocab)), dtype=np.int8)
    for s in symptoms:
        x[0, vocab.index(s)] = 1

    proba = clf.predict_proba(x)[0]
    top_disease = le.inverse_transform([np.argmax(proba)])[0]

    result = recommend_doctors(top_disease, city=city, area=area, top_n=top_n)
    doctors = result["doctors"]
    if len(doctors) == 0:
        print("No matching doctors found.")
        return

    display_cols = ["doctor_name", "specialization", "hospital", "area", "city",
                     "experience_years", "rating"]
    print(doctors[display_cols].to_string(index=False))

def predict_and_recommend(symptom_text: str, city: str | None = None, area: str | None = None, top_n: int = 3):
    """Same logic as run(), but returns a dict instead of printing --
    use this one for Streamlit / any frontend integration."""
    symptoms = extract_symptoms(symptom_text)

    if not symptoms:
        return {"error": "No recognizable symptoms found in that text."}

    x = np.zeros((1, len(vocab)), dtype=np.int8)
    for s in symptoms:
        x[0, vocab.index(s)] = 1

    proba = clf.predict_proba(x)[0]
    top_idx = np.argmax(proba)
    top_disease = le.inverse_transform([top_idx])[0]
    confidence = float(proba[top_idx])

    result = recommend_doctors(top_disease, city=city, area=area, top_n=top_n)
    doctors = result["doctors"]

    return {
        "disease": top_disease,
        "confidence": confidence,
        "specialist": result["specialist"],
        "note": result["note"],
        "doctors": doctors.to_dict(orient="records"),
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Symptom -> disease -> doctor recommender")
    parser.add_argument("text", nargs="*", help="Free-text symptom description")
    parser.add_argument("--city", default=None, help="Filter doctors by city")
    parser.add_argument("--area", default=None, help="Filter doctors by area within the city")
    parser.add_argument("--top_n", type=int, default=3, help="Number of doctors to recommend")
    args = parser.parse_args()

    if args.text:
        run(" ".join(args.text), city=args.city, area=args.area, top_n=args.top_n)
    else:
        print("Enter a free-text description of your symptoms (Ctrl+C to quit).\n")
        city = input("City (optional, press Enter to skip): ").strip() or None
        area = input("Area (optional, press Enter to skip): ").strip() or None
        print()
        while True:
            try:
                user_text = input(">>> ")
                if not user_text.strip():
                    continue
                run(user_text, city=city, area=area)
                print()
            except KeyboardInterrupt:
                print("\nGoodbye.")
                break