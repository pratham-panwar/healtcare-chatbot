"""
doctor_recommender.py
----------------------
Given a disease name (as predicted by predict_from_text.py) and optionally
a city/area, returns the top matching doctors from data/synthetic_doctors.csv.

Disease -> specialist type comes from data/doctor_versus_disease.csv. That file
was hand-labelled and its specialist names don't all line up exactly with
the specialization values actually present in synthetic_doctors.csv (typos,
case differences, and specialties that simply have no doctor in the sample
data, e.g. "Pediatrician", "Phlebologist", "Internal Medcine"). SPECIALIST_
OVERRIDES below fixes the typos/case issues and re-routes the handful of
specialties absent from the doctor roster to the closest available one.
Adjust this dict if your real doctor roster covers different specialties.
"""

from pathlib import Path

import pandas as pd

# <repo root>/data, resolved from this file so it works from any working directory.
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

_dvd = pd.read_csv(
    DATA_DIR / "doctor_versus_disease.csv",
    encoding="latin1", header=None, names=["disease", "specialist"],
)
_dvd["disease"] = _dvd["disease"].str.strip()
_dvd["specialist"] = _dvd["specialist"].str.strip()

DOCTORS = pd.read_csv(DATA_DIR / "synthetic_doctors.csv")
AVAILABLE_SPECIALTIES = set(DOCTORS["specialization"].unique())

# Fixes typos/casing + re-routes specialties with no doctors in the roster
# to the closest one that IS available.
SPECIALIST_OVERRIDES = {
    "hepatologist": "Hepatologist",              # case fix
    "Rheumatologists": "Rheumatologist",          # pluralization fix
    "Gynecologist": "Urologist",                  # UTI -> no gynecologist in roster
    "Internal Medcine": "General Physician",      # typo + no such specialty
    "Osteopathic": "Infectious Disease Specialist",  # AIDS
    "Otolaryngologist": "ENT Specialist",         # vertigo, common cold
    "Pediatrician": "General Physician",          # Typhoid
    "Phlebologist": "General Surgeon",            # Varicose veins
    "Tuberculosis": "Pulmonologist",              # TB -> lungs
}

DISEASE_TO_SPECIALIST = {}
for _, row in _dvd.iterrows():
    raw = row["specialist"]
    DISEASE_TO_SPECIALIST[row["disease"]] = SPECIALIST_OVERRIDES.get(raw, raw)

# Sanity check: every mapped specialist should now exist in the doctor roster.
_unmatched = {s for s in DISEASE_TO_SPECIALIST.values() if s not in AVAILABLE_SPECIALTIES}
if _unmatched:
    raise RuntimeError(
        f"These specialist labels have no doctors in synthetic_doctors.csv "
        f"and need an entry in SPECIALIST_OVERRIDES: {_unmatched}"
    )


def get_specialist(disease: str) -> str | None:
    """Look up which specialist type treats a given disease."""
    return DISEASE_TO_SPECIALIST.get(disease.strip())


def recommend_doctors(disease: str, city: str | None = None,
                       area: str | None = None, top_n: int = 3):
    """Return up to `top_n` recommended doctors for a predicted disease.

    Filters by city/area if given (case-insensitive substring match),
    then ranks by rating (desc) and experience_years (desc) as a tiebreak.
    If fewer than `top_n` doctors match, returns all of them.

    Returns a dict: {
        "disease": ..., "specialist": ..., "doctors": <DataFrame>,
        "note": <str or None>   # e.g. explains a fallback that was applied
    }
    Raises ValueError if the disease isn't recognized at all.
    """
    specialist = get_specialist(disease)
    if specialist is None:
        raise ValueError(
            f"'{disease}' is not in the known disease list. "
            f"Known diseases: {sorted(DISEASE_TO_SPECIALIST)}"
        )

    pool = DOCTORS[DOCTORS["specialization"] == specialist].copy()
    note = None

    if city:
        city_matches = pool[pool["city"].str.contains(city, case=False, na=False)]
        if area:
            area_matches = city_matches[city_matches["area"].str.contains(area, case=False, na=False)]
            if len(area_matches) > 0:
                pool = area_matches
            elif len(city_matches) > 0:
                pool = city_matches
                note = (f"No {specialist} found in '{area}' specifically; "
                        f"showing results for all of {city} instead.")
            else:
                note = (f"No {specialist} found in '{city}'; "
                        f"showing the best-matching {specialist}s from other cities instead.")
        else:
            if len(city_matches) > 0:
                pool = city_matches
            else:
                note = (f"No {specialist} found in '{city}'; "
                        f"showing the best-matching {specialist}s from other cities instead.")

    pool = pool.sort_values(by=["rating", "experience_years"], ascending=[False, False])
    top = pool.head(top_n).reset_index(drop=True)

    if len(top) < top_n and note is None and len(pool) < top_n:
        note = f"Only {len(pool)} {specialist}(s) available in total for this filter."

    return {"disease": disease, "specialist": specialist, "doctors": top, "note": note}
