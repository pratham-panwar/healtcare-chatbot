# Simple demo: put this file in the repo root (next to the src/ folder) and run:
#     python demo.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # adds the repo root

from src.recommend_pipeline import predict_and_recommend

text = input("Describe your symptoms: ")
city = input("City (optional): ").strip() or None
area = input("Area (optional): ").strip() or None

result = predict_and_recommend(text, city, area)   # returns a dictionary

if "error" in result:                              # no symptoms recognized
    print(result["error"])
else:
    print("Disease   :", result["disease"])
    print("Confidence:", f"{result['confidence']:.1%}")
    print("Specialist:", result["specialist"])
    print("Note      :", result["note"])            # None if no fallback was needed
    print("Doctors   :")
    for doctor in result["doctors"]:               # list of dictionaries
        print("  -", doctor["doctor_name"], "|", doctor["hospital"], "|",
              doctor["area"], ",", doctor["city"], "| rating", doctor["rating"])

# Sample text to try:
# 1. fungal infection - I've been having a lot of itching and a skin rash, and I also noticed some nodal skin eruptions.

# 2. malaria - I've had a high fever along with a headache, nausea, and vomiting.

# 3. GRED - I've been having chest pain, acidity, and I've also been vomiting.

# 4. migrane - I've been having a headache along with blurred and distorted vision, and I've also been feeling irritable

# 5. diabetes - I've been feeling very tired, I've had an increased appetite, and I've noticed weight loss

# 6. heart attack - I've been having chest pain and difficulty breathing, and I've also been vomiting