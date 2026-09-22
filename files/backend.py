"""
Booking logic against the 'slots' and 'appointments' tables created by
db_setup.py, plus an optional Google Calendar sync after a successful booking.
"""
import sqlite3
from datetime import datetime

DB_PATH = "hospital.db"


def get_available_slots(doctor_id: int) -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM slots WHERE doctor_id = ? AND is_booked = 0", (doctor_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def book_slot(slot_id: int, patient_name: str, patient_contact: str = "") -> bool:
    """
    Books a slot inside one locked transaction so two users clicking
    'Book' on the same slot at the same time can't both succeed.
    """
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("BEGIN IMMEDIATE")  # locks the db until commit/rollback
        row = conn.execute("SELECT is_booked FROM slots WHERE id = ?", (slot_id,)).fetchone()
        if row is None or row[0] == 1:
            conn.rollback()
            return False  # already booked or doesn't exist

        conn.execute("UPDATE slots SET is_booked = 1 WHERE id = ?", (slot_id,))
        conn.execute(
            "INSERT INTO appointments (slot_id, patient_name, patient_contact) VALUES (?, ?, ?)",
            (slot_id, patient_name, patient_contact),
        )
        conn.commit()
        return True
    except Exception:
        conn.rollback()
        return False
    finally:
        conn.close()


# def add_to_google_calendar(summary: str, date: str, time: str) -> bool:
#     """
#     Adds the confirmed appointment to the user's Google Calendar.
#     One-time setup needed (per machine/deployment):
#       1. pip install google-api-python-client google-auth-oauthlib
#       2. Create a Google Cloud project, enable the Calendar API,
#          download 'credentials.json' into this folder.
#     Returns False (without crashing the app) if that setup isn't done yet.
#     """
#     try:
#         import os
#         from google.auth.transport.requests import Request
#         from google.oauth2.credentials import Credentials
#         from google_auth_oauthlib.flow import InstalledAppFlow
#         from googleapiclient.discovery import build

#         scopes = ["https://www.googleapis.com/auth/calendar.events"]
#         creds = None
#         if os.path.exists("token.json"):
#             creds = Credentials.from_authorized_user_file("token.json", scopes)
#         if not creds or not creds.valid:
#             if creds and creds.expired and creds.refresh_token:
#                 creds.refresh(Request())
#             else:
#                 flow = InstalledAppFlow.from_client_secrets_file("credentials.json", scopes)
#                 creds = flow.run_local_server(port=0)
#             with open("token.json", "w") as f:
#                 f.write(creds.to_json())

#         service = build("calendar", "v3", credentials=creds)
#         start_dt = datetime.strptime(f"{date} {time}", "%Y-%m-%d %I:%M %p")
#         event = {
#             "summary": summary,
#             "start": {"dateTime": start_dt.isoformat(), "timeZone": "Asia/Kolkata"},
#             "end": {"dateTime": start_dt.isoformat(), "timeZone": "Asia/Kolkata"},
#         }
#         service.events().insert(calendarId="primary", body=event).execute()
#         return True
#     except Exception:
#         return False
