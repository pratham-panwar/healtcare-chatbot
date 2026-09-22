"""
Frontend chatbot for symptom-based doctor recommendation + appointment booking.

Run with:   streamlit run app1.py

Expects (built separately by the team):
    disease_prediction.py    -> extract_symptoms(), get_probable_symptoms(), predict_department()
    doctor_recommendation.py -> get_top_doctors()
    backend.py               -> get_available_slots(), book_slot()
    hospital.db              -> created by running `python db_setup.py` once
"""
import html
import streamlit as st
import streamlit.components.v1 as components
from disease_prediction import extract_symptoms, get_probable_symptoms, predict_department
from doctor_recommendation import get_top_doctors
from backend import get_available_slots, book_slot  # add_to_google_calendar disabled

try:
    from streamlit_geolocation import streamlit_geolocation
except ImportError:
    streamlit_geolocation = None


# -----
# Page config
# -----
st.set_page_config(
    page_title="ScoutCare",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# -----
# Dashboard Designing
# -----
st.markdown("""
<style>
    
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap'); /* FONT STYLE */


    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif;
        background-color: #ffffff; /* dark background for contrast */
        color: #e5e7eb;
    }


    /* Hide default Streamlit header/footer decorations for cleaner look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}


    /* Main layout: two columns: left panel + right chat area */
    .dashboard-container {
        display: flex;
        flex-direction: row;
        gap: 24px;
        padding: 24px 32px 32px 32px;
        max-width: 1400px;
        margin: 0 auto;
    }


    /* LEFT PANEL */
    .left-panel {
        flex: 0;
        min-width: 260px;
        max-width: 320px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        padding: 16px 8px 8px 8px;
    }


    .bot-title-block {
        margin-top: 10px;
        margin-bottom: 24px;
    }


    .bot-title {
        font-size: 50px;
        font-weight: 900;
        font-style: italic;
        line-height: 1.25;
        /* Blue,black,Red combination */
        background: linear-gradient(90deg, #1e3a8a 0%, #1e3a8a 35%, #b91c1c 45%, #b91c1c 100%);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        margin-bottom: 8px;
    }


    .bot-subtitle {
        font-size: 14px;
        color: #9ca3af;
        line-height: 1.4;
    }


    /* Lottie animation */
    .lottie-placeholder {
        width: 100%;
        height: 180px;
        border-radius: 14px;
        background: linear-gradient(135deg, #111827, #1f2937);
        border: 1px solid #374151;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #9ca3af;
        font-size: 13px;
        text-align: center;
        padding: 10px;
        margin-top: auto; /* push to bottom of left panel */
    }


    /* Keep the animation visible while the conversation scrolls. */
    [data-testid="stIFrame"] {
        position: fixed;
        left: 32px;
        bottom: 24px;
        width: 280px !important;
        height: 180px !important;
        z-index: 10;
    }


    /* RIGHT CHAT AREA - no gray box background now */
    .chat-area {
        flex: 2;
        min-width: 0;
        display: flex;
        flex-direction: column;
        background: transparent;
        border: none;
        box-shadow: none;
    }


    /*.chat-box {
        background-color: transparent;
        border-radius: 0;
        padding: 10px 0 0 0;
        min-height: 460px;
        max-height: 460px;
        overflow-y: auto;
        border: none;
        box-shadow: none;
    }*?


    /* Chat bubbles - increased gap between messages */
    .message-row {
        display: flex;
        margin: 18px 0; 
    }


    .message-row.user {
        justify-content: flex-end;
    }


    .message-row.bot {
        justify-content: flex-start;
    }


    .bubble {
        max-width: 75%;
        padding: 14px 18px;
        border-radius: 18px;
        font-size: 15px;
        line-height: 1.45;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        word-wrap: break-word;
    }


    .bubble.bot {
        background: #98FB98;
        color: #111827;
        border-bottom-left-radius: 6px;
        margin-right: auto;
    }


    .bubble.user {
        background: #dbeafe; /* soft blue */
        color: #0f172a;
        border-bottom-right-radius: 6px;
        margin-left: auto;
    }


    /* Input area under chat box */
    .input-area {
        margin-top: 14px;
        display: flex;
        gap: 10px;
        align-items: center;
    }


    .input-area input {
        flex: 1;
        padding: 12px 16px;
        border-radius: 12px;
        border: 1px solid #9ca3af;
        background: #f3f4f6;
        color: #111827;
        font-size: 15px;
        font-family: 'Inter', sans-serif;
    }


    [data-testid="stWidgetLabel"] p {
        color: #1e3a8a !important;
    }


    [data-testid="stRadio"] label,
    [data-testid="stRadio"] label p {
        color: #1e3a8a !important;
    }


    .doctor-name {
        color: #b91c1c;
    }


    .no-slots-message {
        color: #6B4C9A;
        font-weight: 600;
    }


    [data-testid="stAlert"] p {
        color: #c4a7e7 !important;
        font-family: Georgia, serif;
        font-style: italic;
        font-weight: 600;
    }


    .input-area button {
        padding: 12px 18px;
        border-radius: 12px;
        border: none;
        background: #1e3a8a; /* deep blue */
        color: #ffffff;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        cursor: pointer;
        transition: background 0.2s ease;
    }


    .input-area button:hover {
        background: #1e40af;
    }


    /* Option buttons during conversation */
    .options-row {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 10px;
    }


    .option-btn {
        background: #e5e7eb;
        color: #111827;
        border: 1px solid #9ca3af;
        border-radius: 999px;
        padding: 8px 14px;
        font-size: 13px;
        font-weight: 500;
        cursor: pointer;
        font-family: 'Inter', sans-serif;
        transition: background 0.15s ease, transform 0.1s ease;
    }


    .option-btn:hover {
        background: #f3f4f6;
        transform: translateY(-1px);
    }


    /* Scrollbar styling for chat box */
    .chat-box::-webkit-scrollbar {
        width: 8px;
    }
    .chat-box::-webkit-scrollbar-track {
        background: #0b0f19;
        border-radius: 10px;
    }
    .chat-box::-webkit-scrollbar-thumb {
        background: #374151;
        border-radius: 10px;
    }
    .chat-box::-webkit-scrollbar-thumb:hover {
        background: #4b5563;
    }


    /* Responsive: stack on small screens */
    @media (max-width: 900px) {
        [data-testid="stIFrame"] {
            left: 16px;
            bottom: 16px;
            width: 180px !important;
            height: 120px !important;
        }

        .dashboard-container {
            flex-direction: column;
            padding: 16px;
        }
        .left-panel {
            max-width: 100%;
            flex-direction: row;
            align-items: flex-start;
            gap: 16px;
        }
        .bot-title-block {
            margin-bottom: 0;
        }
        .lottie-placeholder {
            height: 120px;
            min-width: 160px;
            margin-top: 0;
        }
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------
# Session state (your defaults + chat)
# -----------------------------
defaults = {
    "stage": "welcome",
    "chat": [],
    "profile": {},
    "symptoms": [],
    "probable_options": [],
    "department": None,
    "doctors": [],
    "chosen_doctor": None,
    "slots": [],
    "chosen_slot": None,
}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)


def say(role, msg):
    st.session_state.chat.append((role, msg))


if not st.session_state.chat:
    say("ScoutCare", "Hi! Would you like to continue as a Guest or Sign in?")


# -----------------------------
# Layout: left panel + right chat
# -----------------------------
col_left, col_right = st.columns([1, 2])


with col_left:
    st.markdown("""
    <div class="left-panel">
        <div class="bot-title-block">
            <div class="bot-title">ScoutCare</div>
            <div class="bot-subtitle">
                Your friendly, professional chat companion for guidance, support, and quick answers.
            </div>
        </div>


    </div>
    """, unsafe_allow_html=True)
    components.html(
        """
        <script src="https://unpkg.com/@lottiefiles/lottie-player@latest/dist/lottie-player.js"></script>
        <lottie-player
            src="https://lottie.host/3e558ba6-065f-4327-aab5-7330ce35bbf8/dV1nRVmIxs.json"
            background="transparent"
            speed="0.6"
            style="width: 100%; height: 180px;"
            loop
            autoplay>
        </lottie-player>
        """,
        height=180,
    )


with col_right:
    # Chat area + chat box (conversation bubbles go inside this box)
    st.markdown("""
    <div class="chat-area">
        <div class="chat-box" id="chat-box">
    """, unsafe_allow_html=True)

    # Render chat bubbles inside the chat box (now transparent background)
    for role, msg in st.session_state.chat:
        role_class = "bot" if role == "ScoutCare" else "user"
        st.markdown(f"""
        <div class="message-row {role_class}">
            <div class="bubble {role_class}">{html.escape(msg)}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
        </div>
    """, unsafe_allow_html=True)

    # ---------- Stage-specific controls (below chat box) ----------
    st.markdown('<div class="stage-area">', unsafe_allow_html=True)

    stage = st.session_state.stage

    if stage == "welcome":
        c1, c2 = st.columns(2)
        if c1.button("Be My Honourable Guest", use_container_width=True, key="btn_guest"):
            say("Myjesty", "BE honourable guest!")
            st.session_state.profile = {"name": "My honourable guest!", "lat": None, "lon": None}
            say("ScoutCare", "No problem! How are you feeling today My dear Guest?")
            st.session_state.stage = "ask_feeling"
            st.rerun()
        if c2.button("I'm Myjesty", use_container_width=True, key="btn_myjesty"):
            say("Myjesty", "I'm Myjesty")
            st.session_state.stage = "signin_form"
            st.rerun()

    elif stage == "signin_form":
        with st.form("Myjesty's profile:"):
            name = st.text_input("Name", key="signin_name")
            age = st.number_input("Age", min_value=1, max_value=120, step=1, key="signin_age")
            gender = st.selectbox(
                "Gender",
                ["Female", "Male", "Other", "Prefer not to say"],
                key="signin_gender",
            )
            loc = streamlit_geolocation() if streamlit_geolocation else None
            submitted = st.form_submit_button("Continue")
        if submitted and name:
            say("Myjesty", f"{name}, {age}, {gender}")
            st.session_state.profile = {
                "name": name,
                "age": age,
                "gender": gender,
                "lat": loc.get("latitude") if loc else None,
                "lon": loc.get("longitude") if loc else None,
            }
            say("ScoutCare", f"Thanks My Myjesty {name}! How are you feeling today?")
            st.session_state.stage = "ask_feeling"
            st.rerun()

    elif stage == "ask_feeling":
        with st.form("feeling_form"):
            feeling = st.text_input("Please type here...", key="feeling_input")
            submitted = st.form_submit_button("Send")
        if submitted and feeling:
            say("Myjesty", feeling)
            st.session_state.symptoms = extract_symptoms(feeling)
            say("ScoutCare", "Got it. What else are you experiencing My Myjesty?")
            st.session_state.stage = "ask_more"
            st.rerun()

    elif stage == "ask_more":
        with st.form("more_form"):
            more = st.text_input("Please type here...", key="more_input")
            submitted = st.form_submit_button("Send")
        if submitted and more:
            say("Myjesty", more)
            st.session_state.symptoms += extract_symptoms(more)
            st.session_state.probable_options = get_probable_symptoms(st.session_state.symptoms)
            say("ScoutCare", "My Honour, Are you also experiencing any of these?")
            st.session_state.stage = "ask_probable"
            st.rerun()

    elif stage == "ask_probable":
        if not st.session_state.probable_options:
            st.write("No probable symptoms available yet.")
        else:
            choice = st.radio(
                "Pick one:",
                st.session_state.probable_options,
                key="probable_choice",
            )
            if st.button("Confirm", key="confirm_probable"):
                say("Myjesty", choice)
                st.session_state.symptoms.append(choice)
                dept = predict_department(st.session_state.symptoms)
                st.session_state.department = dept
                say("ScoutCare", f"Based on what you've shared, I'd recommend seeing a **{dept}**.")
                st.session_state.stage = "show_doctors"
                st.rerun()

    elif stage == "show_doctors":
        if not st.session_state.doctors:
            profile = st.session_state.profile
            st.session_state.doctors = get_top_doctors(
                st.session_state.department,
                profile.get("lat"),
                profile.get("lon"),
            )
            say("ScoutCare", "Here are some top available doctors near you My Honour:")
            st.rerun()

        for d in st.session_state.doctors:
            dist = f" · {d['distance_km']} km away" if d["distance_km"] is not None else ""
            st.markdown(
                f'<strong class="doctor-name">{html.escape(d["name"])}</strong> '
                f'— {d["rating"]}/5{html.escape(dist)}',
                unsafe_allow_html=True,
            )
            if st.button(f"Book with {d['name']}", key=f"pick_{d['id']}"):
                say("Myjesty", f"Book with {d['name']}")
                st.session_state.chosen_doctor = d
                st.session_state.slots = get_available_slots(d["id"])
                say("ScoutCare", "Would you like to book an appointment? Here are the open slots:")
                st.session_state.stage = "choose_slot"
                st.rerun()

    elif stage == "choose_slot":
        slot_labels = [f'{s["date"]} at {s["time"]}' for s in st.session_state.slots]
        if not slot_labels:
            st.markdown(
                '<p class="no-slots-message">OPPS!! No open slots right now.</p>',
                unsafe_allow_html=True,
            )
        else:
            picked = st.radio("Choose a slot:", slot_labels, key="slot_choice")
            if st.button("Book this slot", key="book_slot_btn"):
                idx = slot_labels.index(picked)
                st.session_state.chosen_slot = st.session_state.slots[idx]
                say("Myjesty", f"Book {picked}")
                ok = book_slot(
                    st.session_state.chosen_slot["id"],
                    st.session_state.profile.get("name", "Guest"),
                )
                if ok:
                    say("ScoutCare", "My myjesty Your appointment is confirmed!")
                    st.session_state.stage = "done"
                else:
                    say(
                        "ScoutCare",
                        "My appologies My Honour, that slot was just taken please pick another one.",
                    )
                    st.session_state.slots = get_available_slots(st.session_state.chosen_doctor["id"])
                st.rerun()

    # Calendar option disabled for now.
    # elif stage == "ask_calendar":
    #     c1, c2 = st.columns(2)
    #     if c1.button("Yes, add it", key="cal_yes"):
    #         slot, doctor = st.session_state.chosen_slot, st.session_state.chosen_doctor
    #         added = add_to_google_calendar(f"Appointment with {doctor['name']}", slot["date"], slot["time"])
    #         say(
    #             "ScoutCare",
    #             "Added to your calendar!" if added else "Calendar isn't connected yet, but your appointment is booked.",
    #         )
    #         st.session_state.stage = "done"
    #         st.rerun()
    #     if c2.button("No thanks", key="cal_no"):
    #         say("ScoutCare", "It's all good to GO!")
    #         st.session_state.stage = "done"
    #         st.rerun()

    elif stage == "done":
        st.success("Thanks for the conversation, My Honour! Refresh the page to start over.")

    st.markdown('</div>', unsafe_allow_html=True)  # close .stage-area
    st.markdown('</div>', unsafe_allow_html=True)  # close .chat-area