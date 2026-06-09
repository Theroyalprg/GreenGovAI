"""
app.py — GreenGov AI: Streamlit Frontend
Citizen-facing chat interface for discovering sustainability subsidies.
"""

import os
import time
from pathlib import Path

import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv

from rag import initialize_rag
from agents import run_agentic_pipeline

# ─────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="GreenGov AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Load Environment & Custom CSS
# ─────────────────────────────────────────────
load_dotenv()

CUSTOM_CSS = """
<style>
/* ── Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display:ital@0;1&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── Root palette ── */
:root {
    --green-900: #0d3b1e;
    --green-700: #1a6b38;
    --green-500: #2d9d5c;
    --green-300: #6fcf97;
    --green-100: #e8f7ee;
    --earth-600: #7c5c3a;
    --earth-100: #f5ede3;
    --sky-500:   #1a7fc1;
    --neutral-50: #fafaf8;
    --neutral-100: #f2f2ee;
    --neutral-800: #1c1c1a;
    --warning:   #e67e22;
}

/* ── App background ── */
.stApp {
    background-color: var(--neutral-50);
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(160deg, var(--green-900) 0%, #143d22 100%);
    color: #e8f7ee;
    border-right: none;
}
section[data-testid="stSidebar"] * {
    color: #e8f7ee !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    font-family: 'DM Serif Display', serif;
    letter-spacing: -0.3px;
}
section[data-testid="stSidebar"] .scheme-card {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 10px;
    padding: 10px 14px;
    margin: 6px 0;
    font-size: 0.88rem;
    line-height: 1.5;
}
section[data-testid="stSidebar"] .scheme-card strong {
    display: block;
    font-size: 0.92rem;
    color: var(--green-300) !important;
    margin-bottom: 2px;
}

/* ── Hero header ── */
.hero-header {
    background: linear-gradient(135deg, var(--green-900) 0%, var(--green-700) 60%, var(--green-500) 100%);
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 24px;
    color: white;
}
.hero-header h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 2.2rem;
    margin: 0 0 6px 0;
    letter-spacing: -0.5px;
    color: white;
}
.hero-header p {
    font-size: 1rem;
    opacity: 0.88;
    margin: 0;
    color: rgba(255,255,255,0.9);
}
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.2);
    border: 1px solid rgba(255,255,255,0.3);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.78rem;
    letter-spacing: 0.5px;
    margin-bottom: 10px;
    color: white;
}

/* ── Chat messages ── */
.chat-message-user {
    background: var(--green-100);
    border-left: 4px solid var(--green-500);
    border-radius: 0 12px 12px 0;
    padding: 14px 18px;
    margin: 8px 0;
    font-size: 0.97rem;
}
.chat-message-ai {
    background: white;
    border: 1px solid #e4e9e5;
    border-radius: 12px;
    padding: 18px 22px;
    margin: 8px 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}
.chat-message-ai .agent-badge {
    display: inline-block;
    background: var(--green-100);
    color: var(--green-700);
    font-size: 0.72rem;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 10px;
    margin-bottom: 10px;
    letter-spacing: 0.3px;
}

/* ── Profile pill ── */
.profile-pill {
    background: var(--earth-100);
    border: 1px solid #e0d0be;
    border-radius: 8px;
    padding: 8px 14px;
    font-size: 0.82rem;
    color: var(--earth-600);
    margin: 6px 0 14px;
    line-height: 1.6;
}

/* ── Disclaimer box ── */
.disclaimer {
    background: #fff8ed;
    border: 1px solid #f5c97a;
    border-radius: 10px;
    padding: 10px 16px;
    font-size: 0.82rem;
    color: #8a6320;
    margin-top: 16px;
}

/* ── Status banners ── */
.status-ok {
    background: var(--green-100);
    color: var(--green-700);
    border: 1px solid #b2dfc4;
    border-radius: 8px;
    padding: 8px 14px;
    font-size: 0.85rem;
}
.status-warn {
    background: #fff3e0;
    color: #c0692b;
    border: 1px solid #f5c97a;
    border-radius: 8px;
    padding: 8px 14px;
    font-size: 0.85rem;
}

/* ── Input area ── */
.stTextInput > div > div > input,
.stTextArea textarea {
    border: 1.5px solid #cce6d8;
    border-radius: 10px;
    font-size: 0.96rem;
    background: white;
}
.stTextInput > div > div > input:focus,
.stTextArea textarea:focus {
    border-color: var(--green-500);
    box-shadow: 0 0 0 3px rgba(45,157,92,0.12);
}

/* ── Buttons ── */
.stButton > button {
    background: var(--green-700);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    padding: 10px 24px;
    font-family: 'DM Sans', sans-serif;
    transition: background 0.2s;
}
.stButton > button:hover {
    background: var(--green-500);
    color: white;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    font-weight: 600;
    font-size: 0.9rem;
    color: var(--green-700);
}

/* ── Divider ── */
hr {
    border: none;
    border-top: 1px solid #e4e9e5;
    margin: 18px 0;
}

/* ── Spinner override ── */
.stSpinner > div {
    border-top-color: var(--green-500) !important;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Session State Initialization
# ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "scheme_names" not in st.session_state:
    st.session_state.scheme_names = []

if "rag_status" not in st.session_state:
    st.session_state.rag_status = None

if "gemini_model" not in st.session_state:
    st.session_state.gemini_model = None


# ─────────────────────────────────────────────
# Scheme Metadata (sidebar descriptions)
# ─────────────────────────────────────────────
SCHEME_INFO = {
    "PM-KUSUM": {
        "icon": "☀️",
        "desc": "Solar pumps & grid-connected solar for farmers.",
        "ministry": "Ministry of New & Renewable Energy",
    },
    "PMKSY": {
        "icon": "💧",
        "desc": "Irrigation infrastructure & Har Khet Ko Pani.",
        "ministry": "Ministry of Jal Shakti",
    },
    "Soil_Health_Card": {
        "icon": "🧪",
        "desc": "Free soil testing & nutrient recommendations.",
        "ministry": "Ministry of Agriculture",
    },
    "PMFBY": {
        "icon": "🛡️",
        "desc": "Crop insurance against natural calamities.",
        "ministry": "Ministry of Agriculture",
    },
    "Micro_Irrigation": {
        "icon": "🌊",
        "desc": "Drip & sprinkler irrigation subsidies.",
        "ministry": "Ministry of Agriculture",
    },
}


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
def render_sidebar(scheme_names: list, rag_status: str):
    with st.sidebar:
        st.markdown("## 🌿 GreenGov AI")
        st.markdown(
            "<p style='font-size:0.82rem;opacity:0.75;margin-top:-8px;'>"
            "Sustainability Subsidies Assistant</p>",
            unsafe_allow_html=True,
        )
        st.divider()

        # RAG status
        if rag_status:
            if "no_pdfs" in rag_status:
                st.markdown(
                    '<div class="status-warn">⚠️ No PDFs found in <code>/data</code>. '
                    "Add scheme PDFs to enable RAG.</div>",
                    unsafe_allow_html=True,
                )
            else:
                loaded_label = (
                    "📦 Cache loaded" if "cache" in rag_status
                    else f"🔨 Index built"
                )
                st.markdown(
                    f'<div class="status-ok">✅ Knowledge base ready — {loaded_label}</div>',
                    unsafe_allow_html=True,
                )

        st.markdown("### 📋 Supported Schemes")

        displayed = 0
        for name in scheme_names:
            # Match against known scheme metadata (fuzzy key match)
            info = None
            for key in SCHEME_INFO:
                if key.lower().replace("_", "") in name.lower().replace("_", "").replace("-", ""):
                    info = SCHEME_INFO[key]
                    break

            if info:
                st.markdown(
                    f'<div class="scheme-card">'
                    f'<strong>{info["icon"]} {name.replace("_", " ")}</strong>'
                    f'{info["desc"]}<br>'
                    f'<span style="opacity:0.65;font-size:0.78rem;">{info["ministry"]}</span>'
                    f"</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="scheme-card"><strong>📄 {name.replace("_", " ")}</strong></div>',
                    unsafe_allow_html=True,
                )
            displayed += 1

        if displayed == 0:
            for key, info in SCHEME_INFO.items():
                st.markdown(
                    f'<div class="scheme-card">'
                    f'<strong>{info["icon"]} {key.replace("_", " ")}</strong>'
                    f'{info["desc"]}'
                    f"</div>",
                    unsafe_allow_html=True,
                )

        st.divider()
        st.markdown("### 💡 Sample Questions")
        samples = [
            "Am I eligible for PM-KUSUM as a small farmer in Rajasthan?",
            "What documents do I need for PMKSY drip irrigation?",
            "How to apply for crop insurance under PMFBY?",
            "Soil health card scheme details for UP farmers",
            "Solar pump subsidy for 5-acre farm in Maharashtra",
        ]
        for q in samples:
            if st.button(q, key=f"sample_{q[:20]}", use_container_width=True):
                st.session_state["prefill_query"] = q
                st.rerun()

        st.divider()
        st.markdown(
            '<p style="font-size:0.75rem;opacity:0.55;line-height:1.6;">'
            "Built with LangChain · FAISS · Gemini · Streamlit<br>"
            "© 2026 GreenGov AI by Prakarsh Gupta</p>",
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────
# Initialize Gemini + RAG (cached across reruns)
# ─────────────────────────────────────────────
def initialize_app(api_key: str):
    """Set up Gemini model and RAG pipeline. Cached in session state."""
    if st.session_state.gemini_model is None:
        genai.configure(api_key=api_key)
        st.session_state.gemini_model = genai.GenerativeModel("gemini-1.5-flash")

    if st.session_state.vectorstore is None:
        with st.spinner("🔨 Loading knowledge base…"):
            vectorstore, scheme_names, status = initialize_rag(api_key)
        st.session_state.vectorstore = vectorstore
        st.session_state.scheme_names = scheme_names
        st.session_state.rag_status = status


# ─────────────────────────────────────────────
# Render Chat History
# ─────────────────────────────────────────────
def render_chat_history():
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="chat-message-user">🧑‍🌾 <strong>You:</strong> {msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="chat-message-ai">'
                '<span class="agent-badge">🤖 GREENGOV AI · AGENTIC RESPONSE</span>',
                unsafe_allow_html=True,
            )

            # Profile pill
            if msg.get("profile"):
                p = msg["profile"]
                pills = []
                if p.state:         pills.append(f"📍 {p.state}")
                if p.land_size:     pills.append(f"🌾 {p.land_size}")
                if p.crop_type:     pills.append(f"🌱 {p.crop_type}")
                if p.goal:          pills.append(f"🎯 {p.goal}")
                if p.farmer_category: pills.append(f"👤 {p.farmer_category} farmer")
                if pills:
                    st.markdown(
                        '<div class="profile-pill">'
                        + " &nbsp;|&nbsp; ".join(pills)
                        + "</div>",
                        unsafe_allow_html=True,
                    )

            st.markdown(msg["content"])

            # Sources expander
            if msg.get("sources"):
                with st.expander("📚 Document Sources Used", expanded=False):
                    for s in msg["sources"]:
                        st.markdown(f"- {s}")

            st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Main App
# ─────────────────────────────────────────────
def main():
    # ── API Key Resolution ──────────────────
    api_key = os.getenv("GEMINI_API_KEY", "")

    # ── Hero Header ─────────────────────────
    st.markdown(
        """
        <div class="hero-header">
          <span class="hero-badge">🌿 RAG + AGENTIC AI</span>
          <h1>GreenGov AI</h1>
          <p>Discover clean energy, water conservation & sustainable agriculture subsidies 
             tailored to your profile — powered by official scheme documents.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── API Key Input (if not set in env) ──
    if not api_key:
        st.markdown("#### 🔑 Enter Your Gemini API Key")
        col1, col2 = st.columns([3, 1])
        with col1:
            api_key = st.text_input(
                "Gemini API Key",
                type="password",
                placeholder="AIza...",
                label_visibility="collapsed",
            )
        with col2:
            st.link_button("Get API Key →", "https://aistudio.google.com/app/apikey")

        if not api_key:
            st.info("👆 Add your Gemini API key above to start exploring subsidies.")
            render_sidebar([], None)
            return

    # ── Initialize Backend ──────────────────
    try:
        initialize_app(api_key)
    except Exception as e:
        st.error(f"❌ Initialization failed: {e}")
        return

    # ── Sidebar ─────────────────────────────
    render_sidebar(st.session_state.scheme_names, st.session_state.rag_status)

    # ── Chat History ────────────────────────
    render_chat_history()

    # ── Input Area ──────────────────────────
    st.markdown("---")

    prefill = st.session_state.pop("prefill_query", "")

    col_input, col_btn = st.columns([5, 1])
    with col_input:
        user_query = st.text_input(
            "Ask about any government scheme",
            value=prefill,
            placeholder="e.g. I'm a small farmer in Punjab with 3 acres — which solar or water schemes can I get?",
            label_visibility="collapsed",
            key="user_input",
        )
    with col_btn:
        submit = st.button("Ask 🌿", use_container_width=True)

    # Quick action chips
    chips = ["☀️ Solar subsidies", "💧 Water conservation", "🌾 Crop insurance", "🧪 Soil health"]
    chip_cols = st.columns(len(chips))
    for i, chip in enumerate(chips):
        if chip_cols[i].button(chip, key=f"chip_{i}", use_container_width=True):
            user_query = chip.split(" ", 1)[1]
            submit = True

    # ── Process Query ────────────────────────
    if submit and user_query.strip():
        st.session_state.messages.append({"role": "user", "content": user_query})

        with st.spinner("🔍 Agents are analyzing your query…"):
            try:
                if st.session_state.vectorstore:
                    result = run_agentic_pipeline(
                        query=user_query,
                        vectorstore=st.session_state.vectorstore,
                        gemini_model=st.session_state.gemini_model,
                    )
                else:
                    # No PDFs — use Gemini directly with a disclaimer
                    response = st.session_state.gemini_model.generate_content(
                        f"Answer this question about Indian government sustainability schemes "
                        f"(clean energy, water, agriculture): {user_query}\n\n"
                        "Format with sections: Recommended Schemes, Benefits, Application Steps."
                    )
                    result = {
                        "response": response.text
                            + "\n\n---\n⚠️ *No scheme PDFs loaded — response based on general knowledge only.*",
                        "profile": None,
                        "sources": [],
                        "eligibility": "",
                        "doc_count": 0,
                    }

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["response"],
                    "profile": result.get("profile"),
                    "sources": result.get("sources", []),
                })

            except Exception as e:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"❌ Error processing your query: {e}\n\nPlease check your API key and try again.",
                    "profile": None,
                    "sources": [],
                })

        st.rerun()

    # ── Clear Chat ───────────────────────────
    if st.session_state.messages:
        if st.button("🗑️ Clear conversation", key="clear"):
            st.session_state.messages = []
            st.rerun()

    # ── Bottom Disclaimer ────────────────────
    st.markdown(
        '<div class="disclaimer">'
        "⚠️ <strong>Important:</strong> All recommendations are based on available scheme documents "
        "and should be verified through official government portals such as "
        "<a href='https://pmkisan.gov.in' target='_blank'>pmkisan.gov.in</a>, "
        "<a href='https://india.gov.in' target='_blank'>india.gov.in</a>, or your "
        "State Agriculture Department before applying."
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
