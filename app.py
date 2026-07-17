"""
ConsumerGuard AI — Streamlit Application
Single-page educational assistant for Indian consumer rights.

Layout:
    - Title + subtitle
    - Large complaint text area
    - Analyze button
    - Results: Platform, Issue Type, Category, Timeline,
               Relevant Policy (expander), Consumer Rights (success),
               Recommended Action (info), Confidence (warning/success/info),
               Sources Cited (expander)
    - Disclaimer

Run locally:
    streamlit run app.py

Environment:
    Set GEMINI_API_KEY in .env (local) or Azure App Service settings (production).
"""

import streamlit as st
from dotenv import load_dotenv

# Load .env file for local development
# On Azure, environment variables are set via App Service configuration
load_dotenv()

# ─── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="ConsumerGuard AI",
    page_icon="🛡️",
    layout="centered",          # single-column, academic layout
    initial_sidebar_state="collapsed",
)

# ─── Inline CSS (minimal, academic style) ─────────────────────────────────────
st.markdown("""
<style>
    /* Hide Streamlit default hamburger menu and footer for clean look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Slightly reduce top padding */
    .block-container {
        padding-top: 2rem;
        max-width: 800px;
    }

    /* Disclaimer box styling */
    .disclaimer-box {
        background-color: #f0f4ff;
        border-left: 4px solid #4a7fc1;
        padding: 0.75rem 1rem;
        border-radius: 4px;
        font-size: 0.85rem;
        color: #333;
        margin-top: 1.5rem;
    }

    /* Section label styling */
    .section-label {
        font-weight: 600;
        font-size: 0.95rem;
        color: #555;
        margin-bottom: 0.25rem;
    }

    /* Confidence badge */
    .confidence-high  { color: #1a7a3c; font-weight: 700; }
    .confidence-med   { color: #b36a00; font-weight: 700; }
    .confidence-low   { color: #b00020; font-weight: 700; }
</style>
""", unsafe_allow_html=True)


# ─── Header ───────────────────────────────────────────────────────────────────

st.title("🛡️ ConsumerGuard AI")
st.markdown(
    "**Understand your Amazon and Flipkart consumer rights.**  \n"
    "Powered by Indian consumer protection law and platform policies."
)
st.divider()


# ─── Complaint Input ──────────────────────────────────────────────────────────

st.markdown("#### Describe your issue")
complaint_text = st.text_area(
    label="complaint_input",
    label_visibility="collapsed",
    placeholder=(
        "Amazon rejected replacement for my laptop after 5 days "
        "because the box was opened."
    ),
    height=150,
    help="Describe what happened with your order, including the platform name, product, and timeline if possible.",
)

analyze_button = st.button(
    "🔍 Analyze Complaint",
    type="primary",
    use_container_width=True,
    disabled=(not complaint_text.strip()),
)


# ─── Analysis & Results ───────────────────────────────────────────────────────

if analyze_button and complaint_text.strip():

    with st.spinner("Analyzing your complaint… This may take a few seconds."):
        try:
            # Import here so the app can still display the UI if imports fail
            from consumerguard.pipeline import analyze
            result = analyze(complaint_text.strip())
        except ValueError as e:
            # Missing API key — show clear error
            st.error(f"⚠️ Configuration Error: {e}")
            st.stop()
        except Exception as e:
            # ChromaDB not built yet, or other unexpected error
            st.error(
                f"❌ An error occurred: {e}\n\n"
                "If this is your first run, make sure to build the knowledge base:\n"
                "```\npython -m consumerguard.ingest\n```"
            )
            st.stop()

    st.divider()
    st.divider()

    # ── 🛒 Complaint Summary ───────────────────────────────────────────────
    st.markdown("#### 🛒 Complaint Summary")
    st.divider()
    platform = result.get("platform", "Unknown")
    issue_type = result.get("issue_type", "Unknown")
    category = result.get("product_category", "Others")
    timeline = result.get("timeline") or "Not Mentioned"

    st.write(f"- **Platform:** {platform}")
    st.write(f"- **Issue Type:** {issue_type}")
    st.write(f"- **Product Category:** {category}")
    st.write(f"- **Timeline:** {timeline}")
    st.write("")

    # ── 📜 Relevant Platform Policy ────────────────────────────────────────
    st.markdown("#### 📜 Relevant Platform Policy")
    st.divider()
    st.write(result.get("relevant_platform_policy", ""))
    st.write("")

    # ── ⚖️ Consumer Protection Rules ──────────────────────────────────────
    st.markdown("#### ⚖️ Consumer Protection Rules")
    st.divider()
    rules = result.get("consumer_rules", [])
    if isinstance(rules, list) and rules:
        for rule in rules:
            st.markdown(f"- {rule}")
    else:
        st.write(str(rules))
    st.write("")

    # ── 🏛 Consumer Protection Act Reference ──────────────────────────────
    st.markdown("#### 🏛 Consumer Protection Act 2019")
    st.divider()
    rights = result.get("consumer_act_rights", [])
    if isinstance(rights, list) and rights:
        for right in rights:
            st.markdown(f"- {right}")
    else:
        st.write(str(rights))
    st.write("")

    # ── ✅ Recommended Actions ─────────────────────────────────────────────
    st.markdown("#### ✅ Recommended Actions")
    st.divider()
    actions = result.get("recommended_actions", [])
    if isinstance(actions, list) and actions:
        for i, action in enumerate(actions, start=1):
            st.write(f"{i}. {action}")
    else:
        st.write(str(actions))
    st.write("")

    # ── 📊 Confidence Level ────────────────────────────────────────────────
    st.markdown("#### 📊 Confidence Level")
    st.divider()
    confidence = result.get("confidence_level", "Low")
    reason = result.get("confidence_reason", "")
    
    st.write(f"**{confidence.capitalize()}**")
    st.write(reason)
    st.write("")

    # ── ℹ️ Disclaimer ──────────────────────────────────────────────────────
    st.markdown("#### ℹ️ Disclaimer")
    st.divider()
    st.caption("ConsumerGuard AI provides educational information based on publicly available policies and consumer regulations. This is not legal advice.")

# ─── Static Disclaimer (always visible at bottom) ─────────────────────────────
elif not analyze_button:
    st.divider()
    st.markdown("#### ℹ️ Disclaimer")
    st.divider()
    st.caption("ConsumerGuard AI provides educational information based on publicly available policies and consumer regulations. This is not legal advice.")

    # ── How to Use section ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 💡 How to use")
    st.markdown("""
    1. **Type your complaint** in the text box above — mention the platform (Amazon or Flipkart), the product, and what went wrong.
    2. **Click Analyze Complaint** — the system will detect the issue type and search relevant policies.
    3. **Review the results** — understand your rights and the recommended action steps.

    **Example complaints you can try:**
    - *"Amazon rejected replacement for my laptop after 5 days because the box was opened."*
    - *"Flipkart is not accepting my return request for a kurta. It has been only 3 days."*
    - *"I ordered a refrigerator from Amazon. The seller is a third-party and refusing any refund."*
    """)
