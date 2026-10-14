"""RecallGuard - Temporal Memory Reliability Firewall for AI Agents.

Feature 02: Local AI Memory Extraction with Verbatim Grounding.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import streamlit as st
import pandas as pd

from core.database import DatabaseManager, DEFAULT_DB_PATH
from core.llm import OllamaClient
from core.memory_service import MemoryService
from core.schemas import MemoryClaim

# Set page configuration
st.set_page_config(
    page_title="RecallGuard | Temporal Memory Firewall",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(120deg, #3B82F6, #8B5CF6, #EC4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.15rem;
        color: #94A3B8;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }
    .claim-card {
        background-color: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.9rem;
    }
    .quote-box {
        background-color: rgba(15, 23, 42, 0.85);
        border-left: 3px solid #10B981;
        padding: 0.6rem 0.9rem;
        border-radius: 0px 6px 6px 0px;
        font-size: 0.88rem;
        font-style: italic;
        color: #D1FAE5;
        margin-top: 0.5rem;
    }
    .quote-rejected {
        border-left-color: #EF4444;
        color: #FCA5A5;
    }
    .status-badge-ok {
        background-color: #064E3B;
        color: #34D399;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }
    .status-badge-warn {
        background-color: #78350F;
        color: #FBBF24;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }
    .status-badge-err {
        background-color: #7F1D1D;
        color: #F87171;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }
    .callout-box {
        background: rgba(15, 23, 42, 0.7);
        border-left: 4px solid #3B82F6;
        padding: 1rem 1.2rem;
        border-radius: 4px 8px 8px 4px;
        margin: 1rem 0;
    }
    .callout-danger {
        border-left-color: #EF4444;
    }
    .callout-warning {
        border-left-color: #F59E0B;
    }
    .roadmap-pill {
        background-color: rgba(59, 130, 246, 0.15);
        border: 1px solid rgba(59, 130, 246, 0.4);
        color: #60A5FA;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.78rem;
        font-weight: 600;
        display: inline-block;
        margin-right: 6px;
    }
    .roadmap-active {
        background-color: rgba(16, 185, 129, 0.2);
        border: 1px solid rgba(16, 185, 129, 0.5);
        color: #34D399;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_service() -> MemoryService:
    """Initialize and cache the MemoryService instance."""
    db = DatabaseManager(DEFAULT_DB_PATH)
    db.init_db()
    llm = OllamaClient()
    return MemoryService(db_manager=db, llm_client=llm)


service = get_service()

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### 🛡️ RecallGuard Engine")
    st.caption("Temporal Memory Reliability Firewall")
    st.divider()

    st.markdown("#### ⚙️ Engine Status")
    # Check Ollama connection
    ollama_ok = service.llm.check_connection()
    model_ok = service.llm.is_model_available() if ollama_ok else False

    if ollama_ok:
        st.markdown(
            "Ollama Daemon: <span class='status-badge-ok'>CONNECTED</span>",
            unsafe_allow_html=True,
        )
        if model_ok:
            st.markdown(
                f"Model (`{service.llm.model}`): <span class='status-badge-ok'>READY</span>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"Model (`{service.llm.model}`): <span class='status-badge-warn'>MISSING</span>",
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            "Ollama Daemon: <span class='status-badge-warn'>OFFLINE</span>",
            unsafe_allow_html=True,
        )
        st.caption("Local Ollama daemon not reachable at localhost:11434.")

    st.markdown(
        f"Database: <span class='status-badge-ok'>SQLITE</span> `{Path(DEFAULT_DB_PATH).name}`",
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown("#### 📊 Store Overview")
    stats = service.get_stats()
    col_sb1, col_sb2 = st.columns(2)
    col_sb1.metric("Total Claims", stats["total"])
    col_sb2.metric("Active", stats["active"])

    st.divider()
    st.markdown("#### 🛠️ Database Actions")
    if st.button("Safely Re-initialize DB", use_container_width=True):
        service.initialize()
        st.success("Database schema verified and initialized.")
        st.rerun()

    st.caption("Local execution only • Zero cloud API calls • Ollama qwen3.5:0.8b")


# --- MAIN HEADER ---
st.markdown("<div class='main-title'>🛡️ RecallGuard</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-title'>Temporal Memory Reliability Firewall for Autonomous AI Agents</div>",
    unsafe_allow_html=True,
)

# Milestone indicator
st.markdown(
    """
    <span class='roadmap-pill' style='opacity:0.8'>MILESTONE 1: Foundation (Done)</span>
    <span class='roadmap-pill roadmap-active'>MILESTONE 2: Local AI Memory Extraction (Active)</span>
    <span class='roadmap-pill' style='opacity:0.6'>NEXT: Feature 03 Conflict Resolution</span>
    """,
    unsafe_allow_html=True,
)
st.write("")

# --- TABS ---
tab_extract, tab_store, tab_cases, tab_problem, tab_overview = st.tabs(
    [
        "🧠 Local AI Memory Extraction",
        "💾 Claim Store & Schema Inspector",
        "📂 Failure Scenarios (Demo Cases)",
        "⚠️ The Memory Problem Explained",
        "🏛️ Foundation Overview",
    ]
)

# ==============================================================================
# TAB 1: Local AI Memory Extraction (Feature 02)
# ==============================================================================
with tab_extract:
    st.subheader("Extract Memory Claims with Local Qwen 3.5 (0.8B)")
    st.write(
        """
        Input any conversation transcript below. RecallGuard uses **local Ollama** to decompose 
        unstructured dialogue into structured **(Subject, Predicate, Object)** atomic claims.
        
        Every claim must pass the **Verbatim Source Quote Firewall**: if the model extracts an assertion 
        whose supporting quote cannot be verified word-for-word in the transcript, the claim is 
        **quarantined and rejected** to prevent memory hallucinations.
        """
    )

    # Sample presets
    SAMPLE_CONVERSATIONS = {
        "Tech Stack Migration": (
            "User: Hi, we had a team architecture meeting today.\n"
            "Assistant: How can I help document the outcomes?\n"
            "User: We officially decided to migrate our backend from Node.js to Go 1.22 starting next month.\n"
            "Assistant: Got it. Any specific database changes?\n"
            "User: Yes, we are deprecating MongoDB and standardizing on PostgreSQL 16 for all new services."
        ),
        "User Tooling & Preferences": (
            "User: I always use dark mode in VS Code and set tab indentation to 4 spaces.\n"
            "Assistant: Noted! Do you have a preferred timezone for scheduling reminders?\n"
            "User: Yes, my primary timezone is America/New_York."
        ),
        "Project Deadline Update": (
            "User: Executive leadership sent an urgent announcement about Project Apollo.\n"
            "Assistant: What changed?\n"
            "User: The launch date for Project Apollo is postponed from July 2026 to November 15, 2026.\n"
            "Assistant: I will update the project roadmap records accordingly."
        ),
    }

    st.markdown("##### 💡 Load Sample Conversation:")
    p_cols = st.columns(len(SAMPLE_CONVERSATIONS))
    for idx, (label, sample_text) in enumerate(SAMPLE_CONVERSATIONS.items()):
        if p_cols[idx].button(f"📄 {label}", use_container_width=True):
            st.session_state["conv_input"] = sample_text
            st.session_state["session_id_input"] = f"session-{label.lower().replace(' ', '-')}"
            st.rerun()

    # Form for conversation input
    with st.form("extraction_form"):
        col_s1, col_s2 = st.columns([1.5, 1])
        session_id_val = st.session_state.get("session_id_input", "session-conversation-001")
        session_id = col_s1.text_input("Source Session ID", value=session_id_val)
        col_s2.markdown(
            f"**Model:** `{service.llm.model}` (Local Ollama) &nbsp; "
            f"{'🟢 Ready' if model_ok else '🔴 Not Ready'}"
        )

        default_conv = st.session_state.get(
            "conv_input", SAMPLE_CONVERSATIONS["Tech Stack Migration"]
        )
        conv_text = st.text_area(
            "Conversation Transcript",
            value=default_conv,
            height=160,
            help="Enter multi-turn dialogue or agent session text.",
        )

        extract_submitted = st.form_submit_button(
            "🚀 Extract Memory Claims (Local Ollama)",
            type="primary",
            use_container_width=True,
        )

    if extract_submitted:
        if not conv_text.strip():
            st.error("Please enter conversation text to extract claims from.")
        elif not ollama_ok:
            st.error(
                "Local Ollama daemon is offline. Please run `ollama serve` and ensure port 11434 is listening."
            )
        elif not model_ok:
            st.error(
                f"Model `{service.llm.model}` is missing in Ollama. Please run `ollama pull {service.llm.model}`."
            )
        else:
            with st.spinner(f"Extracting atomic claims using local `{service.llm.model}`..."):
                t_start = datetime.now()
                extraction_result = service.extract_claims_from_text(
                    text=conv_text, session_id=session_id
                )
                duration = (datetime.now() - t_start).total_seconds()

            st.session_state["last_result"] = extraction_result
            st.session_state["last_duration"] = duration
            st.session_state["last_conv"] = conv_text
            st.session_state["review_approved"] = {
                claim.id: True for claim in extraction_result.valid_claims
            }

    # Display extraction outcomes if available
    if "last_result" in st.session_state:
        res = st.session_state["last_result"]
        dur = st.session_state.get("last_duration", 0.0)

        st.write("---")
        if not res.success:
            st.error(f"❌ Extraction Failed: {res.error_message}")
            if res.raw_model_output:
                with st.expander("Show Raw Model Output"):
                    st.code(res.raw_model_output)

        # Show Rejections / Firewall Quarantine
        if res.rejected_claims:
            st.markdown(
                f"""
                <div class='callout-box callout-danger'>
                    <h4>⚠️ Firewall Notice: {len(res.rejected_claims)} Candidate Claim(s) Quarantined / Rejected</h4>
                    <p>The following assertions were generated by the model but <b>failed verbatim quote verification</b> or schema constraints.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            for rej in res.rejected_claims:
                with st.container():
                    st.markdown(f"**Rejection Reason:** `{rej.reason}`")
                    st.code(json.dumps(rej.claim_data, indent=2), language="json")

        # Show Valid Claims for User Review
        if res.valid_claims:
            st.markdown(
                f"#### ✅ Validated Claims ({len(res.valid_claims)}) — User Review & Approval "
                f"<span style='font-size:0.85rem; color:#94A3B8;'>(Inference took {dur:.2f}s)</span>",
                unsafe_allow_html=True,
            )
            st.caption(
                "Review the proposed claims below. Uncheck any claim you do not want to persist, then click 'Save Approved Claims'."
            )

            # Review checkboxes
            approved_map = st.session_state.get("review_approved", {})
            for c in res.valid_claims:
                with st.container():
                    chk_val = approved_map.get(c.id, True)
                    col_chk, col_content = st.columns([0.15, 0.85])
                    with col_chk:
                        is_checked = st.checkbox(
                            "Approve",
                            value=chk_val,
                            key=f"chk_claim_{c.id}",
                            label_visibility="visible",
                        )
                        approved_map[c.id] = is_checked

                    with col_content:
                        st.markdown(
                            f"""
                            <div class='claim-card'>
                                <b>Subject:</b> <code>{c.subject}</code> &nbsp;|&nbsp; 
                                <b>Predicate:</b> <code>{c.predicate}</code> &nbsp;|&nbsp; 
                                <b>Object:</b> <code>{c.object_value}</code> &nbsp;|&nbsp; 
                                <b>Confidence:</b> <code>{c.confidence:.2f}</code> &nbsp;
                                <span class='status-badge-ok'>VERBATIM GROUNDED</span>
                                <div class='quote-box'>
                                    " {c.source_quote or 'N/A'} "
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            st.session_state["review_approved"] = approved_map

            # Save Approved Claims Button
            col_save1, col_save2 = st.columns([1, 1])
            with col_save1:
                if st.button("💾 Save Approved Claims to SQLite Database", type="primary", use_container_width=True):
                    claims_to_save = [
                        c for c in res.valid_claims if approved_map.get(c.id, True)
                    ]
                    if claims_to_save:
                        service.save_claims(claims_to_save)
                        st.success(f"🎉 Successfully saved {len(claims_to_save)} claim(s) to SQLite memory store!")
                        # Clear reviewed claims to avoid duplicate saves
                        st.session_state.pop("last_result", None)
                        st.session_state.pop("review_approved", None)
                        st.rerun()
                    else:
                        st.warning("No claims selected for approval.")

            with col_save2:
                if st.button("Discard Proposed Claims", use_container_width=True):
                    st.session_state.pop("last_result", None)
                    st.session_state.pop("review_approved", None)
                    st.rerun()

        elif res.success and not res.rejected_claims:
            st.info("ℹ️ No factual memory claims were detected in the conversation.")

        # Expander for raw JSON
        if res.raw_model_output:
            with st.expander("🔍 Inspect Raw Model Output"):
                st.code(res.raw_model_output, language="json")


# ==============================================================================
# TAB 2: Claim Store & Schema Inspector
# ==============================================================================
with tab_store:
    st.subheader("MemoryClaim Store & Database Inspector")

    col_view, col_add = st.columns([1.3, 0.7])

    with col_view:
        st.markdown("##### 📋 Stored Memory Claims in SQLite")
        claims = service.list_claims(active_only=False)

        if claims:
            data = []
            for c in claims:
                data.append(
                    {
                        "ID": c.id[:8] + "...",
                        "Subject": c.subject,
                        "Predicate": c.predicate,
                        "Object Value": c.object_value,
                        "Source Quote": c.source_quote or "-",
                        "Confidence": c.confidence,
                        "Expired?": "⚠️ EXPIRED" if c.is_expired() else "✅ FRESH",
                        "Active": "Yes" if c.is_active else "No",
                        "Session": c.source_session_id or "-",
                    }
                )
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No claims stored yet in the database. Extract claims from the first tab or import demo cases.")

    with col_add:
        st.markdown("##### ➕ Manually Insert a Test Claim")
        with st.form("new_claim_form"):
            subject = st.text_input("Subject", placeholder="e.g. backend_framework")
            predicate = st.text_input("Predicate", placeholder="e.g. language_version")
            object_val = st.text_input("Object Value", placeholder="e.g. Go 1.22")
            source_quote_input = st.text_input("Source Quote", placeholder="Exact quote from user")
            confidence = st.slider("Confidence", min_value=0.0, max_value=1.0, value=0.95, step=0.05)
            session_id_man = st.text_input("Source Session ID", value="manual-test-session")
            is_active = st.checkbox("Is Active", value=True)

            submitted = st.form_submit_button("Persist Claim to SQLite")
            if submitted:
                if not subject.strip() or not predicate.strip() or not object_val.strip():
                    st.error("Subject, Predicate, and Object Value are required.")
                else:
                    new_claim = MemoryClaim(
                        subject=subject,
                        predicate=predicate,
                        object_value=object_val,
                        source_quote=source_quote_input.strip() or None,
                        confidence=confidence,
                        source_session_id=session_id_man,
                        is_active=is_active,
                    )
                    service.save_claim(new_claim)
                    st.success(f"Claim persisted successfully! ID: {new_claim.id[:8]}...")
                    st.rerun()

    st.write("---")
    st.markdown("##### 📐 Evolved Pydantic Schema Definition (`core/schemas.py`)")
    st.code(
        """class MemoryClaim(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    subject: str = Field(..., min_length=1)
    predicate: str = Field(..., min_length=1)
    object_value: str = Field(..., min_length=1)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source_quote: Optional[str] = Field(default=None, description="Verbatim quote from conversation")
    source_session_id: Optional[str] = None
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    is_active: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)""",
        language="python",
    )


# ==============================================================================
# TAB 3: Failure Scenarios (Demo Cases)
# ==============================================================================
with tab_cases:
    st.subheader("Curated Failure Scenarios")
    st.write(
        "Inspect these concrete failure cases from `data/demo_cases.json` to see how unmanaged agent memory causes bugs."
    )

    demo_file = Path("data/demo_cases.json")
    if demo_file.exists():
        with open(demo_file, "r") as f:
            demo_cases = json.load(f)

        case_titles = [f"{c['title']} ({c['category']})" for c in demo_cases]
        selected_idx = st.selectbox("Select a demo case to examine:", range(len(demo_cases)), format_func=lambda i: case_titles[i])
        case = demo_cases[selected_idx]

        st.markdown(f"#### 🔍 {case['title']}")
        st.write(f"**Category:** `{case['category']}` | **Severity:** `{case['severity'].upper()}`")
        st.info(f"**Scenario:** {case['description']}")

        c_col1, c_col2 = st.columns(2)
        with c_col1:
            st.markdown("##### 🗃️ Stored Memory Claims in Agent Brain")
            for idx, claim in enumerate(case["stored_claims"]):
                st.markdown(
                    f"""
                    - **Claim #{idx+1}:** `({claim['subject']}, {claim['predicate']}, "{claim['object_value']}")`
                      - Valid From: `{claim['valid_from']}`
                      - Valid Until: `{claim['valid_until'] or 'Indefinite'}`
                      - Confidence: `{claim['confidence']}`
                    """
                )

        with c_col2:
            st.markdown("##### 💬 Incoming User Interaction")
            st.code(case["incoming_interaction"], language="markdown")
            st.markdown("##### 💥 Failure & Risk Analysis")
            st.error(case["risk_analysis"])

        if st.button("📥 Import This Case's Claims into SQLite Database"):
            imported = 0
            for raw_claim in case["stored_claims"]:
                claim_obj = MemoryClaim(
                    subject=raw_claim["subject"],
                    predicate=raw_claim["predicate"],
                    object_value=raw_claim["object_value"],
                    valid_from=raw_claim.get("valid_from"),
                    valid_until=raw_claim.get("valid_until"),
                    confidence=raw_claim.get("confidence", 1.0),
                    source_session_id=case["id"],
                    metadata={"case_title": case["title"], "category": case["category"]},
                )
                service.save_claim(claim_obj)
                imported += 1
            st.success(f"Successfully imported {imported} claims into SQLite store!")
            st.rerun()

    else:
        st.warning("Demo cases file `data/demo_cases.json` not found.")


# ==============================================================================
# TAB 4: The Memory Problem Explained
# ==============================================================================
with tab_problem:
    st.subheader("Why Autonomous Agents Fail: The Memory Decay Crisis")

    st.write(
        """
        When developers give agents persistent memory (vector databases, document dumps, or naive key-value stores), 
        they inadvertently create two catastrophic reliability failure modes:
        """
    )

    col_p1, col_p2 = st.columns(2)

    with col_p1:
        st.markdown(
            """
            <div class='callout-box callout-warning'>
                <h3>1. Stale Memory (Temporal Obsolescence)</h3>
                <p><b>What happens:</b> An agent records a fact that was true at timestamp <i>T<sub>0</sub></i>, but expires or changes at <i>T<sub>1</sub></i>.</p>
                <p><b>Example:</b></p>
                <ul>
                    <li><i>T<sub>0</sub> (Jan 2024):</i> "We use Python 3.9 for our microservices."</li>
                    <li><i>T<sub>1</sub> (Oct 2024):</i> Python 3.9 reaches End of Life; production upgrades to Python 3.12.</li>
                    <li><i>T<sub>2</sub> (Nov 2024):</i> When asked to generate a Dockerfile, the agent blindly queries its memory and outputs Python 3.9, introducing security flaws and build failures.</li>
                </ul>
                <p><b>Root Cause:</b> Traditional memory systems perform semantic similarity search without temporal validity boundaries. A stale memory with high semantic similarity will outrank or pollute the prompt.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_p2:
        st.markdown(
            """
            <div class='callout-box callout-danger'>
                <h3>2. Contradictory Memory (Logical Inconsistency)</h3>
                <p><b>What happens:</b> Multi-turn or multi-session interactions record opposing claims about the same subject and predicate.</p>
                <p><b>Example:</b></p>
                <ul>
                    <li><i>Session 1:</i> "User prefers strict TypeScript with functional programming style."</li>
                    <li><i>Session 4:</i> "User switched project codebase to Go with standard OOP."</li>
                    <li><i>Session 7:</i> The agent receives both memories in context and produces a Frankenstein hybrid or unpredictably alternates between styles across prompts.</li>
                </ul>
                <p><b>Root Cause:</b> Lack of temporal resolution and claim-level contradiction graph. The agent treats all historical assertions as concurrently valid.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### The RecallGuard Solution: Temporal Memory Firewall")
    st.write(
        """
        RecallGuard breaks raw text memories into atomic **(Subject, Predicate, Object)** claims tagged with:
        - **Verbatim Grounding Quote:** Verified source quote preventing hallucinated extraction.
        - **Temporal Bounds:** `valid_from`, `valid_until`
        - **Reliability Attributes:** `confidence`, `source_session_id`, `is_active`
        
        Before any memory enters the agent's prompt window, RecallGuard audits the candidate set against current query time,
        quarantines superseded records, and alerts on unresolved logical conflicts.
        """
    )


# ==============================================================================
# TAB 5: Foundation Overview
# ==============================================================================
with tab_overview:
    st.subheader("Welcome to RecallGuard")
    st.write(
        """
        **RecallGuard** acts as an intermediary **Temporal Memory Reliability Firewall** between 
        long-term episodic storage and an autonomous AI agent's inference context.
        """
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div class='metric-card'>
                <h4>🕒 Temporal Validity</h4>
                <p>Every stored claim tracks <code>valid_from</code> and <code>valid_until</code> timestamps. Expired instructions are quarantined before prompt construction.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class='metric-card'>
                <h4>⚔️ Contradiction Firewall</h4>
                <p>Detects logical incompatibilities across sessions (e.g., conflicting tech stacks or shifted deadlines) to prevent chaotic hallucinations.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class='metric-card'>
                <h4>🔒 Local-First Privacy</h4>
                <p>Runs 100% locally on your machine via <b>Ollama (qwen3.5:0.8b)</b> and <b>SQLite</b>. Zero external cloud API calls or credential leaks.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("---")
    st.subheader("System Infrastructure Status")
    status_col1, status_col2 = st.columns(2)

    with status_col1:
        st.markdown("##### 🤖 Local Inference Engine")
        st.write(f"- **Provider:** Ollama Local Daemon (`{service.llm.host}`)")
        st.write(f"- **Target Model:** `{service.llm.model}`")
        st.write(f"- **Ollama Daemon Status:** {'🟢 Online' if ollama_ok else '🔴 Offline'}")
        st.write(f"- **Model Downloaded:** {'🟢 Available' if model_ok else '🔴 Missing'}")

    with status_col2:
        st.markdown("##### 🗄️ Database & Storage Engine")
        st.write(f"- **Type:** SQLite 3 (ACID compliant, zero server daemon)")
        st.write(f"- **File Location:** `{Path(DEFAULT_DB_PATH).resolve()}`")
        st.write(f"- **Recorded Claims:** {stats['total']} ({stats['active']} active)")
        st.write("- **Schema Version:** 1 (Tables: `memory_claims`, `schema_version`)")
