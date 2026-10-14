"""RecallGuard - Temporal Memory Reliability Firewall for AI Agents.

Feature 04: Evidence-Grounded Retrieval and Reliability Evaluation.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import streamlit as st
import pandas as pd

from core.database import DatabaseManager, DEFAULT_DB_PATH
from core.evaluator import ReliabilityEvaluator, build_synthetic_dataset
from core.llm import OllamaClient
from core.memory_service import MemoryService
from core.retrieval_service import RetrievalService
from core.schemas import ClaimStatus, ConflictType, MemoryClaim

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
    .conflict-container {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(245, 158, 11, 0.4);
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 1.2rem;
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
    .status-badge-neutral {
        background-color: #374151;
        color: #D1D5DB;
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
        background-color: rgba(245, 158, 11, 0.2);
        border: 1px solid rgba(245, 158, 11, 0.5);
        color: #FBBF24;
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
    c_sb1, c_sb2 = st.columns(2)
    c_sb1.metric("Total", stats["total"])
    c_sb2.metric("Active", stats["active"])

    c_sb3, c_sb4 = st.columns(2)
    c_sb3.metric("Superseded", stats.get("superseded", 0))
    pending_count = stats.get("needs_review", 0)
    c_sb4.metric("Needs Review", pending_count)

    if pending_count > 0:
        st.warning(f"⚠️ {pending_count} conflict(s) pending review in Tab 1!")

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
    <span class='roadmap-pill' style='opacity:0.8'>MILESTONE 2: Local AI Extraction (Done)</span>
    <span class='roadmap-pill' style='opacity:0.8'>MILESTONE 3: Conflict Resolution (Done)</span>
    <span class='roadmap-pill roadmap-active'>MILESTONE 4: Evidence Retrieval &amp; Reliability Lab (Active)</span>
    """,
    unsafe_allow_html=True,
)
st.write("")

# --- TABS ---
tab_conflict, tab_extract, tab_ask, tab_lab, tab_store, tab_cases, tab_problem, tab_overview = st.tabs(
    [
        "⚖️ Memory Conflict Review",
        "🧠 Local AI Memory Extraction",
        "💬 Ask RecallGuard",
        "🔬 Reliability Lab",
        "💾 Claim Store & History Inspector",
        "📂 Failure Scenarios (Demo Cases)",
        "⚠️ The Memory Problem Explained",
        "🏛️ Foundation Overview",
    ]
)

# ==============================================================================
# TAB 1: Memory Conflict Review (Feature 03)
# ==============================================================================
with tab_conflict:
    st.subheader("⚖️ Temporal Memory Conflict Resolution & Review")
    st.write(
        """
        Autonomous agents frequently encounter contradictory statements across conversations. 
        **RecallGuard** enforces deterministic evidence-grounded conflict decisions:
        - **Exact Duplicates:** Prevent redundant store bloat.
        - **Explicit Updates:** Recognized when verified source quotes contain transition evidence (e.g., *"migrated from X to Y"*).
        - **Ambiguous Changes:** Quarantined as `needs_review` so an unverified assertion never blindly supersedes established facts.
        """
    )

    review_queue = service.list_review_queue()

    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        st.markdown(f"##### 📋 Conflict Review Queue ({len(review_queue)} Pending)")
    with col_btn2:
        if st.button("🧪 Seed Demo Migration Conflict (Node.js ➔ Go 1.22)"):
            # Ensure base Node.js claim exists
            base_claim = MemoryClaim(
                subject="backend_service",
                predicate="framework",
                object_value="Node.js 20",
                source_quote="Our backend service runs on Node.js 20.",
                status=ClaimStatus.ACTIVE.value,
            )
            service.save_claim(base_claim)

            # Insert competing Go 1.22 claim as needs_review
            candidate = MemoryClaim(
                subject="backend_service",
                predicate="framework",
                object_value="Go 1.22",
                source_quote="We decided to migrate the backend service from Node.js 20 to Go 1.22.",
                status=ClaimStatus.NEEDS_REVIEW.value,
                supersedes_id=base_claim.id,
            )
            service.save_claim(candidate)
            st.success("Seeded demo conflict for review!")
            st.rerun()

    if not review_queue:
        st.info("✅ No conflicts pending review. All stored active claims are consistent.")
    else:
        for idx, item in enumerate(review_queue):
            pending = item["pending_claim"]
            existing = item["existing_claim"]
            decision = item["decision"]

            st.markdown(f"""<div class='conflict-container'>
                <h4>Conflict #{idx + 1}: Subject <code>{pending.subject}</code> • Predicate <code>{pending.predicate}</code></h4>
            """, unsafe_allow_html=True)

            c_ex, c_prop = st.columns(2)
            with c_ex:
                st.markdown("##### 📌 Existing Active Claim")
                if existing:
                    st.write(f"- **Current Value:** `{existing.object_value}`")
                    st.write(f"- **Recorded Status:** <span class='status-badge-ok'>{existing.status.upper()}</span>", unsafe_allow_html=True)
                    st.write(f"- **Valid From:** `{existing.valid_from or existing.created_at.strftime('%Y-%m-%d %H:%M')}`")
                    st.markdown(f"<div class='quote-box'>\"{existing.source_quote or 'No quote recorded'}\"</div>", unsafe_allow_html=True)
                else:
                    st.write("*No prior active claim found on this subject/predicate.*")

            with c_prop:
                st.markdown("##### 🆕 Proposed Replacement Claim")
                st.write(f"- **New Value:** `{pending.object_value}`")
                st.write(f"- **Status:** <span class='status-badge-warn'>{pending.status.upper()}</span>", unsafe_allow_html=True)
                st.write(f"- **Session ID:** `{pending.source_session_id or 'unknown'}`")
                st.markdown(f"<div class='quote-box'>\"{pending.source_quote or 'No quote'}\"</div>", unsafe_allow_html=True)

            # Decision analysis callout
            badge_class = "status-badge-ok" if decision.is_explicit_update else "status-badge-warn"
            st.markdown(
                f"""
                <div style='margin-top: 10px; padding: 10px; background: rgba(30, 41, 59, 0.4); border-radius: 6px;'>
                    <b>Conflict Type:</b> <span class='{badge_class}'>{decision.conflict_type.upper()}</span><br>
                    <b>Decision Reason:</b> {decision.decision_reason}<br>
                    <b>Recommended Action:</b> <code>{decision.recommended_action}</code>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Action buttons
            col_act1, col_act2, col_act3 = st.columns([1, 1, 2])
            with col_act1:
                if existing:
                    if st.button(f"✅ Approve Update", key=f"appr_{pending.id}"):
                        service.approve_conflict_update(
                            proposed_claim=pending,
                            existing_claim_id=existing.id,
                        )
                        st.success(f"Approved! Old claim superseded, '{pending.object_value}' is now active.")
                        st.rerun()
                else:
                    if st.button(f"✅ Activate Claim", key=f"act_{pending.id}"):
                        service.db.update_claim_status(pending.id, ClaimStatus.ACTIVE.value, is_active=True)
                        st.success("Claim activated!")
                        st.rerun()

            with col_act2:
                if st.button(f"❌ Reject Update", key=f"rej_{pending.id}"):
                    service.reject_conflict_update(
                        proposed_claim_id=pending.id,
                        reason="Rejected during user conflict review",
                    )
                    st.warning("Update rejected! Existing active claim preserved.")
                    st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)

    # --- Conflict Simulator Sandbox ---
    st.write("---")
    st.markdown("##### 🔬 Interactive Conflict Decision Sandbox")
    st.caption("Test how RecallGuard compares any candidate assertion against your active database.")

    with st.form("sandbox_conflict_form"):
        sb_col1, sb_col2 = st.columns(2)
        sb_subj = sb_col1.text_input("Subject", value="backend_service")
        sb_pred = sb_col2.text_input("Predicate", value="framework")

        sb_val = sb_col1.text_input("Proposed New Value", value="Go 1.22")
        sb_quote = sb_col2.text_input(
            "Verbatim Source Quote",
            value="We decided to migrate our backend from Node.js to Go 1.22 starting next month.",
        )

        sim_submitted = st.form_submit_button("🔍 Evaluate Conflict Decision")

    if sim_submitted:
        temp_claim = MemoryClaim(
            subject=sb_subj,
            predicate=sb_pred,
            object_value=sb_val,
            source_quote=sb_quote,
        )
        sim_decision = service.resolve_conflicts(temp_claim)

        st.markdown("###### 🎯 Firewall Evaluation Result:")
        st.write(f"- **Conflict Classification:** `{sim_decision.conflict_type}`")
        st.write(f"- **Explicit Update Detected:** `{'Yes' if sim_decision.is_explicit_update else 'No'}`")
        st.write(f"- **Recommended Action:** `'{sim_decision.recommended_action}'`")
        st.info(f"**Reason:** {sim_decision.decision_reason}")


# ==============================================================================
# TAB 2: Local AI Memory Extraction (Feature 02)
# ==============================================================================
with tab_extract:
    st.subheader("Extract Memory Claims with Local Qwen 3.5 (0.8B)")
    st.write(
        """
        Input any conversation transcript below. RecallGuard uses **local Ollama** to decompose 
        unstructured dialogue into structured **(Subject, Predicate, Object)** atomic claims.
        
        Every claim must pass the **Verbatim Source Quote Firewall** and undergoes **real-time conflict resolution** 
        against the active database.
        """
    )

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
            st.error("Local Ollama daemon is offline. Please run `ollama serve`.")
        elif not model_ok:
            st.error(f"Model `{service.llm.model}` is missing in Ollama.")
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

    if "last_result" in st.session_state:
        res = st.session_state["last_result"]
        dur = st.session_state.get("last_duration", 0.0)

        st.write("---")
        if not res.success:
            st.error(f"❌ Extraction Failed: {res.error_message}")
            if res.raw_model_output:
                with st.expander("Show Raw Model Output"):
                    st.code(res.raw_model_output)

        if res.rejected_claims:
            st.markdown(
                f"""
                <div class='callout-box callout-danger'>
                    <h4>⚠️ Firewall Notice: {len(res.rejected_claims)} Candidate Claim(s) Quarantined / Rejected</h4>
                    <p>The following assertions failed verbatim quote verification or schema constraints.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            for rej in res.rejected_claims:
                with st.container():
                    st.markdown(f"**Rejection Reason:** `{rej.reason}`")
                    st.code(json.dumps(rej.claim_data, indent=2), language="json")

        if res.valid_claims:
            st.markdown(
                f"#### ✅ Validated Claims ({len(res.valid_claims)}) — User Review & Approval "
                f"<span style='font-size:0.85rem; color:#94A3B8;'>(Inference took {dur:.2f}s)</span>",
                unsafe_allow_html=True,
            )
            st.caption(
                "Review candidate claims and their temporal conflict status before saving to SQLite."
            )

            approved_map = st.session_state.get("review_approved", {})
            for c in res.valid_claims:
                # Pre-evaluate conflict against current DB
                conflict_dec = service.resolve_conflicts(c)

                with st.container():
                    chk_val = approved_map.get(c.id, True)
                    col_chk, col_content = st.columns([0.15, 0.85])
                    with col_chk:
                        is_checked = st.checkbox(
                            "Approve",
                            value=chk_val,
                            key=f"chk_claim_{c.id}",
                        )
                        approved_map[c.id] = is_checked

                    with col_content:
                        badge_html = "<span class='status-badge-ok'>NEW CLAIM</span>"
                        if conflict_dec.conflict_type == ConflictType.EXACT_DUPLICATE.value:
                            badge_html = "<span class='status-badge-neutral'>EXACT DUPLICATE</span>"
                        elif conflict_dec.conflict_type == ConflictType.EXPLICIT_UPDATE.value:
                            badge_html = "<span class='status-badge-warn'>EXPLICIT UPDATE</span>"
                        elif conflict_dec.conflict_type == ConflictType.AMBIGUOUS_CHANGE.value:
                            badge_html = "<span class='status-badge-err'>AMBIGUOUS CONFLICT</span>"

                        st.markdown(
                            f"""
                            <div class='claim-card'>
                                <b>Subject:</b> <code>{c.subject}</code> &nbsp;|&nbsp; 
                                <b>Predicate:</b> <code>{c.predicate}</code> &nbsp;|&nbsp; 
                                <b>Object:</b> <code>{c.object_value}</code> &nbsp;|&nbsp; 
                                {badge_html}
                                <div class='quote-box'>
                                    " {c.source_quote or 'N/A'} "
                                </div>
                                <div style='font-size:0.8rem; color:#94A3B8; margin-top:4px;'>
                                    Conflict Analysis: {conflict_dec.decision_reason}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            st.session_state["review_approved"] = approved_map

            col_save1, col_save2 = st.columns([1, 1])
            with col_save1:
                if st.button("💾 Save Approved Claims with Conflict Resolution", type="primary", use_container_width=True):
                    claims_to_save = [c for c in res.valid_claims if approved_map.get(c.id, True)]
                    saved_count = 0
                    superseded_count = 0
                    staged_count = 0

                    for c in claims_to_save:
                        saved_claim, dec = service.process_and_stage_claim(c, auto_supersede_explicit=True)
                        if dec.conflict_type == ConflictType.EXPLICIT_UPDATE.value:
                            superseded_count += 1
                        elif dec.conflict_type == ConflictType.AMBIGUOUS_CHANGE.value:
                            staged_count += 1
                        else:
                            saved_count += 1

                    msg = f"Processed {len(claims_to_save)} claims: {saved_count} new active, {superseded_count} explicit updates superseded"
                    if staged_count > 0:
                        msg += f", {staged_count} queued for conflict review in Tab 1"
                    st.success(msg)

                    st.session_state.pop("last_result", None)
                    st.session_state.pop("review_approved", None)
                    st.rerun()

            with col_save2:
                if st.button("Discard Proposed Claims", use_container_width=True):
                    st.session_state.pop("last_result", None)
                    st.session_state.pop("review_approved", None)
                    st.rerun()

        elif res.success and not res.rejected_claims:
            st.info("ℹ️ No factual memory claims detected in the conversation.")


# ==============================================================================
# TAB 3: Claim Store & History Inspector
# ==============================================================================
with tab_store:
    st.subheader("💾 Memory Claims & Audit History Inspector")

    filter_col, search_col = st.columns([1, 2])
    with filter_col:
        status_filter = st.selectbox(
            "Filter by Lifecycle Status:",
            ["All Claims", "active", "superseded", "needs_review"],
        )

    claims_to_show = service.list_claims(
        active_only=False,
        status=None if status_filter == "All Claims" else status_filter,
    )

    if claims_to_show:
        data = []
        for c in claims_to_show:
            data.append(
                {
                    "ID": c.id[:8] + "...",
                    "Subject": c.subject,
                    "Predicate": c.predicate,
                    "Object Value": c.object_value,
                    "Status": c.status.upper(),
                    "Source Quote": c.source_quote or "-",
                    "Superseded By": (c.superseded_by[:8] + "...") if c.superseded_by else "-",
                    "Supersedes ID": (c.supersedes_id[:8] + "...") if c.supersedes_id else "-",
                    "Valid Until": c.valid_until.isoformat() if c.valid_until else "Active/Indefinite",
                }
            )
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No claims matching the selected filter.")

    st.write("---")
    st.markdown("##### ➕ Manually Insert a Test Claim")
    with st.form("manual_insert_form"):
        col_m1, col_m2 = st.columns(2)
        subj_m = col_m1.text_input("Subject", placeholder="e.g. backend_framework")
        pred_m = col_m2.text_input("Predicate", placeholder="e.g. language_version")
        val_m = col_m1.text_input("Object Value", placeholder="e.g. Go 1.22")
        quote_m = col_m2.text_input("Source Quote", placeholder="Verbatim quote from user transcript")

        submitted = st.form_submit_button("Persist Claim to SQLite")
        if submitted:
            if not subj_m.strip() or not pred_m.strip() or not val_m.strip():
                st.error("Subject, Predicate, and Object Value are required.")
            else:
                new_claim = MemoryClaim(
                    subject=subj_m,
                    predicate=pred_m,
                    object_value=val_m,
                    source_quote=quote_m.strip() or None,
                    status=ClaimStatus.ACTIVE.value,
                )
                service.save_claim(new_claim)
                st.success("Claim persisted successfully!")
                st.rerun()


# ==============================================================================
# TAB 4: Failure Scenarios (Demo Cases)
# ==============================================================================
with tab_cases:
    st.subheader("Curated Failure Scenarios")
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


# ==============================================================================
# TAB 5: The Memory Problem Explained
# ==============================================================================
with tab_problem:
    st.subheader("Why Autonomous Agents Fail: The Memory Decay Crisis")
    col_p1, col_p2 = st.columns(2)

    with col_p1:
        st.markdown(
            """
            <div class='callout-box callout-warning'>
                <h3>1. Stale Memory (Temporal Obsolescence)</h3>
                <p><b>What happens:</b> An agent records a fact true at <i>T<sub>0</sub></i>, but invalidated at <i>T<sub>1</sub></i>.</p>
                <p><b>Example:</b></p>
                <ul>
                    <li><i>T<sub>0</sub>:</i> "We use Python 3.9 for our microservices."</li>
                    <li><i>T<sub>1</sub>:</i> Production upgrades to Python 3.12.</li>
                    <li><i>T<sub>2</sub>:</i> Agent still generates Python 3.9 code from unpruned memory.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_p2:
        st.markdown(
            """
            <div class='callout-box callout-danger'>
                <h3>2. Contradictory Memory (Logical Inconsistency)</h3>
                <p><b>What happens:</b> Multi-turn or multi-session interactions record opposing claims.</p>
                <p><b>Example:</b></p>
                <ul>
                    <li><i>Session 1:</i> "User prefers strict TypeScript."</li>
                    <li><i>Session 4:</i> "User migrated project to Go."</li>
                    <li><i>Session 7:</i> Agent arbitrarily alternates between Go and TypeScript across prompts.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ==============================================================================
# TAB 6: Foundation Overview
# ==============================================================================
with tab_overview:
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
        st.write(f"- **Audit History:** {stats.get('superseded', 0)} superseded, {stats.get('needs_review', 0)} in review")


# ==============================================================================
# TAB 3: Ask RecallGuard (Feature 04 – Evidence-Grounded Retrieval)
# ==============================================================================
with tab_ask:
    st.subheader("💬 Ask RecallGuard")
    st.write(
        """
        Ask any question about your stored memories. **RecallGuard** retrieves only active,
        non-expired, non-superseded evidence, then generates a grounded local-LLM answer
        that **cites exact source quotes**. If the evidence base is insufficient, the system
        **safely abstains** rather than hallucinating an answer.
        """
    )

    st.markdown(
        """
        <div class='callout-box'>
        🔒 <strong>Zero cloud API calls.</strong> All inference runs locally via Ollama 
        <code>qwen3.5:0.8b</code>. Superseded and expired claims are automatically excluded.
        </div>
        """,
        unsafe_allow_html=True,
    )

    ask_col, result_col = st.columns([1, 1], gap="large")

    with ask_col:
        st.markdown("##### 🔍 Your Question")
        user_question = st.text_area(
            "Enter your question about stored memories:",
            placeholder="e.g. What programming language does the user prefer?",
            height=120,
            key="ask_question_input",
        )

        retrieval_mode = st.radio(
            "Retrieval mode:",
            ["Retrieve evidence only", "Retrieve + generate grounded answer"],
            index=1,
            key="retrieval_mode",
        )

        min_conf = st.slider(
            "Minimum claim confidence:",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
            key="retrieval_min_conf",
        )

        ask_btn = st.button(
            "🔍 Search Memory",
            use_container_width=True,
            key="ask_btn",
            disabled=not user_question.strip(),
        )

    with result_col:
        if ask_btn and user_question.strip():
            from core.retrieval_service import RetrievalConfig

            cfg = RetrievalConfig(min_confidence=min_conf)
            retrieval_svc = RetrievalService(
                db_manager=service.db,
                llm_client=service.llm,
                config=cfg,
            )

            with st.spinner("Searching memory store…"):
                if retrieval_mode == "Retrieve evidence only":
                    rr = retrieval_svc.retrieve(user_question)
                else:
                    rr = retrieval_svc.answer(user_question)

            if rr.abstained:
                st.markdown(
                    f"""
                    <div class='callout-box callout-warning'>
                    <strong>⚠️ RecallGuard Abstained</strong><br/>
                    {rr.abstention_reason}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                if rr.answer:
                    st.markdown("##### 📝 Grounded Answer")
                    st.markdown(
                        f"""
                        <div class='callout-box'>
                        {rr.answer.replace(chr(10), '<br/>')}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.markdown(f"##### 🗂️ Retrieved Evidence ({len(rr.retrieved_claims)} claim{'s' if len(rr.retrieved_claims) != 1 else ''})")
                st.caption(f"Mean evidence confidence: **{rr.confidence:.0%}** | Filtered out: **{rr.filtered_out_count}** stale/expired claim(s)")

                for i, claim in enumerate(rr.retrieved_claims, start=1):
                    with st.expander(f"[E{i}] {claim.subject} → {claim.predicate} = **{claim.object_value}**", expanded=i == 1):
                        col_a, col_b, col_c = st.columns(3)
                        col_a.metric("Confidence", f"{claim.confidence:.0%}")
                        col_b.metric("Status", claim.status.upper())
                        col_c.metric("ID", claim.id[:8] + "…")

                        if claim.source_quote:
                            st.markdown(
                                f"<div class='quote-box'>📎 {claim.source_quote}</div>",
                                unsafe_allow_html=True,
                            )
                        st.caption(f"Created: {claim.created_at}")
                        if claim.valid_until:
                            st.caption(f"Valid until: {claim.valid_until}")
        else:
            st.markdown(
                """
                <div style='text-align:center; color:#475569; padding: 3rem 1rem;'>
                <div style='font-size:3rem'>🔍</div>
                <div style='font-size:1.1rem; margin-top:0.5rem;'>Ask a question to retrieve grounded evidence</div>
                <div style='font-size:0.85rem; margin-top:0.3rem;'>RecallGuard will search your memory store and cite verbatim source quotes</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ==============================================================================
# TAB 4: Reliability Lab (Feature 04 – Evaluation Dashboard)
# ==============================================================================
with tab_lab:
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        HAS_PLOTLY = True
    except ImportError:
        HAS_PLOTLY = False

    st.subheader("🔬 Reliability Lab")
    st.write(
        """
        The **Reliability Lab** runs a synthetic evaluation suite entirely in-process — no
        cloud calls, no external datasets. It measures four core reliability metrics and
        compares RecallGuard against a **naive last-write-wins baseline** to quantify the
        firewall's real-world value.
        """
    )

    lab_col1, lab_col2 = st.columns([1, 2], gap="large")

    with lab_col1:
        st.markdown("##### 📐 Evaluation Metrics")
        st.markdown(
            """
            | Metric | Description |
            |---|---|
            | **Stale Memory Error Rate** | Fraction of answers using superseded/expired claims |
            | **Conflict Detection Recall** | Fraction of true conflicts flagged |
            | **Evidence Coverage** | Fraction of questions answered with evidence |
            | **Safe Abstention Rate** | Fraction of unanswerable questions correctly abstained |
            """
        )

        st.markdown("##### ⚙️ Evaluation Settings")
        eval_dataset_size = st.number_input(
            "Synthetic dataset size (samples):",
            min_value=4,
            max_value=8,
            value=8,
            step=1,
            key="eval_dataset_size",
            help="The synthetic dataset is fixed at 8 samples; this is a display setting.",
        )
        run_eval_btn = st.button(
            "▶️ Run Reliability Evaluation",
            use_container_width=True,
            key="run_eval_btn",
            type="primary",
        )

    with lab_col2:
        if run_eval_btn:
            with st.spinner("Running reliability evaluation on synthetic dataset…"):
                evaluator = ReliabilityEvaluator()
                report = evaluator.run_full_evaluation()
                chart_data = ReliabilityEvaluator.build_metrics_chart_data(report)
                st.session_state["eval_report"] = report
                st.session_state["eval_chart_data"] = chart_data

        report = st.session_state.get("eval_report")
        chart_data = st.session_state.get("eval_chart_data")

        if report is not None and chart_data is not None:
            # ---- Summary metrics ----
            st.markdown("##### 📊 RecallGuard vs Naive Baseline")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric(
                "Stale Error Rate",
                f"{report.stale_memory_error_rate:.0%}",
                delta=f"{report.improvements_over_baseline.get('stale_memory_error_rate', 0):+.0%} vs naive",
                delta_color="inverse",
            )
            m2.metric(
                "Conflict Recall",
                f"{report.conflict_detection_recall:.0%}",
                delta=f"{report.improvements_over_baseline.get('conflict_detection_recall', 0):+.0%} vs naive",
            )
            m3.metric(
                "Evidence Coverage",
                f"{report.evidence_coverage:.0%}",
                delta=f"{report.improvements_over_baseline.get('evidence_coverage', 0):+.0%} vs naive",
            )
            m4.metric(
                "Safe Abstention",
                f"{report.safe_abstention_rate:.0%}",
                delta=f"{report.improvements_over_baseline.get('safe_abstention_rate', 0):+.0%} vs naive",
            )

            # ---- Plotly grouped bar chart ----
            if HAS_PLOTLY:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    name="RecallGuard",
                    x=chart_data["labels"],
                    y=chart_data["recallguard_values"],
                    marker_color="#3B82F6",
                    text=[f"{v:.0%}" for v in chart_data["recallguard_values"]],
                    textposition="outside",
                ))
                fig.add_trace(go.Bar(
                    name="Naive Baseline",
                    x=chart_data["labels"],
                    y=chart_data["naive_baseline_values"],
                    marker_color="#EF4444",
                    text=[f"{v:.0%}" for v in chart_data["naive_baseline_values"]],
                    textposition="outside",
                ))
                fig.update_layout(
                    barmode="group",
                    title="RecallGuard vs Naive Baseline — Reliability Metrics",
                    yaxis=dict(tickformat=".0%", range=[0, 1.25], title="Score"),
                    xaxis_title="Metric",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(15,23,42,0.6)",
                    font=dict(color="#CBD5E1"),
                    height=380,
                )
                st.plotly_chart(fig, use_container_width=True)

                # ---- Improvements radar ----
                labels_radar = chart_data["labels"]
                improvements = chart_data["improvements"]
                radar_fig = go.Figure(go.Scatterpolar(
                    r=chart_data["recallguard_values"] + [chart_data["recallguard_values"][0]],
                    theta=labels_radar + [labels_radar[0]],
                    fill="toself",
                    name="RecallGuard",
                    line_color="#3B82F6",
                    fillcolor="rgba(59,130,246,0.2)",
                ))
                radar_fig.add_trace(go.Scatterpolar(
                    r=chart_data["naive_baseline_values"] + [chart_data["naive_baseline_values"][0]],
                    theta=labels_radar + [labels_radar[0]],
                    fill="toself",
                    name="Naive Baseline",
                    line_color="#EF4444",
                    fillcolor="rgba(239,68,68,0.15)",
                ))
                radar_fig.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 1], tickformat=".0%", color="#94A3B8"),
                        bgcolor="rgba(15,23,42,0.7)",
                    ),
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#CBD5E1"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
                    title="Reliability Radar",
                    height=360,
                )
                st.plotly_chart(radar_fig, use_container_width=True)
            else:
                st.warning("Install `plotly` for interactive charts: `pip install plotly`")
                st.json(chart_data)

            # ---- Per-sample breakdown table ----
            st.markdown("##### 📋 Per-Sample Results")
            rows = []
            for psr in report.per_sample_results:
                rr = psr.retrieval_result
                rows.append({
                    "Question": psr.sample.question[:60] + ("…" if len(psr.sample.question) > 60 else ""),
                    "Should Abstain": "✅" if psr.sample.should_abstain else "-",
                    "Abstained": "✅" if (rr and rr.abstained) else "❌",
                    "Correctly Abstained": "✅" if psr.correctly_abstained else ("-" if not psr.sample.should_abstain else "❌"),
                    "Evidence Hit": "✅" if psr.evidence_hit else ("-" if psr.sample.should_abstain else "❌"),
                    "Stale Claim Used": "🚨" if psr.used_stale_claim else "✅",
                    "Has Conflict": "⚠️" if psr.sample.has_conflict else "-",
                    "Conflict in DB": "✅" if psr.conflict_in_db else ("-" if not psr.sample.has_conflict else "❌"),
                    "Evidence": f"{len(rr.retrieved_claims)} claim(s)" if (rr and not rr.abstained) else "None",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            if report.notes:
                with st.expander("ℹ️ Metric Explanations & Notes", expanded=False):
                    for k, v in report.notes.items():
                        st.markdown(f"- **{k.replace('_', ' ').title()}**: {v}")

            # ---- Synthetic dataset details ----
            with st.expander("🗃️ View Synthetic Dataset Details"):
                dataset, _ = build_synthetic_dataset()
                st.markdown(f"**Seed claims:** {len(dataset.seed_claims)}  |  **Stale claims:** {len(dataset.stale_claims)}  |  **Samples:** {len(dataset.samples)}")
                seed_rows = [
                    {
                        "ID": c.id[:8] + "…",
                        "Subject": c.subject,
                        "Predicate": c.predicate,
                        "Value": c.object_value,
                        "Status": c.status,
                        "Confidence": f"{c.confidence:.0%}",
                    }
                    for c in dataset.seed_claims + dataset.stale_claims
                ]
                st.dataframe(pd.DataFrame(seed_rows), use_container_width=True, hide_index=True)

        else:
            st.markdown(
                """
                <div style='text-align:center; color:#475569; padding: 3rem 1rem;'>
                <div style='font-size:3rem'>🔬</div>
                <div style='font-size:1.1rem; margin-top:0.5rem;'>Reliability Lab ready</div>
                <div style='font-size:0.85rem; margin-top:0.3rem;'>Click "Run Reliability Evaluation" to benchmark RecallGuard vs naive baseline</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # --- Current store reliability audit ---
    st.divider()
    st.markdown("##### 🏥 Live Memory Store Health Audit")
    st.caption("Evaluates the reliability of every claim currently in your production memory store.")

    if st.button("🩺 Run Store Health Audit", key="health_audit_btn"):
        all_claims = service.list_claims()
        if not all_claims:
            st.info("No claims in the memory store yet. Add some via the AI Extraction tab.")
        else:
            evaluator = ReliabilityEvaluator()
            batch_result = evaluator.evaluate_batch(all_claims)
            agg = batch_result["aggregate"]

            ha1, ha2, ha3, ha4 = st.columns(4)
            ha1.metric("Total Claims", batch_result["total"])
            ha2.metric("Stale Rate", f"{agg['stale_rate']:.0%}")
            ha3.metric("Contradiction Risk Rate", f"{agg['contradiction_risk_rate']:.0%}")
            ha4.metric("Mean Veracity Confidence", f"{agg['mean_veracity_confidence']:.0%}")

            if HAS_PLOTLY:
                status_counts = {}
                for ev in batch_result["evaluations"]:
                    status_counts[ev["status"]] = status_counts.get(ev["status"], 0) + 1

                pie_fig = go.Figure(go.Pie(
                    labels=list(status_counts.keys()),
                    values=list(status_counts.values()),
                    hole=0.55,
                    marker_colors=["#10B981", "#EF4444", "#F59E0B"],
                ))
                pie_fig.update_layout(
                    title="Claim Status Distribution",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#CBD5E1"),
                    height=300,
                )
                st.plotly_chart(pie_fig, use_container_width=True)

            quarantine_claims = [
                ev for ev in batch_result["evaluations"] if ev["quarantine_recommendation"]
            ]
            if quarantine_claims:
                st.warning(f"⚠️ {len(quarantine_claims)} claim(s) recommended for quarantine (stale/expired/conflicted).")
                qrows = [
                    {
                        "ID": ev["claim_id"][:8] + "…",
                        "Subject": ev["subject"],
                        "Predicate": ev["predicate"],
                        "Status": ev["status"],
                        "Staleness": f"{ev['staleness_score']:.0%}",
                        "Veracity": f"{ev['veracity_confidence']:.0%}",
                        "Expired": "🚨" if ev["is_expired"] else "-",
                    }
                    for ev in quarantine_claims
                ]
                st.dataframe(pd.DataFrame(qrows), use_container_width=True, hide_index=True)
            else:
                st.success("✅ All claims in the memory store pass reliability checks.")

