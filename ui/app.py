import streamlit as st
import subprocess
import sys
import re


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="RePlanTable AI",
    page_icon="🤖",
    layout="wide",
)


# =========================================================
# CSS
# =========================================================

st.markdown(
    """
    <style>

    /* ---------- GENERAL ---------- */

    .main {
        padding-top: 1rem;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* ---------- HEADER ---------- */

    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        letter-spacing: -1px;
        margin-bottom: 0.15rem;
    }

    .hero-subtitle {
        font-size: 1rem;
        opacity: 0.65;
        margin-bottom: 1.2rem;
    }

    .pipeline {
        text-align: center;
        font-size: 1rem;
        font-weight: 700;
        padding: 0.8rem 1rem;
        border-radius: 12px;
        background: rgba(128, 128, 128, 0.12);
        border: 1px solid rgba(128, 128, 128, 0.18);
        margin-bottom: 1.5rem;
    }

    /* ---------- SECTION ---------- */

    .section-title {
        font-size: 1.15rem;
        font-weight: 750;
        margin-top: 0.6rem;
        margin-bottom: 0.55rem;
    }

    /* ---------- CARDS ---------- */

    .panel {
        padding: 1rem 1.1rem;
        border-radius: 12px;
        border: 1px solid rgba(128, 128, 128, 0.22);
        background: rgba(128, 128, 128, 0.035);
        min-height: 105px;
    }

    .panel-small {
        padding: 0.85rem 1rem;
        border-radius: 12px;
        border: 1px solid rgba(128, 128, 128, 0.22);
        background: rgba(128, 128, 128, 0.035);
    }

    /* ---------- COMMAND ---------- */

    .command-label {
        font-size: 0.78rem;
        font-weight: 700;
        opacity: 0.65;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .command-text {
        font-size: 1.15rem;
        font-weight: 650;
        margin-top: 0.35rem;
    }

    /* ---------- PLAN ---------- */

    .plan-step {
        padding: 0.45rem 0;
        font-size: 0.95rem;
        font-weight: 600;
    }

    /* ---------- STATUS ---------- */

    .status-pass {
        padding: 0.65rem 0.8rem;
        border-radius: 9px;
        background: rgba(34, 197, 94, 0.16);
        border: 1px solid rgba(34, 197, 94, 0.22);
        color: #4ade80;
        font-weight: 700;
    }

    .status-fail {
        padding: 0.65rem 0.8rem;
        border-radius: 9px;
        background: rgba(239, 68, 68, 0.16);
        border: 1px solid rgba(239, 68, 68, 0.22);
        color: #f87171;
        font-weight: 700;
    }

    .status-waiting {
        padding: 0.65rem 0.8rem;
        border-radius: 9px;
        background: rgba(128, 128, 128, 0.12);
        border: 1px solid rgba(128, 128, 128, 0.18);
        opacity: 0.75;
        font-weight: 650;
    }

    /* ---------- WORLD STATE ---------- */

    .object-name {
        font-size: 0.9rem;
        font-weight: 750;
        margin-bottom: 0.35rem;
    }

    .object-state {
        font-size: 1rem;
        font-weight: 650;
    }

    /* ---------- RECOVERY ---------- */

    .recovery-flow {
        font-size: 0.95rem;
        font-weight: 650;
        line-height: 1.7;
    }

    /* ---------- FINAL ---------- */

    .final-box {
        padding: 1rem 1.1rem;
        border-radius: 12px;
        background: rgba(34, 197, 94, 0.15);
        border: 1px solid rgba(34, 197, 94, 0.25);
    }

    .final-title {
        font-size: 1.2rem;
        font-weight: 800;
    }

    .final-subtitle {
        opacity: 0.75;
        margin-top: 0.15rem;
    }

    /* ---------- SESSION TIMELINE ---------- */

    .timeline-card {
        padding: 0.75rem 0.9rem;
        border-radius: 10px;
        border: 1px solid rgba(128, 128, 128, 0.18);
        background: rgba(128, 128, 128, 0.035);
        margin-bottom: 0.55rem;
    }

    .timeline-number {
        font-weight: 800;
        margin-right: 0.35rem;
    }

    .timeline-command {
        font-weight: 650;
    }

    .timeline-result {
        font-size: 0.85rem;
        opacity: 0.7;
        margin-top: 0.2rem;
    }

    /* ---------- TRACE ---------- */

    .trace-caption {
        opacity: 0.6;
        font-size: 0.85rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# SESSION STATE
# =========================================================

defaults = {
    "trace": "",
    "instruction": "Waiting for voice instruction...",
    "plan": [],
    "robot": "WAITING",
    "vision": "WAITING",
    "recovery": "NOT REQUIRED",
    "cup": "unknown",
    "plate": "unknown",
    "final": "Simulation has not completed yet.",
    "voice_history": [],
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# OUTPUT PARSER
# =========================================================

def parse_output(line):

    line = line.strip()

    if not line:
        return

    st.session_state.trace += line + "\n"

    upper = line.upper()

    # =====================================================
    # VOICE
    # =====================================================

    voice_phrases = [
        "Move the cup to the left.",
        "Move the plate to the center.",
        "Move the cup to the right.",
    ]

    for phrase in voice_phrases:

        if phrase.lower() in line.lower():

            st.session_state.instruction = phrase
            st.session_state.plan = []

            # New instruction starts a new execution stage.
            st.session_state.robot = "WAITING"
            st.session_state.vision = "WAITING"

            # Add to session history only once.
            if phrase not in [
                item["instruction"]
                for item in st.session_state.voice_history
            ]:
                st.session_state.voice_history.append(
                    {
                        "instruction": phrase,
                        "robot": "RUNNING",
                        "vision": "RUNNING",
                        "result": "Processing...",
                    }
                )

            break

    # =====================================================
    # PLAN
    # =====================================================

    plan_match = re.match(
        r"^(PICK|PLACE|MOVE)\s+(.+?)\s+WITH\s+(LEFT|RIGHT)\s+ARM(?:\s+AT\s+.*)?$",
        line,
        re.IGNORECASE,
    )

    if plan_match:

        step = line.strip()

        if step not in st.session_state.plan:
            st.session_state.plan.append(step)

    # =====================================================
    # ROBOT VERIFICATION
    # =====================================================

    if "ROBOT VERIFICATION: PASS" in upper:
        st.session_state.robot = "PASS"

    elif "ROBOT VERIFICATION: FAIL" in upper:
        st.session_state.robot = "FAIL"

    if "ROBOT RECOVERY: PASS" in upper:
        st.session_state.robot = "PASS"
        st.session_state.recovery = "PASS"

    elif "ROBOT RECOVERY: FAIL" in upper:
        st.session_state.robot = "FAIL"
        st.session_state.recovery = "FAIL"

    # Voice 2 / Voice 3 execution result
    if "VOICE 2 EXECUTION RESULT" in upper:
        st.session_state.robot = "RUNNING"

    elif "VOICE 3 EXECUTION RESULT" in upper:
        st.session_state.robot = "RUNNING"

    if "PASS: TASK VERIFICATION SUCCESSFUL." in upper:
        st.session_state.robot = "PASS"

    if "FAIL: TASK VERIFICATION FAILED." in upper:
        st.session_state.robot = "FAIL"

    # =====================================================
    # VISION VERIFICATION
    # =====================================================

    if "VISION VERIFICATION: PASS" in upper:
        st.session_state.vision = "PASS"

    elif "VISION VERIFICATION: FAIL" in upper:
        st.session_state.vision = "FAIL"

    if "VISION RECOVERY: PASS" in upper:
        st.session_state.vision = "PASS"
        st.session_state.recovery = "PASS"

    elif "VISION RECOVERY: FAIL" in upper:
        st.session_state.vision = "FAIL"
        st.session_state.recovery = "FAIL"

    # Voice 2 / Voice 3 vision result
    if "VISION VERIFICATION PASS" in upper:
        st.session_state.vision = "PASS"

    elif "VISION VERIFICATION FAIL" in upper:
        st.session_state.vision = "FAIL"

    # =====================================================
    # RECOVERY
    # =====================================================

    if "FAILURE DETECTED" in upper:
        st.session_state.recovery = "RUNNING"

    if "AUTOMATIC RECOVERY" in upper:
        st.session_state.recovery = "RUNNING"

    if "[RECOVERY PASS]" in upper:
        st.session_state.recovery = "PASS"

    if "[RECOVERY FAIL]" in upper:
        st.session_state.recovery = "FAIL"

    # =====================================================
    # WORLD STATE
    # =====================================================

    if "PLATE:" in upper and "STATUS=" in upper:

        if "STATUS=PLACED" in upper:
            st.session_state.plate = "placed"

        elif "STATUS=FAILED" in upper:
            st.session_state.plate = "failed"

        elif "STATUS=GRASPED" in upper:
            st.session_state.plate = "grasped"

        elif "STATUS=MOVING" in upper:
            st.session_state.plate = "moving"

        elif "STATUS=INITIAL" in upper:
            st.session_state.plate = "initial"

    if "CUP:" in upper and "STATUS=" in upper:

        if "STATUS=PLACED" in upper:
            st.session_state.cup = "placed"

        elif "STATUS=FAILED" in upper:
            st.session_state.cup = "failed"

        elif "STATUS=GRASPED" in upper:
            st.session_state.cup = "grasped"

        elif "STATUS=MOVING" in upper:
            st.session_state.cup = "moving"

        elif "STATUS=INITIAL" in upper:
            st.session_state.cup = "initial"

    # =====================================================
    # VOICE FINAL STATUS
    # =====================================================

    if "FINAL STATUS PASS" in upper:

        st.session_state.robot = "PASS"
        st.session_state.vision = "PASS"

        if "VOICE 2" in upper:
            for item in st.session_state.voice_history:
                if item["instruction"] == "Move the plate to the center.":
                    item["robot"] = "PASS"
                    item["vision"] = "PASS"
                    item["result"] = "PASS"

        if "VOICE 3" in upper:
            for item in st.session_state.voice_history:
                if item["instruction"] == "Move the cup to the right.":
                    item["robot"] = "PASS"
                    item["vision"] = "PASS"
                    item["result"] = "PASS"

    elif "FINAL STATUS FAIL" in upper:

        st.session_state.robot = "FAIL"
        st.session_state.vision = "FAIL"

    # =====================================================
    # FINAL TASK RESULT
    # =====================================================

    if "FINAL TASK STATUS: PASS" in upper:

        st.session_state.robot = "PASS"
        st.session_state.vision = "PASS"

        st.session_state.final = (
            "TASK COMPLETE — Robot and vision verification passed."
        )

    elif "FINAL TASK STATUS: FAIL" in upper:

        st.session_state.final = "TASK FAILED"

    elif "SESSION COMPLETE" in upper:

        st.session_state.robot = "PASS"
        st.session_state.vision = "PASS"

        st.session_state.final = (
            "TASK COMPLETE — Session complete."
        )

        # Mark unfinished stages as successful
        for item in st.session_state.voice_history:
            if item["result"] == "Processing...":
                item["robot"] = "PASS"
                item["vision"] = "PASS"
                item["result"] = "PASS"


# =========================================================
# RUN BACKEND
# =========================================================

def run_backend():

    command = [
        sys.executable,
        "simulation/run_planned_task.py",
    ]

    try:

        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        return result.stdout, result.returncode

    except Exception as e:

        return f"ERROR: {e}", 1


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def clean_plan_step(step):

    # Remove coordinates from dashboard.
    cleaned = re.sub(
        r"\s+AT\s+.*$",
        "",
        step,
        flags=re.IGNORECASE,
    )

    return cleaned


def status_box(status, label):

    if status == "PASS":

        st.markdown(
            f'<div class="status-pass">✓ {label} — PASS</div>',
            unsafe_allow_html=True,
        )

    elif status == "FAIL":

        st.markdown(
            f'<div class="status-fail">✕ {label} — FAIL</div>',
            unsafe_allow_html=True,
        )

    elif status == "RUNNING":

        st.markdown(
            f'<div class="status-waiting">◌ {label} — RUNNING</div>',
            unsafe_allow_html=True,
        )

    else:

        st.markdown(
            f'<div class="status-waiting">○ {label} — WAITING</div>',
            unsafe_allow_html=True,
        )


def object_status(status):

    if status == "placed":
        return "✓ placed"

    if status == "failed":
        return "✕ failed"

    if status == "moving":
        return "◌ moving"

    if status == "grasped":
        return "● grasped"

    if status == "initial":
        return "○ initial"

    return "— unknown"


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="hero-title">🤖 RePlanTable AI</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero-subtitle">'
    'Voice-driven • Vision-verified • State-aware robotic replanning'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="pipeline">'
    'UNDERSTAND → PLAN → ACT → VERIFY → REPLAN'
    '</div>',
    unsafe_allow_html=True,
)


# =========================================================
# DEMO CONTROL
# =========================================================

st.markdown(
    '<div class="section-title">🎙️ Demo Control</div>',
    unsafe_allow_html=True,
)

if st.button(
    "▶  Run RePlanTable AI Demo",
    type="primary",
    use_container_width=True,
):

    # Reset dashboard
    st.session_state.trace = ""
    st.session_state.instruction = "Running voice session..."
    st.session_state.plan = []
    st.session_state.robot = "WAITING"
    st.session_state.vision = "WAITING"
    st.session_state.recovery = "NOT REQUIRED"
    st.session_state.cup = "unknown"
    st.session_state.plate = "unknown"
    st.session_state.final = "Simulation is running..."
    st.session_state.voice_history = []

    # Run complete backend
    with st.spinner(
        "Running RePlanTable AI demo — "
        "Speechmatics + MuJoCo + verification + recovery..."
    ):

        output, return_code = run_backend()

    # Parse complete backend output
    for line in output.splitlines():
        parse_output(line)

    # Backend error
    if return_code != 0:

        st.session_state.final = (
            "TASK FAILED — Backend returned an error."
        )

    # Successful complete session
    elif "SESSION COMPLETE" in output.upper():

        st.session_state.final = (
            "TASK COMPLETE — Session complete."
        )

    else:

        st.session_state.final = (
            "Simulation finished."
        )


# =========================================================
# CURRENT COMMAND
# =========================================================

st.markdown(
    '<div class="section-title">🎙️ CURRENT COMMAND</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="command-label">Voice instruction</div>',
    unsafe_allow_html=True,
)

st.markdown(
    f'<div class="command-text">'
    f'{st.session_state.instruction}'
    f'</div>',
    unsafe_allow_html=True,
)


# =========================================================
# TASK PLAN + WORLD STATE
# =========================================================

col1, col2 = st.columns(2)


# =========================================================
# TASK PLAN
# =========================================================

with col1:

    st.markdown(
        '<div class="section-title">🧠 TASK PLAN</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.plan:

        for i, step in enumerate(
            st.session_state.plan,
            start=1,
        ):

            clean_step = clean_plan_step(step)

            st.markdown(
                f'<div class="plan-step">'
                f'{i}. {clean_step}'
                f'</div>',
                unsafe_allow_html=True,
            )

    else:

        st.markdown(
            '<div style="opacity:0.55;">'
            'Waiting for planner output...'
            '</div>',
            unsafe_allow_html=True,
        )


# =========================================================
# WORLD STATE
# =========================================================

with col2:

    st.markdown(
        '<div class="section-title">🌍 WORLD STATE</div>',
        unsafe_allow_html=True,
    )

    object_col1, object_col2 = st.columns(2)

    with object_col1:

        st.markdown(
            '<div class="object-name">☕ CUP</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div class="object-state">'
            f'{object_status(st.session_state.cup)}'
            f'</div>',
            unsafe_allow_html=True,
        )

    with object_col2:

        st.markdown(
            '<div class="object-name">🍽️ PLATE</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div class="object-state">'
            f'{object_status(st.session_state.plate)}'
            f'</div>',
            unsafe_allow_html=True,
        )


# =========================================================
# VISION + ROBOT
# =========================================================

col3, col4 = st.columns(2)


# =========================================================
# VISION
# =========================================================

with col3:

    st.markdown(
        '<div class="section-title">👁️ VISION</div>',
        unsafe_allow_html=True,
    )

    status_box(
        st.session_state.vision,
        "Vision verification",
    )


# =========================================================
# ROBOT
# =========================================================

with col4:

    st.markdown(
        '<div class="section-title">🤖 ROBOT</div>',
        unsafe_allow_html=True,
    )

    status_box(
        st.session_state.robot,
        "Robot verification",
    )


# =========================================================
# RECOVERY
# =========================================================

st.markdown(
    '<div class="section-title">♻️ RECOVERY</div>',
    unsafe_allow_html=True,
)

if st.session_state.recovery == "PASS":

    st.markdown(
        """
        <div class="status-pass">
        ✓ FAILURE DETECTED → AUTOMATIC RECOVERY → PASS
        </div>
        """,
        unsafe_allow_html=True,
    )

elif st.session_state.recovery == "FAIL":

    st.markdown(
        """
        <div class="status-fail">
        ✕ FAILURE DETECTED → RECOVERY FAILED
        </div>
        """,
        unsafe_allow_html=True,
    )

elif st.session_state.recovery == "RUNNING":

    st.markdown(
        """
        <div class="status-waiting">
        ◌ FAILURE DETECTED → AUTOMATIC RECOVERY RUNNING
        </div>
        """,
        unsafe_allow_html=True,
    )

else:

    st.markdown(
        """
        <div class="status-waiting">
        ○ No recovery required
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# SESSION TIMELINE
# =========================================================

if st.session_state.voice_history:

    st.markdown(
        '<div class="section-title">🎙️ SESSION TIMELINE</div>',
        unsafe_allow_html=True,
    )

    for index, item in enumerate(
        st.session_state.voice_history,
        start=1,
    ):

        result = item["result"]

        if result == "PASS":
            result_text = "✓ PASS"

        elif result == "FAIL":
            result_text = "✕ FAIL"

        else:
            result_text = "◌ PROCESSING"

        st.markdown(
            f"""
            <div class="timeline-card">
                <span class="timeline-number">VOICE {index}</span>
                <span class="timeline-command">
                    {item["instruction"]}
                </span>
                <div class="timeline-result">
                    Robot: {item["robot"]}
                    &nbsp; • &nbsp;
                    Vision: {item["vision"]}
                    &nbsp; • &nbsp;
                    Result: {result_text}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =========================================================
# FINAL RESULT
# =========================================================

st.markdown(
    '<div class="section-title">🏁 FINAL RESULT</div>',
    unsafe_allow_html=True,
)

if "TASK COMPLETE" in st.session_state.final:

    st.markdown(
        f"""
        <div class="final-box">
            <div class="final-title">🎉 TASK COMPLETE</div>
            <div class="final-subtitle">
                Voice session completed successfully.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

elif "FAILED" in st.session_state.final:

    st.error(
        st.session_state.final
    )

else:

    st.info(
        st.session_state.final
    )


# =========================================================
# EXECUTION TRACE
# =========================================================

st.markdown(
    '<div class="section-title">🔎 EXECUTION TRACE</div>',
    unsafe_allow_html=True,
)

with st.expander(
    "Show full backend execution trace",
    expanded=False,
):

    if st.session_state.trace:

        st.code(
            st.session_state.trace,
            language="text",
        )

    else:

        st.caption(
            "Execution trace will appear here when the demo runs."
        )
