from datetime import datetime
from pathlib import Path
import hmac
import json
import os
import sys
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import BEST_MODEL_FILE, CATEGORICAL_FEATURES, DEMO_PATIENTS_DIR, NUMERIC_FEATURES
from src.data_pipeline import build_inference_dataframe
from src.recommendations import build_personalized_recommendations

DEFAULT_MANUAL_PAYLOAD = {
    "age": 50,
    "bmi": 28.0,
    "systolic_bp": 130,
    "diastolic_bp": 85,
    "glucose": 120,
    "hba1c": 6.0,
    "cholesterol": 200,
    "creatinine": 1.0,
    "egfr": 85,
    "physical_activity_days": 3,
    "sleep_hours": 7.0,
    "gender": "female",
    "smoking_status": "never",
    "family_history": "no",
    "diet_quality": "average",
    "alcohol_intake": "low",
}


def init_page() -> None:
    st.set_page_config(
        page_title="Chronic Disease Detection",
        page_icon="+",
        layout="wide",
    )

    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1180px;
            padding-top: 1rem;
            padding-bottom: 2rem;
        }
        .hero {
            background: linear-gradient(130deg, #102c46 0%, #1f5c86 65%, #4e98a9 100%);
            border: 1px solid #57a7be;
            border-radius: 14px;
            padding: 16px 20px;
            color: #f6fbff;
            margin-bottom: 12px;
        }
        .hero h2 {
            margin: 0;
            font-size: 1.75rem;
            line-height: 1.25;
        }
        .hero p {
            margin: 0.45rem 0 0 0;
            opacity: 0.95;
        }
        .risk-badge {
            display: inline-block;
            padding: 0.34rem 0.72rem;
            border-radius: 999px;
            font-size: 0.84rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            letter-spacing: 0.02em;
        }
        .risk-badge-high {
            background: #4a1b1b;
            border: 1px solid #8f3535;
            color: #ffd7d7;
        }
        .risk-badge-low {
            background: #163824;
            border: 1px solid #2b6b45;
            color: #d8ffe8;
        }
        .subtle {
            opacity: 0.86;
            font-size: 0.93rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="hero">
          <h2>Chronic Disease Risk Prediction</h2>
          <p>Fast screening support with patient-case uploads and action-oriented guidance.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Educational decision-support demo. Not a standalone medical diagnosis tool.")


def get_demo_password() -> Optional[str]:
    env_password = os.getenv("APP_DEMO_PASSWORD", "").strip()
    if env_password:
        return env_password

    try:
        secret_password = str(st.secrets.get("APP_DEMO_PASSWORD", "")).strip()
        return secret_password or None
    except Exception:
        return None


def require_streamlit_login() -> None:
    configured_password = get_demo_password()
    if not configured_password:
        return

    if st.session_state.get("authenticated", False):
        st.sidebar.success("Authenticated")
        if st.sidebar.button("Sign Out"):
            st.session_state["authenticated"] = False
            st.rerun()
        return

    st.subheader("Demo Access Login")
    st.info("This demo requires a shared password configured by APP_DEMO_PASSWORD.")
    entered_password = st.text_input("Password", type="password")

    col1, _ = st.columns([1, 3])
    with col1:
        if st.button("Unlock", type="primary"):
            if hmac.compare_digest(entered_password, configured_password):
                st.session_state["authenticated"] = True
                st.rerun()
            st.error("Incorrect password.")

    st.stop()


@st.cache_resource
def load_artifact(model_path: Path):
    return joblib.load(model_path)


@st.cache_data
def load_demo_cases(demo_dir: Path) -> Dict[str, Dict[str, Any]]:
    cases: Dict[str, Dict[str, Any]] = {}
    if not demo_dir.exists():
        return cases

    for file_path in sorted(demo_dir.glob("*.json")):
        try:
            case_label = file_path.stem.replace("_", " ").title()
            cases[case_label] = json.loads(file_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

    return cases


def init_state() -> None:
    for key, value in DEFAULT_MANUAL_PAYLOAD.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if "selected_demo_case" not in st.session_state:
        st.session_state["selected_demo_case"] = "Custom"
    if "prediction_history" not in st.session_state:
        st.session_state["prediction_history"] = []
    if "manual_result" not in st.session_state:
        st.session_state["manual_result"] = None
    if "upload_results" not in st.session_state:
        st.session_state["upload_results"] = []
    if "upload_summary" not in st.session_state:
        st.session_state["upload_summary"] = []
    if "selected_uploaded_file" not in st.session_state:
        st.session_state["selected_uploaded_file"] = ""


def reset_manual_inputs() -> None:
    for key, value in DEFAULT_MANUAL_PAYLOAD.items():
        st.session_state[key] = value


def load_payload_into_state(payload: Dict[str, Any]) -> None:
    for key in NUMERIC_FEATURES + CATEGORICAL_FEATURES:
        if key in payload:
            st.session_state[key] = payload[key]


def validate_payload(payload: Dict[str, Any]) -> List[str]:
    return [key for key in NUMERIC_FEATURES + CATEGORICAL_FEATURES if key not in payload]


def build_manual_payload() -> Dict[str, Any]:
    return {key: st.session_state[key] for key in NUMERIC_FEATURES + CATEGORICAL_FEATURES}


def add_prediction_history(source: str, prediction: Dict[str, Any]) -> None:
    entry = {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "source": source,
        "risk_category": prediction["risk_category"],
        "predicted_probability": round(prediction["predicted_probability"], 4),
    }
    history = st.session_state["prediction_history"]
    history.insert(0, entry)
    st.session_state["prediction_history"] = history[:40]


def run_prediction_from_payload(payload: Dict[str, Any], threshold: float, pipeline) -> Dict[str, Any]:
    sample_df = build_inference_dataframe(payload)
    probability = float(pipeline.predict_proba(sample_df)[0, 1])
    predicted_label = int(probability >= threshold)

    recommendations = build_personalized_recommendations(
        payload=payload,
        predicted_probability=probability,
        threshold=threshold,
    )

    return {
        "predicted_probability": probability,
        "predicted_label": predicted_label,
        "risk_category": "high_risk" if predicted_label == 1 else "low_risk",
        "recommendations": recommendations,
    }


def render_list_section(title: str, values: List[str]) -> None:
    st.markdown(f"#### {title}")
    for value in values:
        st.write(f"- {value}")


def render_prediction_details(prediction: Dict[str, Any], threshold: float, source: Optional[str] = None) -> None:
    probability = prediction["predicted_probability"]
    predicted_label = prediction["predicted_label"]
    recommendations = prediction["recommendations"]

    risk_text = "HIGH RISK" if predicted_label == 1 else "LOW RISK"
    risk_class = "risk-badge-high" if predicted_label == 1 else "risk-badge-low"
    margin = probability - threshold

    st.markdown("### Prediction Result")
    st.markdown(f'<span class="risk-badge {risk_class}">{risk_text}</span>', unsafe_allow_html=True)
    if source:
        st.caption(f"Source: {source}")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Predicted Probability", f"{probability:.2%}")
    with col2:
        st.metric("Decision Threshold", f"{threshold:.2f}")
    with col3:
        st.metric("Distance To Threshold", f"{margin:+.2f}")

    st.progress(min(max(probability, 0.0), 1.0))

    if predicted_label == 1:
        st.warning("Patient appears in a high-risk zone. Recommend clinical follow-up.")
    else:
        st.success("Patient appears in a low-risk zone based on current inputs.")

    st.markdown("### Guidance")
    st.info(recommendations["summary"])

    tab_factors, tab_exercise, tab_nutrition, tab_safety, tab_daily = st.tabs(
        ["Risk Factors", "Exercise", "Nutrition", "Safety", "Daily Plan"]
    )

    with tab_factors:
        render_list_section("Input Risk Factors", recommendations["risk_factors_detected"])

    with tab_exercise:
        render_list_section("Physical Exercise Plan", recommendations["exercise_plan"])

    with tab_nutrition:
        render_list_section("Foods To Take More", recommendations["foods_to_take_more"])
        render_list_section("Foods To Limit Or Avoid", recommendations["foods_to_limit_or_avoid"])

    with tab_safety:
        render_list_section("Natural Support Options", recommendations["natural_support_options"])
        render_list_section("Medication Safety Notes", recommendations["medication_safety_notes"])
        render_list_section("When To Seek Urgent Care", recommendations["when_to_seek_urgent_care"])
        st.markdown("#### Evidence Sources")
        for src in recommendations["evidence_sources"]:
            st.write(f"- {src}")
        st.caption(recommendations["safety_disclaimer"])

    with tab_daily:
        render_list_section("Daily Action Plan", recommendations["daily_action_plan"])


def render_sidebar(model_name: str, default_threshold: float) -> float:
    st.sidebar.header("Model Settings")
    threshold = st.sidebar.slider(
        "Decision threshold",
        min_value=0.05,
        max_value=0.95,
        value=float(default_threshold),
        step=0.01,
    )
    st.sidebar.write(f"Active model: {model_name}")
    st.sidebar.markdown(
        '<p class="subtle">Higher threshold gives fewer high-risk flags. Lower threshold gives more sensitivity.</p>',
        unsafe_allow_html=True,
    )

    if get_demo_password():
        st.sidebar.caption("Access mode: password protected")
    else:
        st.sidebar.caption("Access mode: open")

    return threshold


def render_manual_tab(threshold: float, pipeline, demo_cases: Dict[str, Dict[str, Any]]) -> None:
    st.subheader("Manual Patient Entry")

    with st.container(border=True):
        st.markdown("#### Quick Demo Loader")
        case_options = ["Custom"] + list(demo_cases.keys())
        if st.session_state["selected_demo_case"] not in case_options:
            st.session_state["selected_demo_case"] = "Custom"

        col_case, col_load, col_reset = st.columns([2, 1, 1])
        with col_case:
            st.selectbox("Select case", options=case_options, key="selected_demo_case")
        with col_load:
            if st.button("Load Case"):
                if st.session_state["selected_demo_case"] != "Custom":
                    load_payload_into_state(demo_cases[st.session_state["selected_demo_case"]])
                    st.success(f"Loaded case: {st.session_state['selected_demo_case']}")
                    st.rerun()
        with col_reset:
            if st.button("Reset Inputs"):
                reset_manual_inputs()
                st.success("Inputs reset.")
                st.rerun()

    with st.form("manual_prediction_form"):
        st.markdown("#### Clinical Inputs")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.number_input("Age", min_value=0, max_value=120, key="age")
            st.number_input("BMI", min_value=10.0, max_value=60.0, key="bmi")
            st.number_input("Systolic BP", min_value=70, max_value=250, key="systolic_bp")
            st.number_input("Diastolic BP", min_value=40, max_value=150, key="diastolic_bp")
        with c2:
            st.number_input("Glucose", min_value=40, max_value=400, key="glucose")
            st.number_input("HbA1c", min_value=3.0, max_value=18.0, key="hba1c")
            st.number_input("Cholesterol", min_value=80, max_value=450, key="cholesterol")
            st.number_input("Creatinine", min_value=0.1, max_value=8.0, key="creatinine")
        with c3:
            st.number_input("eGFR", min_value=5, max_value=150, key="egfr")
            st.number_input("Physical Activity Days/Week", min_value=0, max_value=7, key="physical_activity_days")
            st.number_input("Sleep Hours", min_value=2.0, max_value=12.0, key="sleep_hours")

        st.markdown("#### Lifestyle Inputs")
        l1, l2, l3 = st.columns(3)
        with l1:
            st.selectbox("Gender", ["female", "male"], key="gender")
            st.selectbox("Smoking Status", ["never", "former", "current"], key="smoking_status")
        with l2:
            st.selectbox("Family History", ["no", "yes"], key="family_history")
            st.selectbox("Diet Quality", ["good", "average", "poor"], key="diet_quality")
        with l3:
            st.selectbox("Alcohol Intake", ["low", "moderate", "high"], key="alcohol_intake")

        submitted = st.form_submit_button("Run Prediction", type="primary")

    if submitted:
        payload = build_manual_payload()
        missing_payload = validate_payload(payload)
        if missing_payload:
            st.error(f"Missing required inputs: {', '.join(missing_payload)}")
        else:
            prediction = run_prediction_from_payload(payload=payload, threshold=threshold, pipeline=pipeline)
            st.session_state["manual_result"] = prediction
            add_prediction_history(source="manual", prediction=prediction)

    if st.session_state["manual_result"] is not None:
        render_prediction_details(st.session_state["manual_result"], threshold, source="manual form")


def render_upload_tab(threshold: float, pipeline) -> None:
    st.subheader("Upload JSON Cases")

    with st.container(border=True):
        left, right = st.columns([2, 1])
        with left:
            st.write("Upload one or more patient JSON files.")
            uploaded_files = st.file_uploader(
                "Patient files",
                type=["json"],
                accept_multiple_files=True,
                key="uploaded_files_input",
            )
            run_upload = st.button("Run Uploaded Predictions", type="primary")
        with right:
            st.markdown("#### JSON Template")
            st.download_button(
                label="Download Template",
                data=json.dumps(DEFAULT_MANUAL_PAYLOAD, indent=2),
                file_name="patient_template.json",
                mime="application/json",
            )
            st.caption("You can also use files from data/raw/demo_patients/.")

    if run_upload:
        if not uploaded_files:
            st.warning("Please upload at least one JSON file.")
        else:
            summary_rows = []
            detailed_results = []

            for uploaded_file in uploaded_files:
                file_name = uploaded_file.name

                try:
                    payload = json.loads(uploaded_file.getvalue().decode("utf-8"))
                except json.JSONDecodeError:
                    summary_rows.append(
                        {
                            "file_name": file_name,
                            "status": "error",
                            "message": "Invalid JSON format",
                        }
                    )
                    continue

                missing_payload = validate_payload(payload)
                if missing_payload:
                    summary_rows.append(
                        {
                            "file_name": file_name,
                            "status": "error",
                            "message": f"Missing keys: {', '.join(missing_payload)}",
                        }
                    )
                    continue

                try:
                    prediction = run_prediction_from_payload(payload=payload, threshold=threshold, pipeline=pipeline)
                except Exception as exc:
                    summary_rows.append(
                        {
                            "file_name": file_name,
                            "status": "error",
                            "message": f"Prediction failed: {exc}",
                        }
                    )
                    continue

                add_prediction_history(source=f"upload:{file_name}", prediction=prediction)

                summary_rows.append(
                    {
                        "file_name": file_name,
                        "status": "ok",
                        "predicted_probability": round(prediction["predicted_probability"], 4),
                        "risk_category": prediction["risk_category"],
                    }
                )
                detailed_results.append({"file_name": file_name, **prediction})

            st.session_state["upload_summary"] = summary_rows
            st.session_state["upload_results"] = detailed_results
            if detailed_results:
                st.session_state["selected_uploaded_file"] = detailed_results[0]["file_name"]

    if st.session_state["upload_summary"]:
        st.markdown("### Batch Summary")
        summary_df = pd.DataFrame(st.session_state["upload_summary"])
        if "predicted_probability" in summary_df.columns:
            summary_df = summary_df.sort_values(by="predicted_probability", ascending=False)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

    if st.session_state["upload_results"]:
        json_payload = json.dumps(st.session_state["upload_results"], indent=2)
        st.download_button(
            label="Download Full Results (JSON)",
            data=json_payload,
            file_name="uploaded_batch_predictions.json",
            mime="application/json",
        )

        summary_df = pd.DataFrame(st.session_state["upload_summary"])
        st.download_button(
            label="Download Summary (CSV)",
            data=summary_df.to_csv(index=False),
            file_name="uploaded_batch_summary.csv",
            mime="text/csv",
        )

        options = [item["file_name"] for item in st.session_state["upload_results"]]
        if st.session_state["selected_uploaded_file"] not in options:
            st.session_state["selected_uploaded_file"] = options[0]

        selected_file_name = st.selectbox(
            "Show detailed result for file",
            options=options,
            key="selected_uploaded_file",
        )
        selected_result = next(item for item in st.session_state["upload_results"] if item["file_name"] == selected_file_name)
        render_prediction_details(selected_result, threshold, source=f"upload file: {selected_file_name}")


def render_history() -> None:
    if not st.session_state["prediction_history"]:
        return

    with st.expander("Recent Prediction History", expanded=False):
        _, clear_col = st.columns([4, 1])
        with clear_col:
            if st.button("Clear History"):
                st.session_state["prediction_history"] = []
                st.rerun()

        history_df = pd.DataFrame(st.session_state["prediction_history"])
        st.dataframe(history_df, use_container_width=True, hide_index=True)


def main() -> None:
    init_page()
    require_streamlit_login()

    if not BEST_MODEL_FILE.exists():
        st.error("Trained model not found. Run: python -m src.train")
        st.stop()

    artifact = load_artifact(BEST_MODEL_FILE)
    pipeline = artifact["pipeline"]
    model_name = artifact.get("model_name", "unknown_model")
    default_threshold = float(artifact.get("threshold", 0.5))

    init_state()
    demo_cases = load_demo_cases(DEMO_PATIENTS_DIR)
    threshold = render_sidebar(model_name=model_name, default_threshold=default_threshold)

    manual_tab, upload_tab = st.tabs(["Manual Entry", "Upload JSON Cases"])
    with manual_tab:
        render_manual_tab(threshold=threshold, pipeline=pipeline, demo_cases=demo_cases)
    with upload_tab:
        render_upload_tab(threshold=threshold, pipeline=pipeline)

    render_history()


if __name__ == "__main__":
    main()
