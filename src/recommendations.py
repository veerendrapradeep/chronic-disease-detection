from typing import Any, Dict, List


def _safe_float(payload: Dict[str, Any], key: str, default: float = 0.0) -> float:
    value = payload.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_str(payload: Dict[str, Any], key: str, default: str = "") -> str:
    value = payload.get(key, default)
    if value is None:
        return default
    return str(value).strip().lower()


def build_personalized_recommendations(
    payload: Dict[str, Any],
    predicted_probability: float,
    threshold: float,
) -> Dict[str, Any]:
    bmi = _safe_float(payload, "bmi")
    systolic_bp = _safe_float(payload, "systolic_bp")
    diastolic_bp = _safe_float(payload, "diastolic_bp")
    glucose = _safe_float(payload, "glucose")
    hba1c = _safe_float(payload, "hba1c")
    activity_days = _safe_float(payload, "physical_activity_days")
    sleep_hours = _safe_float(payload, "sleep_hours")

    smoking_status = _safe_str(payload, "smoking_status")
    family_history = _safe_str(payload, "family_history")
    diet_quality = _safe_str(payload, "diet_quality")
    alcohol_intake = _safe_str(payload, "alcohol_intake")

    high_risk = predicted_probability >= threshold

    risk_factors_detected: List[str] = []
    if bmi >= 30:
        risk_factors_detected.append("BMI is in a higher-risk range.")
    if systolic_bp >= 140 or diastolic_bp >= 90:
        risk_factors_detected.append("Blood pressure is in a high range.")
    if glucose >= 126:
        risk_factors_detected.append("Glucose is elevated.")
    if hba1c >= 6.5:
        risk_factors_detected.append("HbA1c is elevated.")
    if activity_days < 3:
        risk_factors_detected.append("Weekly physical activity is below the target range.")
    if sleep_hours < 7:
        risk_factors_detected.append("Sleep duration is below recommended range.")
    if smoking_status == "current":
        risk_factors_detected.append("Current smoking increases cardiometabolic risk.")
    if diet_quality == "poor":
        risk_factors_detected.append("Diet quality appears low.")
    if alcohol_intake == "high":
        risk_factors_detected.append("Alcohol intake appears high.")
    if family_history == "yes":
        risk_factors_detected.append("Family history indicates elevated baseline risk.")

    exercise_plan = [
        "Target at least 150 minutes per week of moderate activity (for example brisk walking 30 minutes on 5 days).",
        "Add strength training 2 non-consecutive days per week (bodyweight or resistance bands).",
        "Include 8 to 10 minutes of stretching or mobility daily.",
    ]
    if activity_days < 3:
        exercise_plan.append(
            "Start with 10 to 20 minutes daily and increase by about 10 minutes each week until consistent."
        )
    if high_risk:
        exercise_plan.append(
            "If symptoms or major health conditions exist, get clinician clearance before vigorous exercise."
        )

    foods_to_take_more = [
        "Vegetables, beans, and whole grains for fiber-rich meals.",
        "Lean proteins such as fish, eggs, tofu, and pulses.",
        "Nuts and seeds in moderate portions.",
        "Water and unsweetened beverages as primary hydration.",
    ]
    if glucose >= 126 or hba1c >= 6.5:
        foods_to_take_more.append(
            "Low-glycemic carbohydrate choices such as oats, barley, millets, and legumes."
        )
    if systolic_bp >= 140 or diastolic_bp >= 90:
        foods_to_take_more.append(
            "DASH-style foods: fruits, vegetables, low-fat dairy, and potassium-rich options."
        )

    foods_to_limit_or_avoid = [
        "Sugary drinks, sweets, and refined flour snacks.",
        "Ultra-processed foods and deep-fried foods.",
        "Packaged foods high in sodium and trans fats.",
        "Smoking and tobacco in any form.",
    ]
    if systolic_bp >= 140 or diastolic_bp >= 90:
        foods_to_limit_or_avoid.append("Limit added salt and salty packaged foods as much as possible.")
    if alcohol_intake in {"moderate", "high"}:
        foods_to_limit_or_avoid.append("Reduce alcohol; avoid binge intake.")

    natural_support_options = [
        "Use food-first natural supports: vegetables, whole grains, pulses, and probiotic curd/yogurt if tolerated.",
        "Use turmeric, ginger, garlic, and cinnamon in normal culinary amounts for flavor.",
        "Use daily sunlight exposure and regular sleep timing to support metabolic health.",
        "Avoid concentrated herbal supplements unless approved by your clinician."
    ]

    medication_safety_notes = [
        "This system does not prescribe medicines.",
        "Do not start, stop, or change prescription medicines without clinician advice.",
        "If you are already on blood pressure, sugar, or cholesterol medicines, continue exactly as prescribed.",
        "Check with a clinician or pharmacist before OTC painkillers, herbal products, or supplements, especially with kidney disease risk.",
    ]

    daily_action_plan = [
        "Keep fixed meal timings and avoid late-night heavy meals.",
        "Sleep 7 to 8 hours daily.",
        "Track weekly weight, blood pressure, and activity minutes.",
        "Repeat clinical checkup based on risk level and clinician advice.",
    ]

    when_to_seek_urgent_care = [
        "Seek urgent care immediately for chest pain, severe breathlessness, one-sided weakness, confusion, or fainting.",
    ]
    if systolic_bp >= 180 or diastolic_bp >= 120:
        when_to_seek_urgent_care.append(
            "Input blood pressure is very high; obtain urgent medical evaluation."
        )
    if glucose >= 300:
        when_to_seek_urgent_care.append("Input glucose is very high; seek urgent medical evaluation.")

    summary = (
        "Risk is above the active threshold. Prioritize medical follow-up and structured lifestyle changes."
        if high_risk
        else "Risk is below the active threshold. Continue prevention habits and periodic health review."
    )

    if not risk_factors_detected:
        risk_factors_detected.append("No major input-based red flags were detected from the current payload.")

    return {
        "summary": summary,
        "risk_factors_detected": risk_factors_detected,
        "exercise_plan": exercise_plan,
        "foods_to_take_more": foods_to_take_more,
        "foods_to_limit_or_avoid": foods_to_limit_or_avoid,
        "natural_support_options": natural_support_options,
        "medication_safety_notes": medication_safety_notes,
        "daily_action_plan": daily_action_plan,
        "when_to_seek_urgent_care": when_to_seek_urgent_care,
        "evidence_sources": [
            "https://www.who.int/news-room/fact-sheets/detail/healthy-diet",
            "https://www.who.int/initiatives/behealthy/physical-activity",
            "https://www.cdc.gov/heart-disease/prevention/index.html",
            "https://www.nimh.nih.gov/health/topics/caring-for-your-mental-health",
        ],
        "safety_disclaimer": "Educational guidance only. This tool is not a diagnosis or prescription system.",
    }
