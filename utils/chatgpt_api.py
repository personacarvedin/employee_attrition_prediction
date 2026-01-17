# utils/chatgpt_api.py
import os
import re
import json
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from typing import Optional, Dict, Any

# --- INITIALIZATION ---
load_dotenv()
client = OpenAI()

# --- HELPERS ---
NUMERIC_WORD_MAP = {
    'very low': 1, 'low': 1,
    'medium': 2, 'moderate': 2,
    'high': 3, 'very high': 4, 'excellent': 4,
    'yes': 1, 'no': 0,
    'poor': 1, 'fair': 2, 'good': 3, 'very good': 4
}

def extract_first_number(val) -> Optional[float]:
    """
    Extract first numeric value from strings like '6 hours', '$5,000', '2 years'.
    Returns float or None if no numeric token found.
    """
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, (int, float)):
        try:
            return float(val)
        except Exception:
            return None
    s = str(val).strip()
    if s == '':
        return None
    # Remove common currency symbols and thousands separators
    s_clean = s.replace(',', '').replace('$', '').replace('€', '').replace('£', '')
    m = re.search(r'[-+]?\d*\.?\d+', s_clean)
    if m:
        try:
            return float(m.group())
        except Exception:
            return None
    # Fallback to mapping textual categories (e.g., "low" -> 1)
    lower = s.lower()
    for word, num in NUMERIC_WORD_MAP.items():
        if word in lower:
            return float(num)
    return None

def interpret_categorical_as_number(val) -> Optional[float]:
    """
    Interpret categorical strings (e.g., 'Low (1)', 'Low', '3') as numeric ordinals.
    Returns float or None.
    """
    if val is None:
        return None
    if isinstance(val, (int, float)):
        try:
            return float(val)
        except Exception:
            return None
    s = str(val).strip()
    # Prefer parenthetical numeric e.g. 'Low (1)'
    m = re.search(r'\((\d+)\)', s)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass
    # Pure numeric string
    if re.match(r'^\s*[-+]?\d+(\.\d+)?\s*$', s):
        try:
            return float(s)
        except Exception:
            pass
    # Word mapping
    lower = s.lower()
    for word, num in NUMERIC_WORD_MAP.items():
        if word in lower:
            return float(num)
    return None

def clamp_probability(p: float, low: float = 0.01, high: float = 0.95) -> float:
    try:
        pv = float(p)
    except Exception:
        pv = low
    return max(low, min(high, pv))

# --- MAIN API FUNCTION ---
def get_full_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Sends employee data (DataFrame) to the LLM and requests a structured analysis.
    Attempts to coerce numeric fields; falls back to a safe local heuristic if the LLM fails.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("get_full_analysis expects a pandas DataFrame")

    # Build employee JSON lines for the LLM prompt
    employee_data_string = ""
    for i, row in df.iterrows():
        row_dict = {}
        for key, value in row.to_dict().items():
            # Normalize NaN to None for JSON
            if pd.isna(value):
                row_dict[key] = None
                continue

            lower_key = str(key).lower()
            # Numeric-like keys: try to coerce
            if lower_key in (
                'age', 'yearsatcompany', 'years_at_company', 'years',
                'monthlyincome', 'monthly_income',
                'overtimehours', 'over_time_hours', 'ot',
                'distancefromhome_km', 'distancefromhome', 'distance',
                'promotionlast5years', 'promotion_last_5_years'
            ):
                num = extract_first_number(value)
                if num is None:
                    num = interpret_categorical_as_number(value)
                row_dict[key] = num if num is not None else str(value)
                continue

            # Ordinal-like keys: job satisfaction / work-life balance
            if lower_key in ('jobsatisfaction', 'job_satisfaction', 'jobsat',
                             'worklifebalance', 'work_life_balance', 'wlb'):
                num = interpret_categorical_as_number(value)
                row_dict[key] = num if num is not None else str(value)
                continue

            # Default: keep strings/numbers as is (JSON-serializable)
            if isinstance(value, (int, float)):
                row_dict[key] = float(value)
            else:
                row_dict[key] = str(value)

        # Ensure a Name exists
        if 'Name' not in row_dict or not row_dict['Name'] or row_dict['Name'] == 'None':
            row_dict['Name'] = f"Employee_{i+1}"

        employee_data_string += f"{json.dumps(row_dict)}\n"

    # Compose the LLM prompt
    prompt = (
        "You are an expert HR Data Scientist analyzing employee attrition risk.\n\n"
        "CRITICAL: You MUST return ONLY valid JSON with this EXACT structure:\n"
        "{\n"
        '  "predictions": [\n'
        '    {\n'
        '      "name": "Employee_1",\n'
        '      "prediction": "Yes",\n'
        '      "probability": 0.75,\n'
        '      "key_factors": [\n'
        '        {"feature": "Job Satisfaction", "value": "Low (1)", "impact": "High"},\n'
        '        {"feature": "Overtime Hours", "value": "20 hours", "impact": "High"}\n'
        '      ]\n'
        '    }, ...\n'
        '  ],\n'
        '  "feature_importance": [{"feature": "Job Satisfaction", "score": 0.85}, ...],\n'
        '  "demographics_plot_data": {"1 - Low": 5, "2 - Medium": 3, "3 - High": 1, "4 - Very High": 0}\n'
        "}\n\n"
        "CRITICAL RULES FOR KEY_FACTORS:\n"
        "1. For employees with HIGH RISK (prediction='Yes'):\n"
        "   - ONLY show factors with 'High' impact\n"
        "   - If NO High impact factors exist, show 'Medium' impact factors\n"
        "   - Show 2-5 most critical factors\n"
        "2. For employees with LOW RISK (prediction='No'):\n"
        "   - Return EMPTY array for key_factors: []\n"
        "   - Do NOT show any factors for low-risk employees\n"
        "3. Impact classification:\n"
        "   - HIGH: JobSatisfaction ≤2, WorkLifeBalance ≤2, OverTimeHours >10, YearsAtCompany <3, MonthlyIncome <4000, DistanceFromHome >25km, CustomFeature with negative value\n"
        "   - MEDIUM: JobSatisfaction =3, WorkLifeBalance =3, OverTimeHours 5-10, YearsAtCompany 3-5, MonthlyIncome 4000-6000, DistanceFromHome 15-25km, PromotionLast5Years=0\n"
        "   - LOW: All other values (don't include these)\n"
        "4. Value formatting:\n"
        "   - JobSatisfaction/WorkLifeBalance: 'Low (1)', 'Medium (2)', 'High (3)', 'Very High (4)'\n"
        "   - MonthlyIncome: '$5000' format\n"
        "   - OverTimeHours: '15 hours' format\n"
        "   - YearsAtCompany: '2 years' format\n"
        "   - Age: '28 years' format\n"
        "   - DistanceFromHome_km: '25 km' format\n"
        "   - PromotionLast5Years: 'Yes' or 'No'\n"
        "5. Analyze ALL features: JobSatisfaction, WorkLifeBalance, OverTimeHours, YearsAtCompany, MonthlyIncome, Age, DistanceFromHome_km, PromotionLast5Years, CustomFeature\n\n"
        "FEATURE IMPORTANCE:\n"
        "- List top 6-8 features across all employees with importance scores 0.0-1.0\n\n"
        "DEMOGRAPHICS:\n"
        "- Map job satisfaction to attrition counts using exact keys: '1 - Low', '2 - Medium', '3 - High', '4 - Very High'\n\n"
        "Employee Data:\n"
        f"{employee_data_string}\n\n"
        "Return ONLY the JSON object, no other text."
    )

    # Call the LLM
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a data analysis assistant. Respond ONLY with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )

        content = response.choices[0].message.content
        full_analysis = json.loads(content)

        # Basic validation of expected structure
        if not all(k in full_analysis for k in ('predictions', 'feature_importance', 'demographics_plot_data')):
            raise ValueError("LLM returned JSON missing required keys")

        # Clamp probabilities to reasonable bounds
        for p in full_analysis.get('predictions', []):
            p['probability'] = clamp_probability(p.get('probability', 0.01))

        # Ensure demographics is a dict
        if not isinstance(full_analysis['demographics_plot_data'], dict):
            raise ValueError("demographics_plot_data must be a dict")

        return full_analysis

    except Exception as e:
        # Fallback local heuristic if LLM fails
        print(f"[chatgpt_api] LLM error or validation failed: {e}. Using fallback logic.")

        fallback_predictions = []
        for i, row in df.iterrows():
            name = row.get('Name', f"Employee_{i+1}")

            # Robust parsing with defaults
            job_sat = interpret_categorical_as_number(row.get('JobSatisfaction')) or extract_first_number(row.get('JobSatisfaction')) or 3.0
            try:
                job_sat = float(job_sat)
            except Exception:
                job_sat = 3.0

            wlb = interpret_categorical_as_number(row.get('WorkLifeBalance')) or extract_first_number(row.get('WorkLifeBalance')) or 3.0
            try:
                wlb = float(wlb)
            except Exception:
                wlb = 3.0

            ot_val = extract_first_number(row.get('OverTimeHours')) or 0.0
            try:
                ot_val = float(ot_val)
            except Exception:
                ot_val = 0.0

            years_val = extract_first_number(row.get('YearsAtCompany')) or 3.0
            try:
                years_val = float(years_val)
            except Exception:
                years_val = 3.0

            income_val = extract_first_number(row.get('MonthlyIncome')) or 5000.0
            try:
                income_val = float(income_val)
            except Exception:
                income_val = 5000.0

            dist_val = extract_first_number(row.get('DistanceFromHome_km')) or 0.0
            try:
                dist_val = float(dist_val)
            except Exception:
                dist_val = 0.0

            promo_val = interpret_categorical_as_number(row.get('PromotionLast5Years'))
            if promo_val is None:
                promo_val = extract_first_number(row.get('PromotionLast5Years'))
            try:
                promo_val = int(promo_val) if promo_val is not None else 0
            except Exception:
                promo_val = 0

            # Simple heuristic: lower job satisfaction increases risk
            risk_prob = max(0.01, min(0.99, (5 - job_sat) / 4))
            is_high_risk = risk_prob > 0.5

            key_factors = []
            if is_high_risk:
                high = []
                medium = []

                if job_sat <= 2:
                    js_label = {1: 'Low (1)', 2: 'Medium (2)'}.get(int(job_sat), f'Level {job_sat}')
                    high.append({"feature": "Job Satisfaction", "value": js_label, "impact": "High"})
                elif job_sat == 3:
                    medium.append({"feature": "Job Satisfaction", "value": "High (3)", "impact": "Medium"})

                if wlb <= 2:
                    wlb_label = {1: 'Poor (1)', 2: 'Fair (2)'}.get(int(wlb), f'Level {wlb}')
                    high.append({"feature": "Work Life Balance", "value": wlb_label, "impact": "High"})
                elif wlb == 3:
                    medium.append({"feature": "Work Life Balance", "value": "Good (3)", "impact": "Medium"})

                if ot_val > 10:
                    high.append({"feature": "Overtime Hours", "value": f"{int(ot_val)} hours", "impact": "High"})
                elif ot_val >= 5:
                    medium.append({"feature": "Overtime Hours", "value": f"{int(ot_val)} hours", "impact": "Medium"})

                if years_val < 3:
                    high.append({"feature": "Years At Company", "value": f"{int(years_val)} years", "impact": "High"})
                elif years_val <= 5:
                    medium.append({"feature": "Years At Company", "value": f"{int(years_val)} years", "impact": "Medium"})

                if income_val < 4000:
                    high.append({"feature": "Monthly Income", "value": f"${int(income_val)}", "impact": "High"})
                elif income_val < 6000:
                    medium.append({"feature": "Monthly Income", "value": f"${int(income_val)}", "impact": "Medium"})

                if dist_val > 25:
                    high.append({"feature": "Distance From Home", "value": f"{int(dist_val)} km", "impact": "High"})
                elif dist_val > 15:
                    medium.append({"feature": "Distance From Home", "value": f"{int(dist_val)} km", "impact": "Medium"})

                if promo_val == 0:
                    medium.append({"feature": "No Promotion in 5 Years", "value": "No", "impact": "Medium"})

                # Custom feature included if present
                if 'CustomFeature' in row and 'CustomFeatureValue' in row and pd.notna(row.get('CustomFeature')) and pd.notna(row.get('CustomFeatureValue')):
                    high.append({"feature": str(row.get('CustomFeature')), "value": str(row.get('CustomFeatureValue')), "impact": "High"})

                key_factors = high[:5] if high else medium[:5]

            fallback_predictions.append({
                "name": str(name),
                "prediction": "Yes" if is_high_risk else "No",
                "probability": clamp_probability(risk_prob),
                "key_factors": key_factors
            })

        fallback_feature_importance = [
            {"feature": "Job Satisfaction", "score": 0.85},
            {"feature": "Work Life Balance", "score": 0.78},
            {"feature": "Overtime Hours", "score": 0.72},
            {"feature": "Years At Company", "score": 0.65},
            {"feature": "Monthly Income", "score": 0.58},
            {"feature": "Distance From Home", "score": 0.45},
            {"feature": "Promotion Last 5 Years", "score": 0.38}
        ]

        demography = {"1 - Low": 0, "2 - Medium": 0, "3 - High": 0, "4 - Very High": 0}
        for pred, row in zip(fallback_predictions, df.to_dict(orient='records')):
            js = interpret_categorical_as_number(row.get('JobSatisfaction'))
            try:
                js_int = int(js) if js is not None else None
            except Exception:
                js_int = None
            if js_int == 1:
                demography["1 - Low"] += 1
            elif js_int == 2:
                demography["2 - Medium"] += 1
            elif js_int == 3:
                demography["3 - High"] += 1
            elif js_int == 4:
                demography["4 - Very High"] += 1

        return {
            "predictions": fallback_predictions,
            "feature_importance": fallback_feature_importance,
            "demographics_plot_data": demography
        }