# utils/data_cleaning.py
import re
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List, Optional

# (Keep or reuse your existing canonicalization / parsing functions.)
CANONICAL_NAMES = {
    'name': 'Name',
    'employee_name': 'Name',
    'age': 'Age',
    'yearsatcompany': 'YearsAtCompany',
    'years_at_company': 'YearsAtCompany',
    'years': 'YearsAtCompany',
    'monthlyincome': 'MonthlyIncome',
    'monthly_income': 'MonthlyIncome',
    'jobsatisfaction': 'JobSatisfaction',
    'job_satisfaction': 'JobSatisfaction',
    'jobsat': 'JobSatisfaction',
    'worklifebalance': 'WorkLifeBalance',
    'work_life_balance': 'WorkLifeBalance',
    'wlb': 'WorkLifeBalance',
    'overtimehours': 'OverTimeHours',
    'overtime_hours': 'OverTimeHours',
    'ot': 'OverTimeHours',
    'distancefromhome_km': 'DistanceFromHome_km',
    'distancefromhome': 'DistanceFromHome_km',
    'distance': 'DistanceFromHome_km',
    'promotionlast5years': 'PromotionLast5Years',
    'promotion_last_5_years': 'PromotionLast5Years',
    'promotion': 'PromotionLast5Years',
    'customfeature': 'CustomFeature',
    'customfeaturevalue': 'CustomFeatureValue',
    'custom_feature': 'CustomFeature',
    'custom_feature_value': 'CustomFeatureValue'
}

NUMERIC_WORD_MAP = {
    'very low': 1, 'low': 1,
    'medium': 2, 'moderate': 2,
    'high': 3, 'very high': 4, 'excellent': 4,
    'yes': 1, 'no': 0,
    'poor': 1, 'fair': 2, 'good': 3, 'very good': 4
}

def extract_first_number(val) -> Optional[float]:
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
    s_clean = s.replace(',', '').replace('$', '').replace('€', '').replace('£', '')
    m = re.search(r'[-+]?\d*\.?\d+', s_clean)
    if m:
        try:
            return float(m.group())
        except Exception:
            return None
    lower = s.lower()
    for word, num in NUMERIC_WORD_MAP.items():
        if word in lower:
            return float(num)
    return None

def interpret_categorical_as_number(val) -> Optional[float]:
    if val is None:
        return None
    if isinstance(val, (int, float)):
        try:
            return float(val)
        except Exception:
            return None
    s = str(val).strip()
    m = re.search(r'\((\d+)\)', s)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass
    m2 = re.match(r'^\s*[-+]?\d+(\.\d+)?\s*$', s)
    if m2:
        try:
            return float(s)
        except Exception:
            pass
    lower = s.lower()
    for word, num in NUMERIC_WORD_MAP.items():
        if word in lower:
            return float(num)
    return None

def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for col in df.columns:
        key = re.sub(r'\W+', '', col).lower()
        if key in CANONICAL_NAMES:
            rename_map[col] = CANONICAL_NAMES[key]
    if rename_map:
        df = df.rename(columns=rename_map)
    return df

def ensure_id_and_name(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Ensure every row has a Name and an internal Employee_ID column.
    Return df and metadata about assigned names/ids.
    """
    df = df.copy()
    metadata = {'assigned_names': 0, 'employee_id_col': 'Employee_ID'}
    # Assign Name column if missing
    if 'Name' not in df.columns:
        df['Name'] = None
    # Trim and normalize name strings
    df['Name'] = df['Name'].apply(lambda v: None if (v is None or (isinstance(v, float) and pd.isna(v)) or str(v).strip() == '') else str(v).strip())
    # Assign fallback names and create stable Employee_ID
    ids = []
    for i, name in enumerate(df['Name']):
        if not name:
            fallback = f"Employee_{i+1}"
            df.at[i, 'Name'] = fallback
            metadata['assigned_names'] += 1
        ids.append(f"EID_{i+1}")
    df[metadata['employee_id_col']] = ids
    return df, metadata

def deduplicate_dataframe(
    df: pd.DataFrame,
    subset: Optional[List[str]] = None,
    keep: str = 'first',
    aggregate: Optional[Dict[str, str]] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Deduplicate a DataFrame.

    Parameters:
    - df: DataFrame to dedupe
    - subset: list of column names to consider when defining duplicates. If None, defaults to ['Name'].
    - keep: 'first' or 'last' or 'drop' (drop all duplicates — keep none); if aggregate is provided, duplicates will be aggregated instead.
    - aggregate: dict mapping column -> aggregation function (e.g. {'MonthlyIncome':'mean', 'YearsAtCompany':'max'}). If provided, duplicates grouped and aggregated.

    Returns:
    - deduped_df, metadata (with counts removed and strategy used)
    """
    if subset is None:
        subset = ['Name']
    df = df.copy()
    metadata = {'original_rows': len(df), 'subset': subset, 'strategy': 'drop_duplicates', 'removed': 0}

    # If no Name column, create fallback names and IDs first
    if 'Name' not in df.columns:
        df, name_meta = ensure_id_and_name(df)
        metadata.update(name_meta)

    # If there are Employee_IDs, keep them stable (should already be unique)
    # Option: prefer deduping on Employee_ID if present and intended
    # By default, dedupe on subset columns
    if aggregate:
        # Group by subset, apply aggregations, and rebuild a row for each group
        grouped = df.groupby(subset, dropna=False)
        # For columns not in aggregate, we will keep the first() occurrence
        agg_dict = {}
        for c in df.columns:
            if c in aggregate:
                agg_dict[c] = aggregate[c]
            else:
                agg_dict[c] = 'first'
        df_agg = grouped.agg(agg_dict).reset_index()
        removed = metadata['original_rows'] - len(df_agg)
        metadata['removed'] = int(removed)
        metadata['strategy'] = 'aggregate'
        return df_agg, metadata

    else:
        # Simple drop_duplicates behaviour
        before = len(df)
        if keep == 'drop':
            # remove all rows that are duplicated (i.e., keep only unique rows that appear once)
            dup_mask = df.duplicated(subset=subset, keep=False)
            df_unique = df[~dup_mask].reset_index(drop=True)
            removed = before - len(df_unique)
            metadata['removed'] = int(removed)
            metadata['strategy'] = 'drop_all_duplicates'
            return df_unique, metadata
        else:
            # keep 'first' or 'last'
            df_deduped = df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)
            removed = before - len(df_deduped)
            metadata['removed'] = int(removed)
            metadata['strategy'] = f'drop_duplicates_keep_{keep}'
            return df_deduped, metadata