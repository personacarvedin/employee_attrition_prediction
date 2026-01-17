import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# -------------------------------
# Safe path handling
# -------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "employees.csv")

# -------------------------------
# Feature engineering function
# -------------------------------
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Income per year (avoid div by zero)
    df['IncomePerYear'] = df['MonthlyIncome'] / (df['YearsAtCompany'] + 1)
    df['AgeBand'] = pd.cut(df['Age'], bins=[19,29,39,49,59,70], labels=['20s','30s','40s','50s','60+'])
    df['TenureBand'] = pd.cut(df['YearsAtCompany'], bins=[-1,1,3,5,10,20], labels=['<1','1-3','3-5','5-10','10+'])
    # Parse OverTimeHours ("No OT" or "8 hours") to numeric
    def parse_ot(x):
        if isinstance(x, str) and "No OT" in x:
            return 0
        try:
            return int(str(x).split()[0])
        except Exception:
            return 0
    df['OverTimeHoursNum'] = df['OverTimeHours'].apply(parse_ot)
    df['OvertimeWLB'] = df['OverTimeHours'].astype(str) + "_" + df['WorkLifeBalance'].astype(str)
    return df

# -------------------------------
# Top-level column selectors (for pickling!)
# -------------------------------
def select_numeric_columns(df):
    # Exclude index columns and target if present
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    # Exclude EmpID if it's numerical in some datasets
    return [c for c in num_cols if c not in ["EmpID"]]

def select_categorical_columns(df):
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    # Exclude Name and Attrition and EmpID if present
    return [c for c in cat_cols if c not in ["Name", "Attrition", "EmpID"]]

# -------------------------------
# Build full preprocessing pipeline
# -------------------------------
def build_pipeline(model):
    feature_engineering_transformer = FunctionTransformer(feature_engineering)

    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ])

    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipe, select_numeric_columns),
        ("cat", cat_pipe, select_categorical_columns)
    ])

    pipe = Pipeline([
        ("features", feature_engineering_transformer),
        ("preprocess", preprocessor),
        ("model", model)
    ])
    return pipe

# Quick check (for dev only)
if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    # Drop ID and Name columns for training
    X = df.drop(columns=["EmpID", "Name", "Attrition"])
    y = (df["Attrition"]=="Yes").astype(int)
    from sklearn.linear_model import LogisticRegression
    pipe = build_pipeline(LogisticRegression(max_iter=500))
    pipe.fit(X, y)
    print("✅ Pipeline builds & fits correctly")