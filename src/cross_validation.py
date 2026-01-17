import os
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from preprocess import DATA_PATH, build_pipeline

# Load dataset
df = pd.read_csv(DATA_PATH)
# Drop non-feature columns for consistency with training and preprocessing
X = df.drop(columns=["EmpID", "Name", "Attrition"])
y = (df["Attrition"] == "Yes").astype(int)

# Models (can tune further for accuracy if desired)
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "RandomForest": RandomForestClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "KNN": KNeighborsClassifier(),
    "XGB": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
}

# Cross-validation setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = {}

for name, model in models.items():
    pipe = build_pipeline(model)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
    results[name] = {
        "mean_accuracy": np.mean(scores),
        "std_accuracy": np.std(scores)
    }
    print(f"{name}: {scores.mean():.4f} ± {scores.std():.4f}")

# Save results
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
pd.DataFrame(results).T.to_csv(os.path.join(RESULTS_DIR, "cross_validation.csv"))
print(f"✅ Cross-validation results saved to {RESULTS_DIR}/cross_validation.csv")