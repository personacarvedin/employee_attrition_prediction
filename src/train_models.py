import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from preprocess import DATA_PATH, build_pipeline

# Load raw dataset
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["Attrition"])
y = (df["Attrition"] == "Yes").astype(int)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# Define models and parameters for high accuracy
models = {
    "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "DecisionTree": DecisionTreeClassifier(max_depth=8, min_samples_leaf=10, random_state=42, class_weight="balanced"),
    "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=10, random_state=42, class_weight="balanced"),
    "AdaBoost": AdaBoostClassifier(n_estimators=150, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=150, max_depth=8, random_state=42),
    "SVM": SVC(C=1.5, probability=True, random_state=42, class_weight="balanced"),
    "KNN": KNeighborsClassifier(n_neighbors=10, weights="distance"),
    "XGB": XGBClassifier(use_label_encoder=False, eval_metric="logloss", 
                         scale_pos_weight=1.5, n_estimators=150, max_depth=8, random_state=42)
}

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Optionally use cross-validation for accuracy reporting
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = []
for name, model in models.items():
    pipe = build_pipeline(model)
    # Optionally show cross-validation score for each model
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy")
    print(f"🔎 {name} CV Accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")
    pipe.fit(X_train, y_train)
    test_acc = pipe.score(X_test, y_test)
    print(f"✅ {name} Test Accuracy: {test_acc:.3f}")
    model_path = os.path.join(MODEL_DIR, f"{name}.pkl")
    joblib.dump(pipe, model_path)
    print(f"📦 Saved {name} → {model_path}")
    results.append((name, scores.mean(), test_acc))

# Summary
print("\n=== Model Summary (CV & Test) ===")
for name, cv_acc, test_acc in results:
    print(f"{name:18} | CV Acc: {cv_acc:.3f} | Test Acc: {test_acc:.3f}")