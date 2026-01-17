# src/interpret.py

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.inspection import permutation_importance, PartialDependenceDisplay
import shap

from preprocess import DATA_PATH

# =====================
# Quick / Full toggle
# =====================
QUICK = True   # 🔹 set to False for full detailed run

# =====================
# Paths
# =====================
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# =====================
# Load Data
# =====================
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["Attrition"])
y = (df["Attrition"] == "Yes").astype(int)

# Apply sampling in QUICK mode
if QUICK and len(X) > 500:
    X = X.sample(500, random_state=42)
    y = y.loc[X.index]

# =====================
# Load Models
# =====================
model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pkl")]

# =====================
# Feature Names Helper
# =====================
def get_feature_names_from_column_transformer(preprocessor):
    feature_names = []
    for name, transformer, cols in preprocessor.transformers_:
        if name == "remainder" and transformer == "drop":
            continue
        if hasattr(transformer, "get_feature_names_out"):
            feature_names.extend(transformer.get_feature_names_out(cols))
        else:
            feature_names.extend(cols)
    return feature_names

# =====================
# Loop over models
# =====================
for model_file in model_files:
    model_path = os.path.join(MODEL_DIR, model_file)
    pipe = joblib.load(model_path)
    model_name = model_file.replace(".pkl", "")
    print(f"\n🔎 Analyzing {model_name}")

    # Extract features after preprocessing
    preprocessor = pipe.named_steps["preprocess"]
    feature_names = get_feature_names_from_column_transformer(preprocessor)
    print(f"📋 {model_name}: {len(feature_names)} features")

    # Transform X for downstream tasks
    X_transformed = pipe[:-1].transform(X)
    if hasattr(X_transformed, "toarray"):  # convert sparse → dense
        X_dense = X_transformed.toarray()
    else:
        X_dense = X_transformed

    # -----------------------
    # 1. Feature Importances
    # -----------------------
    if hasattr(pipe[-1], "feature_importances_"):
        importances = pipe[-1].feature_importances_
        fi_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
        fi_df = fi_df.sort_values("Importance", ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x="Importance", y="Feature", data=fi_df.head(15))
        plt.title(f"Top 15 Feature Importances - {model_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f"{model_name}_feature_importances.png"))
        plt.close()
        print(f"📊 {model_name}: feature importance plot saved.")

    # -----------------------
    # 2. Permutation Importance
    # -----------------------
    try:
        repeats = 3 if QUICK else 10
        r = permutation_importance(pipe[-1], X_dense, y, n_repeats=repeats, random_state=42, n_jobs=-1)
        perm_df = pd.DataFrame({"Feature": feature_names, "Importance": r.importances_mean})
        perm_df = perm_df.sort_values("Importance", ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x="Importance", y="Feature", data=perm_df.head(15))
        plt.title(f"Top 15 Permutation Importances - {model_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f"{model_name}_permutation_importance.png"))
        plt.close()
        print(f"📊 {model_name}: permutation importance plot saved.")
    except Exception as e:
        print(f"⚠️ {model_name}: permutation importance skipped → {e}")

    # -----------------------
    # 3. Partial Dependence Plots
    # -----------------------
    features_to_plot = ["JobSatisfaction", "WorkLifeBalance", "MonthlyIncome", "YearsAtCompany"]
    for feat in features_to_plot:
        if feat in X.columns:
            try:
                fig, ax = plt.subplots(figsize=(6, 4))
                PartialDependenceDisplay.from_estimator(pipe, X, [feat], ax=ax)
                plt.tight_layout()
                plt.savefig(os.path.join(FIGURES_DIR, f"{model_name}_pdp_{feat}.png"))
                plt.close()
                print(f"📊 {model_name}: PDP for {feat} saved.")
            except Exception as e:
                print(f"⚠️ {model_name}: could not plot PDP for {feat} → {e}")

    # -----------------------
    # 4. SHAP Analysis
    # -----------------------
    try:
        if QUICK:
            X_shap = X_dense[:200]  # only explain 200 samples
        else:
            X_shap = X_dense

        if hasattr(pipe[-1], "predict_proba") and "Forest" in model_name or "Boost" in model_name or "Tree" in model_name:
            explainer = shap.TreeExplainer(pipe[-1])
        else:
            explainer = shap.Explainer(pipe[-1], feature_names=feature_names)

        shap_values = explainer(X_shap)

        # Summary Plot
        plt.figure()
        shap.summary_plot(shap_values, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f"{model_name}_shap_summary.png"))
        plt.close()
        print(f"📊 {model_name}: SHAP summary plot saved.")
    except Exception as e:
        print(f"⚠️ {model_name}: SHAP skipped → {e}")

print(f"\n✅ All interpretability plots saved in {FIGURES_DIR}")
