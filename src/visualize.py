# src/visualize.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import shap

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "employees.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "RandomForest.pkl")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# -------------------------------
# Load raw data for EDA
# -------------------------------
df_raw = pd.read_csv(DATA_PATH)

# -------------------------------
# EDA - Numeric features
# -------------------------------
numeric_cols = ["Age", "YearsAtCompany", "MonthlyIncome", "DistanceFromHome_km"]
for col in numeric_cols:
    if col in df_raw.columns:
        plt.figure(figsize=(6,4))
        sns.histplot(df_raw[col], kde=True, bins=30)
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f"dist_{col}.png"))
        plt.close()

        plt.figure(figsize=(6,4))
        sns.boxplot(x="Attrition", y=col, data=df_raw)
        plt.title(f"{col} vs Attrition")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f"box_{col}.png"))
        plt.close()

# -------------------------------
# EDA - Categorical features
# -------------------------------
categorical_cols = ["OverTimeHours", "JobSatisfaction", "WorkLifeBalance", "PromotionLast5Years"]
for col in categorical_cols:
    if col in df_raw.columns:
        plt.figure(figsize=(6,4))
        sns.countplot(x=col, hue="Attrition", data=df_raw)
        plt.title(f"{col} vs Attrition")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f"cat_{col}.png"))
        plt.close()

# Attrition distribution
plt.figure(figsize=(5,4))
sns.countplot(x="Attrition", data=df_raw)
plt.title("Attrition Distribution")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "attrition_distribution.png"))
plt.close()

# -------------------------------
# Load trained model & data
# -------------------------------
pipe = joblib.load(MODEL_PATH)
X = df_raw.drop(columns=["Attrition"])
y = (df_raw["Attrition"]=="Yes").astype(int)

# -------------------------------
# Model Evaluation
# -------------------------------
y_pred = pipe.predict(X)
y_prob = pipe.predict_proba(X)[:,1]

# Confusion Matrix
cm = confusion_matrix(y, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "confusion_matrix.png"))
plt.close()

# ROC Curve
fpr, tpr, _ = roc_curve(y, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0,1],[0,1],"--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "roc_curve.png"))
plt.close()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y, y_prob)
plt.figure(figsize=(6,4))
plt.plot(recall, precision, color="purple")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "pr_curve.png"))
plt.close()

# -------------------------------
# Feature Importance
# -------------------------------
try:
    # Get transformed feature names
    preprocessor = pipe.named_steps["preprocess"]
    feature_names = []
    for name, transformer, cols in preprocessor.transformers_:
        if hasattr(transformer, "get_feature_names_out"):
            feature_names.extend(transformer.get_feature_names_out(cols))
        else:
            feature_names.extend(cols)

    if hasattr(pipe[-1], "feature_importances_"):
        importances = pipe[-1].feature_importances_

        # Align lengths
        if len(importances) != len(feature_names):
            print(f"⚠️ Mismatch: {len(importances)} importances vs {len(feature_names)} features")
            feature_names = feature_names[:len(importances)]  # trim safely

        fi_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        })
        fi_df = fi_df.sort_values("Importance", ascending=False)

        plt.figure(figsize=(10,6))
        sns.barplot(x="Importance", y="Feature", data=fi_df.head(15))
        plt.title("Top 15 Feature Importances")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "feature_importances.png"))
        plt.close()
except Exception as e:
    print(f"⚠️ Feature importance skipped → {e}")

# -------------------------------
# SHAP Summary Plot
# -------------------------------
try:
    X_transformed = pipe[:-1].transform(X)
    if hasattr(X_transformed, "toarray"):
        X_dense = X_transformed.toarray()
    else:
        X_dense = X_transformed

    X_dense = X_dense[:500]  # limit for speed
    explainer = shap.Explainer(pipe[-1], X_dense)
    shap_values = explainer(X_dense)

    plt.figure()
    shap.summary_plot(shap_values, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "shap_summary.png"))
    plt.close()
except Exception as e:
    print(f"⚠️ SHAP plot skipped → {e}")
    
print("✅ All visualizations saved in 'figures/'")
