import os
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

from preprocess import DATA_PATH, build_pipeline

# Load dataset
df = pd.read_csv(DATA_PATH)
# Drop non-feature columns for consistency with training
X = df.drop(columns=["EmpID", "Name", "Attrition"])
y = (df["Attrition"] == "Yes").astype(int)

# Paths
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

results = {}

for model_file in os.listdir(MODEL_DIR):
    if model_file.endswith(".pkl"):
        model_name = model_file.replace(".pkl", "")
        pipe = joblib.load(os.path.join(MODEL_DIR, model_file))
        
        y_pred = pipe.predict(X)

        results[model_name] = {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "f1": f1_score(y, y_pred, zero_division=0)
        }

        print(f"📊 {model_name}")
        print(classification_report(y, y_pred, zero_division=0))
        print("Confusion Matrix:\n", confusion_matrix(y, y_pred), "\n")

# Save evaluation results
pd.DataFrame(results).T.to_csv(os.path.join(RESULTS_DIR, "evaluation.csv"))
print(f"✅ Evaluation results saved to {RESULTS_DIR}/evaluation.csv")