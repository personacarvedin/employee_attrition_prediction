# src/model_selection.py
import os
import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")

cv_results = pd.read_csv(os.path.join(RESULTS_DIR, "cross_validation.csv"), index_col=0)
eval_results = pd.read_csv(os.path.join(RESULTS_DIR, "evaluation.csv"), index_col=0)

# Merge results on model name
combined = cv_results.join(eval_results, how="inner")

# Rank by F1 score (primary) and accuracy (secondary)
combined = combined.sort_values(by=["f1", "accuracy"], ascending=False)
combined["rank"] = range(1, len(combined) + 1)

# Save combined results
combined.to_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"))

best_model = combined.iloc[0]

print("🏆 Best Model Selected:")
print(best_model)
print(f"✅ Full model comparison saved to {RESULTS_DIR}/model_comparison.csv")
