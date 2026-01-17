# utils/plot_utils.py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

def plot_to_base64(fig) -> str:
    """Converts a Matplotlib figure object to a Base64 encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def generate_feature_plot(feature_data: list) -> str:
    """
    Generates a horizontal bar chart based on data provided by the LLM.
    """
    if not feature_data or len(feature_data) == 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, 'No feature data available',
                ha='center', va='center', fontsize=14, color='red')
        ax.axis('off')
        return plot_to_base64(fig)

    feature_data.sort(key=lambda x: x.get('score', 0), reverse=True)

    features = [d.get('feature', 'Unknown') for d in feature_data]
    scores = [d.get('score', 0) for d in feature_data]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(features, scores, color='#DC143C')

    for i, (bar, score) in enumerate(zip(bars, scores)):
        ax.text(score + 0.02, i, f'{score:.2f}',
                va='center', fontsize=9, color='black')

    ax.set_title("Feature Importance (SHAP-Style)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Relative Importance Score", fontsize=11)
    ax.set_xlim(0, max(scores) * 1.15 if scores else 1)
    ax.tick_params(axis='y', labelsize=10)
    ax.invert_yaxis()
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    plt.tight_layout()

    return plot_to_base64(fig)

def generate_demographic_plot(demo_data: dict) -> str:
    """
    Generates a bar chart showing predicted attrition counts across Job Satisfaction levels.
    """
    if not demo_data or len(demo_data) == 0:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(0.5, 0.5, 'No demographic data available',
                ha='center', va='center', fontsize=14, color='red')
        ax.axis('off')
        return plot_to_base64(fig)

    ordered_keys = ['1 - Low', '2 - Medium', '3 - High', '4 - Very High']
    labels = [k for k in ordered_keys if k in demo_data]
    counts = [demo_data[k] for k in labels]

    if not labels:
        labels = list(demo_data.keys())
        counts = list(demo_data.values())

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, counts, color='#1E90FF', edgecolor='black', linewidth=1.2)

    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(count)}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_title("Predicted Attrition by Job Satisfaction", fontsize=13, fontweight='bold')
    ax.set_ylabel("Count of 'Yes' Predictions", fontsize=11)
    ax.set_xlabel("Job Satisfaction Level", fontsize=11)
    ax.set_ylim(0, max(counts) * 1.2 if counts and max(counts) > 0 else 5)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()

    return plot_to_base64(fig)