import pandas as pd
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT / "artifacts" / "phase4"
METRICS_DIR = PROJECT_ROOT / "metrics"


CSV_PATH = PROJECT_ROOT / "metrics" / "robust_failure_analysis.csv"
df = pd.read_csv(CSV_PATH)
total = len(df)
print(f"Loaded {total} rows from {CSV_PATH}")

thresholds = [0.5, 0.6,0.7,0.8,0.9,0.95,0.98]

results = []

for threshold in thresholds:
    accepted_mask = df["confidence"] >= threshold
    accepted = df[accepted_mask]
    rejected = df[~accepted_mask]

    accepted_count = len(accepted)
    rejected_count = len(rejected)
    correct_accepted = accepted["is_correct"].sum() if accepted_count > 0 else 0

    #compute metrics
    coverage = accepted_count / total
    accepted_accuracy = correct_accepted / accepted_count if accepted_count >0 else 0.0
    rejection_rate = rejected_count / total

    results.append({
        "threshold": threshold,
        "coverage": round(coverage, 4),
        "accepted_accuracy": round(accepted_accuracy, 4),
        "rejection_rate": round(rejection_rate, 4),
        "accepted_count": accepted_count,
        "rejected_count": rejected_count,
    })

    print(
        f"threshold={threshold:.2f} | "
        f"coverage={coverage*100:.1f}% | "
        f"accepted_acc={accepted_accuracy*100:.2f}% | "
        f"rejection_rate={rejection_rate*100:.1f}%"
    )

#save csv report

results_df = pd.DataFrame(results)
output_csv = METRICS_DIR / "threshold_evaluation.csv"
results_df.to_csv(output_csv, index=False)
print(f"Saved metrics to -> {output_csv}")

#plot threshold curves
fig, axes = plt.subplots(1,2,figsize=(12,5))
fig.suptitle("Threshold Evaluation - Uncertainty Rejection", fontsize = 14, fontweight = "bold")
x= results_df["threshold"]

#left: accepted accuracy vs threshold
ax1 = axes[0]
ax1.plot(x, results_df["accepted_accuracy"] * 100, marker="o", color="#2196F3", linewidth=2, markersize=7)
for _, row in results_df.iterrows():
    ax1.annotate(
        f'{row["accepted_accuracy"]*100:.1f}%',
        (row["threshold"], row["accepted_accuracy"] * 100),
        textcoords="offset points", xytext=(0, 8),
        ha="center", fontsize=8, color="#1565C0"
    )
ax1.set_title("Accepted Accuracy vs Threshold")
ax1.set_xlabel("Confidence Threshold")
ax1.set_ylabel("Accepted Accuracy (%)")
ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
ax1.set_xticks(x)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(
    max(0, results_df["accepted_accuracy"].min() * 100 - 2),
    min(100, results_df["accepted_accuracy"].max() * 100 + 3)
)

# Right: Coverage & Rejection Rate vs Threshold
ax2 = axes[1]
ax2.plot(x, results_df["coverage"] * 100,        marker="s", color="#4CAF50", linewidth=2, markersize=7, label="Coverage")
ax2.plot(x, results_df["rejection_rate"] * 100,   marker="^", color="#F44336", linewidth=2, markersize=7, label="Rejection Rate", linestyle="--")
ax2.set_title("Coverage & Rejection Rate vs Threshold")
ax2.set_xlabel("Confidence Threshold")
ax2.set_ylabel("Percentage (%)")
ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
ax2.set_xticks(x)
ax2.legend()
ax2.grid(True, alpha=0.3)
 
plt.tight_layout()
output_png = OUT_DIR / "threshold_curve.png"
plt.savefig(output_png, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved plot -> {output_png}")


print("\n── Summary ────────────────────────────────────────────────────────────")
print(results_df.to_string(index=False))