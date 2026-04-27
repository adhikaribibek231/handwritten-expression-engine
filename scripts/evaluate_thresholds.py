"""Evaluate simple confidence thresholds for the robust CNN.

The input is the per-sample csv produced by `scripts/analyze_failures.py`.
For each threshold we ask a practical question:
if we only trust predictions above this confidence, how much accuracy do we
gain, how many samples do we still cover, and how many do we reject?
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT / "artifacts" / "phase4"
METRICS_DIR = PROJECT_ROOT / "metrics"
CSV_PATH = METRICS_DIR / "robust_failure_analysis.csv"
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98]


# step 1 - load the per-sample failure table
def load_failure_table(path: Path) -> pd.DataFrame:
    """Load the per-sample prediction table produced by failure analysis."""

    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows from {path}")
    return df


# step 2 - score a list of confidence thresholds
def evaluate_threshold_table(df: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    """Summarize coverage, accepted accuracy, and rejection rate for each threshold."""

    total = len(df)
    results: list[dict[str, float | int]] = []

    for threshold in thresholds:
        accepted_mask = df["confidence"] >= threshold
        accepted = df[accepted_mask]
        rejected = df[~accepted_mask]

        accepted_count = len(accepted)
        rejected_count = len(rejected)
        correct_accepted = int(accepted["is_correct"].sum()) if accepted_count > 0 else 0

        coverage = accepted_count / total
        accepted_accuracy = correct_accepted / accepted_count if accepted_count > 0 else 0.0
        rejection_rate = rejected_count / total

        results.append(
            {
                "threshold": threshold,
                "coverage": round(coverage, 4),
                "accepted_accuracy": round(accepted_accuracy, 4),
                "rejection_rate": round(rejection_rate, 4),
                "accepted_count": accepted_count,
                "rejected_count": rejected_count,
            }
        )

        print(
            f"threshold={threshold:.2f} | "
            f"coverage={coverage * 100:.1f}% | "
            f"accepted_acc={accepted_accuracy * 100:.2f}% | "
            f"rejection_rate={rejection_rate * 100:.1f}%"
        )

    return pd.DataFrame(results)


# step 3 - draw the threshold trade-off plots
def plot_threshold_curves(results_df: pd.DataFrame, out_path: Path) -> None:
    """Plot the tradeoff between stricter confidence and retained coverage."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Threshold Evaluation - Uncertainty Rejection", fontsize=14, fontweight="bold")
    x = results_df["threshold"]

    # Left: if we keep only confident predictions, how clean does the kept set become?
    ax1 = axes[0]
    ax1.plot(
        x,
        results_df["accepted_accuracy"] * 100,
        marker="o",
        color="#2196F3",
        linewidth=2,
        markersize=7,
    )
    for _, row in results_df.iterrows():
        ax1.annotate(
            f'{row["accepted_accuracy"] * 100:.1f}%',
            (row["threshold"], row["accepted_accuracy"] * 100),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
            color="#1565C0",
        )
    ax1.set_title("Accepted Accuracy vs Threshold")
    ax1.set_xlabel("Confidence Threshold")
    ax1.set_ylabel("Accepted Accuracy (%)")
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax1.set_xticks(x)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(
        max(0, results_df["accepted_accuracy"].min() * 100 - 2),
        min(100, results_df["accepted_accuracy"].max() * 100 + 3),
    )

    # Right: the same threshold also trades away coverage, so show both views together.
    ax2 = axes[1]
    ax2.plot(
        x,
        results_df["coverage"] * 100,
        marker="s",
        color="#4CAF50",
        linewidth=2,
        markersize=7,
        label="Coverage",
    )
    ax2.plot(
        x,
        results_df["rejection_rate"] * 100,
        marker="^",
        color="#F44336",
        linewidth=2,
        markersize=7,
        label="Rejection Rate",
        linestyle="--",
    )
    ax2.set_title("Coverage & Rejection Rate vs Threshold")
    ax2.set_xlabel("Confidence Threshold")
    ax2.set_ylabel("Percentage (%)")
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax2.set_xticks(x)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot -> {out_path}")


# step 4 - run the threshold evaluation end to end
def main() -> None:
    """Load failure data, score several thresholds, and save the summary artifacts."""

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # step 1 - read the csv made by failure analysis
    df = load_failure_table(CSV_PATH)

    # step 2 - compute the threshold summary table
    results_df = evaluate_threshold_table(df, THRESHOLDS)

    # step 3 - save the numeric report
    output_csv = METRICS_DIR / "threshold_evaluation.csv"
    results_df.to_csv(output_csv, index=False)
    print(f"Saved metrics to -> {output_csv}")

    # step 4 - save the plot and print a text summary
    plot_threshold_curves(results_df, OUT_DIR / "threshold_curve.png")

    print("\n-- Summary ------------------------------------------------------------")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
