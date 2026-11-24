"""Plot final-test Qini curves from the saved CSVs (no models are re-run)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "reports" / "final_test"
OUT = ROOT / "reports" / "figures" / "qini_final_test.png"


def main() -> None:
    lgbm = pd.read_csv(FINAL / "lightgbm_test_qini_curve.csv")
    logit = pd.read_csv(FINAL / "logistic_test_qini_curve.csv")
    metrics = pd.read_csv(FINAL / "final_test_metrics.csv").set_index("model")

    q_lgbm = metrics.loc["LightGBM T-Learner (final)", "qini_coefficient"]
    q_logit = metrics.loc["Logistic T-Learner (baseline)", "qini_coefficient"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(lgbm["target_percentage"], lgbm["qini_gain_pp"],
            label=f"LightGBM T-Learner (Qini {q_lgbm:.3f})", linewidth=2)
    ax.plot(logit["target_percentage"], logit["qini_gain_pp"],
            label=f"Logistic T-Learner (Qini {q_logit:.3f})", linewidth=2)
    ax.plot(lgbm["target_percentage"], lgbm["random_baseline_pp"],
            "k--", label="Random targeting", linewidth=1.5)

    ax.set_xlabel("% of customers targeted (highest predicted uplift first)")
    ax.set_ylabel("Cumulative incremental conversions (pp)")
    ax.set_title("Held-out test split: Men's Email vs No Email")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print(f"Saved {OUT}")
    print("End of curve (should be close to the observed ATE, 0.70 pp):",
          round(lgbm['qini_gain_pp'].iloc[-1], 3),
          round(logit['qini_gain_pp'].iloc[-1], 3))


if __name__ == "__main__":
    main()