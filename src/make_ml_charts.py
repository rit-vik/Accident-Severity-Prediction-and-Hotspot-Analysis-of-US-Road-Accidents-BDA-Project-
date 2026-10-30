import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = pd.read_csv("results/ml_experiments.csv")
d = d[(d["features"] == "-") | (d["features"] == "without_month_season")].copy()
def label(r):
    if r["model"] == "baseline_majority":
        return "baseline"
    return ("LR" if r["model"] == "logistic_regression" else "RF") + " (" + r["weights"] + ")"
d["label"] = d.apply(label, axis=1)
metrics = ["accuracy", "weighted_f1", "macro_f1", "balanced_acc"]
x = np.arange(len(d)); w = 0.2
fig, ax = plt.subplots(figsize=(9, 4))
for i, m in enumerate(metrics):
    ax.bar(x + i * w, d[m], w, label=m)
ax.set_xticks(x + 1.5 * w); ax.set_xticklabels(d["label"])
ax.set_title("Model comparison (test set, without month/season)"); ax.legend()
plt.tight_layout(); plt.savefig("charts/ml_comparison.png", dpi=150); plt.close()

cm = pd.read_csv("results/cm_logistic_regression_without_month_season_sqrt.csv", index_col=0)
cmn = cm.div(cm.sum(axis=1), axis=0)
labs = ["Sev1", "Sev2", "Sev3", "Sev4"]
fig, ax = plt.subplots(figsize=(5, 4.2))
ax.imshow(cmn.values, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(4)); ax.set_xticklabels(labs)
ax.set_yticks(range(4)); ax.set_yticklabels(labs)
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title("Logistic Regression (sqrt weights)\nrow-normalised confusion matrix")
for i in range(4):
    for j in range(4):
        v = cmn.values[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="white" if v > 0.5 else "black")
plt.tight_layout(); plt.savefig("charts/ml_confusion_lr_sqrt.png", dpi=150); plt.close()

imp = pd.read_csv("results/rf_importance_without_month_season_inverse.csv").iloc[::-1]
plt.figure(figsize=(7, 4.5))
plt.barh(imp["feature"], imp["importance"])
plt.title("Random Forest feature importance (top 15)")
plt.tight_layout(); plt.savefig("charts/ml_rf_importance.png", dpi=150)
