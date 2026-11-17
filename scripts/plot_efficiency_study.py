"""Plot every registered training-efficacy result, including rejected candidates."""
import argparse
import csv
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--study", type=Path, required=True)
parser.add_argument("--followup", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "指导/临时/mpl"))
import matplotlib  # noqa: E402 -- configure cache before importing the plotting runtime

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402 -- configure the headless backend first
import numpy as np  # noqa: E402

first = json.loads(args.study.read_text(encoding="utf-8"))
second = json.loads(args.followup.read_text(encoding="utf-8"))
if first["status"] != "completed" or second["status"] != "completed":
    raise ValueError("Both studies must be complete")
rows = first["A"] + second["A"]
variants = ["control", "weighted", "scheduled", "memory", "unweighted_scheduled",
            "unweighted_memory", "soft_scheduled"]
labels = ["Control", "Weight 1.5", "Weight 1.5\n+ schedule", "Weight 1.5\n+ schedule + cap",
          "Schedule only", "Schedule\n+ cap", "Weight 0.5\n+ schedule"]
fig, axes = plt.subplots(2, 1, figsize=(12, 7.3), sharex=True)
for strategy, shift, color in [("always", -0.18, "#3278b8"),
                                ("performance", 0.18, "#118477")]:
    selected = [next(r for r in rows if r["variant"] == v and r["strategy"] == strategy)
                for v in variants]
    for axis, key in zip(axes, ["prequential_sample_weighted_accuracy",
                                "prequential_mean_batch_macro_f1"], strict=True):
        bars = axis.bar(np.arange(len(variants)) + shift, [r["summary"][key] for r in selected],
                        width=0.34, color=color, label=strategy)
        for bar in bars:
            axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                      f"{bar.get_height():.4f}", ha="center", fontsize=7)
axes[0].set_ylabel("Sample-weighted accuracy")
axes[1].set_ylabel("Mean batch Macro-F1")
axes[1].set_xticks(range(len(variants)), labels)
for axis in axes:
    axis.set_ylim(0, 1)
    axis.grid(axis="y", alpha=0.2)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
axes[0].axhline(0.85, color="#777777", linestyle="--", linewidth=1, label="85% target")
axes[0].legend(loc="upper center", ncol=3, frameon=False, fontsize=9)
fig.suptitle("A training-efficacy ablations: predictions before current-batch decisions")
fig.text(0.5, 0.015, "Repeated public development replay. Follow-up variants are exploratory."
         " Higher training budgets are not equal-cost improvements.", ha="center", fontsize=8)
fig.tight_layout(rect=(0, 0.045, 1, 0.96))
args.output.mkdir(parents=True, exist_ok=True)
for suffix in ["png", "svg"]:
    fig.savefig(args.output / f"A_all_ablations.{suffix}", dpi=180)

brows = first["B"] + second["B"]
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4.9))
blabels = [f"h={r['hidden']}, e={r['epochs']}\nlr={r.get('lr', 0.001):g}" for r in brows]
for axis, metric, baseline in zip(axes2, ["rmse", "mae"], [681.1598326043112,
                                                          563.1058414514887], strict=True):
    axis.bar(range(len(brows)), [r["after"][metric] for r in brows],
             color=["#118477" if r["accepted"] else "#ae7373" for r in brows])
    axis.axhline(baseline, color="#555555", linestyle="--", label="Original checkpoint")
    axis.set_xticks(range(len(brows)), blabels, rotation=35, ha="right", fontsize=8)
    axis.set_ylabel(f"Candidate {metric.upper()} (minutes)")
    axis.spines[["top", "right"]].set_visible(False)
axes2[0].legend(fontsize=8)
fig2.suptitle("B capacity and learning-rate study: green accepted, red rejected")
fig2.tight_layout(rect=(0, 0.04, 1, 0.94))
for suffix in ["png", "svg"]:
    fig2.savefig(args.output / f"B_all_candidates.{suffix}", dpi=180)
with (args.output / "A_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["variant", "strategy", "experiment_id", "accuracy", "mean_batch_macro_f1",
                     "updates", "training_seconds"])
    for row in rows:
        summary = row["summary"]
        writer.writerow([row["variant"], row["strategy"], row["experiment_id"],
                         summary["prequential_sample_weighted_accuracy"],
                         summary["prequential_mean_batch_macro_f1"], summary["updates_charged"],
                         summary["training_seconds_charged"]])
print(json.dumps({"output": str(args.output.resolve()), "A_runs": len(rows), "B_runs": len(brows)}))
