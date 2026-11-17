"""Plot actual prequential metrics; selection-biased same-batch scores are not the headline."""
import argparse
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--run", action="append", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "指导/临时/mpl"))
import matplotlib  # noqa: E402 -- configure cache location before importing the plotting runtime

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402 -- choose the headless backend before pyplot

reports = [json.loads((p / "run.json").read_text(encoding="utf-8")) for p in args.run]
if len(reports) != 3 or {r["strategy"] for r in reports} != {"static", "always", "performance"}:
    raise ValueError("Need the three completed comparison strategies")
args.output.mkdir(parents=True, exist_ok=True)
colors = {"static": "#707884", "always": "#3176b8", "performance": "#118477"}
names = {"static": "Static", "always": "Train every batch", "performance": "Performance trigger"}
fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), gridspec_kw={"width_ratios": [1.4, 1.4, 0.8]})
for report in reports:
    strategy = report["strategy"]
    events = [e for e in report["events"] if e["status"] == "completed"]
    for axis, metric in zip(axes[:2], ["accuracy", "macro_f1"], strict=True):
        axis.plot([e["batch_index"] + 2 for e in events],
                  [e["prequential"][metric] for e in events], marker="o", markersize=4,
                  linewidth=1.8, color=colors[strategy], label=names[strategy])
    axes[2].bar(names[strategy], report["summary"]["updates_charged"], color=colors[strategy])
axes[0].set_title("Accuracy before current-batch decision")
axes[1].set_title("Macro-F1 before current-batch decision")
axes[2].set_title("Training updates")
for axis in axes[:2]:
    axis.set_xlabel("UCI chronological batch")
    axis.set_xticks(range(2, 11))
    axis.set_ylim(0, 1)
    axis.grid(alpha=0.22)
    axis.spines[["top", "right"]].set_visible(False)
axes[0].legend(fontsize=8, loc="lower left")
axes[2].set_ylim(0, 10)
axes[2].tick_params(axis="x", labelrotation=35, labelsize=8)
axes[2].spines[["top", "right"]].set_visible(False)
fig.suptitle("Gas sensor drift: fixed protocol, three real replay strategies", fontsize=13)
fig.text(0.5, 0.02, "Local labeled development replay; all six classes; no hidden-test claim.",
         ha="center", fontsize=9, color="#555555")
fig.tight_layout(rect=(0, 0.06, 1, 0.94))
fig.savefig(args.output / "prequential_comparison.png", dpi=200)
fig.savefig(args.output / "prequential_comparison.svg")
comparison = {"protocol": reports[0]["config"],
              "results": [{"experiment_id": r["experiment_id"], "strategy": r["strategy"],
                           **r["summary"], "elapsed_seconds": r["elapsed_seconds"]}
                          for r in reports],
              "primary_scope": "prediction_before_current_label_based_decision",
              "limitations": ["one ordered UCI stream; not an independent replication",
                              "same-batch selection scores are separately labeled",
                              "performance policy does not reach 85% aggregate accuracy",
                              "mean batch Macro-F1 is lower than always-training control"]}
(args.output / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n",
                                             encoding="utf-8")
print(json.dumps({"output": str(args.output.resolve()),
                  "results": comparison["results"]}, indent=2))
