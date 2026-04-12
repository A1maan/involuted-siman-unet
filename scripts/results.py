import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ------------------------------------------------------------------
# Registry: add a new entry here whenever a model is evaluated
# ------------------------------------------------------------------
RESULTS_REGISTRY = {
    "involuted": {
        "name":   "Involuted SIMAN UNet",
        "folder": "involuted",
        "prefix": "involuted_siman_unet",
        "data": {
            "ISIC2017": {
                "model": "Involuted SIMAN UNet",
                "dataset": "ISIC2017",
                "mIoU": 81.83,
                "DSC": 88.96,
                "Sensitivity": 89.38,
                "Specificity": 98.00,
            },
            "ISIC2018": {
                "model": "Involuted SIMAN UNet",
                "dataset": "ISIC2018",
                "mIoU": 81.45,
                "DSC": 88.78,
                "Sensitivity": 90.62,
                "Specificity": 97.36,
            },
        },
    },
    "convnext": {
        "name":   "ConvNeXt SIMAN UNet",
        "folder": "convnext",
        "prefix": "convnext_siman_unet",
        "data": {
            "ISIC2017": {
                "model": "ConvNeXt SIMAN UNet",
                "dataset": "ISIC2017",
                "mIoU": 81.46,
                "DSC": 88.59,
                "Sensitivity": 90.39,
                "Specificity": 97.99,
            },
            "ISIC2018": {
                "model": "ConvNeXt SIMAN UNet",
                "dataset": "ISIC2018",
                "mIoU": 80.75,
                "DSC": 88.40,
                "Sensitivity": 91.65,
                "Specificity": 97.00,
            },
        },
    },
    "deformcnext": {
        "name":   "Deform ConvNeXt SIMAN UNet",
        "folder": "deformcnext",
        "prefix": "deformcnext_siman_unet",
        "data": None,   # populate after training/evaluation
    },
}

# ------------------------------------------------------------------
# Parse model argument
# ------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Generate results report and plots for a trained model.")
parser.add_argument(
    "--model",
    choices=list(RESULTS_REGISTRY.keys()),
    default="convnext",
    help="Model to report results for (default: convnext)",
)
args = parser.parse_args()

entry       = RESULTS_REGISTRY[args.model]
MODEL_NAME  = entry["name"]
results     = entry["data"]
prefix      = entry["prefix"]
plots_dir   = os.path.join(_project_root, "plots", entry["folder"])

if results is None:
    raise ValueError(
        f"No results found for '{args.model}' in RESULTS_REGISTRY. "
        "Run evaluation first and populate the 'data' field."
    )

# Stamp evaluation date on load
for ds_data in results.values():
    ds_data["evaluation_date"] = datetime.now().isoformat()

os.makedirs(plots_dir, exist_ok=True)
print(f"Model : {MODEL_NAME}")
print(f"Output: {plots_dir}")

# ------------------------------------------------------------------
# Save JSON
# ------------------------------------------------------------------
results_json_path = os.path.join(plots_dir, f"{prefix}_results.json")
with open(results_json_path, "w") as f:
    json.dump(results, f, indent=4)
print(f"✅ Results saved to {results_json_path}")

# ------------------------------------------------------------------
# Save text report
# ------------------------------------------------------------------
report_path = os.path.join(plots_dir, f"{prefix}_results.txt")
with open(report_path, "w") as f:
    f.write("=" * 60 + "\n")
    f.write(f"{MODEL_NAME} Evaluation Results\n")
    f.write("=" * 60 + "\n\n")
    for data in results.values():
        f.write(f"Dataset: {data['dataset']}\n")
        f.write(f"Model: {data['model']}\n")
        f.write(f"Evaluation Date: {data['evaluation_date']}\n")
        f.write("-" * 60 + "\n")
        f.write(f"mIoU (Mean Intersection over Union): {data['mIoU']:.2f}%\n")
        f.write(f"DSC (Dice Similarity Coefficient):   {data['DSC']:.2f}%\n")
        f.write(f"Sensitivity (Recall):                {data['Sensitivity']:.2f}%\n")
        f.write(f"Specificity:                         {data['Specificity']:.2f}%\n")
        f.write("\n\n")
print(f"✅ Report saved to {report_path}")

# ------------------------------------------------------------------
# Visualizations
# ------------------------------------------------------------------
datasets      = ["ISIC2017", "ISIC2018"]
metrics_names = ["mIoU", "DSC", "Sensitivity", "Specificity"]
colors        = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A"]

isic2017_metrics = [results["ISIC2017"][m] for m in metrics_names]
isic2018_metrics = [results["ISIC2018"][m] for m in metrics_names]

fig = plt.figure(figsize=(16, 12))

# 1. All metrics across datasets
ax1 = plt.subplot(2, 2, 1)
x = np.arange(len(datasets))
width = 0.2
for i, metric in enumerate(metrics_names):
    values = [results[ds][metric] for ds in datasets]
    ax1.bar(x + i * width, values, width, label=metric, color=colors[i])
    for j, v in enumerate(values):
        ax1.text(j + i * width, v + 1, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
ax1.set_ylabel("Score (%)", fontsize=12, fontweight="bold")
ax1.set_title(f"{MODEL_NAME}: All Metrics Comparison", fontsize=14, fontweight="bold")
ax1.set_xticks(x + 1.5 * width)
ax1.set_xticklabels(datasets, fontsize=11)
ax1.legend(fontsize=10)
ax1.set_ylim([70, 102])
ax1.grid(axis="y", alpha=0.3)

# 2. ISIC2017 horizontal bar
ax2 = plt.subplot(2, 2, 2)
bars2 = ax2.barh(metrics_names, isic2017_metrics, color=colors)
for i, (bar, val) in enumerate(zip(bars2, isic2017_metrics)):
    ax2.text(val + 0.5, i, f"{val:.2f}%", va="center", fontsize=10, fontweight="bold")
ax2.set_xlabel("Score (%)", fontsize=12, fontweight="bold")
ax2.set_title(f"{MODEL_NAME} on ISIC2017", fontsize=14, fontweight="bold")
ax2.set_xlim([70, 102])
ax2.grid(axis="x", alpha=0.3)

# 3. ISIC2018 horizontal bar
ax3 = plt.subplot(2, 2, 3)
bars3 = ax3.barh(metrics_names, isic2018_metrics, color=colors)
for i, (bar, val) in enumerate(zip(bars3, isic2018_metrics)):
    ax3.text(val + 0.5, i, f"{val:.2f}%", va="center", fontsize=10, fontweight="bold")
ax3.set_xlabel("Score (%)", fontsize=12, fontweight="bold")
ax3.set_title(f"{MODEL_NAME} on ISIC2018", fontsize=14, fontweight="bold")
ax3.set_xlim([70, 102])
ax3.grid(axis="x", alpha=0.3)

# 4. ISIC2017 vs ISIC2018 side-by-side
ax4 = plt.subplot(2, 2, 4)
x_pos = np.arange(len(metrics_names))
w = 0.35
bars_2017 = ax4.bar(x_pos - w / 2, isic2017_metrics, w, label="ISIC2017",
                    color="#3498db", alpha=0.8, edgecolor="black", linewidth=1.5)
bars_2018 = ax4.bar(x_pos + w / 2, isic2018_metrics, w, label="ISIC2018",
                    color="#e74c3c", alpha=0.8, edgecolor="black", linewidth=1.5)
for bars in [bars_2017, bars_2018]:
    for bar in bars:
        h = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2., h + 0.5,
                 f"{h:.1f}%", ha="center", va="bottom", fontsize=9)
ax4.set_ylabel("Score (%)", fontsize=12, fontweight="bold")
ax4.set_title("ISIC2017 vs ISIC2018 Performance", fontsize=14, fontweight="bold")
ax4.set_xticks(x_pos)
ax4.set_xticklabels(metrics_names, fontsize=11)
ax4.legend(fontsize=11, loc="lower right")
ax4.set_ylim([70, 102])
ax4.grid(axis="y", alpha=0.3)

plt.suptitle(f"{MODEL_NAME} Evaluation Results Summary", fontsize=16, fontweight="bold", y=0.995)
plt.tight_layout()
viz_path = os.path.join(plots_dir, f"{prefix}_results_visualization.png")
plt.savefig(viz_path, dpi=300, bbox_inches="tight")
print(f"✅ Visualization saved to {viz_path}")
plt.show()

# ------------------------------------------------------------------
# Metrics table figure
# ------------------------------------------------------------------
fig2, ax = plt.subplots(figsize=(12, 6))
ax.axis("tight")
ax.axis("off")

table_data = [["Metric", "ISIC2017", "ISIC2018", "Difference"]]
for metric in metrics_names:
    v17   = results["ISIC2017"][metric]
    v18   = results["ISIC2018"][metric]
    diff  = v18 - v17
    d_str = f"+{diff:.2f}%" if diff >= 0 else f"{diff:.2f}%"
    table_data.append([metric, f"{v17:.2f}%", f"{v18:.2f}%", d_str])

avg17    = np.mean(isic2017_metrics)
avg18    = np.mean(isic2018_metrics)
avg_diff = avg18 - avg17
table_data.append(["Average", f"{avg17:.2f}%", f"{avg18:.2f}%",
                   f"+{avg_diff:.2f}%" if avg_diff >= 0 else f"{avg_diff:.2f}%"])

table = ax.table(cellText=table_data, cellLoc="center", loc="center",
                 colWidths=[0.25, 0.25, 0.25, 0.25])
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 2.5)

for i in range(4):
    table[(0, i)].set_facecolor("#34495e")
    table[(0, i)].set_text_props(weight="bold", color="white")
for i in range(1, len(table_data) - 1):
    clr = "#ecf0f1" if i % 2 == 0 else "#ffffff"
    for j in range(4):
        table[(i, j)].set_facecolor(clr)
        table[(i, j)].set_text_props(weight="bold" if j == 0 else "normal")
for j in range(4):
    table[(len(table_data) - 1, j)].set_facecolor("#f39c12")
    table[(len(table_data) - 1, j)].set_text_props(weight="bold", color="white")

plt.title(f"{MODEL_NAME} Performance Metrics - Detailed Comparison",
          fontsize=14, fontweight="bold", pad=20)
table_path = os.path.join(plots_dir, f"{prefix}_results_table.png")
plt.savefig(table_path, dpi=300, bbox_inches="tight")
print(f"✅ Results table saved to {table_path}")
plt.show()

# ------------------------------------------------------------------
# Console summary
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print(f"{MODEL_NAME.upper()} EVALUATION RESULTS SUMMARY")
print("=" * 60)
print("\n📊 ISIC2017 Results:")
print(f"  • mIoU:        {results['ISIC2017']['mIoU']:.2f}%")
print(f"  • DSC:         {results['ISIC2017']['DSC']:.2f}%")
print(f"  • Sensitivity: {results['ISIC2017']['Sensitivity']:.2f}%")
print(f"  • Specificity: {results['ISIC2017']['Specificity']:.2f}%")
print(f"  • Average:     {np.mean(isic2017_metrics):.2f}%")
print("\n📊 ISIC2018 Results:")
print(f"  • mIoU:        {results['ISIC2018']['mIoU']:.2f}%")
print(f"  • DSC:         {results['ISIC2018']['DSC']:.2f}%")
print(f"  • Sensitivity: {results['ISIC2018']['Sensitivity']:.2f}%")
print(f"  • Specificity: {results['ISIC2018']['Specificity']:.2f}%")
print(f"  • Average:     {np.mean(isic2018_metrics):.2f}%")
print("\n📈 Improvements (ISIC2017 → ISIC2018):")
for metric in metrics_names:
    diff  = results["ISIC2018"][metric] - results["ISIC2017"][metric]
    arrow = "↑" if diff > 0 else "↓" if diff < 0 else "→"
    print(f"  • {metric:12s}: {arrow} {diff:+.2f}%")
print("\n" + "=" * 60)
