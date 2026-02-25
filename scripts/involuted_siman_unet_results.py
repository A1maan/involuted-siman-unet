import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Results for Involuted SIMAN UNet on ISIC2017 and ISIC2018
results = {
    "ISIC2017": {
        "model": "Involuted SIMAN UNet",
        "dataset": "ISIC2017",
        "mIoU": 81.83,
        "DSC": 88.96,
        "Sensitivity": 89.38,
        "Specificity": 98.00,
        "evaluation_date": datetime.now().isoformat()
    },
    "ISIC2018": {
        "model": "Involuted SIMAN UNet",
        "dataset": "ISIC2018",
        "mIoU": 81.45,
        "DSC": 88.78,
        "Sensitivity": 90.62,
        "Specificity": 97.36,
        "evaluation_date": datetime.now().isoformat()
    }
}

# Save results to JSON
results_json_path = "involuted_siman_unet_results.json"
with open(results_json_path, 'w') as f:
    json.dump(results, f, indent=4)
print(f"✅ Results saved to {results_json_path}")

# Create a detailed text report
report_path = "involuted_siman_unet_results.txt"
with open(report_path, 'w') as f:
    f.write("=" * 60 + "\n")
    f.write("Involuted SIMAN UNet Evaluation Results\n")
    f.write("=" * 60 + "\n\n")
    
    for dataset_name, data in results.items():
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

# Create comprehensive visualizations
fig = plt.figure(figsize=(16, 12))

# 1. Bar chart comparing all metrics across datasets
ax1 = plt.subplot(2, 2, 1)
datasets = ['ISIC2017', 'ISIC2018']
metrics_names = ['mIoU', 'DSC', 'Sensitivity', 'Specificity']
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']

x = np.arange(len(datasets))
width = 0.2

for i, metric in enumerate(metrics_names):
    values = [results[ds][metric] for ds in datasets]
    ax1.bar(x + i*width, values, width, label=metric, color=colors[i])

ax1.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
ax1.set_title('Involuted SIMAN UNet: All Metrics Comparison', fontsize=14, fontweight='bold')
ax1.set_xticks(x + 1.5*width)
ax1.set_xticklabels(datasets, fontsize=11)
ax1.legend(fontsize=10)
ax1.set_ylim([70, 102])
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, metric in enumerate(metrics_names):
    values = [results[ds][metric] for ds in datasets]
    for j, v in enumerate(values):
        ax1.text(j + i*width, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)

# 2. ISIC2017 metrics radar-like bar chart
ax2 = plt.subplot(2, 2, 2)
isic2017_metrics = [results['ISIC2017'][m] for m in metrics_names]
bars2 = ax2.barh(metrics_names, isic2017_metrics, color=colors)
ax2.set_xlabel('Score (%)', fontsize=12, fontweight='bold')
ax2.set_title('Involuted SIMAN UNet on ISIC2017', fontsize=14, fontweight='bold')
ax2.set_xlim([70, 102])
ax2.grid(axis='x', alpha=0.3)

# Add value labels
for i, (bar, val) in enumerate(zip(bars2, isic2017_metrics)):
    ax2.text(val + 0.5, i, f'{val:.2f}%', va='center', fontsize=10, fontweight='bold')

# 3. ISIC2018 metrics radar-like bar chart
ax3 = plt.subplot(2, 2, 3)
isic2018_metrics = [results['ISIC2018'][m] for m in metrics_names]
bars3 = ax3.barh(metrics_names, isic2018_metrics, color=colors)
ax3.set_xlabel('Score (%)', fontsize=12, fontweight='bold')
ax3.set_title('Involuted SIMAN UNet on ISIC2018', fontsize=14, fontweight='bold')
ax3.set_xlim([70, 102])
ax3.grid(axis='x', alpha=0.3)

# Add value labels
for i, (bar, val) in enumerate(zip(bars3, isic2018_metrics)):
    ax3.text(val + 0.5, i, f'{val:.2f}%', va='center', fontsize=10, fontweight='bold')

# 4. Side-by-side comparison with better visualization
ax4 = plt.subplot(2, 2, 4)
x_pos = np.arange(len(metrics_names))
width = 0.35

bars_2017 = ax4.bar(x_pos - width/2, isic2017_metrics, width, label='ISIC2017', 
                     color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
bars_2018 = ax4.bar(x_pos + width/2, isic2018_metrics, width, label='ISIC2018', 
                     color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)

ax4.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
ax4.set_title('ISIC2017 vs ISIC2018 Performance', fontsize=14, fontweight='bold')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(metrics_names, fontsize=11)
ax4.legend(fontsize=11, loc='lower right')
ax4.set_ylim([70, 102])
ax4.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars_2017, bars_2018]:
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

plt.suptitle('Involuted SIMAN UNet Evaluation Results Summary', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()

# Save the comprehensive plot
plot_path = "involuted_siman_unet_results_visualization.png"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"✅ Comprehensive visualization saved to {plot_path}")
plt.show()

# Create an additional detailed metrics table visualization
fig2, ax = plt.subplots(figsize=(12, 6))
ax.axis('tight')
ax.axis('off')

# Prepare table data
table_data = []
table_data.append(['Metric', 'ISIC2017', 'ISIC2018', 'Difference'])

for metric in metrics_names:
    val_2017 = results['ISIC2017'][metric]
    val_2018 = results['ISIC2018'][metric]
    diff = val_2018 - val_2017
    diff_str = f"+{diff:.2f}%" if diff >= 0 else f"{diff:.2f}%"
    table_data.append([metric, f"{val_2017:.2f}%", f"{val_2018:.2f}%", diff_str])

# Add average row
avg_2017 = np.mean(isic2017_metrics)
avg_2018 = np.mean(isic2018_metrics)
avg_diff = avg_2018 - avg_2017
avg_diff_str = f"+{avg_diff:.2f}%" if avg_diff >= 0 else f"{avg_diff:.2f}%"
table_data.append(['Average', f"{avg_2017:.2f}%", f"{avg_2018:.2f}%", avg_diff_str])

table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                colWidths=[0.25, 0.25, 0.25, 0.25])

table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 2.5)

# Style header row
for i in range(4):
    table[(0, i)].set_facecolor('#34495e')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Style data rows with alternating colors
for i in range(1, len(table_data) - 1):
    color = '#ecf0f1' if i % 2 == 0 else '#ffffff'
    for j in range(4):
        table[(i, j)].set_facecolor(color)
        table[(i, j)].set_text_props(weight='bold' if j == 0 else 'normal')

# Style average row
for j in range(4):
    table[(len(table_data) - 1, j)].set_facecolor('#f39c12')
    table[(len(table_data) - 1, j)].set_text_props(weight='bold', color='white')

plt.title('Involuted SIMAN UNet Performance Metrics - Detailed Comparison', 
         fontsize=14, fontweight='bold', pad=20)
plt.savefig('involuted_siman_unet_results_table.png', dpi=300, bbox_inches='tight')
print(f"✅ Results table saved to involuted_siman_unet_results_table.png")
plt.show()

# Print summary to console
print("\n" + "=" * 60)
print("INVOLUTED SIMAN UNET EVALUATION RESULTS SUMMARY")
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
    diff = results['ISIC2018'][metric] - results['ISIC2017'][metric]
    arrow = "↑" if diff > 0 else "↓" if diff < 0 else "→"
    print(f"  • {metric:12s}: {arrow} {diff:+.2f}%")

print("\n" + "=" * 60)
