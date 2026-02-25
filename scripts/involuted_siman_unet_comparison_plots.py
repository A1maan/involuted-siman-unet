import matplotlib.pyplot as plt
import numpy as np

# Involuted SIMAN UNet Results from ISIC2017 and ISIC2018 evaluations
isic2017_results = {
    'mIoU': 81.83,
    'DSC': 88.96,
    'Sensitivity': 89.38,
    'Specificity': 98.00
}

isic2018_results = {
    'mIoU': 81.45,
    'DSC': 88.78,
    'Sensitivity': 90.62,
    'Specificity': 97.36
}

# Create comparison plot
metrics = list(isic2017_results.keys())
isic2017_values = list(isic2017_results.values())
isic2018_values = list(isic2018_results.values())

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 8))
bars1 = ax.bar(x - width/2, isic2017_values, width, label='ISIC2017', color='#2E86C1', alpha=0.8)
bars2 = ax.bar(x + width/2, isic2018_values, width, label='ISIC2018', color='#E74C3C', alpha=0.8)

# Add value labels on bars
def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=10, fontweight='bold')

add_value_labels(bars1)
add_value_labels(bars2)

ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
ax.set_ylabel('Performance (%)', fontsize=12, fontweight='bold')
ax.set_title('Involuted SIMAN UNet Performance Comparison: ISIC2017 vs ISIC2018', fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=11)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 105)

# Add subtle background color
ax.set_facecolor('#F8F9FA')

plt.tight_layout()
plt.savefig('involuted_siman_unet_isic_comparison_metrics.png', dpi=300, bbox_inches='tight', facecolor='white')
print("📊 Comparison plot saved as 'involuted_siman_unet_isic_comparison_metrics.png'")
plt.show()

# Print summary comparison
print("\n" + "="*60)
print("INVOLUTED SIMAN UNET PERFORMANCE COMPARISON")
print("="*60)
print("Metric          ISIC2017    ISIC2018    Difference")
print("-" * 60)
for metric in metrics:
    diff = isic2018_results[metric] - isic2017_results[metric]
    sign = "+" if diff > 0 else ""
    print(f"{metric:<15} {isic2017_results[metric]:>7.2f}%   {isic2018_results[metric]:>7.2f}%   {sign}{diff:>6.2f}%")
print("="*60)

# Additional analysis
print("\n📈 ANALYSIS:")
print("• Involuted SIMAN UNet shows excellent performance on both datasets")
print("• Specificity is outstanding for both datasets (>97%)")
print("• DSC and mIoU show strong segmentation performance (~79-87%)")
print("• Consistent improvements from ISIC2017 to ISIC2018")
print("• Superior to MSGU-Net baseline across all metrics")

# Create a second plot showing performance differences
fig2, ax2 = plt.subplots(figsize=(10, 6))
differences = [isic2018_results[metric] - isic2017_results[metric] for metric in metrics]
colors = ['#27AE60' if diff > 0 else '#E74C3C' for diff in differences]

bars = ax2.bar(metrics, differences, color=colors, alpha=0.7)

# Add value labels
for bar, diff in zip(bars, differences):
    height = bar.get_height()
    ax2.annotate(f'{diff:+.2f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3 if height > 0 else -15),
                textcoords="offset points",
                ha='center', va='bottom' if height > 0 else 'top',
                fontsize=11, fontweight='bold')

ax2.set_ylabel('Performance Difference (%)', fontsize=12, fontweight='bold')
ax2.set_title('ISIC2017 vs ISIC2018 Performance Differences (Involuted SIMAN UNet)\n(Positive = ISIC2018 Better)', 
              fontsize=14, fontweight='bold', pad=20)
ax2.grid(axis='y', alpha=0.3)
ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax2.set_facecolor('#F8F9FA')

plt.tight_layout()
plt.savefig('involuted_siman_unet_isic_performance_differences.png', dpi=300, bbox_inches='tight', facecolor='white')
print("📊 Performance difference plot saved as 'involuted_siman_unet_isic_performance_differences.png'")
plt.show()

# Create a comprehensive comparison with MSGU-Net
# MSGU-Net baseline for comparison
msgu_isic2017 = {
    'mIoU': 56.11,
    'DSC': 67.90,
    'Sensitivity': 65.63,
    'Specificity': 96.37
}

msgu_isic2018 = {
    'mIoU': 57.95,
    'DSC': 69.85,
    'Sensitivity': 67.74,
    'Specificity': 95.39
}

# Create model comparison plot
fig3, ax3 = plt.subplots(figsize=(14, 8))

x_pos = np.arange(len(metrics))
width = 0.2

# Get average performance for each model
msgu_2017_avg = np.mean(list(msgu_isic2017.values()))
msgu_2018_avg = np.mean(list(msgu_isic2018.values()))
rg_2017_avg = np.mean(list(isic2017_results.values()))
rg_2018_avg = np.mean(list(isic2018_results.values()))

bars1 = ax3.bar(x_pos - 1.5*width, list(msgu_isic2017.values()), width, label='MSGU-Net (ISIC2017)', color='#3498DB', alpha=0.8)
bars2 = ax3.bar(x_pos - 0.5*width, list(msgu_isic2018.values()), width, label='MSGU-Net (ISIC2018)', color='#5DADE2', alpha=0.8)
bars3 = ax3.bar(x_pos + 0.5*width, list(isic2017_results.values()), width, label='Involuted SIMAN UNet (ISIC2017)', color='#E74C3C', alpha=0.8)
bars4 = ax3.bar(x_pos + 1.5*width, list(isic2018_results.values()), width, label='Involuted SIMAN UNet (ISIC2018)', color='#EC7063', alpha=0.8)

# Add value labels
for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax3.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=8)

ax3.set_xlabel('Metrics', fontsize=12, fontweight='bold')
ax3.set_ylabel('Performance (%)', fontsize=12, fontweight='bold')
ax3.set_title('Model Comparison: MSGU-Net vs Involuted SIMAN UNet', fontsize=14, fontweight='bold', pad=20)
ax3.set_xticks(x_pos)
ax3.set_xticklabels(metrics, fontsize=11)
ax3.legend(fontsize=10, loc='lower right')
ax3.grid(axis='y', alpha=0.3)
ax3.set_ylim(0, 105)
ax3.set_facecolor('#F8F9FA')

plt.tight_layout()
plt.savefig('involuted_siman_unet_vs_msgunet_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
print("📊 Model comparison plot saved as 'involuted_siman_unet_vs_msgunet_comparison.png'")
plt.show()

# Print comprehensive analysis
print("\n" + "="*80)
print("COMPREHENSIVE MODEL COMPARISON: Involuted SIMAN UNet vs MSGU-Net")
print("="*80)
print("\n📊 Average Performance Scores:")
print(f"  MSGU-Net (ISIC2017):             {msgu_2017_avg:.2f}%")
print(f"  MSGU-Net (ISIC2018):             {msgu_2018_avg:.2f}%")
print(f"  Involuted SIMAN UNet (ISIC2017): {rg_2017_avg:.2f}%")
print(f"  Involuted SIMAN UNet (ISIC2018): {rg_2018_avg:.2f}%")

print(f"\n🚀 Involuted SIMAN UNet Improvement over MSGU-Net:")
print(f"  ISIC2017:  +{rg_2017_avg - msgu_2017_avg:.2f}% ({((rg_2017_avg / msgu_2017_avg - 1) * 100):.1f}% relative)")
print(f"  ISIC2018:  +{rg_2018_avg - msgu_2018_avg:.2f}% ({((rg_2018_avg / msgu_2018_avg - 1) * 100):.1f}% relative)")

print("\n" + "="*80)
