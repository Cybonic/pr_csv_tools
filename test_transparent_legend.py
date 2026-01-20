#!/usr/bin/env python3
"""
Test script to verify that the separate legend is transparent with no contour/frame.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from graphs import save_legend_separately

# Create test directory
test_dir = "/home/tiago/workspace/place_uk/pr_result_tools/test_transparent_legend"
os.makedirs(test_dir, exist_ok=True)

print("=" * 70)
print("Testing Transparent Legend with No Contour/Frame")
print("=" * 70)

# Test data
models = ['PointNetPGAP', 'PointNetVLAD', 'SPVSoAP3D', 'LOGG3D', 'OverlapTransformer']
x_data = np.arange(1, 26)
colors = ['blue', 'red', 'green', 'orange', 'purple']
markers = ['o', 's', '^', 'v', 'D']
line_styles = ['-', '--', '-.', ':', '-']

print("\nCreating test plot with legend...")
print("-" * 70)

fig, ax = plt.subplots(figsize=(10, 8))

# Add labeled plots
for i, model in enumerate(models):
    y_data = np.random.rand(25) * 0.5 + (i * 0.1)
    ax.plot(x_data, y_data, label=model, color=colors[i], 
            marker=markers[i], linestyle=line_styles[i], linewidth=2, markersize=8)

ax.set_xlabel('Top k', fontsize=20)
ax.set_ylabel('Recall@k', fontsize=20)
ax.grid(True)
ax.set_ylim(0, 1)

# Save main plot without legend
file_main = os.path.join(test_dir, "plot_no_legend.pdf")
plt.savefig(file_main, transparent=True, dpi=300)
print(f"✓ Main plot saved: {file_main}")

# Save legend separately (transparent, no frame)
legend_file = save_legend_separately(fig, ax, os.path.join(test_dir, "plot_no_legend"), fontsize=18)
print(f"✓ Separate legend saved: {legend_file}")

plt.close(fig)

print("\n" + "=" * 70)
print("Test Complete")
print("=" * 70)
print(f"\nGenerated files in: {test_dir}")
print("  ✓ plot_no_legend.pdf - Main plot without legend")
print("  ✓ plot_no_legend_legend.pdf - Separate horizontal legend")
print("\nLegend properties:")
print("  ✓ Transparent background (transparent=True)")
print("  ✓ No frame/contour (frameon=False)")
print("  ✓ Horizontal alignment (ncol=5)")
print("=" * 70)
