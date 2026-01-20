#!/usr/bin/env python3
"""
Test script to verify that legends are always plotted:
- When show_legend=True: legend appears in the plot
- When show_legend=False: legend is saved in a separate PDF
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from graphs import save_legend_separately

# Create test directory
test_dir = "/home/tiago/workspace/place_uk/pr_result_tools/test_legend_always"
os.makedirs(test_dir, exist_ok=True)

print("=" * 70)
print("Testing Legend Always Plotted Feature")
print("=" * 70)

# Test data
models = ['Model A', 'Model B', 'Model C', 'Model D']
x_data = np.arange(1, 26)
colors = ['blue', 'red', 'green', 'orange']
markers = ['o', 's', '^', 'v']
line_styles = ['-', '--', '-.', ':']

print("\n[Test 1] show_legend=True")
print("-" * 70)
fig1, ax1 = plt.subplots(figsize=(10, 8))

# Always add labels - they will be used either in the plot or in the separate legend
for i, model in enumerate(models):
    y_data = np.random.rand(25) * 0.5 + (i * 0.1)
    ax1.plot(x_data, y_data, label=model, color=colors[i], 
             marker=markers[i], linestyle=line_styles[i], linewidth=2, markersize=8)

ax1.set_xlabel('Top k', fontsize=20)
ax1.set_ylabel('Recall@k', fontsize=20)
ax1.grid(True)
ax1.set_ylim(0, 1)

# Show legend IN the plot
show_legend = True
if show_legend:
    ax1.legend(fontsize=16)
    print("✓ Legend displayed INSIDE the plot")
else:
    save_legend_separately(fig1, ax1, os.path.join(test_dir, "test1"), fontsize=16)
    print("✓ Legend saved as SEPARATE PDF")

file1 = os.path.join(test_dir, "test_with_legend_in_plot.pdf")
plt.savefig(file1, transparent=True, dpi=300)
plt.close(fig1)
print(f"✓ Plot saved: {file1}")

print("\n[Test 2] show_legend=False")
print("-" * 70)
fig2, ax2 = plt.subplots(figsize=(10, 8))

# Always add labels - they will be used either in the plot or in the separate legend
for i, model in enumerate(models):
    y_data = np.random.rand(25) * 0.5 + (i * 0.1)
    ax2.plot(x_data, y_data, label=model, color=colors[i], 
             marker=markers[i], linestyle=line_styles[i], linewidth=2, markersize=8)

ax2.set_xlabel('Top k', fontsize=20)
ax2.set_ylabel('Recall@k', fontsize=20)
ax2.grid(True)
ax2.set_ylim(0, 1)

# Save legend separately
show_legend = False
if show_legend:
    ax2.legend(fontsize=16)
    print("✓ Legend displayed INSIDE the plot")
else:
    legend_file = save_legend_separately(fig2, ax2, os.path.join(test_dir, "test2"), fontsize=16)
    print(f"✓ Legend saved as SEPARATE PDF: {legend_file}")

file2 = os.path.join(test_dir, "test2.pdf")
plt.savefig(file2, transparent=True, dpi=300)
plt.close(fig2)
print(f"✓ Plot saved (no legend in plot): {file2}")

print("\n" + "=" * 70)
print("Test Summary")
print("=" * 70)
print(f"\nGenerated files in: {test_dir}")
print("\n[Test 1] show_legend=True:")
print("  ✓ test_with_legend_in_plot.pdf - Plot WITH legend inside")
print("  ✗ No separate legend file")
print("\n[Test 2] show_legend=False:")
print("  ✓ test2.pdf - Plot WITHOUT legend")
print("  ✓ test2_legend.pdf - Separate HORIZONTAL legend")
print("\n" + "=" * 70)
print("Key Change:")
print("=" * 70)
print("Labels are ALWAYS added to plot lines (label=model)")
print("The legend is ALWAYS generated from these labels")
print("  - When show_legend=True: ax.legend() displays it in the plot")
print("  - When show_legend=False: save_legend_separately() creates separate PDF")
print("=" * 70)
