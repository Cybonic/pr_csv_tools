#!/usr/bin/env python3
"""
Test the separate legend saving functionality with horizontal alignment.

This script demonstrates:
1. When show_legend=False, a separate horizontal legend is generated
2. When show_legend=True, legend appears in the plot (no separate file)
"""

import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Add the parent directory to the path
sys.path.insert(0, '/home/tiago/workspace/place_uk/pr_result_tools')

from graphs import save_legend_separately

# Create sample data
x = np.linspace(0, 25, 25)
y1 = 0.8 - 0.3 * np.exp(-x/5)
y2 = 0.7 - 0.2 * np.exp(-x/5)
y3 = 0.65 - 0.25 * np.exp(-x/5)
y4 = 0.6 - 0.2 * np.exp(-x/5)
y5 = 0.5 - 0.15 * np.exp(-x/5)

output_dir = '/home/tiago/workspace/place_uk/pr_result_tools/test_output'
os.makedirs(output_dir, exist_ok=True)

print("="*70)
print("Testing Separate Legend Feature")
print("="*70)

# Test 1: show_legend=False (generates separate horizontal legend)
print("\n[Test 1] show_legend=False")
print("-" * 70)

fig, ax = plt.subplots(figsize=(10, 8))

# Plot multiple lines with labels
ax.plot(x, y1, '-', linewidth=3, label='PointNetPGAP')
ax.plot(x, y2, '--', linewidth=3, label='PointNetVLAD')
ax.plot(x, y3, '-.', linewidth=3, label='SPVSoAP3D')
ax.plot(x, y4, ':', linewidth=3, label='LOGG3D')
ax.plot(x, y5, '-', linewidth=3, label='OverlapTransformer')

ax.set_xlabel('Top k', fontsize=20)
ax.set_ylabel('Recall@k', fontsize=20)
ax.grid(True)
ax.set_ylim(0, 1)
ax.tick_params(axis='both', labelsize=18)

# NO legend in main plot
show_legend = False

file_path = os.path.join(output_dir, 'test_no_legend')

if show_legend:
    ax.legend(fontsize=18)
else:
    # Save separate horizontal legend
    save_legend_separately(fig, ax, file_path, fontsize=18)

plt.savefig(f'{file_path}.pdf', transparent=True, bbox_inches='tight')
print(f"✓ Main plot saved: {file_path}.pdf (NO legend in plot)")
print(f"✓ Separate legend saved: {file_path}_legend.pdf (HORIZONTAL)")

plt.close()

# Test 2: show_legend=True (legend in plot, no separate file)
print("\n[Test 2] show_legend=True")
print("-" * 70)

fig, ax = plt.subplots(figsize=(10, 8))

# Plot multiple lines with labels
ax.plot(x, y1, '-', linewidth=3, label='PointNetPGAP')
ax.plot(x, y2, '--', linewidth=3, label='PointNetVLAD')
ax.plot(x, y3, '-.', linewidth=3, label='SPVSoAP3D')
ax.plot(x, y4, ':', linewidth=3, label='LOGG3D')
ax.plot(x, y5, '-', linewidth=3, label='OverlapTransformer')

ax.set_xlabel('Top k', fontsize=20)
ax.set_ylabel('Recall@k', fontsize=20)
ax.grid(True)
ax.set_ylim(0, 1)
ax.tick_params(axis='both', labelsize=18)

# Legend in main plot
show_legend = True

file_path = os.path.join(output_dir, 'test_with_legend')

if show_legend:
    ax.legend(fontsize=18)
else:
    # This won't execute in this test
    save_legend_separately(fig, ax, file_path, fontsize=18)

plt.savefig(f'{file_path}.pdf', transparent=True, bbox_inches='tight')
print(f"✓ Main plot saved: {file_path}.pdf (legend INSIDE plot)")
print(f"✗ No separate legend file generated (show_legend=True)")

plt.close()

print("\n" + "="*70)
print("Test Summary")
print("="*70)
print(f"\nGenerated files in: {output_dir}")
print("\n[Test 1] show_legend=False:")
print("  ✓ test_no_legend.pdf - Plot WITHOUT legend")
print("  ✓ test_no_legend_legend.pdf - Separate HORIZONTAL legend")
print("\n[Test 2] show_legend=True:")
print("  ✓ test_with_legend.pdf - Plot WITH legend inside")
print("  ✗ No separate legend file")
print("\n" + "="*70)
print("Usage in graphs.py:")
print("="*70)
print("""
When show_legend=False:
  - Main plot has NO legend (clean plot area)
  - Separate horizontal legend PDF is generated
  - Legend file: {filename}_legend.pdf

When show_legend=True:
  - Main plot has legend inside
  - NO separate legend file generated
  
This is ideal for publications where you want:
  - Clean plots without legends
  - A single horizontal legend to place below all plots
""")
print("="*70)
