#!/usr/bin/env python3
"""
Test the separate legend saving functionality.

This script creates a simple plot and saves the legend separately
to demonstrate the new functionality.
"""

import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Add the parent directory to the path
sys.path.insert(0, '/home/tiago/workspace/place_uk/pr_result_tools')

from graphs import save_legend_separately

# Create sample data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.sin(x) * np.cos(x)
y4 = np.exp(-x/10)

# Create a figure
fig, ax = plt.subplots(figsize=(10, 8))

# Plot multiple lines with labels
ax.plot(x, y1, '-', linewidth=3, label='Model A: PointNetPGAP')
ax.plot(x, y2, '--', linewidth=3, label='Model B: PointNetVLAD')
ax.plot(x, y3, '-.', linewidth=3, label='Model C: SPVSoAP3D')
ax.plot(x, y4, ':', linewidth=3, label='Model D: LOGG3D')

# Add labels and grid
ax.set_xlabel('Top k', fontsize=20)
ax.set_ylabel('Recall@k', fontsize=20)
ax.grid(True)
ax.tick_params(axis='both', labelsize=18)

# Add legend to the plot
ax.legend(fontsize=18, loc='best')

# Save the main plot
output_dir = '/home/tiago/workspace/place_uk/pr_result_tools/test_output'
os.makedirs(output_dir, exist_ok=True)

main_file = os.path.join(output_dir, 'test_plot')
plt.savefig(f'{main_file}.pdf', transparent=True, bbox_inches='tight')
print(f"Main plot saved to: {main_file}.pdf")

# Save legend separately
legend_file = save_legend_separately(fig, ax, main_file, fontsize=18)

# Also save a version without legend for comparison
ax.get_legend().remove()
plt.savefig(f'{main_file}_no_legend.pdf', transparent=True, bbox_inches='tight')
print(f"Plot without legend saved to: {main_file}_no_legend.pdf")

plt.close()

print("\n" + "="*60)
print("Test completed successfully!")
print("="*60)
print(f"\nGenerated files in: {output_dir}")
print("  1. test_plot.pdf - Plot with legend")
print("  2. test_plot_legend.pdf - Standalone legend (adjusted size)")
print("  3. test_plot_no_legend.pdf - Plot without legend")
print("\nFor publications, you can now:")
print("  - Use the no_legend version in your figure")
print("  - Place the legend PDF separately or combine them")
print("="*60)
