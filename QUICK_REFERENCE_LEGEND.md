# Quick Reference: Separate Legend Feature

## What You Get

When you run `graphs.py` with `show_legend=True`, you now automatically get:

```
For each plot:
  ✓ {name}.pdf          - Plot with legend
  ✓ {name}_legend.pdf   - Legend only (auto-sized) ← NEW!
```

## Example Usage

### Run Your Analysis

```bash
cd /home/tiago/workspace/place_uk/pr_result_tools
python graphs.py
```

### Results

```
hortov2_uk/graphs_top25/10m/w_label/
  ├── PCD_EASY.pdf              # Plot with legend
  ├── PCD_EASY_legend.pdf       # ← Auto-sized legend!
  ├── PCD_Easy_DARK.pdf         # Plot with legend
  └── PCD_Easy_DARK_legend.pdf  # ← Auto-sized legend!

hortov2_uk/graphs_range/top1_range/w_label/
  ├── PCD_MED.pdf
  ├── PCD_MED_legend.pdf        # ← Auto-sized legend!
  └── ...
```

## Use Cases

### 1. LaTeX Multi-Panel Figure

```latex
% Four plots without legend
\subfigure{\includegraphics[width=0.45\textwidth]{plot1.pdf}}
\subfigure{\includegraphics[width=0.45\textwidth]{plot2.pdf}}
\subfigure{\includegraphics[width=0.45\textwidth]{plot3.pdf}}
\subfigure{\includegraphics[width=0.45\textwidth]{plot4.pdf}}

% Shared legend below
\centering
\includegraphics[width=0.4\textwidth]{plot1_legend.pdf}
```

### 2. Side-by-Side Comparison

```latex
\begin{minipage}{0.7\textwidth}
    \includegraphics[width=\textwidth]{results.pdf}
\end{minipage}
\begin{minipage}{0.25\textwidth}
    \includegraphics[width=\textwidth]{results_legend.pdf}
\end{minipage}
```

### 3. Margin Legend

```latex
\begin{marginfigure}
    \includegraphics[width=\marginparwidth]{plot_legend.pdf}
\end{marginfigure}
```

## Key Features

| Feature | Benefit |
|---------|---------|
| **Auto-sized** | PDF exactly fits legend content |
| **Transparent** | Works on any background |
| **High quality** | 300 DPI, vector graphics |
| **Consistent** | Same font/style as main plot |
| **Automatic** | No code changes needed |

## File Naming

| Main Plot | Legend File |
|-----------|-------------|
| `results.pdf` | `results_legend.pdf` |
| `PCD_EASY.pdf` | `PCD_EASY_legend.pdf` |
| `analysis.pdf` | `analysis_legend.pdf` |

## Quick Workflow

1. **Generate plots** with `show_legend=True`
2. **Get two files** per plot automatically
3. **Use main plot** for data visualization
4. **Use legend file** for flexible placement

## Benefits

✅ **Save space** - Remove legend from plot area  
✅ **Flexible layout** - Place legend anywhere  
✅ **Share legend** - One legend for multiple plots  
✅ **Publication ready** - Professional appearance  
✅ **Easy updates** - Legend separate from plot  

## Common Patterns

### Pattern 1: Remove Main Legend (Manual)

If you want ONLY the separate legend (no legend in main plot):

Edit `graphs.py` after legend is saved:

```python
if show_legend:
    ax.legend(fontsize=size_param)
    save_legend_separately(fig, ax, file, fontsize=size_param)
    ax.get_legend().remove()  # ← Add this line
```

### Pattern 2: Different Legend Positions

In LaTeX, control position:

```latex
% Top right
\put(80,90){\includegraphics[width=3cm]{legend.pdf}}

% Bottom center  
\put(40,10){\includegraphics[width=4cm]{legend.pdf}}

% Outside plot
\put(110,50){\includegraphics[width=2.5cm]{legend.pdf}}
```

### Pattern 3: Shared Legend Grid

```latex
% 2x2 grid of plots
\includegraphics{plot1.pdf}
\includegraphics{plot2.pdf}
\includegraphics{plot3.pdf}
\includegraphics{plot4.pdf}

% Single legend for all
\centerline{\includegraphics[width=0.3\textwidth]{plot1_legend.pdf}}
```

## No Changes Needed!

Your existing `graphs.py` workflow continues to work:

```python
# Your current code - unchanged
run_range_graphs(root, sequences, model_order,
    save_dir    = graph_path,
    size_param  = 30, 
    show_legend = True)

# Now automatically generates:
# - Plot with legend
# - Separate legend file ← NEW!
```

## Summary

🎉 **You get separate legend PDFs automatically!**

- No code changes required
- Same workflow as before  
- Bonus legend file per plot
- Perfect for publications
- Sized to fit content

Just look for `*_legend.pdf` files in your output directory!
