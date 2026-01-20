# Separate Legend Export Feature

## Overview

The `graphs.py` script has been updated to automatically save plot legends as **separate PDF files** with size adjusted to fit the legend content. This is particularly useful for creating publication-quality figures where you want precise control over legend placement.

## What Changed

### New Function: `save_legend_separately()`

```python
def save_legend_separately(fig, ax, filename, fontsize=None, **legend_kwargs):
    """
    Save the legend of a plot as a separate PDF file.
    
    Args:
        fig: Matplotlib figure object
        ax: Matplotlib axis object
        filename: Path to save the legend PDF (without .pdf extension)
        fontsize: Font size for legend text
        **legend_kwargs: Additional keyword arguments to pass to ax.legend()
    
    Returns:
        Path to the saved legend file
    """
```

### Key Features

1. **Automatic Size Adjustment**: Legend PDF is sized to fit the content exactly
2. **Transparent Background**: Legend can be overlaid on other figures
3. **Maintains Formatting**: Font size and styling preserved from original
4. **Minimal Padding**: Smart padding added for clean appearance

## Usage

### Automatic (Default Behavior)

When `show_legend=True`, both the main plot and a separate legend file are generated:

```python
run_range_graphs(root, sequences, model_order,
    save_dir    = graph_path,
    size_param  = 30, 
    show_legend = True)  # Enables separate legend export
```

### Output Files

For each plot, two files are created:

```
output_dir/
  ├── PCD_EASY.pdf              # Main plot with legend
  ├── PCD_EASY_legend.pdf       # Standalone legend (auto-sized)
  ├── PCD_Easy_DARK.pdf         # Main plot with legend
  └── PCD_Easy_DARK_legend.pdf  # Standalone legend (auto-sized)
```

## File Naming Convention

| Original File | Legend File |
|--------------|-------------|
| `sequence_name.pdf` | `sequence_name_legend.pdf` |
| `PCD_EASY.pdf` | `PCD_EASY_legend.pdf` |
| `results_plot.pdf` | `results_plot_legend.pdf` |

## Use Cases

### 1. Publication Figures

Create multi-panel figures with shared legend:

```
┌─────────────┬─────────────┐
│   Plot A    │   Plot B    │
│  (no legend)│  (no legend)│
├─────────────┼─────────────┤
│   Plot C    │   Plot D    │
│  (no legend)│  (no legend)│
└─────────────┴─────────────┘
        ┌─────────────┐
        │   Legend    │
        │  (separate) │
        └─────────────┘
```

### 2. Custom Legend Placement

- Position legend anywhere in your LaTeX document
- Resize without affecting plot dimensions
- Rotate or transform independently

### 3. Space Optimization

- Remove legend from plots to maximize data area
- Place legend in document margin
- Share one legend across multiple related plots

## Integration with Existing Code

### Top-25 Graphs

```python
# In gen_top25_fig()
if show_legend:
    ax.legend(fontsize=size_param)
    # Automatically saves separate legend
    save_legend_separately(fig, ax, file, fontsize=size_param)
```

### Range Graphs

```python
# In gen_range_fig()
if show_legend:
    ax.legend(fontsize=size_param)
    # Automatically saves separate legend
    save_legend_separately(fig, ax, file, fontsize=size_param)
```

## Technical Details

### Size Calculation

The legend PDF size is calculated as:

1. Create temporary figure with legend only
2. Draw legend and get bounding box dimensions
3. Add 10% padding on all sides
4. Crop PDF to exact legend size

### Transparency

- Background is transparent by default
- Can be overlaid on any background color
- Suitable for PowerPoint, Keynote, and LaTeX

### DPI Setting

- Default: 300 DPI (publication quality)
- Matches typical journal requirements
- Ensures crisp text at any scale

## LaTeX Integration

### Example: Include Both Files

```latex
\begin{figure}[t]
    \centering
    \includegraphics[width=0.8\textwidth]{PCD_EASY.pdf}
    \caption{Recall performance on PCD\_EASY sequence.}
    \label{fig:pcd_easy}
\end{figure}

% Separate legend figure
\begin{figure}[t]
    \centering
    \includegraphics[width=0.4\textwidth]{PCD_EASY_legend.pdf}
    \caption{Legend for Figure~\ref{fig:pcd_easy}.}
    \label{fig:pcd_easy_legend}
\end{figure}
```

### Example: Multi-Panel with Shared Legend

```latex
\begin{figure*}[t]
    \centering
    \begin{subfigure}[b]{0.45\textwidth}
        \includegraphics[width=\textwidth]{PCD_EASY_no_legend.pdf}
        \caption{Easy conditions}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.45\textwidth}
        \includegraphics[width=\textwidth]{PCD_HARD_no_legend.pdf}
        \caption{Hard conditions}
    \end{subfigure}
    
    \vspace{1em}
    
    % Shared legend centered below
    \begin{subfigure}[b]{0.4\textwidth}
        \centering
        \includegraphics[width=\textwidth]{PCD_EASY_legend.pdf}
    \end{subfigure}
    
    \caption{Performance comparison across different conditions.}
    \label{fig:comparison}
\end{figure*}
```

## Testing

Run the test script to see examples:

```bash
cd /home/tiago/workspace/place_uk/pr_result_tools
python test_separate_legend.py
```

This generates:
- `test_plot.pdf` - Plot with legend
- `test_plot_legend.pdf` - **Standalone legend (auto-sized)**
- `test_plot_no_legend.pdf` - Plot without legend

## Customization

### Adjust Legend Size

Modify font size when calling:

```python
save_legend_separately(fig, ax, file, fontsize=25)
```

### Additional Legend Options

Pass matplotlib legend kwargs:

```python
save_legend_separately(fig, ax, file, 
                      fontsize=20,
                      framealpha=0.9,
                      fancybox=True,
                      shadow=True)
```

### Change Padding

Edit the function:

```python
# Current: 10% padding
padding = 0.1

# Increase padding:
padding = 0.2  # 20% padding
```

## Benefits

### For Publications
✅ Professional appearance  
✅ Flexible layout options  
✅ Easy to update legend separately  
✅ Consistent with journal requirements  

### For Presentations
✅ Control legend placement  
✅ Animate legend separately  
✅ Resize without affecting plot  
✅ Reuse across slides  

### For Collaboration
✅ Share plots without legend clutter  
✅ Let collaborators design legend placement  
✅ Easy version control (separate files)  
✅ Modular figure composition  

## Backward Compatibility

The changes are **fully backward compatible**:

- Existing code continues to work
- Legend still appears in main plot
- Separate legend is bonus feature
- Can be ignored if not needed

## File Size

Legend PDFs are very small:
- Typical size: 10-50 KB
- Vector format (scalable)
- No quality loss at any zoom level

## Common Issues

### Issue: Legend Not Found

**Symptom**: Warning message "No legend found"

**Solution**: Ensure plot has labels:
```python
ax.plot(x, y, label='Model Name')
ax.legend()
```

### Issue: Legend Cut Off

**Symptom**: Legend partially visible

**Solution**: Increase padding in function:
```python
padding = 0.2  # Increase from 0.1
```

### Issue: Font Size Mismatch

**Symptom**: Legend text size differs

**Solution**: Pass same fontsize to both:
```python
ax.legend(fontsize=size_param)
save_legend_separately(fig, ax, file, fontsize=size_param)
```

## Future Enhancements

Possible improvements:
1. Option to exclude legend from main plot
2. Horizontal vs vertical legend orientation
3. Multiple columns in legend
4. Custom legend background colors
5. Border styling options

## Summary

The separate legend feature provides:
- ✅ **Automatic generation** - No extra code needed
- ✅ **Perfect sizing** - Adjusted to content
- ✅ **Publication ready** - 300 DPI, transparent
- ✅ **Easy integration** - Works with LaTeX, PowerPoint, etc.
- ✅ **Fully compatible** - No breaking changes

Generated files follow pattern: `{filename}_legend.pdf`
