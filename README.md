# Place Recognition Result Tools (pr_result_tools)

A comprehensive Python toolkit for visualizing and analyzing place recognition benchmark results. This package provides utilities for generating publication-quality graphs, tables, heatmaps, and statistical analyses from place recognition evaluation data.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Modules](#modules)
- [Usage Examples](#usage-examples)
- [Output Formats](#output-formats)
- [Documentation](#documentation)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Contributing](#contributing)

## 🎯 Overview

`pr_csv_tools` is designed to streamline the analysis and visualization of place recognition experiments. It processes CSV result files from various place recognition methods and generates:

- **Recall curves** (Top-K and range-based)
- **Performance tables** (LaTeX/CSV compatible)
- **Heatmaps** for multi-sequence comparisons
- **Statistical analyses** across datasets
- **Publication-ready figures** with separate legend exports

The toolkit is particularly useful for researchers evaluating lidar-based place recognition methods across multiple sequences and environmental conditions.

## ✨ Features

### Core Functionality

- **Automatic CSV Parsing**: Recursively loads result files with pattern matching
- **Multi-Sequence Analysis**: Compare performance across different datasets simultaneously
- **Model Comparison**: Visualize multiple place recognition methods side-by-side
- **Flexible Visualization**: Customizable graphs with publication-quality output
- **LaTeX Integration**: Generate ready-to-use LaTeX tables and figures
- **Separate Legend Export**: Automatically save plot legends as standalone PDF files

### Visualization Types

1. **Top-K Recall Graphs**: Performance vs. number of candidates retrieved
2. **Range-Based Recall**: Performance vs. distance threshold
4. **Heatmaps**: Color-coded performance matrices
5. **Density Analysis**: Performance correlation with point cloud density

### Advanced Features

- **Automatic legend sizing and export** for flexible figure layouts
- **Multi-column table generation** with hierarchical headers
- **Tag-based file filtering** for selective data loading
- **Customizable plotting styles** (colors, markers, line styles)
- **Transparent backgrounds** for overlay compositions
- **High-DPI output** (300 DPI) for print publications

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Install Dependencies

```bash
cd pr_csv_tools
pip install -r requirements.txt
```

### Required Packages

```
pandas
numpy
matplotlib
seaborn
tabulate
tqdm
```

Or install individually:

```bash
pip install pandas numpy matplotlib seaborn tabulate tqdm
```

## 🚀 Quick Start

### Basic Usage

```python
from graphs import run_top25_graphs, run_range_graphs

# Define your data
root_dir = "/path/to/results"
sequences = ["SEQ1", "SEQ2", "SEQ3"]
models = ["PointNetVLAD", "OverlapTransformer", "LoGG3D"]

# Generate Top-K graphs
run_top25_graphs(
    root=root_dir,
    seq_order=sequences,
    model_order=models,
    save_dir="output/graphs_top25",
    size_param=25,
    linewidth=5,
    show_legend=True
)

# Generate range-based graphs
run_range_graphs(
    root=root_dir,
    seq_order=sequences,
    model_order=models,
    save_dir="output/graphs_range",
    size_param=25,
    topk=1,
    show_legend=True
)
```

### Generate Tables

```python
from segment_table import run_table

# Create performance table
table = run_table(
    root=root_dir,
    files_to_show=["recall.csv"],
    seq_order=sequences,
    model_order=models,
    topk=1,
    range=10,
    tag="global",
    save_dir="output/tables",
    save_latex=True
)
```

### Create Heatmaps

```python
from heatmap import main_heatmap

main_heatmap(
    root=root_dir,
    sequences=sequences,
    model_order=models,
    heatmap_path="output/heatmaps",
    new_model=models,
    ROWS=models,
    size_param=25,
    topk=1,
    target_range=10
)
```

## 📚 Modules

### `graphs.py`

Main visualization module for generating recall curves.

**Key Functions:**

- `run_top25_graphs()`: Generate Top-K recall curves (K=1 to 25)
- `run_range_graphs()`: Generate range-based recall curves
- `gen_top25_fig()`: Create individual Top-K figures
- `gen_range_fig()`: Create individual range figures
- `save_legend_separately()`: Export plot legends as separate PDFs

**Example:**

```python
from graphs import run_top25_graphs

run_top25_graphs(
    root="/path/to/results",
    seq_order=["SEQ1", "SEQ2"],
    model_order=["MODEL1", "MODEL2"],
    save_dir="output",
    size_param=30,
    linewidth=5,
    show_legend=True,
    marker_size=15
)
```

### `table.py`

Generate LaTeX and CSV tables from results.

**Key Functions:**

- `generate_table()`: Create formatted performance tables
- Support for multi-column headers
- Automatic mean and standard deviation calculation

**Example:**

```python
from table import generate_table

table = generate_table(
    table=results_dict,
    rows=models,
    columns=sequences,
    ranges=[5, 10, 15, 20],
    k_cand=[1, 5, 10],
    res=3  # decimal precision
)
```

### `segment_table.py`

Per-segment performance analysis, mainly used in horticultural environments, where one needs to compare performance at the row level.

**Key Functions:**

- `run_table()`: Generate segment-wise performance tables
- `generate_table()`: Format segment results

**Example:**

```python
from segment_table import run_table

pd_table = run_table(
    root="/path/to/results",
    files_to_show=["recall_0.csv", "recall_1.csv"],
    seq_order=sequences,
    model_order=models,
    topk=1,
    range=10,
    tag="segments",
    save_dir="output"
)
```

### `heatmap.py`

Create color-coded performance heatmaps.

**Key Functions:**

- `main_heatmap()`: Generate heatmaps for all sequences
- Automatic segment and global performance visualization

**Example:**

```python
from heatmap import main_heatmap

main_heatmap(
    root="/path/to/results",
    sequences=["SEQ1", "SEQ2"],
    model_order=models,
    heatmap_path="output/heatmaps",
    new_model=models,
    ROWS=models,
    size_param=25,
    topk=1,
    target_range=10
)
```

### `utils.py`

Utility functions for data loading and processing.

**Key Functions:**

- `load_results()`: Recursively load CSV result files
- `parse_result_path()`: Extract metadata from file paths
- `find_file()`: Tag-based file search
- `find_tags_in_str()`: String pattern matching with logic operators

**Example:**

```python
from utils import load_results, find_file

# Load all results
matches, sequences, models = load_results(
    dir="/path/to/results",
    model_key="L2",
    seq_key="eval-",
    score_key="@"
)

# Find specific files
indices = find_file(
    data_struct=results,
    tags=["recall", "10m"],
    op="and"  # Both tags must be present
)
```

### `graphs_density.py`

Analyze performance correlation with point cloud density.

**Key Functions:**

- `run_graphs_density()`: Generate density-based performance graphs
- `generate_density()`: Create density distribution plots
- `parse_results()`: Custom result parsing with flexible keys

### `loss_study.py`

Training hyperparameter sensitivity analysis.

**Key Functions:**

- `loss_study()`: Analyze effect of loss function parameters
- `gen_top25_fig()`: Visualize Top-K performance trends

## 💡 Usage Examples

### Example 1: Complete Analysis Pipeline

```python
from graphs import run_top25_graphs, run_range_graphs
from segment_table import run_table
from heatmap import main_heatmap

# Configuration
root = "/path/to/benchmark/results"
sequences = ["PCD_EASY", "PCD_Easy_DARK", "PCD_MED", "PCD_HARD"]
models = ["PointNetVLAD", "LOGG3D", "OverlapTransformer", "ScanContext"]
output_dir = "output_analysis"

# 1. Generate Top-K graphs
print("Generating Top-K graphs...")
run_top25_graphs(
    root=root,
    seq_order=sequences,
    model_order=models,
    save_dir=f"{output_dir}/graphs_top25",
    size_param=30,
    show_legend=True
)

# 2. Generate range-based graphs
print("Generating range graphs...")
run_range_graphs(
    root=root,
    seq_order=sequences,
    model_order=models,
    save_dir=f"{output_dir}/graphs_range",
    size_param=30,
    topk=1
)

# 3. Create performance tables
print("Generating tables...")
table = run_table(
    root=root,
    files_to_show=["recall.csv"],
    seq_order=sequences,
    model_order=models,
    topk=1,
    range=10,
    tag="global",
    save_dir=f"{output_dir}/tables",
    save_latex=True
)

# 4. Generate heatmaps
print("Generating heatmaps...")
main_heatmap(
    root=root,
    sequences=sequences,
    model_order=models,
    heatmap_path=f"{output_dir}/heatmaps",
    new_model=models,
    ROWS=models,
    size_param=25,
    topk=1,
    target_range=10
)

print(f"Analysis complete! Results saved to {output_dir}/")
```

### Example 2: Custom Plotting Style

```python
from graphs import gen_top25_fig
from utils import load_results

# Load results
matches, sequences, models = load_results("/path/to/results")

# Generate custom styled plot
gen_top25_fig(
    results=matches,
    save_dir="custom_output",
    size_param=20,       # Font size
    linewidth=4,         # Line thickness
    marker_size=12,      # Marker size
    show_legend=False,   # Hide legend in plot
    colors=["red", "blue", "green"],
    linestyles=["-", "--", "-."]
)
```

### Example 3: Selective File Processing

```python
from utils import find_file, load_results

# Load results
results, seqs, models = load_results("/path/to/results")

# Find files with specific tags (AND logic)
recall_10m = find_file(
    data_struct=results["PCD_EASY"]["PointNetVLAD"],
    tags=["recall", "10m"],
    op="and"
)

# Find files with any matching tag (OR logic)
any_recall = find_file(
    data_struct=results["PCD_EASY"]["PointNetVLAD"],
    tags=["recall_0", "recall_1", "recall_2"],
    op="or"
)
```

### Example 4: LaTeX Integration

After generating figures with `show_legend=True`:

```latex
\begin{figure*}[t]
    \centering
    
    % Four plots without legends
    \begin{subfigure}[b]{0.45\textwidth}
        \includegraphics[width=\textwidth]{PCD_EASY.pdf}
        \caption{Easy conditions}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.45\textwidth}
        \includegraphics[width=\textwidth]{PCD_MED.pdf}
        \caption{Medium conditions}
    \end{subfigure}
    
    \vspace{0.5em}
    
    \begin{subfigure}[b]{0.45\textwidth}
        \includegraphics[width=\textwidth]{PCD_HARD.pdf}
        \caption{Hard conditions}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.45\textwidth}
        \includegraphics[width=\textwidth]{PCD_Easy_DARK.pdf}
        \caption{Dark conditions}
    \end{subfigure}
    
    % Shared legend at bottom
    \vspace{1em}
    \begin{subfigure}[b]{0.6\textwidth}
        \centering
        \includegraphics[width=\textwidth]{PCD_EASY_legend.pdf}
    \end{subfigure}
    
    \caption{Place recognition performance across different environmental conditions.}
    \label{fig:multi_condition}
\end{figure*}
```

## 📊 Output Formats

### File Naming Convention

**Graphs:**
- `{sequence_name}.pdf` - Main plot
- `{sequence_name}_legend.pdf` - Separate legend (auto-generated)

**Tables:**
- `global_recall_{range}m@{topk}.csv` - CSV format
- `global_recall_{range}m@{topk}.tex` - LaTeX format
- `segments_recall_{range}m@{topk}.csv` - Segment tables

**Heatmaps:**
- `global_r{range}m_top{k}.pdf` - Global performance
- `segments_{seq}_r{range}m_top{k}.pdf` - Per-sequence segments

### Directory Structure

```
output/
├── graphs_top25/
│   ├── 10m/
│   │   ├── w_label/
│   │   │   ├── PCD_EASY.pdf
│   │   │   ├── PCD_EASY_legend.pdf
│   │   │   └── ...
│   │   └── no_label/
│   └── mean.pdf
├── graphs_range/
│   ├── top1_range/
│   │   ├── w_label/
│   │   └── no_label/
│   └── ...
├── tables/
│   ├── global_recall_10m@1.csv
│   ├── global_recall_10m@1.tex
│   └── segments_recall_10m@1.csv
└── heatmaps/
    ├── global_r10m_top1.pdf
    └── segments_PCD_EASY_r10m_top1.pdf
```

## 📖 Documentation

### Additional Resources

- **[QUICK_REFERENCE_LEGEND.md](QUICK_REFERENCE_LEGEND.md)**: Quick guide for using the separate legend feature
- **[SEPARATE_LEGEND_DOCUMENTATION.md](SEPARATE_LEGEND_DOCUMENTATION.md)**: Detailed documentation on legend export functionality
- **Demo Examples**: See `demo/` directory for example scripts

### Key Concepts

#### Result File Structure

The toolkit expects CSV files organized with specific naming patterns:

```
results/
├── {model_name}/
│   └── eval-{sequence_name}/
│       ├── recall@{range}m.csv
│       ├── recall_0.csv  # Segment 0
│       ├── recall_1.csv  # Segment 1
│       └── ...
```

#### CSV Format

Result CSV files should contain recall values:

```csv
range,1,2,3,4,5,...,25
5,0.45,0.62,0.73,0.81,...
10,0.38,0.55,0.68,0.76,...
15,0.32,0.48,0.61,0.71,...
```

- **Rows**: Distance thresholds (meters)
- **Columns**: Top-K candidates
- **Values**: Recall@K for given range

#### Tag-Based Filtering

Use logical operators to filter files:

```python
# AND: All tags must be present
find_file(data, tags=["recall", "10m", "segment_0"], op="and")

# OR: Any tag must be present  
find_file(data, tags=["easy", "medium"], op="or")
```

## 🗂️ Project Structure

```
pr_result_tools/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .gitignore                        # Git ignore patterns
│
├── Core Modules
├── graphs.py                         # Main visualization (Top-K, range)
├── table.py                          # Table generation (LaTeX/CSV)
├── segment_table.py                  # Per-segment analysis
├── heatmap.py                        # Heatmap generation
├── utils.py                          # Utility functions
├── graphs_density.py                 # Density analysis
├── loss_study.py                     # Loss function study
├── multi_topk_table.py              # Multi-column tables
│
├── Documentation
├── QUICK_REFERENCE_LEGEND.md        # Legend feature quick guide
├── SEPARATE_LEGEND_DOCUMENTATION.md  # Detailed legend docs
│
├── Test Scripts
├── test_horizontal_legend.py         # Test horizontal legends
├── test_legend_always.py            # Test legend generation
├── test_separate_legend.py          # Test separate export
├── test_transparent_legend.py       # Test transparency
│
├── Utilities
├── copy_file.sh                      # File copying script
│
└── demo/
    └── multi-coloum.py              # Multi-column example
```

## 📋 Requirements

### System Requirements

- **OS**: Linux, macOS, Windows
- **Python**: 3.8+
- **Memory**: 4GB+ recommended
- **Disk**: Sufficient space for output files

### Python Packages

```txt
pandas>=1.3.0
numpy>=1.20.0
matplotlib>=3.4.0
seaborn>=0.11.0
tabulate>=0.8.9
tqdm>=4.62.0
```

Install with:

```bash
pip install pandas numpy matplotlib seaborn tabulate tqdm
```

### Optional Dependencies

For development:

```bash
pip install pytest black flake8 mypy
```

## 🔧 Configuration

### Customizing Plotting Styles

Edit global parameters in `graphs.py`:

```python
# Plot styling
COLORS = ["blue", "gray", "red", "green", "orange"]
LINESTYLES = ["-", "--", "-.", ":"]
MARKERS = ['s', '^', 'v', 'D', 'p', 'o']

# Default sizes
SIZE_PARAM = 25    # Font size
LINEWIDTH = 5      # Line width
```

### File Pattern Matching

Customize pattern matching in `utils.py`:

```python
# Default patterns
model_key = 'L2'        # Model identifier
seq_key = 'eval-'       # Sequence identifier  
score_key = "@"         # Score file marker
```

## 🤝 Contributing

### Development Setup

```bash
# Clone repository
git clone https://github.com/Cybonic/pr_csv_tools.git
cd pr_result_tools

# Install in development mode
pip install -e .

# Install development dependencies
pip install pytest black flake8
```

### Code Style

- Follow PEP 8 guidelines
- Use type hints where possible
- Document all functions with docstrings
- Add unit tests for new features

### Testing

```bash
# Run test scripts
python test_separate_legend.py
python test_horizontal_legend.py
```

## 📝 License

[Specify your license here]

## 👥 Authors

- **Tiago Barros** - *Initial work and maintenance*

## 🙏 Acknowledgments

- Built for place recognition benchmarking research
- Designed to work with lidar-based place recognition datasets
- Compatible with various place recognition methods (PointNetVLAD, OverlapTransformer, LoGG3D, etc.)

## 📧 Contact

For questions, issues, or contributions:
- Repository: https://github.com/Cybonic/pr_csv_tools
- Issues: https://github.com/Cybonic/pr_csv_tools/issues

## 🔄 Changelog

### Version 2.0 (Current)
- Added separate legend export functionality
- Improved multi-sequence handling
- Enhanced table generation with multi-column support
- Added transparent background option
- Improved documentation

### Version 1.0
- Initial release
- Basic graph and table generation
- CSV result parsing
- Heatmap visualization

---

**Note**: This tool is actively maintained. For the latest updates, check the [GitHub repository](https://github.com/Cybonic/pr_csv_tools).
