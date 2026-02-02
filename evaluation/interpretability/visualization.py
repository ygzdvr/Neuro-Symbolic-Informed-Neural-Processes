"""
Shared visualization utilities for interpretability experiments.

Provides consistent styling and helper functions for all plots.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import warnings

# Matplotlib and seaborn setup
def setup_plotting():
    """Setup matplotlib and seaborn with consistent styling."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set seaborn style
        sns.set(font_scale=1.3)
        sns.set_style("whitegrid")
        sns.set_palette("rocket", n_colors=10)
        
        # Additional matplotlib settings
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams['savefig.facecolor'] = 'white'
        plt.rcParams['savefig.dpi'] = 150
        plt.rcParams['savefig.bbox'] = 'tight'
        plt.rcParams['font.family'] = 'sans-serif'
        
        return plt, sns
    except ImportError as e:
        warnings.warn(f"Visualization libraries not available: {e}")
        return None, None


def create_output_dir(base_dir: Path, experiment_name: str) -> Path:
    """Create organized output directory for plots."""
    plots_dir = base_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir


def save_figure(fig, path: Path, name: str, formats: List[str] = ["png", "pdf"]):
    """Save figure in multiple formats."""
    for fmt in formats:
        save_path = path / f"{name}.{fmt}"
        fig.savefig(save_path, format=fmt, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {name}.{formats}")


def add_value_labels(ax, bars, fmt=".2f", fontsize=10, offset=3):
    """Add value labels on top of bars."""
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:{fmt}}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, offset),
                   textcoords="offset points",
                   ha='center', va='bottom',
                   fontsize=fontsize, fontweight='bold')


def create_heatmap_data(data_dict: Dict, row_labels: List, col_labels: List) -> np.ndarray:
    """Convert dictionary data to heatmap array."""
    n_rows = len(row_labels)
    n_cols = len(col_labels)
    matrix = np.zeros((n_rows, n_cols))
    
    for i, row in enumerate(row_labels):
        for j, col in enumerate(col_labels):
            key = f"{row}_{col}"
            if key in data_dict:
                matrix[i, j] = data_dict[key]
    
    return matrix


# Color palettes for specific use cases
COLORS = {
    "with_knowledge": "#1f77b4",  # Blue
    "without_knowledge": "#d62728",  # Red
    "goldilocks": "#2ca02c",  # Green
    "warning": "#ff7f0e",  # Orange
    "neutral": "#7f7f7f",  # Gray
}


def get_rocket_palette(n_colors: int = 10):
    """Get rocket palette with specified number of colors.
    
    This function ensures we always get enough colors to avoid
    'list index out of range' errors when accessing palette[5] or palette[6].
    
    Args:
        n_colors: Number of colors to generate (default 10)
        
    Returns:
        List of RGB tuples representing the color palette
    """
    try:
        import seaborn as sns
        return sns.color_palette("rocket", n_colors=n_colors)
    except ImportError:
        # Fallback colors if seaborn not available
        return [
            (0.02, 0.05, 0.15),  # Dark blue
            (0.22, 0.10, 0.30),  # Purple
            (0.45, 0.15, 0.35),  # Magenta
            (0.65, 0.20, 0.30),  # Red
            (0.85, 0.35, 0.25),  # Orange
            (0.95, 0.55, 0.30),  # Light orange
            (0.98, 0.75, 0.50),  # Yellow
            (0.99, 0.88, 0.70),  # Light yellow
            (0.99, 0.95, 0.85),  # Cream
            (1.00, 0.98, 0.95),  # White
        ][:n_colors]

# Standard figure sizes
FIGSIZE = {
    "single": (8, 6),
    "double": (14, 5),
    "triple": (16, 5),
    "quad": (14, 10),
    "large": (16, 12),
    "heatmap": (10, 8),
}
