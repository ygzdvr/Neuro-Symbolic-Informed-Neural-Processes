#!/usr/bin/env python
"""
Concatenate plots from multiple models into side-by-side comparison images.

Usage:
    python concat_plots.py --batch-dir interpretability_results/sinusoids_batch/batch_20260201_172406

This script:
1. Finds all model subdirectories in the batch directory
2. For each experiment (M1-M10), finds matching plot files
3. Concatenates them horizontally with model names as titles
4. Saves combined images to a 'combined' folder
"""

import argparse
import os
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict

# PDF/image handling
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("Warning: PIL not available. Install with: pip install Pillow")

try:
    from pdf2image import convert_from_path
    HAS_PDF2IMAGE = True
except ImportError:
    HAS_PDF2IMAGE = False
    print("Warning: pdf2image not available. Install with: pip install pdf2image")
    print("         Also need poppler: conda install -c conda-forge poppler")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def discover_models(batch_dir: Path) -> List[str]:
    """Find all model subdirectories."""
    models = []
    for item in sorted(batch_dir.iterdir()):
        if item.is_dir() and not item.name.startswith(('.', 'combined')):
            models.append(item.name)
    return models


def find_plot_files(batch_dir: Path, models: List[str]) -> Dict[str, Dict[str, Path]]:
    """
    Find all plot files organized by relative path.
    
    Returns:
        Dict mapping relative_path -> {model_name: full_path}
    """
    plot_files = defaultdict(dict)
    
    for model in models:
        model_dir = batch_dir / model
        
        # Walk through all subdirectories to find plot files
        for root, dirs, files in os.walk(model_dir):
            root_path = Path(root)
            
            for filename in files:
                # Only process image/pdf files
                if filename.lower().endswith(('.png', '.pdf', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                    full_path = root_path / filename
                    
                    # Get relative path from model directory
                    rel_path = full_path.relative_to(model_dir)
                    
                    plot_files[str(rel_path)][model] = full_path
    
    return dict(plot_files)


def load_image(path: Path) -> Image.Image:
    """Load an image or PDF page as a PIL Image."""
    suffix = path.suffix.lower()
    
    if suffix == '.pdf':
        if not HAS_PDF2IMAGE:
            raise RuntimeError("pdf2image required for PDF files. Install: pip install pdf2image")
        # Convert first page of PDF to image
        images = convert_from_path(str(path), first_page=1, last_page=1, dpi=150)
        return images[0].convert('RGB')
    elif suffix in ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'):
        img = Image.open(path)
        # Convert to RGB if needed (handles RGBA, P mode, etc.)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img
    else:
        raise RuntimeError(f"Unsupported image format: {suffix}")


def concatenate_images_horizontal(
    images: List[Tuple[str, Image.Image]], 
    output_path: Path,
    title: str = None
):
    """Concatenate images horizontally with model names."""
    if not images:
        return
    
    # Get dimensions
    widths = [img.width for _, img in images]
    heights = [img.height for _, img in images]
    
    max_height = max(heights)
    total_width = sum(widths)
    
    # Add space for titles
    title_height = 40
    
    # Create new image
    combined = Image.new('RGB', (total_width, max_height + title_height), 'white')
    
    # Paste images
    x_offset = 0
    for model_name, img in images:
        # Center vertically
        y_offset = title_height + (max_height - img.height) // 2
        combined.paste(img, (x_offset, y_offset))
        x_offset += img.width
    
    # Add title text using matplotlib if available
    if HAS_MATPLOTLIB and title:
        # Save temp, add title, resave
        import io
        import numpy as np
        
        fig, ax = plt.subplots(figsize=(total_width / 100, (max_height + title_height) / 100), dpi=100)
        ax.imshow(np.array(combined))
        ax.axis('off')
        
        # Add model names
        x_offset = 0
        for i, (model_name, img) in enumerate(images):
            x_center = x_offset + img.width / 2
            ax.text(x_center, 20, model_name, ha='center', va='center', 
                   fontsize=10, fontweight='bold')
            x_offset += img.width
        
        plt.tight_layout(pad=0)
        plt.savefig(output_path, bbox_inches='tight', dpi=150, facecolor='white')
        plt.close()
    else:
        combined.save(output_path)


def concatenate_plots(batch_dir: Path, output_dir: Path = None):
    """Main function to concatenate all matching plots."""
    
    if not HAS_PIL:
        print("Error: PIL (Pillow) is required. Install with: pip install Pillow")
        return
    
    # Find models
    models = discover_models(batch_dir)
    print(f"Found {len(models)} models: {', '.join(models)}")
    
    if len(models) < 2:
        print("Need at least 2 models to concatenate")
        return
    
    # Find plot files
    plot_files = find_plot_files(batch_dir, models)
    
    # Count by extension
    ext_counts = defaultdict(int)
    for path in plot_files.keys():
        ext = os.path.splitext(path)[1].lower()
        ext_counts[ext] += 1
    ext_summary = ", ".join(f"{ext}: {count}" for ext, count in sorted(ext_counts.items()))
    print(f"Found {len(plot_files)} unique plot paths ({ext_summary})")
    
    # Filter to only plots that exist for multiple models
    multi_model_plots = {
        path: model_paths 
        for path, model_paths in plot_files.items() 
        if len(model_paths) >= 2
    }
    print(f"Found {len(multi_model_plots)} plots present in 2+ models")
    
    # Create output directory
    if output_dir is None:
        output_dir = batch_dir / "combined"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each plot
    for rel_path, model_paths in sorted(multi_model_plots.items()):
        print(f"\nProcessing: {rel_path}")
        print(f"  Found in: {', '.join(model_paths.keys())}")
        
        # Load images
        images = []
        for model in models:  # Maintain consistent order
            if model in model_paths:
                try:
                    img = load_image(model_paths[model])
                    images.append((model, img))
                    print(f"    Loaded: {model}")
                except Exception as e:
                    print(f"    Failed to load {model}: {e}")
        
        if len(images) < 2:
            print(f"  Skipping: less than 2 images loaded")
            continue
        
        # Create output path (flatten directory structure)
        output_filename = rel_path.replace('/', '_').replace('\\', '_')
        # Always output as PNG for consistency
        base = os.path.splitext(output_filename)[0]
        output_filename = f"{base}_combined.png"
        
        output_path = output_dir / output_filename
        
        # Concatenate
        try:
            concatenate_images_horizontal(images, output_path, title=rel_path)
            print(f"  Saved: {output_path}")
        except Exception as e:
            print(f"  Failed to concatenate: {e}")
    
    print(f"\n{'=' * 60}")
    print(f"Combined plots saved to: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Concatenate plots from multiple models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--batch-dir", type=str, required=True,
                        help="Path to batch results directory")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: batch-dir/combined)")
    return parser.parse_args()


def main():
    args = parse_args()
    
    batch_dir = Path(args.batch_dir)
    if not batch_dir.exists():
        print(f"Error: Directory does not exist: {batch_dir}")
        return
    
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    concatenate_plots(batch_dir, output_dir)


if __name__ == "__main__":
    main()
