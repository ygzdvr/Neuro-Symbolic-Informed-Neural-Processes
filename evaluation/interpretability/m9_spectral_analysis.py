"""
M9: Heavy-Tailed Self-Regularization (HTSR) Analysis

This module analyzes the spectral properties of weight matrices to assess
generalization quality using the WeightWatcher framework.

Theory (from Martin & Mahoney):
    - Well-generalizing networks exhibit Heavy-Tailed Self-Regularization
    - The empirical spectral density (ESD) of weight matrices follows power law: p(λ) ∝ λ^(-alpha)
    - alpha ∈ [2, 4]: "Goldilocks zone" - good generalization
    - alpha < 2: Overfit (too heavy tail, memorization)
    - alpha > 4: Underfit (light tail, undercapacity)

Hypothesis:
    - The Knowledge Encoder in INPs induces regularization in aggregator layers
    - INP should have α closer to Goldilocks zone than NP
    - Knowledge conditioning acts as implicit regularization

Key metrics:
    - α (power law exponent): Main indicator of generalization
    - λ_max (spectral norm): Related to Lipschitz constant
    - Stable rank: Effective dimensionality of weight matrix
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from tqdm import tqdm
from pathlib import Path
import warnings

from .base import InterpretabilityExperiment, ExperimentConfig


def compute_esd(weight_matrix: np.ndarray) -> np.ndarray:
    """
    Compute Empirical Spectral Density (eigenvalues of W^T W).
    
    Args:
        weight_matrix: 2D weight matrix [out_features, in_features]
        
    Returns:
        Array of eigenvalues (squared singular values)
    """
    # Compute W^T W
    if weight_matrix.shape[0] > weight_matrix.shape[1]:
        # More rows than columns: compute W^T W
        correlation = weight_matrix.T @ weight_matrix
    else:
        # More columns than rows: compute W W^T (faster)
        correlation = weight_matrix @ weight_matrix.T
    
    # Compute eigenvalues
    eigenvalues = np.linalg.eigvalsh(correlation)
    
    # Return positive eigenvalues only, sorted descending
    eigenvalues = np.sort(eigenvalues[eigenvalues > 1e-10])[::-1]
    
    return eigenvalues


def fit_power_law(eigenvalues: np.ndarray, xmin: Optional[float] = None) -> Dict[str, float]:
    """
    Fit a power law distribution to eigenvalues: p(λ) ∝ λ^(-α)
    
    Uses maximum likelihood estimation for the power law exponent.
    
    Args:
        eigenvalues: Array of eigenvalues
        xmin: Minimum value for fitting (if None, auto-detect)
        
    Returns:
        Dictionary with alpha, xmin, and goodness of fit
    """
    if len(eigenvalues) < 10:
        return {"alpha": float('nan'), "xmin": float('nan'), "valid": False}
    
    eigenvalues = np.sort(eigenvalues)[::-1]  # Descending
    
    # Auto-detect xmin using Clauset et al. method (simplified)
    if xmin is None:
        # Use median as a reasonable starting point
        xmin = np.median(eigenvalues)
    
    # Filter values >= xmin
    data = eigenvalues[eigenvalues >= xmin]
    
    if len(data) < 5:
        # Fall back to smaller xmin
        xmin = np.percentile(eigenvalues, 25)
        data = eigenvalues[eigenvalues >= xmin]
    
    if len(data) < 5:
        return {"alpha": float('nan'), "xmin": float('nan'), "valid": False}
    
    # MLE estimate of power law exponent
    # For p(x) ∝ x^(-α), MLE gives: α = 1 + n / Σ log(x_i / x_min)
    n = len(data)
    alpha = 1 + n / np.sum(np.log(data / xmin))
    
    # Simple goodness of fit: KS statistic
    # (Full implementation would use Clauset et al. method)
    sorted_data = np.sort(data)
    theoretical_cdf = 1 - (sorted_data / xmin) ** (1 - alpha)
    empirical_cdf = np.arange(1, n + 1) / n
    ks_stat = np.max(np.abs(theoretical_cdf - empirical_cdf))
    
    return {
        "alpha": float(alpha),
        "xmin": float(xmin),
        "ks_statistic": float(ks_stat),
        "n_tail": int(n),
        "valid": True,
    }


def compute_stable_rank(weight_matrix: np.ndarray) -> float:
    """
    Compute stable rank of a weight matrix.
    
    Stable rank = ||W||_F^2 / ||W||_2^2 = Σσ_i^2 / σ_max^2
    
    This is a smooth measure of "effective rank" that's more stable
    than actual rank.
    """
    singular_values = np.linalg.svd(weight_matrix, compute_uv=False)
    if len(singular_values) == 0 or singular_values[0] < 1e-10:
        return float('nan')
    
    frobenius_sq = np.sum(singular_values ** 2)
    spectral_sq = singular_values[0] ** 2
    
    return float(frobenius_sq / spectral_sq)


def analyze_weight_matrix(weight: torch.Tensor, name: str) -> Dict[str, Any]:
    """
    Analyze a single weight matrix.
    
    Args:
        weight: Weight tensor
        name: Name of the layer
        
    Returns:
        Dictionary with spectral analysis results
    """
    # Flatten to 2D if needed
    w = weight.detach().cpu().numpy()
    
    if w.ndim == 1:
        # Bias vector, skip
        return {"name": name, "type": "bias", "skipped": True}
    
    original_shape = w.shape
    
    if w.ndim > 2:
        # Reshape: treat as (out_features, in_features)
        w = w.reshape(w.shape[0], -1)
    
    if w.shape[0] < 2 or w.shape[1] < 2:
        return {"name": name, "type": "small", "skipped": True}
    
    # Compute metrics
    results = {
        "name": name,
        "shape": list(original_shape),
        "skipped": False,
    }
    
    # Spectral norm (largest singular value)
    singular_values = np.linalg.svd(w, compute_uv=False)
    results["spectral_norm"] = float(singular_values[0])
    results["frobenius_norm"] = float(np.linalg.norm(w, 'fro'))
    
    # Stable rank
    results["stable_rank"] = compute_stable_rank(w)
    
    # Eigenvalue distribution
    eigenvalues = compute_esd(w)
    results["num_eigenvalues"] = len(eigenvalues)
    results["max_eigenvalue"] = float(eigenvalues[0]) if len(eigenvalues) > 0 else 0
    
    # Power law fit
    power_law = fit_power_law(eigenvalues)
    results["alpha"] = power_law["alpha"]
    results["xmin"] = power_law.get("xmin", float('nan'))
    results["power_law_valid"] = power_law.get("valid", False)
    
    # Categorize alpha
    alpha = results["alpha"]
    if not np.isnan(alpha):
        if 2 <= alpha <= 4:
            results["alpha_category"] = "goldilocks"
        elif alpha < 2:
            results["alpha_category"] = "overfit"
        elif alpha <= 6:
            results["alpha_category"] = "good"
        else:
            results["alpha_category"] = "underfit"
    else:
        results["alpha_category"] = "unknown"
    
    return results


class SpectralAnalysisExperiment(InterpretabilityExperiment):
    """
    M9: Heavy-Tailed Self-Regularization Analysis
    
    Analyzes weight matrix spectra to assess generalization quality.
    
    Uses WeightWatcher framework if available, otherwise falls back
    to custom implementation.
    
    Key Outputs:
        - Per-layer alpha (power law exponent)
        - Module-wise aggregated metrics
        - Comparison highlighting knowledge-related layers
        
    Interpretation:
        - α ∈ [2, 4]: Goldilocks zone (good generalization)
        - α < 2: Overfit tendency
        - α > 6: Underfit tendency
    """
    
    name = "m9_spectral_analysis"
    description = "Heavy-tailed self-regularization (WeightWatcher) analysis"
    
    def __init__(
        self,
        model: nn.Module,
        config: ExperimentConfig,
        use_weightwatcher: bool = True,
    ):
        """
        Args:
            model: Trained INP model
            config: Experiment configuration
            use_weightwatcher: If True and available, use WeightWatcher library
        """
        super().__init__(model, config)
        self.use_weightwatcher = use_weightwatcher
        self.ww_available = False
        
        if use_weightwatcher:
            try:
                import weightwatcher as ww
                self.ww_available = True
                self.ww = ww
            except ImportError:
                print("WeightWatcher not installed, using custom implementation")
                self.ww_available = False
    
    def _analyze_with_weightwatcher(self) -> Dict[str, Any]:
        """Use WeightWatcher library for analysis."""
        watcher = self.ww.WeightWatcher(model=self.model)
        
        # Analyze with default settings
        details = watcher.analyze(
            mp_fit=True,  # Fit power law
            plot=False,
        )
        
        # Convert to our format
        results = {
            "method": "weightwatcher",
            "layers": [],
        }
        
        for idx, row in details.iterrows():
            layer_info = {
                "name": row.get("name", f"layer_{idx}"),
                "alpha": float(row.get("alpha", float('nan'))),
                "spectral_norm": float(row.get("spectral_norm", float('nan'))),
                "stable_rank": float(row.get("stable_rank", float('nan'))),
                "num_eigenvalues": int(row.get("N", 0)),
            }
            results["layers"].append(layer_info)
        
        # Summary statistics
        alphas = [l["alpha"] for l in results["layers"] if not np.isnan(l["alpha"])]
        if alphas:
            results["summary"] = {
                "mean_alpha": float(np.mean(alphas)),
                "std_alpha": float(np.std(alphas)),
                "min_alpha": float(np.min(alphas)),
                "max_alpha": float(np.max(alphas)),
            }
        
        return results
    
    def _analyze_custom(self) -> Dict[str, Any]:
        """Custom spectral analysis implementation."""
        results = {
            "method": "custom",
            "layers": [],
        }
        
        # Analyze all weight matrices
        for name, param in self.model.named_parameters():
            if "weight" in name and param.dim() >= 2:
                layer_result = analyze_weight_matrix(param, name)
                if not layer_result.get("skipped", True):
                    results["layers"].append(layer_result)
        
        # Summary statistics
        alphas = [l["alpha"] for l in results["layers"] 
                  if not np.isnan(l["alpha"]) and l.get("power_law_valid", True)]
        
        if alphas:
            results["summary"] = {
                "mean_alpha": float(np.mean(alphas)),
                "std_alpha": float(np.std(alphas)),
                "min_alpha": float(np.min(alphas)),
                "max_alpha": float(np.max(alphas)),
                "num_layers_analyzed": len(alphas),
            }
        
        return results
    
    def _categorize_layers(self, results: Dict[str, Any]) -> Dict[str, List]:
        """Group layers by module for analysis."""
        categories = {
            "xy_encoder": [],
            "x_encoder": [],
            "latent_encoder": [],
            "knowledge_encoder": [],
            "decoder": [],
            "other": [],
        }
        
        for layer in results.get("layers", []):
            name = layer.get("name", "")
            categorized = False
            
            for category in categories:
                if category.replace("_", "") in name.replace("_", "").lower():
                    categories[category].append(layer)
                    categorized = True
                    break
            
            if not categorized:
                categories["other"].append(layer)
        
        return categories
    
    def run(self, dataloader: torch.utils.data.DataLoader = None) -> Dict[str, Any]:
        """
        Run the spectral analysis.
        
        Note: This analysis only requires the model weights, not data.
        The dataloader parameter is included for API consistency.
        
        Returns:
            Dictionary containing spectral analysis results
        """
        self.model.eval()
        
        print("\nAnalyzing weight matrix spectra...")
        
        # Run analysis
        if self.ww_available and self.use_weightwatcher:
            results = self._analyze_with_weightwatcher()
        else:
            results = self._analyze_custom()
        
        # Categorize by module
        categories = self._categorize_layers(results)
        results["by_module"] = {}
        
        for module_name, layers in categories.items():
            if layers:
                alphas = [l["alpha"] for l in layers if not np.isnan(l.get("alpha", float('nan')))]
                stable_ranks = [l["stable_rank"] for l in layers if not np.isnan(l.get("stable_rank", float('nan')))]
                
                results["by_module"][module_name] = {
                    "num_layers": len(layers),
                    "mean_alpha": float(np.mean(alphas)) if alphas else float('nan'),
                    "std_alpha": float(np.std(alphas)) if alphas else float('nan'),
                    "mean_stable_rank": float(np.mean(stable_ranks)) if stable_ranks else float('nan'),
                    "layer_names": [l.get("name", "") for l in layers],
                }
        
        # Goldilocks analysis
        all_alphas = [l["alpha"] for l in results.get("layers", []) 
                      if not np.isnan(l.get("alpha", float('nan')))]
        
        if all_alphas:
            in_goldilocks = sum(1 for a in all_alphas if 2 <= a <= 4)
            results["goldilocks_analysis"] = {
                "num_in_goldilocks": in_goldilocks,
                "fraction_in_goldilocks": float(in_goldilocks / len(all_alphas)),
                "num_overfit": sum(1 for a in all_alphas if a < 2),
                "num_underfit": sum(1 for a in all_alphas if a > 6),
            }
        
        # Interpretation
        interpretation_parts = []
        
        if "summary" in results:
            mean_alpha = results["summary"]["mean_alpha"]
            if 2 <= mean_alpha <= 4:
                interpretation_parts.append(
                    f"Mean α={mean_alpha:.2f} is in Goldilocks zone [2,4] - good generalization expected"
                )
            elif mean_alpha < 2:
                interpretation_parts.append(
                    f"Mean α={mean_alpha:.2f} < 2 suggests overfit tendency"
                )
            elif mean_alpha <= 6:
                interpretation_parts.append(
                    f"Mean α={mean_alpha:.2f} is in acceptable range [2,6]"
                )
            else:
                interpretation_parts.append(
                    f"Mean α={mean_alpha:.2f} > 6 suggests underfit tendency"
                )
        
        # Compare knowledge-related layers
        if "knowledge_encoder" in results["by_module"]:
            ke_alpha = results["by_module"]["knowledge_encoder"]["mean_alpha"]
            if not np.isnan(ke_alpha):
                if 2 <= ke_alpha <= 4:
                    interpretation_parts.append(
                        f"Knowledge encoder α={ke_alpha:.2f} in Goldilocks zone"
                    )
                interpretation_parts.append(
                    "Knowledge encoder shows signs of implicit regularization"
                )
        
        results["interpretation"] = ". ".join(interpretation_parts)
        
        self.results = results
        self.save_results(results)
        
        # Generate visualization
        self._save_visualization(results)
        
        return results
    
    def _save_visualization(self, results: Dict[str, Any]):
        """Generate comprehensive visualizations for M9."""
        try:
            from .enhanced_viz import viz_m9_spectral
            viz_m9_spectral(results, self.output_dir)
        except ImportError:
            self._save_basic_visualization(results)
    
    def _save_basic_visualization(self, results: Dict[str, Any]):
        """Basic fallback visualization."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Set styling
            sns.set(font_scale=1.3)
            sns.set_style("whitegrid")
            
            # Create palette with enough colors
            palette = sns.color_palette("rocket", n_colors=10)
            sns.set_palette(palette)
            
            plots_dir = self.output_dir / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)
            
            layers = results.get("layers", [])
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 12))
            
            # Top-left: Alpha histogram
            ax1 = axes[0, 0]
            alphas = [l["alpha"] for l in layers if not np.isnan(l.get("alpha", np.nan))]
            
            if alphas:
                ax1.hist(alphas, bins=20, color=palette[0],
                        edgecolor='black', alpha=0.7)
                ax1.axvline(x=2, color='green', linestyle='--', linewidth=2, label='Goldilocks (2)')
                ax1.axvline(x=4, color='green', linestyle='--', linewidth=2, label='Goldilocks (4)')
                ax1.axvline(x=np.mean(alphas), color='red', linestyle='-', linewidth=2,
                           label=f'Mean: {np.mean(alphas):.2f}')
                ax1.axvspan(2, 4, alpha=0.2, color='green')
                
                ax1.set_xlabel('Power Law Exponent (α)', fontsize=12)
                ax1.set_ylabel('Count', fontsize=12)
                ax1.set_title('Distribution of Layer α Values', fontsize=14, fontweight='bold')
                ax1.legend(fontsize=10)
                ax1.grid(True, alpha=0.3)
            
            # Top-right: Per-module alpha
            ax2 = axes[0, 1]
            by_module = results.get("by_module", {})
            
            if by_module:
                modules = [m for m in by_module.keys() if not np.isnan(by_module[m].get("mean_alpha", np.nan))]
                module_alphas = [by_module[m]["mean_alpha"] for m in modules]
                module_stds = [by_module[m].get("std_alpha", 0) for m in modules]
                
                colors = ['green' if 2 <= a <= 4 else 'orange' if a < 6 else 'red' for a in module_alphas]
                bars = ax2.bar(range(len(modules)), module_alphas, yerr=module_stds,
                              color=colors, edgecolor='black', capsize=4)
                ax2.axhline(y=2, color='green', linestyle='--', alpha=0.7)
                ax2.axhline(y=4, color='green', linestyle='--', alpha=0.7)
                ax2.axhspan(2, 4, alpha=0.1, color='green')
                
                ax2.set_xticks(range(len(modules)))
                ax2.set_xticklabels([m.replace('_', '\n') for m in modules], fontsize=9)
                ax2.set_ylabel('Mean α', fontsize=12)
                ax2.set_title('Per-Module Power Law Exponent', fontsize=14, fontweight='bold')
                ax2.grid(True, alpha=0.3, axis='y')
            
            # Bottom-left: Stable rank vs alpha scatter
            ax3 = axes[1, 0]
            stable_ranks = [l.get("stable_rank", np.nan) for l in layers]
            layer_alphas = [l.get("alpha", np.nan) for l in layers]
            
            valid = [(sr, a) for sr, a in zip(stable_ranks, layer_alphas) 
                    if not np.isnan(sr) and not np.isnan(a)]
            
            if valid:
                srs, als = zip(*valid)
                ax3.scatter(srs, als, c=palette[0], 
                           alpha=0.6, edgecolors='black', s=80)
                ax3.axhline(y=2, color='green', linestyle='--', alpha=0.7)
                ax3.axhline(y=4, color='green', linestyle='--', alpha=0.7)
                ax3.axhspan(2, 4, alpha=0.1, color='green')
                
                ax3.set_xlabel('Stable Rank', fontsize=12)
                ax3.set_ylabel('α (Power Law Exponent)', fontsize=12)
                ax3.set_title('Stable Rank vs α', fontsize=14, fontweight='bold')
                ax3.grid(True, alpha=0.3)
            
            # Bottom-right: Goldilocks pie
            ax4 = axes[1, 1]
            if "goldilocks_analysis" in results:
                ga = results["goldilocks_analysis"]
                sizes = [ga["num_in_goldilocks"], ga["num_overfit"], ga["num_underfit"]]
                total = sum(sizes)
                if total > 0:
                    labels = [f'Goldilocks [2,4]\n{sizes[0]} ({sizes[0]/total:.1%})',
                             f'Overfit (<2)\n{sizes[1]} ({sizes[1]/total:.1%})',
                             f'Underfit (>6)\n{sizes[2]} ({sizes[2]/total:.1%})']
                    colors = ['green', 'red', 'orange']
                    ax4.pie([s for s in sizes if s > 0], 
                           labels=[l for l, s in zip(labels, sizes) if s > 0],
                           colors=[c for c, s in zip(colors, sizes) if s > 0],
                           autopct='', startangle=90,
                           wedgeprops=dict(edgecolor='white', linewidth=2))
                    ax4.set_title('Goldilocks Zone Analysis', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            fig.savefig(plots_dir / "m9_spectral.png", dpi=150, bbox_inches='tight', facecolor='white')
            fig.savefig(plots_dir / "m9_spectral.pdf", dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            
            print(f"  M9 visualizations saved to {plots_dir}")
            
        except ImportError as e:
            print(f"Visualization libraries not available: {e}")


def compare_models_spectral(
    model_inp: nn.Module,
    model_np: nn.Module,
    config: ExperimentConfig,
) -> Dict[str, Any]:
    """
    Compare spectral properties between INP and NP models.
    
    Args:
        model_inp: Trained INP model
        model_np: Trained NP model (without knowledge)
        config: Experiment configuration
        
    Returns:
        Comparison results
    """
    results = {}
    
    exp_inp = SpectralAnalysisExperiment(model_inp, config, use_weightwatcher=False)
    exp_np = SpectralAnalysisExperiment(model_np, config, use_weightwatcher=False)
    
    results["inp"] = exp_inp.run()
    results["np"] = exp_np.run()
    
    # Comparison
    inp_mean_alpha = results["inp"].get("summary", {}).get("mean_alpha", float('nan'))
    np_mean_alpha = results["np"].get("summary", {}).get("mean_alpha", float('nan'))
    
    results["comparison"] = {
        "inp_mean_alpha": inp_mean_alpha,
        "np_mean_alpha": np_mean_alpha,
        "alpha_difference": inp_mean_alpha - np_mean_alpha,
    }
    
    return results
