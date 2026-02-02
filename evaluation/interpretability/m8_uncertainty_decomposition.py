"""
M8: Epistemic Uncertainty Decomposition

This module decomposes predictive uncertainty into aleatoric and epistemic components
to quantify the "bit-value" of knowledge in INPs.

Uncertainty Decomposition:
    H[y|x] = E_q(z)[H[y|x,z]] + I(y,z)
    
    Total = Aleatoric + Epistemic

Where:
    - Total Uncertainty H[y|x]: Entropy of the predictive distribution
    - Aleatoric (noise): E_q(z)[H[y|x,z]] - irreducible noise, inherent to the data
    - Epistemic (model): I(y,z) - model ignorance, reducible with more data/knowledge

Key Hypothesis:
    - INP at N=0 (zero-shot): LOW epistemic uncertainty (knowledge provides prior)
    - NP at N=0: HIGH epistemic uncertainty (no prior information)
    - Both decrease as N increases, but INP starts lower and may converge faster

The difference in epistemic uncertainty at N=0 quantifies the "bit-value" of knowledge K.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from tqdm import tqdm
from pathlib import Path
import math

from .base import InterpretabilityExperiment, ExperimentConfig


def gaussian_entropy(scale: torch.Tensor) -> torch.Tensor:
    """
    Compute entropy of a Gaussian distribution.
    
    H[N(μ, σ²)] = 0.5 * log(2πeσ²) = 0.5 * (1 + log(2π) + 2*log(σ))
    
    Args:
        scale: Standard deviation tensor
        
    Returns:
        Entropy in nats
    """
    return 0.5 * (1.0 + math.log(2 * math.pi) + 2 * torch.log(scale + 1e-8))


def mixture_of_gaussians_entropy(
    means: torch.Tensor,
    scales: torch.Tensor,
    n_samples: int = 1000,
) -> torch.Tensor:
    """
    Estimate entropy of a mixture of Gaussians via Monte Carlo sampling.
    
    For a mixture p(y) = (1/K) Σ_k N(y; μ_k, σ_k²), we estimate:
    H[p] ≈ -E_p[log p(y)]
    
    Args:
        means: [K, ...] means of mixture components
        scales: [K, ...] standard deviations of mixture components
        n_samples: Number of Monte Carlo samples
        
    Returns:
        Estimated entropy
    """
    K = means.shape[0]
    
    # Sample from the mixture
    # First sample component indices uniformly
    component_idx = torch.randint(0, K, (n_samples,), device=means.device)
    
    # Sample from selected components
    selected_means = means[component_idx]  # [n_samples, ...]
    selected_scales = scales[component_idx]
    samples = torch.randn_like(selected_means) * selected_scales + selected_means
    
    # Compute log probability under the mixture
    # log p(y) = log( (1/K) Σ_k N(y; μ_k, σ_k²) )
    #          = log(1/K) + logsumexp_k( log N(y; μ_k, σ_k²) )
    
    # Expand samples for broadcasting: [n_samples, 1, ...]
    samples_expanded = samples.unsqueeze(1)
    
    # Compute log probability under each component
    # log N(y; μ, σ²) = -0.5 * log(2πσ²) - 0.5 * ((y-μ)/σ)²
    log_probs = (
        -0.5 * math.log(2 * math.pi)
        - torch.log(scales + 1e-8)
        - 0.5 * ((samples_expanded - means) / (scales + 1e-8)) ** 2
    )  # [n_samples, K, ...]
    
    # Sum over output dimensions if needed
    if log_probs.dim() > 2:
        log_probs = log_probs.sum(dim=tuple(range(2, log_probs.dim())))
    
    # logsumexp over mixture components, subtract log(K) for uniform mixing
    log_mixture_prob = torch.logsumexp(log_probs, dim=1) - math.log(K)
    
    # Entropy estimate: -E[log p]
    entropy = -log_mixture_prob.mean()
    
    return entropy


class UncertaintyDecompositionExperiment(InterpretabilityExperiment):
    """
    M8: Epistemic Uncertainty Decomposition
    
    Decomposes predictive uncertainty to quantify the value of knowledge.
    
    Procedure:
        1. For each context size N in [0, 1, 3, 5, 10, 20, 30]:
        2. Compute predictions with multiple z samples
        3. Total uncertainty: Entropy of mixture of Gaussians
        4. Aleatoric: Average entropy of individual Gaussians
        5. Epistemic: Total - Aleatoric
        6. Compare INP vs NP curves
    
    Key Metrics:
        - Zero-shot epistemic gap: Epistemic(NP, N=0) - Epistemic(INP, N=0)
        - "Bit-value" of knowledge: Reduction in uncertainty from K
        - Convergence rate: How fast does epistemic uncertainty decay with N?
    """
    
    name = "m8_uncertainty_decomposition"
    description = "Epistemic vs aleatoric uncertainty decomposition"
    
    def __init__(
        self,
        model: nn.Module,
        config: ExperimentConfig,
        context_sizes: List[int] = [0, 1, 3, 5, 10, 20, 30],
        num_z_samples: int = 50,
        num_tasks: int = 100,
        mc_samples: int = 500,
    ):
        """
        Args:
            model: Trained INP model
            config: Experiment configuration
            context_sizes: List of N values to evaluate
            num_z_samples: Number of z samples for uncertainty estimation
            num_tasks: Number of tasks to average over
            mc_samples: Monte Carlo samples for mixture entropy
        """
        super().__init__(model, config)
        self.context_sizes = context_sizes
        self.num_z_samples = num_z_samples
        self.num_tasks = num_tasks
        self.mc_samples = mc_samples
    
    def _disable_knowledge_dropout(self) -> Optional[float]:
        latent_encoder = getattr(self.model, "latent_encoder", None)
        if latent_encoder is None or not hasattr(latent_encoder, "knowledge_dropout"):
            return None
        original = float(latent_encoder.knowledge_dropout)
        latent_encoder.knowledge_dropout = 0.0
        return original
    
    def _compute_uncertainty_for_task(
        self,
        x_full: torch.Tensor,
        y_full: torch.Tensor,
        x_target: torch.Tensor,
        knowledge: Optional[torch.Tensor],
        num_context: int,
        context_idx: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Compute uncertainty decomposition for a single task.
        
        Args:
            x_full: All x values [1, total_points, x_dim]
            y_full: All y values [1, total_points, y_dim]
            x_target: Target x values for prediction [1, num_targets, x_dim]
            knowledge: Knowledge tensor (or None for NP)
            num_context: Number of context points to use
            
        Returns:
            Dictionary with total, aleatoric, epistemic uncertainty
        """
        # Select context points
        if num_context > 0:
            if context_idx is None:
                context_idx = torch.randperm(x_full.shape[1], device=x_full.device)[:num_context]
            else:
                if not torch.is_tensor(context_idx):
                    context_idx = torch.tensor(context_idx, device=x_full.device)
                else:
                    context_idx = context_idx.to(x_full.device)
            x_context = x_full[:, context_idx, :]
            y_context = y_full[:, context_idx, :]
        else:
            # Zero-shot: empty context
            x_context = x_full[:, :0, :]
            y_context = y_full[:, :0, :]
        
        # Encode
        x_context_enc = self.model.x_encoder(x_context)
        x_target_enc = self.model.x_encoder(x_target)
        R = self.model.encode_globally(x_context_enc, y_context, x_target_enc)
        
        # Handle empty context
        if num_context == 0:
            R = torch.zeros((1, 1, R.shape[-1]), device=R.device)
        
        # Get latent distribution
        q_z = self.model.infer_latent_dist(R, knowledge, num_context)
        
        # Sample multiple z values
        z_samples = q_z.rsample([self.num_z_samples])  # [num_z, batch, 1, hidden_dim]
        
        # Decode each z sample
        R_target = z_samples.expand(-1, -1, x_target.shape[1], -1)
        p_y_stats = self.model.decoder(x_target_enc, R_target)
        p_y_loc, p_y_scale = p_y_stats.split(self.model.config.output_dim, dim=-1)
        p_y_scale = 0.1 + 0.9 * F.softplus(p_y_scale)
        
        # Shapes: [num_z_samples, batch, num_targets, output_dim]
        means = p_y_loc.squeeze(1)  # [num_z, num_targets, output_dim]
        scales = p_y_scale.squeeze(1)
        
        # Aleatoric uncertainty: E_z[H[y|x,z]]
        # For Gaussian: H = 0.5 * log(2πeσ²)
        component_entropies = gaussian_entropy(scales)  # [num_z, num_targets, output_dim]
        aleatoric = component_entropies.mean(dim=0).sum().item()  # Sum over targets and dims
        
        # Total uncertainty: H[y|x] where p(y|x) is mixture of Gaussians
        # Estimate via Monte Carlo
        try:
            total_entropy = 0.0
            for t in range(x_target.shape[1]):
                means_t = means[:, t, :]  # [num_z, output_dim]
                scales_t = scales[:, t, :]
                
                # For 1D output, simpler computation
                if means_t.shape[-1] == 1:
                    means_t = means_t.squeeze(-1)
                    scales_t = scales_t.squeeze(-1)
                    
                    entropy_t = mixture_of_gaussians_entropy(
                        means_t, scales_t, n_samples=self.mc_samples
                    )
                    total_entropy += entropy_t.item()
                else:
                    # Multi-dimensional: sum entropies (assuming independence)
                    for d in range(means_t.shape[-1]):
                        entropy_td = mixture_of_gaussians_entropy(
                            means_t[:, d], scales_t[:, d], n_samples=self.mc_samples
                        )
                        total_entropy += entropy_td.item()
            
            total = total_entropy
        except Exception as e:
            # Fallback: use variance-based approximation
            # Total variance ≈ E[σ²] + Var[μ]
            total_var = scales.pow(2).mean(dim=0) + means.var(dim=0)
            total = gaussian_entropy(total_var.sqrt()).sum().item()
        
        # Epistemic = Total - Aleatoric
        epistemic = max(0, total - aleatoric)  # Ensure non-negative
        
        return {
            "total": total,
            "aleatoric": aleatoric,
            "epistemic": epistemic,
        }
    
    def run(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, Any]:
        """
        Run the uncertainty decomposition analysis.
        
        Computes uncertainty curves vs context size for INP and NP.
        
        Returns:
            Dictionary containing uncertainty curves and analysis
        """
        self.model.eval()
        
        original_dropout = self._disable_knowledge_dropout()
        
        results = {
            "config": {
                "context_sizes": self.context_sizes,
                "num_z_samples": self.num_z_samples,
                "num_tasks": self.num_tasks,
                "mc_samples": self.mc_samples,
                "aligned_context": True,
                "knowledge_dropout_original": original_dropout,
                "knowledge_dropout_set_to": 0.0 if original_dropout is not None else None,
            },
            "with_knowledge": {n: [] for n in self.context_sizes},
            "without_knowledge": {n: [] for n in self.context_sizes},
        }
        
        # Collect tasks
        tasks = []
        for batch in dataloader:
            context, target, knowledge, _ = batch
            x_full, y_full = target  # Use targets as full data
            
            # Process each item in batch
            for i in range(x_full.shape[0]):
                if len(tasks) >= self.num_tasks:
                    break
                    
                x_i = x_full[i:i+1].to(self.device)
                y_i = y_full[i:i+1].to(self.device)
                
                if isinstance(knowledge, torch.Tensor):
                    k_i = knowledge[i:i+1].to(self.device)
                elif isinstance(knowledge, list):
                    item = knowledge[i]
                    if isinstance(item, str):
                        k_i = [item]
                    elif isinstance(item, torch.Tensor):
                        k_i = item.unsqueeze(0).to(self.device)
                    else:
                        try:
                            k_i = torch.tensor(item).unsqueeze(0).to(self.device)
                        except Exception:
                            k_i = None
                else:
                    k_i = None
                
                tasks.append((x_i, y_i, k_i))
            
            if len(tasks) >= self.num_tasks:
                break
        
        print(f"\nAnalyzing uncertainty over {len(tasks)} tasks...")
        
        try:
            with torch.no_grad():
                for task_idx, (x_full, y_full, knowledge) in enumerate(tqdm(tasks)):
                    # Use subset as targets
                    num_points = x_full.shape[1]
                    target_idx = torch.randperm(num_points, device=x_full.device)[:min(20, num_points)]
                    x_target = x_full[:, target_idx, :]
                    
                    for n in self.context_sizes:
                        if n > num_points - 5:  # Need some points for targets
                            continue
                        
                        context_idx = None
                        if n > 0:
                            context_idx = torch.randperm(num_points, device=x_full.device)[:n]
                        
                        # With knowledge (INP)
                        unc_k = self._compute_uncertainty_for_task(
                            x_full, y_full, x_target, knowledge, n, context_idx=context_idx
                        )
                        results["with_knowledge"][n].append(unc_k)
                        
                        # Without knowledge (NP ablation)
                        unc_no_k = self._compute_uncertainty_for_task(
                            x_full, y_full, x_target, None, n, context_idx=context_idx
                        )
                        results["without_knowledge"][n].append(unc_no_k)
        finally:
            if original_dropout is not None:
                self.model.latent_encoder.knowledge_dropout = original_dropout
        
        # Aggregate results
        def aggregate_uncertainties(data: Dict[int, List]) -> Dict[str, Dict]:
            aggregated = {}
            for n, uncertainties in data.items():
                if not uncertainties:
                    continue
                    
                totals = [u["total"] for u in uncertainties]
                aleatorics = [u["aleatoric"] for u in uncertainties]
                epistemics = [u["epistemic"] for u in uncertainties]
                
                aggregated[n] = {
                    "total_mean": float(np.mean(totals)),
                    "total_std": float(np.std(totals)),
                    "aleatoric_mean": float(np.mean(aleatorics)),
                    "aleatoric_std": float(np.std(aleatorics)),
                    "epistemic_mean": float(np.mean(epistemics)),
                    "epistemic_std": float(np.std(epistemics)),
                }
            return aggregated
        
        results["with_knowledge_aggregated"] = aggregate_uncertainties(results["with_knowledge"])
        results["without_knowledge_aggregated"] = aggregate_uncertainties(results["without_knowledge"])
        
        # Clean up raw data for JSON
        results["with_knowledge"] = {str(k): len(v) for k, v in results["with_knowledge"].items()}
        results["without_knowledge"] = {str(k): len(v) for k, v in results["without_knowledge"].items()}
        
        # Analysis
        agg_k = results["with_knowledge_aggregated"]
        agg_no_k = results["without_knowledge_aggregated"]
        
        # Zero-shot comparison (N=0)
        if 0 in agg_k and 0 in agg_no_k:
            epistemic_k_0 = agg_k[0]["epistemic_mean"]
            epistemic_no_k_0 = agg_no_k[0]["epistemic_mean"]
            
            results["zero_shot_analysis"] = {
                "epistemic_with_k": epistemic_k_0,
                "epistemic_without_k": epistemic_no_k_0,
                "epistemic_reduction": epistemic_no_k_0 - epistemic_k_0,
                "bit_value_of_knowledge": (epistemic_no_k_0 - epistemic_k_0) / math.log(2),  # Convert nats to bits
            }
        
        # Convergence analysis
        n_values = sorted([n for n in agg_k.keys() if n > 0])
        if len(n_values) >= 2:
            # Fit exponential decay: epistemic(N) ≈ A * exp(-N/τ) + B
            # Simpler: just compare slopes
            epistemic_k = [agg_k[n]["epistemic_mean"] for n in n_values]
            epistemic_no_k = [agg_no_k[n]["epistemic_mean"] for n in n_values if n in agg_no_k]
            
            if len(epistemic_k) >= 2:
                decay_rate_k = (epistemic_k[0] - epistemic_k[-1]) / (n_values[-1] - n_values[0])
                results["convergence_rate_with_k"] = float(decay_rate_k)
            
            if len(epistemic_no_k) >= 2:
                n_no_k = [n for n in n_values if n in agg_no_k]
                decay_rate_no_k = (epistemic_no_k[0] - epistemic_no_k[-1]) / (n_no_k[-1] - n_no_k[0])
                results["convergence_rate_without_k"] = float(decay_rate_no_k)
        
        # Interpretation
        interpretation_parts = []
        
        if "zero_shot_analysis" in results:
            zs = results["zero_shot_analysis"]
            if zs["epistemic_reduction"] > 0:
                interpretation_parts.append(
                    f"Knowledge reduces zero-shot epistemic uncertainty by {zs['epistemic_reduction']:.2f} nats "
                    f"({zs['bit_value_of_knowledge']:.2f} bits)"
                )
            else:
                interpretation_parts.append(
                    "Knowledge does not reduce zero-shot epistemic uncertainty"
                )
        
        # Check if aleatoric is consistent (should be similar regardless of knowledge)
        if 5 in agg_k and 5 in agg_no_k:
            aleatoric_k = agg_k[5]["aleatoric_mean"]
            aleatoric_no_k = agg_no_k[5]["aleatoric_mean"]
            aleatoric_diff = abs(aleatoric_k - aleatoric_no_k)
            if aleatoric_diff < 0.5:
                interpretation_parts.append(
                    "Aleatoric uncertainty is consistent (sanity check passed)"
                )
            else:
                interpretation_parts.append(
                    f"WARNING: Aleatoric uncertainty differs by {aleatoric_diff:.2f}"
                )
        
        results["interpretation"] = ". ".join(interpretation_parts)
        
        self.results = results
        self.save_results(results)
        
        # Generate visualization
        self._save_visualization(results)
        
        return results
    
    def _save_visualization(self, results: Dict[str, Any]):
        """Generate comprehensive visualizations for M8."""
        try:
            from .enhanced_viz import viz_m8_uncertainty
            viz_m8_uncertainty(results, self.output_dir)
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
            
            agg_k = results.get("with_knowledge_aggregated", {})
            agg_no_k = results.get("without_knowledge_aggregated", {})
            
            n_values_k = sorted([int(k) for k in agg_k.keys()])
            n_values_no_k = sorted([int(k) for k in agg_no_k.keys()])
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 12))
            
            # Top-left: Epistemic uncertainty vs N
            ax1 = axes[0, 0]
            epistemic_k = [agg_k[n]["epistemic_mean"] for n in n_values_k]
            epistemic_k_std = [agg_k[n]["epistemic_std"] for n in n_values_k]
            epistemic_no_k = [agg_no_k.get(n, {"epistemic_mean": 0})["epistemic_mean"] for n in n_values_no_k]
            epistemic_no_k_std = [agg_no_k.get(n, {"epistemic_std": 0})["epistemic_std"] for n in n_values_no_k]
            
            ax1.plot(n_values_k, epistemic_k, 'o-', color=palette[0],
                    linewidth=3, markersize=8, label='With Knowledge (INP)')
            ax1.fill_between(n_values_k, 
                            np.array(epistemic_k) - np.array(epistemic_k_std),
                            np.array(epistemic_k) + np.array(epistemic_k_std),
                            alpha=0.2, color=palette[0])
            
            ax1.plot(n_values_no_k, epistemic_no_k, 's--', color=palette[5],
                    linewidth=3, markersize=8, label='Without Knowledge (NP)')
            ax1.fill_between(n_values_no_k,
                            np.array(epistemic_no_k) - np.array(epistemic_no_k_std),
                            np.array(epistemic_no_k) + np.array(epistemic_no_k_std),
                            alpha=0.2, color=palette[5])
            
            ax1.set_xlabel('Number of Context Points (N)', fontsize=12)
            ax1.set_ylabel('Epistemic Uncertainty (nats)', fontsize=12)
            ax1.set_title('Epistemic Uncertainty vs Context Size', fontsize=14, fontweight='bold')
            ax1.legend(fontsize=11)
            ax1.grid(True, alpha=0.3)
            
            # Top-right: Stacked bars
            ax2 = axes[0, 1]
            n_example = n_values_k[len(n_values_k)//2] if n_values_k else 5
            
            if n_example in agg_k:
                categories = ['With K\n(INP)', 'Without K\n(NP)']
                aleatoric = [agg_k[n_example]["aleatoric_mean"], 
                            agg_no_k.get(n_example, {}).get("aleatoric_mean", 0)]
                epistemic = [agg_k[n_example]["epistemic_mean"],
                            agg_no_k.get(n_example, {}).get("epistemic_mean", 0)]
                
                ax2.bar(categories, aleatoric, label='Aleatoric', 
                       color=palette[2], edgecolor='black')
                ax2.bar(categories, epistemic, bottom=aleatoric, label='Epistemic',
                       color=palette[0], edgecolor='black')
                
                ax2.set_ylabel('Uncertainty (nats)', fontsize=12)
                ax2.set_title(f'Uncertainty Decomposition (N={n_example})', fontsize=14, fontweight='bold')
                ax2.legend(fontsize=11)
                ax2.grid(True, alpha=0.3, axis='y')
            
            # Bottom-left: Bit-value of knowledge
            ax3 = axes[1, 0]
            if "zero_shot_analysis" in results:
                zs = results["zero_shot_analysis"]
                metrics = ['Epistemic\nReduction\n(nats)', 'Bit Value\nof Knowledge']
                values = [zs["epistemic_reduction"], zs["bit_value_of_knowledge"]]
                colors = ['green' if v > 0 else 'red' for v in values]
                
                bars = ax3.bar(metrics, values, color=colors, edgecolor='black')
                ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
                ax3.set_ylabel('Value', fontsize=12)
                ax3.set_title('Zero-Shot: Value of Knowledge', fontsize=14, fontweight='bold')
                ax3.grid(True, alpha=0.3, axis='y')
                
                for bar, val in zip(bars, values):
                    ax3.annotate(f'{val:.3f}',
                                xy=(bar.get_x() + bar.get_width() / 2, val),
                                xytext=(0, 5), textcoords="offset points",
                                ha='center', fontsize=12, fontweight='bold')
            
            # Bottom-right: Heatmap over N
            ax4 = axes[1, 1]
            n_common = sorted(set(n_values_k) & set(n_values_no_k))
            
            if n_common:
                heatmap_data = []
                for n in n_common:
                    heatmap_data.append([
                        agg_k[n]["epistemic_mean"],
                        agg_no_k.get(n, {}).get("epistemic_mean", 0)
                    ])
                
                heatmap_arr = np.array(heatmap_data).T
                sns.heatmap(heatmap_arr, ax=ax4, cmap='rocket_r',
                           xticklabels=n_common, yticklabels=['With K', 'Without K'],
                           annot=True, fmt='.2f', cbar_kws={'label': 'Epistemic Unc.'},
                           linewidths=1, linecolor='white')
                ax4.set_xlabel('Context Size (N)', fontsize=12)
                ax4.set_title('Epistemic Uncertainty Heatmap', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            fig.savefig(plots_dir / "m8_uncertainty.png", dpi=150, bbox_inches='tight', facecolor='white')
            fig.savefig(plots_dir / "m8_uncertainty.pdf", dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            
            print(f"  M8 visualizations saved to {plots_dir}")
            
        except ImportError as e:
            print(f"Visualization libraries not available: {e}")
