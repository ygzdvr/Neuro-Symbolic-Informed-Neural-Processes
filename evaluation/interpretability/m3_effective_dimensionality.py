"""
M3: Effective Dimensionality Analysis

This module analyzes the intrinsic dimensionality of the INP latent space using SVD.

Hypothesis (Manifold Constraint): Knowledge K forces latent codes z to reside on a
lower-dimensional manifold that maps directly to physical parameters. For sinusoids
defined by (a, b, c), the intrinsic dimensionality should be ~3, not the full 128D.

Key metric: Effective Dimensionality (ED) via participation ratio:
    ED = (Σσᵢ)² / Σσᵢ²

Expected behavior:
- INP: ED close to true generative parameters (e.g., 3 for sinusoids)
- NP: Higher ED (modeling noise in spurious dimensions)
- INP reaches 95% explained variance with fewer components
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from tqdm import tqdm
from pathlib import Path

from .base import InterpretabilityExperiment, ExperimentConfig


class EffectiveDimensionalityExperiment(InterpretabilityExperiment):
    """
    M3: Effective Dimensionality Analysis
    
    Collects latent codes from the model and analyzes their intrinsic
    dimensionality using SVD and participation ratio.
    
    Key metrics:
        - Effective Dimensionality (ED): Participation ratio of singular values
        - Explained variance curve: Cumulative variance by component
        - Components to 95%: Number of components for 95% variance
        
    Comparison:
        - With knowledge (INP): Should have lower ED
        - Baseline (NP or ablation): Often higher ED
    """
    
    name = "m3_effective_dimensionality"
    description = "SVD-based latent space dimensionality analysis"
    
    def __init__(
        self,
        model: nn.Module,
        config: ExperimentConfig,
        variance_threshold: float = 0.95,
        max_samples: int = 5000,
        baseline_model: Optional[nn.Module] = None,
        baseline_label: Optional[str] = None,
        baseline_info: Optional[Dict[str, Any]] = None,
        max_visual_samples: int = 1000,
        compute_tsne: bool = True,
    ):
        """
        Args:
            model: Trained INP model
            config: Experiment configuration
            variance_threshold: Threshold for "components to X%" metric
            max_samples: Maximum number of latent samples to collect
            baseline_model: Optional NP baseline model (separate checkpoint)
            baseline_label: Label for baseline in plots
            baseline_info: Metadata describing the baseline source
            max_visual_samples: Max samples per condition for projections
            compute_tsne: Whether to compute t-SNE projection if available
        """
        super().__init__(model, config)
        self.variance_threshold = variance_threshold
        self.max_samples = max_samples
        self.baseline_model = baseline_model
        if baseline_label is None:
            self.baseline_label = "NP Baseline" if baseline_model is not None else "Without Knowledge (Ablation)"
        else:
            self.baseline_label = baseline_label
        self.baseline_info = baseline_info or {}
        self.max_visual_samples = max_visual_samples
        self.compute_tsne = compute_tsne

        if self.baseline_model is not None:
            self.baseline_model.to(self.device)
            self.baseline_model.eval()
    
    def _collect_latents(
        self,
        dataloader: torch.utils.data.DataLoader,
        use_knowledge: bool = True,
        model: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """
        Collect latent means from the model.
        
        Args:
            dataloader: Data source
            use_knowledge: Whether to use knowledge conditioning
            
        Returns:
            Tensor of latent means [N_samples, latent_dim]
        """
        latents = []
        total_samples = 0
        
        model = model or self.model
        model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Collecting latents"):
                if total_samples >= self.max_samples:
                    break
                
                context, target, knowledge, _ = batch
                x_context, y_context = context
                x_target, y_target = target
                
                x_context = x_context.to(self.device)
                y_context = y_context.to(self.device)
                x_target = x_target.to(self.device)
                
                if isinstance(knowledge, torch.Tensor):
                    knowledge = knowledge.to(self.device)
                
                # Encode context
                x_enc = model.x_encoder(x_context)
                x_target_enc = model.x_encoder(x_target)
                R = model.encode_globally(x_enc, y_context, x_target_enc)
                
                # Get latent distribution
                if use_knowledge:
                    q_z = model.infer_latent_dist(R, knowledge, x_context.shape[1])
                else:
                    q_z = model.infer_latent_dist(R, None, x_context.shape[1])
                
                # Extract latent mean (deterministic core)
                mu_z = q_z.base_dist.loc.squeeze(1)  # [batch, latent_dim]
                latents.append(mu_z.cpu())
                
                total_samples += mu_z.shape[0]
        
        # Concatenate all latents
        Z = torch.cat(latents, dim=0)[:self.max_samples]
        return Z
    
    def _compute_svd_analysis(
        self,
        Z: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Perform SVD analysis on latent codes.
        
        Args:
            Z: Latent codes [N_samples, latent_dim]
            
        Returns:
            Dictionary with SVD analysis results
        """
        # Center the latent codes
        Z_centered = Z - Z.mean(dim=0, keepdim=True)
        
        # SVD
        try:
            U, S, V = torch.svd(Z_centered)
        except RuntimeError:
            # Fall back to numpy for numerical stability
            Z_np = Z_centered.numpy()
            U, S, V = np.linalg.svd(Z_np, full_matrices=False)
            S = torch.from_numpy(S)
        
        singular_values = S.numpy()
        
        # Explained variance
        variance = singular_values ** 2
        total_variance = variance.sum()
        explained_variance_ratio = variance / total_variance
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        # Effective Dimensionality (Participation Ratio)
        # ED = (Σσᵢ)² / Σσᵢ²  (PR on singular values)
        ed = (singular_values.sum() ** 2) / (singular_values ** 2).sum()

        # Alternative ED using variance eigenvalues (σᵢ²)
        ed_variance = (variance.sum() ** 2) / (variance ** 2).sum()
        
        # Components needed for threshold
        components_to_threshold = np.searchsorted(cumulative_variance, self.variance_threshold) + 1
        
        # Alternative ED metrics
        # Shannon entropy-based ED
        p = explained_variance_ratio + 1e-10  # Add small epsilon for stability
        entropy = -np.sum(p * np.log(p))
        ed_entropy = np.exp(entropy)
        
        return {
            "singular_values": singular_values.tolist(),
            "explained_variance_ratio": explained_variance_ratio.tolist(),
            "cumulative_variance": cumulative_variance.tolist(),
            "effective_dimensionality": float(ed),
            "effective_dimensionality_variance": float(ed_variance),
            "effective_dimensionality_entropy": float(ed_entropy),
            "components_to_threshold": int(components_to_threshold),
            "threshold": self.variance_threshold,
            "total_variance": float(total_variance),
            "latent_dim": Z.shape[1],
            "num_samples": Z.shape[0],
        }
    
    def run(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, Any]:
        """
        Run the Effective Dimensionality analysis.
        
        Collects latents with and without knowledge, performs SVD analysis
        on both, and compares the results.
        
        Returns:
            Dictionary containing:
                - with_knowledge: SVD analysis for INP
                - without_knowledge: SVD analysis for NP baseline
                - comparison: ED ratio and interpretation
        """
        baseline_model = self.baseline_model or self.model
        baseline_type = "np_checkpoint" if self.baseline_model is not None else "ablation_same_model"

        results = {
            "config": {
                "variance_threshold": self.variance_threshold,
                "max_samples": self.max_samples,
                "baseline_type": baseline_type,
                "baseline_label": self.baseline_label,
                "baseline_info": self.baseline_info,
            }
        }

        def _disable_knowledge_dropout(model: nn.Module) -> Optional[float]:
            if not hasattr(model, "latent_encoder"):
                return None
            if not hasattr(model.latent_encoder, "knowledge_dropout"):
                return None
            prev = model.latent_encoder.knowledge_dropout
            model.latent_encoder.knowledge_dropout = 0.0
            return prev

        prev_dropout_main = _disable_knowledge_dropout(self.model)
        prev_dropout_baseline = None
        if baseline_model is not self.model:
            prev_dropout_baseline = _disable_knowledge_dropout(baseline_model)
        
        try:
            # Collect and analyze latents WITH knowledge
            print("\nCollecting latents WITH knowledge (INP)...")
            Z_with_k = self._collect_latents(dataloader, use_knowledge=True, model=self.model)
            print(f"Collected {Z_with_k.shape[0]} samples, dim={Z_with_k.shape[1]}")
            
            results["with_knowledge"] = self._compute_svd_analysis(Z_with_k)
            
            # Collect and analyze latents WITHOUT knowledge (baseline)
            print(f"\nCollecting latents WITHOUT knowledge ({self.baseline_label})...")
            Z_without_k = self._collect_latents(dataloader, use_knowledge=False, model=baseline_model)
            print(f"Collected {Z_without_k.shape[0]} samples, dim={Z_without_k.shape[1]}")
            
            results["without_knowledge"] = self._compute_svd_analysis(Z_without_k)
        finally:
            if prev_dropout_main is not None:
                self.model.latent_encoder.knowledge_dropout = prev_dropout_main
            if prev_dropout_baseline is not None and baseline_model is not self.model:
                baseline_model.latent_encoder.knowledge_dropout = prev_dropout_baseline
        
        # Comparison
        ed_with_k = results["with_knowledge"]["effective_dimensionality"]
        ed_without_k = results["without_knowledge"]["effective_dimensionality"]
        
        comp_threshold_with_k = results["with_knowledge"]["components_to_threshold"]
        comp_threshold_without_k = results["without_knowledge"]["components_to_threshold"]
        
        results["comparison"] = {
            "ed_with_knowledge": ed_with_k,
            "ed_without_knowledge": ed_without_k,
            "ed_ratio": ed_with_k / max(ed_without_k, 1e-8),
            "ed_reduction": ed_without_k - ed_with_k,
            "components_with_knowledge": comp_threshold_with_k,
            "components_without_knowledge": comp_threshold_without_k,
            "components_reduction": comp_threshold_without_k - comp_threshold_with_k,
        }
        
        # Interpretation
        ed_ratio = results["comparison"]["ed_ratio"]
        baseline_name = self.baseline_label
        if ed_ratio < 0.5:
            results["interpretation"] = (
                f"Strong manifold constraint: INP uses {ed_ratio:.2%} of {baseline_name} dimensionality. "
                f"ED={ed_with_k:.1f} vs {ed_without_k:.1f}"
            )
        elif ed_ratio < 0.8:
            results["interpretation"] = (
                f"Moderate manifold constraint: INP uses {ed_ratio:.2%} of {baseline_name} dimensionality. "
                f"ED={ed_with_k:.1f} vs {ed_without_k:.1f}"
            )
        else:
            results["interpretation"] = (
                f"Weak manifold constraint: Similar dimensionality. "
                f"ED={ed_with_k:.1f} vs {ed_without_k:.1f}"
            )
        
        self.results = results
        self.save_results(results)
        
        # Generate visualization
        self._save_visualization(results, Z_with_k=Z_with_k, Z_without_k=Z_without_k)
        
        return results
    
    def _save_visualization(
        self,
        results: Dict[str, Any],
        Z_with_k: Optional[torch.Tensor] = None,
        Z_without_k: Optional[torch.Tensor] = None,
    ):
        """Generate comprehensive visualizations for M3."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Set styling
            sns.set(font_scale=1.3)
            sns.set_style("whitegrid")
            
            # Create a fixed palette with enough colors
            palette = sns.color_palette("rocket", n_colors=10)
            sns.set_palette(palette)
            
            plots_dir = self.output_dir / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)
            
            sv_with_k = results["with_knowledge"]["singular_values"]
            sv_without_k = results["without_knowledge"]["singular_values"]
            cum_var_with_k = results["with_knowledge"]["cumulative_variance"]
            cum_var_without_k = results["without_knowledge"]["cumulative_variance"]
            with_label = "With Knowledge (INP)"
            without_label = self.baseline_label
            
            # Ensure we have enough data
            if len(sv_with_k) == 0 or len(sv_without_k) == 0:
                print("  No singular values to visualize")
                return
            
            max_components = min(50, len(sv_with_k), len(sv_without_k))
            
            # ================================================================
            # PLOT 1: Main Analysis (2x2 grid)
            # ================================================================
            fig, axes = plt.subplots(2, 2, figsize=(14, 12))
            
            # Top-left: Singular value spectrum (log scale)
            ax1 = axes[0, 0]
            ax1.semilogy(range(1, max_components + 1), sv_with_k[:max_components], 
                        color=palette[0], linewidth=2.5,
                        marker='o', markersize=5, label=with_label)
            ax1.semilogy(range(1, max_components + 1), sv_without_k[:max_components], 
                        color=palette[5], linewidth=2.5,
                        marker='s', markersize=5, linestyle='--', label=without_label)
            ax1.set_xlabel("Component Index", fontsize=12)
            ax1.set_ylabel("Singular Value (log scale)", fontsize=12)
            ax1.set_title("Singular Value Spectrum", fontsize=14, fontweight='bold')
            ax1.legend(fontsize=11)
            ax1.grid(True, alpha=0.3)
            
            # Top-right: Explained variance ratio (individual)
            ax2 = axes[0, 1]
            ev_with_k = results["with_knowledge"]["explained_variance_ratio"]
            ev_without_k = results["without_knowledge"]["explained_variance_ratio"]
            
            ax2.bar(np.arange(max_components) - 0.2, ev_with_k[:max_components], 0.4,
                   color=palette[0], alpha=0.7, label=with_label)
            ax2.bar(np.arange(max_components) + 0.2, ev_without_k[:max_components], 0.4,
                   color=palette[5], alpha=0.7, label=without_label)
            ax2.set_xlabel("Component Index", fontsize=12)
            ax2.set_ylabel("Explained Variance Ratio", fontsize=12)
            ax2.set_title("Per-Component Explained Variance", fontsize=14, fontweight='bold')
            ax2.legend(fontsize=10)
            ax2.set_xlim(-0.5, min(20, max_components))
            ax2.grid(True, alpha=0.3, axis='y')
            
            # Bottom-left: Cumulative explained variance
            ax3 = axes[1, 0]
            ax3.plot(range(1, len(cum_var_with_k) + 1), cum_var_with_k, 
                    color=palette[0], linewidth=3, label=with_label)
            ax3.plot(range(1, len(cum_var_without_k) + 1), cum_var_without_k, 
                    color=palette[5], linewidth=3, linestyle='--', 
                    label=without_label)
            ax3.axhline(y=self.variance_threshold, color='green', linestyle=':', 
                       linewidth=2, label=f'{self.variance_threshold:.0%} threshold')
            
            comp_k = results["with_knowledge"]["components_to_threshold"]
            comp_no_k = results["without_knowledge"]["components_to_threshold"]
            ax3.axvline(x=comp_k, color=palette[0], linestyle=':', alpha=0.7)
            ax3.axvline(x=comp_no_k, color=palette[5], linestyle=':', alpha=0.7)
            
            ax3.scatter([comp_k], [self.variance_threshold], s=100, 
                       color=palette[0], zorder=5, marker='*')
            ax3.scatter([comp_no_k], [self.variance_threshold], s=100,
                       color=palette[5], zorder=5, marker='*')
            
            ax3.set_xlabel("Number of Components", fontsize=12)
            ax3.set_ylabel("Cumulative Explained Variance", fontsize=12)
            ax3.set_title("Cumulative Explained Variance Curve", fontsize=14, fontweight='bold')
            ax3.legend(fontsize=10, loc='lower right')
            ax3.grid(True, alpha=0.3)
            ax3.set_xlim(0, max_components)
            ax3.set_ylim(0, 1.05)
            
            # Bottom-right: Effective Dimensionality comparison
            ax4 = axes[1, 1]
            ed_pr = [results["with_knowledge"]["effective_dimensionality"],
                    results["without_knowledge"]["effective_dimensionality"]]
            ed_var = [results["with_knowledge"]["effective_dimensionality_variance"],
                     results["without_knowledge"]["effective_dimensionality_variance"]]
            comp_to_thresh = [comp_k, comp_no_k]
            
            x = np.arange(2)
            width = 0.25
            
            bars1 = ax4.bar(x - width, ed_pr, width, label='PR (Singular Values)', 
                           color=palette[0], edgecolor='black')
            bars2 = ax4.bar(x, ed_var, width, label='PR (Variance)',
                           color=palette[3], edgecolor='black')
            bars3 = ax4.bar(x + width, comp_to_thresh, width, label=f'Components to {self.variance_threshold:.0%}',
                           color=palette[6], edgecolor='black')
            
            ax4.set_ylabel('Dimensionality', fontsize=12)
            ax4.set_title('Effective Dimensionality Metrics', fontsize=14, fontweight='bold')
            ax4.set_xticks(x)
            ax4.set_xticklabels([with_label.replace(" ", "\n"), without_label.replace(" ", "\n")])
            ax4.legend(fontsize=9, loc='upper right')
            ax4.grid(True, alpha=0.3, axis='y')
            
            for bars in [bars1, bars2, bars3]:
                for bar in bars:
                    height = bar.get_height()
                    ax4.annotate(f'{height:.1f}',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3), textcoords="offset points",
                                ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            plt.tight_layout()
            fig.savefig(plots_dir / "m3_dimensionality.png", dpi=150, bbox_inches='tight', facecolor='white')
            fig.savefig(plots_dir / "m3_dimensionality.pdf", dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            
            # ================================================================
            # PLOT 2: Singular Value Heatmap
            # ================================================================
            fig, ax = plt.subplots(figsize=(12, 4))
            
            sv_matrix = np.vstack([
                np.array(sv_with_k[:max_components]),
                np.array(sv_without_k[:max_components])
            ])
            sv_matrix_log = np.log10(sv_matrix + 1e-10)
            
            sns.heatmap(
                sv_matrix_log,
                ax=ax,
                cmap='rocket',
                xticklabels=5,
                yticklabels=[with_label, without_label],
                cbar_kws={'label': 'log10(Singular Value)'},
            )
            ax.set_xlabel("Component Index", fontsize=12)
            ax.set_title("Singular Value Spectrum Heatmap", fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            fig.savefig(plots_dir / "m3_sv_heatmap.png", dpi=150, bbox_inches='tight', facecolor='white')
            fig.savefig(plots_dir / "m3_sv_heatmap.pdf", dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            
            # ================================================================
            # PLOT 3: Latent Projections (PCA / t-SNE)
            # ================================================================
            if Z_with_k is not None and Z_without_k is not None:
                try:
                    rng = np.random.RandomState(self.config.seed)
                    Zk = Z_with_k.detach().cpu().numpy()
                    Zb = Z_without_k.detach().cpu().numpy()
                    num_vis = min(self.max_visual_samples, Zk.shape[0], Zb.shape[0])
                    if num_vis < 2:
                        raise ValueError("Not enough samples for projection plot.")
                    
                    idx_k = rng.choice(Zk.shape[0], size=num_vis, replace=False)
                    idx_b = rng.choice(Zb.shape[0], size=num_vis, replace=False)
                    Zk_vis = Zk[idx_k]
                    Zb_vis = Zb[idx_b]
                    
                    Z_all = np.vstack([Zk_vis, Zb_vis])
                    Z_mean = Z_all.mean(axis=0, keepdims=True)
                    Z_centered = Z_all - Z_mean
                    U, S, Vt = np.linalg.svd(Z_centered, full_matrices=False)
                    components = Vt[:2].T
                    Z_pca = Z_centered @ components
                    Zk_pca = Z_pca[:num_vis]
                    Zb_pca = Z_pca[num_vis:]
                    
                    pca_var = (S ** 2) / max(Z_centered.shape[0] - 1, 1)
                    pca_var_total = pca_var.sum()
                    if pca_var_total > 0:
                        pca_var_ratio = pca_var / pca_var_total
                    else:
                        pca_var_ratio = np.zeros_like(pca_var)
                    
                    tsne_result = None
                    if self.compute_tsne:
                        try:
                            from sklearn.manifold import TSNE
                            tsne_total = min(1000, num_vis)
                            if tsne_total >= 10:
                                idx_k_tsne = rng.choice(Zk_vis.shape[0], size=tsne_total, replace=False)
                                idx_b_tsne = rng.choice(Zb_vis.shape[0], size=tsne_total, replace=False)
                                Zk_tsne = Zk_vis[idx_k_tsne]
                                Zb_tsne = Zb_vis[idx_b_tsne]
                                Z_tsne_all = np.vstack([Zk_tsne, Zb_tsne])
                                perplexity = min(30, max(5, (Z_tsne_all.shape[0] - 1) // 3))
                                tsne = TSNE(n_components=2, perplexity=perplexity, init="pca",
                                            random_state=self.config.seed)
                                Z_tsne = tsne.fit_transform(Z_tsne_all)
                                tsne_result = (Z_tsne[:tsne_total], Z_tsne[tsne_total:])
                        except Exception as tsne_err:
                            print(f"  t-SNE skipped: {tsne_err}")
                    
                    if tsne_result is not None:
                        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
                        ax_pca, ax_tsne = axes
                    else:
                        fig, ax_pca = plt.subplots(1, 1, figsize=(7, 6))
                        ax_tsne = None
                    
                    ax_pca.scatter(Zk_pca[:, 0], Zk_pca[:, 1], s=10, alpha=0.6,
                                   color=palette[0], label=with_label)
                    ax_pca.scatter(Zb_pca[:, 0], Zb_pca[:, 1], s=10, alpha=0.6,
                                   color=palette[5], label=without_label)
                    ax_pca.set_title(
                        f"PCA Projection\nPC1+PC2 Var={pca_var_ratio[:2].sum():.2%}",
                        fontsize=12, fontweight="bold"
                    )
                    ax_pca.set_xlabel("PC1")
                    ax_pca.set_ylabel("PC2")
                    ax_pca.legend(fontsize=10, loc="best")
                    ax_pca.grid(True, alpha=0.2)
                    
                    if tsne_result is not None and ax_tsne is not None:
                        Zk_tsne_plot, Zb_tsne_plot = tsne_result
                        ax_tsne.scatter(Zk_tsne_plot[:, 0], Zk_tsne_plot[:, 1], s=10, alpha=0.6,
                                        color=palette[0], label=with_label)
                        ax_tsne.scatter(Zb_tsne_plot[:, 0], Zb_tsne_plot[:, 1], s=10, alpha=0.6,
                                        color=palette[5], label=without_label)
                        ax_tsne.set_title("t-SNE Projection", fontsize=12, fontweight="bold")
                        ax_tsne.set_xlabel("t-SNE 1")
                        ax_tsne.set_ylabel("t-SNE 2")
                        ax_tsne.legend(fontsize=10, loc="best")
                        ax_tsne.grid(True, alpha=0.2)
                    
                    plt.tight_layout()
                    fig.savefig(plots_dir / "m3_projection.png", dpi=150, bbox_inches="tight", facecolor="white")
                    fig.savefig(plots_dir / "m3_projection.pdf", dpi=150, bbox_inches="tight", facecolor="white")
                    plt.close(fig)
                except Exception as proj_err:
                    print(f"  Projection plot skipped: {proj_err}")
            
            # ================================================================
            # PLOT 4: Manifold Constraint (Schematic)
            # ================================================================
            fig, ax = plt.subplots(figsize=(10, 8))
            
            ed_ratio = results["comparison"]["ed_ratio"]
            
            # Create a visual representation of dimensionality reduction
            theta = np.linspace(0, 2*np.pi, 100)
            r_full = 1.0  # Full dimensionality
            r_reduced = ed_ratio  # Reduced dimensionality
            
            # Baseline circle (outer)
            ax.fill(r_full * np.cos(theta), r_full * np.sin(theta), 
                   alpha=0.3, color=palette[5], label=f'Baseline Dim: {ed_pr[1]:.1f}')
            ax.plot(r_full * np.cos(theta), r_full * np.sin(theta), 
                   color=palette[5], linewidth=2)
            
            # INP circle (inner)
            ax.fill(r_reduced * np.cos(theta), r_reduced * np.sin(theta), 
                   alpha=0.5, color=palette[0], label=f'INP Dim: {ed_pr[0]:.1f}')
            ax.plot(r_reduced * np.cos(theta), r_reduced * np.sin(theta), 
                   color=palette[0], linewidth=2)
            
            ax.set_xlim(-1.3, 1.3)
            ax.set_ylim(-1.3, 1.3)
            ax.set_aspect('equal')
            ax.legend(fontsize=12, loc='upper right')
            ax.set_title(f'Manifold Constraint (Schematic)\nED Ratio: {ed_ratio:.2%}', 
                        fontsize=14, fontweight='bold')
            ax.axis('off')
            
            # Add annotation
            ax.text(0, -1.15, results["interpretation"], ha='center', fontsize=11, 
                   style='italic', wrap=True)
            
            plt.tight_layout()
            fig.savefig(plots_dir / "m3_manifold.png", dpi=150, bbox_inches='tight', facecolor='white')
            fig.savefig(plots_dir / "m3_manifold.pdf", dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            
            print(f"  M3 visualizations saved to {plots_dir}")
            
        except ImportError as e:
            print(f"Visualization libraries not available: {e}")


def compute_ed_for_knowledge_types(
    model: nn.Module,
    dataloader_factory,
    config: ExperimentConfig,
    knowledge_types: List[str],
) -> Dict[str, float]:
    """
    Compare effective dimensionality across different knowledge types.
    
    Useful for understanding how different types of knowledge
    (e.g., 'full', 'abc', 'a', 'none') affect latent structure.
    
    Args:
        model: Trained INP model
        dataloader_factory: Function that creates dataloader for a knowledge_type
        config: Experiment configuration
        knowledge_types: List of knowledge types to compare
        
    Returns:
        Dictionary mapping knowledge_type -> effective_dimensionality
    """
    results = {}
    
    for kt in knowledge_types:
        print(f"\nAnalyzing knowledge_type='{kt}'...")
        dataloader = dataloader_factory(kt)
        
        exp = EffectiveDimensionalityExperiment(model, config)
        Z = exp._collect_latents(dataloader, use_knowledge=True)
        analysis = exp._compute_svd_analysis(Z)
        
        results[kt] = {
            "effective_dimensionality": analysis["effective_dimensionality"],
            "components_to_95": analysis["components_to_threshold"],
        }
    
    return results
