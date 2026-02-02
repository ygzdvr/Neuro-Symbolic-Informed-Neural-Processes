"""
M10: Centered Kernel Alignment (CKA) Representation Similarity Analysis

This module compares the internal representations of INP (with knowledge)
vs NP (without knowledge) to understand where knowledge affects processing.

Key Question: Does knowledge K fundamentally change how data is processed,
or just shift/scale the same underlying features?

CKA (Centered Kernel Alignment):
    - Measures similarity between representation spaces
    - Invariant to orthogonal transformations and isotropic scaling
    - CKA ∈ [0, 1]: 0 = completely different, 1 = identical structure

Hypotheses:
    - Low CKA in early layers: Knowledge modulates data perception early
    - High CKA in early, low in latent: Info-fusion happens late
    - Low CKA in z: Knowledge fundamentally restructures the latent space

This helps us understand the "information fusion" mechanism in INPs.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from tqdm import tqdm
from pathlib import Path

from .base import InterpretabilityExperiment, ExperimentConfig, HookManager


def centering_matrix(n: int, device: str = "cpu") -> torch.Tensor:
    """
    Create centering matrix H = I - (1/n) * 11^T
    
    Used to center kernel matrices for CKA computation.
    """
    H = torch.eye(n, device=device) - torch.ones(n, n, device=device) / n
    return H


def linear_kernel(X: torch.Tensor) -> torch.Tensor:
    """
    Compute linear kernel K = X @ X^T
    
    Args:
        X: Feature matrix [n_samples, n_features]
        
    Returns:
        Kernel matrix [n_samples, n_samples]
    """
    return X @ X.T


def rbf_kernel(X: torch.Tensor, sigma: Optional[float] = None) -> torch.Tensor:
    """
    Compute RBF (Gaussian) kernel.
    
    K_ij = exp(-||x_i - x_j||^2 / (2σ^2))
    
    Args:
        X: Feature matrix [n_samples, n_features]
        sigma: Kernel bandwidth (if None, use median heuristic)
        
    Returns:
        Kernel matrix [n_samples, n_samples]
    """
    # Compute pairwise squared distances
    X_sq = (X ** 2).sum(dim=1, keepdim=True)
    dist_sq = X_sq + X_sq.T - 2 * X @ X.T
    dist_sq = torch.clamp(dist_sq, min=0)  # Numerical stability
    
    if sigma is None:
        # Median heuristic
        sigma = torch.median(torch.sqrt(dist_sq[dist_sq > 0]))
        if sigma == 0:
            sigma = 1.0
    
    K = torch.exp(-dist_sq / (2 * sigma ** 2))
    return K


def hsic(K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """
    Compute Hilbert-Schmidt Independence Criterion (HSIC).
    
    HSIC(K, L) = (1/(n-1)^2) * tr(KHLH)
    
    where H is the centering matrix.
    
    Args:
        K: First kernel matrix [n, n]
        L: Second kernel matrix [n, n]
        
    Returns:
        HSIC value (scalar)
    """
    n = K.shape[0]
    H = centering_matrix(n, K.device)
    
    # Center kernels
    K_c = H @ K @ H
    L_c = H @ L @ H
    
    # HSIC = trace(K_c @ L_c) / (n-1)^2
    hsic_value = torch.trace(K_c @ L_c) / ((n - 1) ** 2)
    
    return hsic_value


def cka(X: torch.Tensor, Y: torch.Tensor, kernel: str = "linear") -> float:
    """
    Compute Centered Kernel Alignment (CKA) between two representation matrices.
    
    CKA(X, Y) = HSIC(K_X, K_Y) / sqrt(HSIC(K_X, K_X) * HSIC(K_Y, K_Y))
    
    Args:
        X: First representation matrix [n_samples, n_features_X]
        Y: Second representation matrix [n_samples, n_features_Y]
        kernel: "linear" or "rbf"
        
    Returns:
        CKA score in [0, 1]
    """
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have same number of samples")
    
    # Flatten if needed
    X = X.reshape(X.shape[0], -1).float()
    Y = Y.reshape(Y.shape[0], -1).float()
    
    # Compute kernels
    if kernel == "linear":
        K_X = linear_kernel(X)
        K_Y = linear_kernel(Y)
    elif kernel == "rbf":
        K_X = rbf_kernel(X)
        K_Y = rbf_kernel(Y)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")
    
    # Compute HSIC values
    hsic_xy = hsic(K_X, K_Y)
    hsic_xx = hsic(K_X, K_X)
    hsic_yy = hsic(K_Y, K_Y)
    
    # CKA
    denominator = torch.sqrt(hsic_xx * hsic_yy)
    if denominator < 1e-10:
        return 0.0
    
    cka_value = hsic_xy / denominator
    
    return float(cka_value.clamp(0, 1))


class CKASimilarityExperiment(InterpretabilityExperiment):
    """
    M10: CKA Representation Similarity Analysis
    
    Compares internal representations when processing data with vs without
    knowledge to understand where and how knowledge affects the network.
    
    Procedure:
        1. Forward pass with knowledge → collect activations at each layer
        2. Forward pass without knowledge → collect activations
        3. Compute CKA between corresponding layers
        4. Identify where representations diverge most
    
    Key Outputs:
        - Layer-wise CKA scores
        - CKA heatmap across layers
        - Identification of "info-fusion" point
        
    Interpretation:
        - High CKA early, low late: Late fusion (knowledge affects decoding)
        - Low CKA throughout: Deep modulation (knowledge affects all processing)
        - Low CKA in z only: Knowledge primarily affects latent structure
    """
    
    name = "m10_cka_similarity"
    description = "CKA analysis of representation similarity with/without knowledge"
    
    def __init__(
        self,
        model: nn.Module,
        config: ExperimentConfig,
        kernel: str = "linear",
        max_samples: int = 500,
    ):
        """
        Args:
            model: Trained INP model
            config: Experiment configuration
            kernel: Kernel for CKA ("linear" or "rbf")
            max_samples: Maximum samples for CKA computation
        """
        super().__init__(model, config)
        self.kernel = kernel
        self.max_samples = max_samples
    
    def _get_layer_names(self) -> List[str]:
        """Get names of layers to analyze."""
        layers = []
        
        # Key representations to compare
        key_layers = [
            "x_encoder",
            "xy_encoder",  
            "latent_encoder",
            "decoder",
        ]
        
        for name, module in self.model.named_modules():
            # Check if this is a key module or submodule
            for key in key_layers:
                if key in name and isinstance(module, (nn.Linear, nn.Module)):
                    if name not in layers:
                        layers.append(name)
                    break
        
        return layers

    def _gather_batches(
        self, dataloader: torch.utils.data.DataLoader
    ) -> Tuple[List[Tuple], int]:
        """Collect a fixed list of batches to ensure paired comparisons."""
        batches: List[Tuple] = []
        total_samples = 0

        for batch in dataloader:
            batches.append(batch)
            context, _, _, _ = batch
            x_context, _ = context
            total_samples += x_context.shape[0]
            if total_samples >= self.max_samples:
                break

        return batches, total_samples

    def _standardize_representation(self, tensor: torch.Tensor) -> torch.Tensor:
        """Ensure representations are [batch, features] tensors."""
        if tensor.dim() == 2:
            return tensor
        if tensor.dim() == 3 and tensor.shape[1] == 1:
            return tensor.squeeze(1)
        return tensor.reshape(tensor.shape[0], -1)

    def _pool_activation(self, act: torch.Tensor) -> torch.Tensor:
        """Pool variable-shaped activations to [batch, features]."""
        if act.dim() == 2:
            return act
        if act.dim() == 3:
            return act.mean(1)
        if act.dim() == 4:
            return act.mean(dim=(1, 2))
        return act.reshape(act.shape[0], -1)

    def _disable_knowledge_dropout(self) -> Optional[float]:
        """Temporarily disable knowledge dropout for deterministic comparisons."""
        if hasattr(self.model, "latent_encoder") and hasattr(
            self.model.latent_encoder, "knowledge_dropout"
        ):
            original = self.model.latent_encoder.knowledge_dropout
            self.model.latent_encoder.knowledge_dropout = 0.0
            return float(original)
        return None
    
    def _collect_representations(
        self,
        batches: List[Tuple],
        use_knowledge: bool,
    ) -> Dict[str, torch.Tensor]:
        """
        Collect intermediate representations from forward passes.
        
        Args:
            dataloader: Data source
            use_knowledge: Whether to use knowledge conditioning
            
        Returns:
            Dictionary mapping layer name to stacked activations
        """
        representations = {
            "R": [],           # Context representation (after XY encoder)
            "z_mean": [],      # Latent mean
            "z_std": [],       # Latent std
            "pred_mean": [],   # Prediction mean
        }
        
        # Also collect intermediate activations via hooks
        hook_activations = {}
        hooks = []
        
        def make_hook(name):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    out = output[0]
                else:
                    out = output
                if name not in hook_activations:
                    hook_activations[name] = []
                hook_activations[name].append(out.detach().cpu())
            return hook_fn
        
        # Register hooks on key modules
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                hook = module.register_forward_hook(make_hook(name))
                hooks.append(hook)
        
        self.model.eval()
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(batches, desc=f"Collecting (K={use_knowledge})"):
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

                knowledge_input = knowledge if use_knowledge else None

                # Forward pass
                x_context_enc = self.model.x_encoder(x_context)
                x_target_enc = self.model.x_encoder(x_target)
                R = self.model.encode_globally(x_context_enc, y_context, x_target_enc)

                q_z = self.model.infer_latent_dist(R, knowledge_input, x_context.shape[1])
                z_mean = q_z.base_dist.loc
                z_std = q_z.base_dist.scale

                # Decode deterministically using z_mean
                z_mean_expanded = z_mean.unsqueeze(0)
                R_target = z_mean_expanded.expand(1, -1, x_target.shape[1], -1)
                p_y_stats = self.model.decoder(x_target_enc, R_target)
                pred_mean = p_y_stats[..., :self.model.config.output_dim]

                # Store - ensure consistent shapes [batch, features]
                representations["R"].append(self._standardize_representation(R).cpu())
                representations["z_mean"].append(self._standardize_representation(z_mean).cpu())
                representations["z_std"].append(self._standardize_representation(z_std).cpu())
                pred_mean_pooled = pred_mean.mean(0)
                pred_mean_pooled = pred_mean_pooled.mean(1)
                representations["pred_mean"].append(pred_mean_pooled.cpu())

                total_samples += x_context.shape[0]
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Concatenate
        for key in representations:
            if representations[key]:
                representations[key] = torch.cat(representations[key], dim=0)[:self.max_samples]
        
        # Add hook activations - handle variable shapes by pooling
        for name, acts in hook_activations.items():
            if acts:
                try:
                    pooled_acts = [self._pool_activation(act) for act in acts]
                    if len(pooled_acts) > 0:
                        feat_dim = pooled_acts[0].shape[-1]
                        compatible = [a for a in pooled_acts if a.shape[-1] == feat_dim]
                        if len(compatible) > 0:
                            stacked = torch.cat(compatible, dim=0)[:self.max_samples]
                            representations[f"hook_{name}"] = stacked
                except Exception:
                    continue
        
        return representations
    
    def run(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, Any]:
        """
        Run the CKA similarity analysis.
        
        Compares representations with vs without knowledge at each layer.
        
        Returns:
            Dictionary containing CKA scores and analysis
        """
        results = {
            "config": {
                "kernel": self.kernel,
                "max_samples": self.max_samples,
            },
            "cka_scores": {},
        }

        original_dropout = self._disable_knowledge_dropout()
        try:
            # Collect fixed batches to ensure paired comparisons
            batches, total_samples = self._gather_batches(dataloader)
            results["config"]["paired_batches"] = True
            results["config"]["num_samples_collected"] = int(total_samples)
            results["config"]["knowledge_dropout_original"] = original_dropout
            results["config"]["knowledge_dropout_set_to"] = 0.0 if original_dropout is not None else None

            # Collect representations
            print("\nCollecting representations WITH knowledge...")
            reps_with_k = self._collect_representations(batches, use_knowledge=True)

            print("\nCollecting representations WITHOUT knowledge...")
            reps_without_k = self._collect_representations(batches, use_knowledge=False)
        finally:
            if original_dropout is not None:
                self.model.latent_encoder.knowledge_dropout = original_dropout
        
        # Compute CKA for each representation
        print("\nComputing CKA scores...")
        
        common_keys = set(reps_with_k.keys()) & set(reps_without_k.keys())
        
        for key in tqdm(sorted(common_keys)):
            X = reps_with_k[key]
            Y = reps_without_k[key]
            
            if X.shape[0] != Y.shape[0]:
                min_n = min(X.shape[0], Y.shape[0])
                X = X[:min_n]
                Y = Y[:min_n]
            
            if X.shape[0] < 10:
                continue
            
            try:
                cka_score = cka(X, Y, kernel=self.kernel)
                results["cka_scores"][key] = cka_score
            except Exception as e:
                print(f"CKA failed for {key}: {e}")
                results["cka_scores"][key] = float('nan')
        
        # Analyze key representations
        key_reps = ["R", "z_mean", "z_std", "pred_mean"]
        results["key_representations"] = {}
        
        for key in key_reps:
            if key in results["cka_scores"]:
                results["key_representations"][key] = results["cka_scores"][key]
        
        # Find where knowledge has most impact (lowest CKA)
        valid_scores = {k: v for k, v in results["cka_scores"].items() 
                       if not np.isnan(v)}
        
        if valid_scores:
            min_cka_layer = min(valid_scores, key=valid_scores.get)
            max_cka_layer = max(valid_scores, key=valid_scores.get)
            
            results["analysis"] = {
                "min_cka_layer": min_cka_layer,
                "min_cka_score": valid_scores[min_cka_layer],
                "max_cka_layer": max_cka_layer,
                "max_cka_score": valid_scores[max_cka_layer],
                "mean_cka": float(np.mean(list(valid_scores.values()))),
                "std_cka": float(np.std(list(valid_scores.values()))),
            }
        
        # Interpretation
        interpretation_parts = []
        
        z_cka = results["cka_scores"].get("z_mean", float('nan'))
        R_cka = results["cka_scores"].get("R", float('nan'))
        
        if not np.isnan(z_cka):
            if z_cka < 0.5:
                interpretation_parts.append(
                    f"Low CKA in latent (z_mean={z_cka:.3f}): Knowledge fundamentally restructures latent space"
                )
            elif z_cka < 0.8:
                interpretation_parts.append(
                    f"Moderate CKA in latent (z_mean={z_cka:.3f}): Knowledge partially affects latent structure"
                )
            else:
                interpretation_parts.append(
                    f"High CKA in latent (z_mean={z_cka:.3f}): Latent structure similar with/without knowledge"
                )
        
        if not np.isnan(R_cka) and not np.isnan(z_cka):
            if R_cka > z_cka + 0.2:
                interpretation_parts.append(
                    "Late fusion pattern: Data encoding similar, knowledge affects later stages"
                )
            elif R_cka < z_cka - 0.2:
                interpretation_parts.append(
                    "Early modulation pattern: Knowledge affects data encoding"
                )
            else:
                interpretation_parts.append(
                    "Distributed effect: Knowledge affects processing throughout"
                )
        
        results["interpretation"] = ". ".join(interpretation_parts) if interpretation_parts else "Insufficient data for interpretation"
        
        self.results = results
        self.save_results(results)
        
        # Generate visualization
        self._save_visualization(results)
        
        return results
    
    def _save_visualization(self, results: Dict[str, Any]):
        """Generate comprehensive visualizations for M10."""
        try:
            from .enhanced_viz import viz_m10_cka
            viz_m10_cka(results, self.output_dir)
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
            
            cka_scores = results.get("cka_scores", {})
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            # Left: Key representations bar chart
            ax1 = axes[0]
            key_reps = ["R", "z_mean", "z_std", "pred_mean"]
            scores = [cka_scores.get(k, np.nan) for k in key_reps]
            valid_keys = [k for k, s in zip(key_reps, scores) if not np.isnan(s)]
            valid_scores = [s for s in scores if not np.isnan(s)]
            
            if valid_keys:
                colors = ['green' if s > 0.7 else 'orange' if s > 0.4 else 'red' for s in valid_scores]
                bars = ax1.bar(valid_keys, valid_scores, color=colors, edgecolor='black')
                ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, linewidth=2)
                ax1.set_ylim(0, 1.1)
                ax1.set_ylabel('CKA Score', fontsize=12)
                ax1.set_title('CKA: With K vs Without K', fontsize=14, fontweight='bold')
                ax1.grid(True, alpha=0.3, axis='y')
                
                for bar, score in zip(bars, valid_scores):
                    ax1.annotate(f'{score:.3f}',
                                xy=(bar.get_x() + bar.get_width() / 2, score),
                                xytext=(0, 5), textcoords="offset points",
                                ha='center', fontsize=12, fontweight='bold')
            
            # Right: Hook layers horizontal bar
            ax2 = axes[1]
            hook_scores = {k: v for k, v in cka_scores.items() 
                          if k.startswith("hook_") and not np.isnan(v)}
            
            if hook_scores:
                sorted_layers = sorted(hook_scores.keys())[:20]
                scores = [hook_scores[k] for k in sorted_layers]
                short_names = [k.replace("hook_", "").split(".")[-1][:20] for k in sorted_layers]
                
                colors = ['green' if s > 0.7 else 'orange' if s > 0.4 else 'red' for s in scores]
                ax2.barh(range(len(short_names)), scores, color=colors, edgecolor='black')
                ax2.set_yticks(range(len(short_names)))
                ax2.set_yticklabels(short_names, fontsize=9)
                ax2.set_xlim(0, 1)
                ax2.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7, linewidth=2)
                ax2.set_xlabel('CKA Score', fontsize=12)
                ax2.set_title('Layer-wise CKA', fontsize=14, fontweight='bold')
                ax2.invert_yaxis()
                ax2.grid(True, alpha=0.3, axis='x')
            
            plt.tight_layout()
            fig.savefig(plots_dir / "m10_cka_similarity.png", dpi=150, bbox_inches='tight', facecolor='white')
            fig.savefig(plots_dir / "m10_cka_similarity.pdf", dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            
            # Heatmap
            if len(cka_scores) > 0:
                fig, ax = plt.subplots(figsize=(10, 8))
                
                all_keys = sorted([k for k in cka_scores.keys() if not np.isnan(cka_scores[k])])[:30]
                
                if len(all_keys) > 1:
                    values = [cka_scores[k] for k in all_keys]
                    short_names = [k.replace("hook_", "")[-20:] for k in all_keys]
                    
                    heatmap_data = np.array(values).reshape(-1, 1)
                    sns.heatmap(heatmap_data, ax=ax, cmap='rocket', vmin=0, vmax=1,
                               yticklabels=short_names, xticklabels=['CKA'],
                               annot=True, fmt='.3f', cbar_kws={'label': 'CKA Score'},
                               linewidths=0.5, linecolor='white')
                    ax.set_title('CKA Similarity Heatmap\n(With K vs Without K)', 
                                fontsize=14, fontweight='bold')
                
                plt.tight_layout()
                fig.savefig(plots_dir / "m10_cka_heatmap.png", dpi=150, bbox_inches='tight', facecolor='white')
                fig.savefig(plots_dir / "m10_cka_heatmap.pdf", dpi=150, bbox_inches='tight', facecolor='white')
                plt.close(fig)
            
            print(f"  M10 visualizations saved to {plots_dir}")
            
        except ImportError as e:
            print(f"Visualization libraries not available: {e}")


def cross_model_cka(
    model1: nn.Module,
    model2: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cpu",
    max_samples: int = 500,
) -> Dict[str, float]:
    """
    Compute CKA between corresponding layers of two different models.
    
    Useful for comparing an INP model to a separately trained NP model.
    
    Args:
        model1: First model (e.g., INP)
        model2: Second model (e.g., NP)
        dataloader: Data for forward passes
        device: Computation device
        max_samples: Maximum samples
        
    Returns:
        Dictionary of layer-wise CKA scores
    """
    from .base import ExperimentConfig
    
    config = ExperimentConfig(device=device)
    
    exp1 = CKASimilarityExperiment(model1, config, max_samples=max_samples)
    exp2 = CKASimilarityExperiment(model2, config, max_samples=max_samples)

    # Collect fixed batches to align samples across models
    batches, _ = exp1._gather_batches(dataloader)

    # Collect with same knowledge setting for fair comparison
    reps1 = exp1._collect_representations(batches, use_knowledge=True)
    reps2 = exp2._collect_representations(batches, use_knowledge=False)
    
    results = {}
    common_keys = set(reps1.keys()) & set(reps2.keys())
    
    for key in common_keys:
        X = reps1[key]
        Y = reps2[key]
        
        min_n = min(X.shape[0], Y.shape[0])
        if min_n < 10:
            continue
            
        X = X[:min_n]
        Y = Y[:min_n]
        
        try:
            results[key] = cka(X, Y, kernel="linear")
        except Exception:
            pass
    
    return results
