"""
M2: Loss Landscape Visualization

This module analyzes the optimization geometry of INPs using filter-normalized
1D interpolation to visualize the loss landscape.

Hypothesis: Knowledge K acts as a prior that convexifies the landscape. Without K,
the landscape is riddled with local minima corresponding to contradictory functions.
With K, these minima vanish, leaving a smoother basin around the true function.

Key technique: Filter normalization ensures scale-invariance, making "flatness"
well-defined across layers with different weight magnitudes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from tqdm import tqdm
import copy
import json
from pathlib import Path

from .base import InterpretabilityExperiment, ExperimentConfig
from models.loss import sum_log_prob


def get_filter_normalized_direction(model: nn.Module) -> List[torch.Tensor]:
    """
    Generates a random direction vector d with the same norm as the weights w
    for each filter/neuron.
    
    This normalization is critical for meaningful flatness comparisons:
    - Neural networks are scale-invariant (especially with BatchNorm/LayerNorm)
    - Without normalization, "flatness" is ill-defined
    - We normalize: ||d^(l)|| = ||θ^(l)|| for each layer l
    
    Args:
        model: The neural network model
        
    Returns:
        List of direction tensors, one per parameter
    """
    direction = []
    for p in model.parameters():
        d = torch.randn_like(p)
        
        if d.dim() > 1:  # Conv/Linear weights - normalize per filter (dim 0)
            for i in range(d.size(0)):
                w_norm = p[i].norm()
                d_norm = d[i].norm()
                if d_norm > 1e-10:
                    d[i] = d[i] * (w_norm / d_norm)
        else:  # Biases, scalars - normalize entire tensor
            w_norm = p.norm()
            d_norm = d.norm()
            if d_norm > 1e-10:
                d = d * (w_norm / d_norm)
        
        direction.append(d)
    
    return direction


def get_random_direction(model: nn.Module, normalize: bool = True) -> List[torch.Tensor]:
    """
    Generates a simple random direction (optionally normalized to unit norm).
    
    Args:
        model: The neural network model
        normalize: If True, normalize direction to unit norm
        
    Returns:
        List of direction tensors
    """
    direction = []
    for p in model.parameters():
        d = torch.randn_like(p)
        direction.append(d)
    
    if normalize:
        # Compute total norm and normalize
        total_norm = sum(d.norm().item() ** 2 for d in direction) ** 0.5
        direction = [d / total_norm for d in direction]
    
    return direction


def set_weights_interpolated(
    model: nn.Module,
    theta_star: List[torch.Tensor],
    direction: List[torch.Tensor],
    alpha: float
):
    """
    Sets model weights to θ* + α * direction.
    
    Args:
        model: The model to modify
        theta_star: Original trained weights
        direction: Perturbation direction
        alpha: Interpolation coefficient
    """
    for p, w_star, d in zip(model.parameters(), theta_star, direction):
        p.data.copy_(w_star + alpha * d)


def set_weights_interpolated_2d(
    model: nn.Module,
    theta_star: List[torch.Tensor],
    direction_a: List[torch.Tensor],
    direction_b: List[torch.Tensor],
    alpha_a: float,
    alpha_b: float,
):
    """
    Sets model weights to θ* + α_a * direction_a + α_b * direction_b.
    """
    for p, w_star, d_a, d_b in zip(
        model.parameters(), theta_star, direction_a, direction_b
    ):
        p.data.copy_(w_star + alpha_a * d_a + alpha_b * d_b)


class LossLandscapeExperiment(InterpretabilityExperiment):
    """
    M2: Loss Landscape Visualization
    
    Visualizes the 1D loss landscape along filter-normalized random directions.
    Compares the "flatness" of minima between INP (with knowledge) and NP (without).
    
    Key metrics:
        - Basin width: Width of region where L(θ) < L(θ*) + ε
        - Curvature: Second derivative approximation at minimum
        - Barrier height: Maximum loss increase within perturbation range
        
    Expected behavior:
        - INP should have broader, flatter basins than NP
        - Knowledge conditioning smooths out local minima
    """
    
    name = "m2_loss_landscape"
    description = "Filter-normalized 1D loss landscape visualization"
    
    def __init__(
        self,
        model: nn.Module,
        config: ExperimentConfig,
        alpha_range: Tuple[float, float] = (-1.0, 1.0),
        num_steps: int = 51,
        num_directions: int = 5,
        use_filter_normalization: bool = True,
        flatness_epsilon: float = 0.1,
        flatness_epsilon_ratio: float = 0.1,
        num_eval_batches: int = 10,
        loss_mode: str = "elbo",
        beta: float = 1.0,
        num_z_samples: Optional[int] = None,
        compute_plane: bool = True,
        plane_num_steps: int = 21,
        plane_alpha_range: Optional[Tuple[float, float]] = None,
    ):
        """
        Args:
            model: Trained INP model
            config: Experiment configuration
            alpha_range: Range of interpolation coefficients (min, max)
            num_steps: Number of points to evaluate along each direction
            num_directions: Number of random directions to average over
            use_filter_normalization: If True, use filter-normalized directions
            flatness_epsilon: Absolute threshold for basin width calculation
            flatness_epsilon_ratio: Relative threshold (fraction of loss at origin)
            num_eval_batches: Number of batches to average loss over
            loss_mode: "nll" or "elbo"
            beta: KL weight for ELBO
            num_z_samples: Number of z samples for loss (defaults to model train samples)
            compute_plane: If True, compute a true 2D loss surface
            plane_num_steps: Resolution for 2D surface per axis
            plane_alpha_range: Optional alpha range for 2D surface
        """
        super().__init__(model, config)
        
        self.alpha_range = alpha_range
        self.num_steps = num_steps
        self.num_directions = num_directions
        self.use_filter_normalization = use_filter_normalization
        self.flatness_epsilon = flatness_epsilon
        self.flatness_epsilon_ratio = flatness_epsilon_ratio
        self.num_eval_batches = num_eval_batches
        self.loss_mode = loss_mode.lower()
        self.beta = beta
        self.num_z_samples = num_z_samples
        self.compute_plane = compute_plane
        self.plane_num_steps = plane_num_steps
        self.plane_alpha_range = plane_alpha_range or alpha_range

        if self.loss_mode not in {"nll", "elbo"}:
            raise ValueError(f"Unsupported loss_mode: {self.loss_mode}")
    
    def _compute_loss(
        self,
        dataloader: torch.utils.data.DataLoader,
        use_knowledge: bool = True,
        fixed_batches: Optional[List] = None,
    ) -> float:
        """
        Compute average loss over batches.
        
        Args:
            dataloader: Data source
            use_knowledge: If True, use knowledge; if False, simulate NP
            fixed_batches: If provided, use these specific batches
            
        Returns:
            Average loss value
        """
        self.model.eval()
        total_loss = 0.0
        count = 0
        
        if fixed_batches is not None:
            batch_iter = fixed_batches
        else:
            batch_iter = dataloader
        
        num_z_samples = self.num_z_samples
        if num_z_samples is None:
            num_z_samples = getattr(self.model, "train_num_z_samples", 1)

        with torch.no_grad():
            for batch_idx, batch in enumerate(batch_iter):
                if fixed_batches is None and batch_idx >= self.num_eval_batches:
                    break
                
                context, target, knowledge, _ = batch
                x_context, y_context = context
                x_target, y_target = target
                
                x_context = x_context.to(self.device)
                y_context = y_context.to(self.device)
                x_target = x_target.to(self.device)
                y_target = y_target.to(self.device)
                
                if isinstance(knowledge, torch.Tensor):
                    knowledge = knowledge.to(self.device)
                
                knowledge_used = knowledge if use_knowledge else None
                
                # Encode inputs
                x_context_enc = self.model.x_encoder(x_context)
                x_target_enc = self.model.x_encoder(x_target)
                
                # Context-dependent latent
                R_context = self.model.encode_globally(
                    x_context_enc, y_context, x_target_enc
                )
                q_zCc = self.model.infer_latent_dist(
                    R_context, knowledge_used, x_context.shape[1]
                )
                
                if self.loss_mode == "elbo":
                    # Target-dependent latent (for KL)
                    R_target = self.model.encode_globally(
                        x_target_enc, y_target, x_target_enc
                    )
                    q_zCct = self.model.infer_latent_dist(
                        R_target, knowledge_used, x_target.shape[1]
                    )
                    z_dist = q_zCct
                elif self.loss_mode == "nll":
                    q_zCct = None
                    z_dist = q_zCc
                else:
                    raise ValueError(f"Unknown loss_mode: {self.loss_mode}")
                
                # Sample z and decode
                z_samples = z_dist.rsample([num_z_samples])
                R_target = self.model.target_dependent_representation(
                    R_context, x_target_enc, z_samples
                )
                p_yCc = self.model.decode_target(x_target_enc, R_target)
                
                # Compute NLL
                sum_log_p_yCz = sum_log_prob(p_yCc, y_target)
                E_z_sum_log_p_yCz = torch.mean(sum_log_p_yCz, dim=0)
                nll = -E_z_sum_log_p_yCz
                
                if self.loss_mode == "elbo":
                    kl_z = torch.distributions.kl.kl_divergence(q_zCct, q_zCc)
                    kl_z = torch.sum(kl_z, dim=1)
                    loss = nll + self.beta * kl_z
                else:
                    loss = nll
                
                total_loss += loss.mean().item()
                count += 1
        
        return total_loss / max(count, 1)
    
    def _plot_landscape_1d(
        self,
        dataloader: torch.utils.data.DataLoader,
        direction: List[torch.Tensor],
        use_knowledge: bool = True,
        fixed_batches: Optional[List] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute 1D loss landscape along a filter-normalized random direction.
        
        Args:
            dataloader: Data source
            direction: Perturbation direction
            use_knowledge: Whether to use knowledge in forward pass
            fixed_batches: Fixed batches for consistent evaluation
            
        Returns:
            Tuple of (alphas array, losses array)
        """
        # Store original weights
        theta_star = [p.data.clone() for p in self.model.parameters()]
        direction = [d.to(self.device) for d in direction]
        
        alphas = np.linspace(self.alpha_range[0], self.alpha_range[1], self.num_steps)
        losses = []
        
        for alpha in alphas:
            set_weights_interpolated(self.model, theta_star, direction, alpha)
            loss = self._compute_loss(dataloader, use_knowledge, fixed_batches)
            losses.append(loss)
        
        # Restore original weights
        set_weights_interpolated(self.model, theta_star, direction, 0.0)
        
        return alphas, np.array(losses)

    def _plot_landscape_2d(
        self,
        dataloader: torch.utils.data.DataLoader,
        direction_a: List[torch.Tensor],
        direction_b: List[torch.Tensor],
        use_knowledge: bool = True,
        fixed_batches: Optional[List] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute a 2D loss surface along two directions.
        
        Returns:
            Tuple of (alphas array, loss grid [num_steps, num_steps])
        """
        theta_star = [p.data.clone() for p in self.model.parameters()]
        direction_a = [d.to(self.device) for d in direction_a]
        direction_b = [d.to(self.device) for d in direction_b]
        
        alphas = np.linspace(
            self.plane_alpha_range[0], self.plane_alpha_range[1], self.plane_num_steps
        )
        grid = np.zeros((len(alphas), len(alphas)), dtype=float)
        
        for i, alpha_a in enumerate(alphas):
            for j, alpha_b in enumerate(alphas):
                set_weights_interpolated_2d(
                    self.model, theta_star, direction_a, direction_b, alpha_a, alpha_b
                )
                grid[i, j] = self._compute_loss(
                    dataloader, use_knowledge=use_knowledge, fixed_batches=fixed_batches
                )
        
        set_weights_interpolated_2d(
            self.model, theta_star, direction_a, direction_b, 0.0, 0.0
        )
        
        return alphas, grid
    
    def _compute_flatness_metrics(
        self,
        alphas: np.ndarray,
        losses: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute flatness metrics from a 1D loss profile.
        
        Args:
            alphas: Interpolation coefficients
            losses: Corresponding loss values
            
        Returns:
            Dictionary of flatness metrics
        """
        # Origin (alpha ~= 0) for local flatness around trained weights
        origin_idx = int(np.argmin(np.abs(alphas)))
        origin_alpha = float(alphas[origin_idx])
        origin_loss = float(losses[origin_idx])

        # Global minimum (for reference)
        min_idx = int(np.argmin(losses))
        min_loss = float(losses[min_idx])
        min_alpha = float(alphas[min_idx])

        # Basin width around origin using relative + absolute thresholds
        epsilon_abs = max(
            float(self.flatness_epsilon),
            abs(origin_loss) * float(self.flatness_epsilon_ratio),
        )
        threshold = origin_loss + epsilon_abs
        in_basin = losses <= threshold

        basin_width = 0.0
        if np.any(in_basin):
            basin_indices = np.where(in_basin)[0]
            split_points = np.where(np.diff(basin_indices) != 1)[0] + 1
            segments = np.split(basin_indices, split_points)
            segment = None
            for seg in segments:
                if seg[0] <= origin_idx <= seg[-1]:
                    segment = seg
                    break
            if segment is not None:
                basin_width = float(alphas[segment[-1]] - alphas[segment[0]])

        # Curvature approximation at origin (finite difference)
        if origin_idx > 0 and origin_idx < len(losses) - 1:
            h = alphas[1] - alphas[0]
            curvature = (losses[origin_idx + 1] - 2 * losses[origin_idx] + losses[origin_idx - 1]) / (h ** 2)
        else:
            curvature = float("nan")

        # Barrier height: max loss relative to origin
        barrier_height = float(np.max(losses) - origin_loss)

        # Loss increase at boundaries relative to origin
        loss_at_neg_boundary = float(losses[0] - origin_loss)
        loss_at_pos_boundary = float(losses[-1] - origin_loss)

        return {
            "origin_loss": origin_loss,
            "origin_alpha": origin_alpha,
            "min_loss": min_loss,
            "min_alpha": min_alpha,
            "basin_width": float(basin_width),
            "curvature": float(curvature),
            "barrier_height": float(barrier_height),
            "loss_increase_negative": loss_at_neg_boundary,
            "loss_increase_positive": loss_at_pos_boundary,
            "epsilon_abs": float(epsilon_abs),
        }
    
    def run(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, Any]:
        """
        Run the loss landscape analysis.
        
        Computes 1D loss profiles along multiple random directions and
        aggregates flatness metrics.
        
        Returns:
            Dictionary containing:
                - profiles: List of (alphas, losses) for each direction
                - metrics: Aggregated flatness metrics
                - comparison: INP vs NP comparison (if applicable)
        """
        results = {
            "config": {
                "alpha_range": self.alpha_range,
                "num_steps": self.num_steps,
                "num_directions": self.num_directions,
                "use_filter_normalization": self.use_filter_normalization,
                "flatness_epsilon": self.flatness_epsilon,
                "flatness_epsilon_ratio": self.flatness_epsilon_ratio,
                "num_eval_batches": self.num_eval_batches,
                "loss_mode": self.loss_mode,
                "beta": self.beta,
                "num_z_samples": self.num_z_samples,
                "compute_plane": self.compute_plane,
                "plane_num_steps": self.plane_num_steps,
                "plane_alpha_range": self.plane_alpha_range,
            },
            "profiles_with_knowledge": [],
            "profiles_without_knowledge": [],
            "metrics_with_knowledge": [],
            "metrics_without_knowledge": [],
        }

        # Disable knowledge dropout for stable evaluation
        original_knowledge_dropout = self.model.latent_encoder.knowledge_dropout
        self.model.latent_encoder.knowledge_dropout = 0.0
        
        # Cache fixed batches for consistent evaluation
        print("Caching evaluation batches...")
        fixed_batches = []
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= self.num_eval_batches:
                break
            fixed_batches.append(batch)
        
        # Compute profiles along multiple directions
        print(f"\nComputing loss landscapes along {self.num_directions} directions...")
        
        for dir_idx in tqdm(range(self.num_directions), desc="Directions"):
            # Use the same direction for with/without knowledge
            if self.use_filter_normalization:
                direction = get_filter_normalized_direction(self.model)
            else:
                direction = get_random_direction(self.model)

            # With knowledge (INP)
            alphas, losses_k = self._plot_landscape_1d(
                dataloader,
                direction=direction,
                use_knowledge=True,
                fixed_batches=fixed_batches,
            )
            metrics_k = self._compute_flatness_metrics(alphas, losses_k)
            
            results["profiles_with_knowledge"].append({
                "alphas": alphas.tolist(),
                "losses": losses_k.tolist(),
            })
            results["metrics_with_knowledge"].append(metrics_k)
            
            # Without knowledge (NP baseline)
            _, losses_no_k = self._plot_landscape_1d(
                dataloader,
                direction=direction,
                use_knowledge=False,
                fixed_batches=fixed_batches,
            )
            metrics_no_k = self._compute_flatness_metrics(alphas, losses_no_k)
            
            results["profiles_without_knowledge"].append({
                "alphas": alphas.tolist(),
                "losses": losses_no_k.tolist(),
            })
            results["metrics_without_knowledge"].append(metrics_no_k)

        # Compute a true 2D loss surface (same directions for both conditions)
        if self.compute_plane:
            if self.use_filter_normalization:
                direction_a = get_filter_normalized_direction(self.model)
                direction_b = get_filter_normalized_direction(self.model)
            else:
                direction_a = get_random_direction(self.model)
                direction_b = get_random_direction(self.model)

            plane_alphas, surface_k = self._plot_landscape_2d(
                dataloader,
                direction_a=direction_a,
                direction_b=direction_b,
                use_knowledge=True,
                fixed_batches=fixed_batches,
            )
            _, surface_no_k = self._plot_landscape_2d(
                dataloader,
                direction_a=direction_a,
                direction_b=direction_b,
                use_knowledge=False,
                fixed_batches=fixed_batches,
            )

            results["surface_with_knowledge"] = {
                "alphas": plane_alphas.tolist(),
                "losses": surface_k.tolist(),
            }
            results["surface_without_knowledge"] = {
                "alphas": plane_alphas.tolist(),
                "losses": surface_no_k.tolist(),
            }
        
        # Aggregate metrics
        def aggregate_metrics(metrics_list: List[Dict]) -> Dict[str, float]:
            aggregated = {}
            keys = metrics_list[0].keys()
            for key in keys:
                values = [m[key] for m in metrics_list if not np.isnan(m[key])]
                if values:
                    aggregated[f"{key}_mean"] = float(np.mean(values))
                    aggregated[f"{key}_std"] = float(np.std(values))
            return aggregated
        
        results["aggregated_with_knowledge"] = aggregate_metrics(results["metrics_with_knowledge"])
        results["aggregated_without_knowledge"] = aggregate_metrics(results["metrics_without_knowledge"])
        
        # Comparison
        agg_k = results["aggregated_with_knowledge"]
        agg_no_k = results["aggregated_without_knowledge"]
        
        results["comparison"] = {
            "basin_width_ratio": agg_k.get("basin_width_mean", 0) / max(agg_no_k.get("basin_width_mean", 1e-8), 1e-8),
            "curvature_ratio": agg_k.get("curvature_mean", 0) / max(agg_no_k.get("curvature_mean", 1e-8), 1e-8),
            "barrier_reduction": agg_no_k.get("barrier_height_mean", 0) - agg_k.get("barrier_height_mean", 0),
        }
        
        # Interpretation
        basin_ratio = results["comparison"]["basin_width_ratio"]
        curvature_ratio = results["comparison"]["curvature_ratio"]
        if basin_ratio > 1.5 and curvature_ratio < 0.7:
            results["interpretation"] = (
                f"INP has {basin_ratio:.2f}x wider basin and lower curvature "
                f"(ratio={curvature_ratio:.2f}) - knowledge significantly flattens landscape"
            )
        elif basin_ratio > 1.1 and curvature_ratio < 0.9:
            results["interpretation"] = (
                f"INP has {basin_ratio:.2f}x wider basin with modest curvature reduction "
                f"(ratio={curvature_ratio:.2f})"
            )
        elif basin_ratio < 0.9 and curvature_ratio > 1.1:
            results["interpretation"] = "Knowledge appears to sharpen the landscape (unexpected)"
        else:
            results["interpretation"] = "Similar flatness with and without knowledge"
        
        self.results = results
        self.save_results(results)
        
        # Generate visualization
        self._save_visualization(results)

        # Restore knowledge dropout
        self.model.latent_encoder.knowledge_dropout = original_knowledge_dropout
        
        return results
    
    def _save_visualization(self, results: Dict[str, Any]):
        """Generate comprehensive visualizations for M2."""
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
            
            alphas = np.array(results["profiles_with_knowledge"][0]["alphas"])
            losses_k = np.array([p["losses"] for p in results["profiles_with_knowledge"]])
            losses_no_k = np.array([p["losses"] for p in results["profiles_without_knowledge"]])
            mean_k = np.mean(losses_k, axis=0)
            std_k = np.std(losses_k, axis=0)
            mean_no_k = np.mean(losses_no_k, axis=0)
            std_no_k = np.std(losses_no_k, axis=0)
            
            # ================================================================
            # PLOT 1: Main Loss Landscape Comparison (4-panel)
            # ================================================================
            fig, axes = plt.subplots(2, 2, figsize=(14, 12))
            
            # Top-left: All profiles overlay
            ax1 = axes[0, 0]
            for i, profile in enumerate(results["profiles_with_knowledge"]):
                alpha_val = 0.4 if i > 0 else 0.9
                label = "With Knowledge (INP)" if i == 0 else None
                ax1.plot(profile["alphas"], profile["losses"], 
                        color=palette[0], alpha=alpha_val, 
                        linewidth=2 if i == 0 else 1, label=label)
            
            for i, profile in enumerate(results["profiles_without_knowledge"]):
                alpha_val = 0.4 if i > 0 else 0.9
                label = "Without Knowledge (NP)" if i == 0 else None
                ax1.plot(profile["alphas"], profile["losses"], 
                        color=palette[5], alpha=alpha_val,
                        linestyle='--', linewidth=2 if i == 0 else 1, label=label)
            
            ax1.set_xlabel("α (perturbation magnitude)", fontsize=12)
            ax1.set_ylabel("Loss", fontsize=12)
            ax1.set_title("All 1D Loss Landscape Profiles", fontsize=14, fontweight='bold')
            ax1.legend(fontsize=11, loc='upper right')
            ax1.grid(True, alpha=0.3)
            
            # Top-right: Mean profiles with confidence bands
            ax2 = axes[0, 1]
            ax2.plot(alphas, mean_k, color=palette[0], 
                    linewidth=3, label="With Knowledge (INP)")
            ax2.fill_between(alphas, mean_k - std_k, mean_k + std_k, 
                           alpha=0.3, color=palette[0])
            ax2.plot(alphas, mean_no_k, color=palette[5], 
                    linewidth=3, linestyle='--', label="Without Knowledge (NP)")
            ax2.fill_between(alphas, mean_no_k - std_no_k, mean_no_k + std_no_k, 
                           alpha=0.3, color=palette[5])
            
            min_loss = min(np.min(mean_k), np.min(mean_no_k))
            ax2.axhline(y=min_loss + self.flatness_epsilon, color='green', 
                       linestyle=':', linewidth=2, label=f'Basin threshold (ε={self.flatness_epsilon})')
            ax2.axvline(x=0, color='gray', linestyle='-', alpha=0.5, linewidth=1)
            
            ax2.set_xlabel("α (perturbation magnitude)", fontsize=12)
            ax2.set_ylabel("Loss", fontsize=12)
            ax2.set_title("Mean Loss Landscape (±1 std)", fontsize=14, fontweight='bold')
            ax2.legend(fontsize=10, loc='upper right')
            ax2.grid(True, alpha=0.3)
            
            # Bottom-left: Flatness metrics comparison
            ax3 = axes[1, 0]
            agg_k = results["aggregated_with_knowledge"]
            agg_no_k = results["aggregated_without_knowledge"]
            
            metrics = ["Basin Width", "Curvature", "Barrier Height"]
            vals_k = [
                agg_k.get("basin_width_mean", 0),
                agg_k.get("curvature_mean", 0) / 10,  # Scale for visibility
                agg_k.get("barrier_height_mean", 0)
            ]
            vals_no_k = [
                agg_no_k.get("basin_width_mean", 0),
                agg_no_k.get("curvature_mean", 0) / 10,
                agg_no_k.get("barrier_height_mean", 0)
            ]
            
            x = np.arange(len(metrics))
            width = 0.35
            bars1 = ax3.bar(x - width/2, vals_k, width, label='With Knowledge (INP)', 
                           color=palette[0], edgecolor='black')
            bars2 = ax3.bar(x + width/2, vals_no_k, width, label='Without Knowledge (NP)',
                           color=palette[5], edgecolor='black')
            
            ax3.set_ylabel('Value', fontsize=12)
            ax3.set_title('Flatness Metrics Comparison', fontsize=14, fontweight='bold')
            ax3.set_xticks(x)
            ax3.set_xticklabels(metrics)
            ax3.legend(fontsize=10)
            ax3.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax3.annotate(f'{height:.2f}',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3), textcoords="offset points",
                                ha='center', va='bottom', fontsize=9)
            
            # Bottom-right: Basin width ratio visualization
            ax4 = axes[1, 1]
            ratio = results["comparison"]["basin_width_ratio"]
            colors = ['green' if ratio > 1 else 'red']
            
            ax4.barh(['Basin Width\nRatio (INP/NP)'], [ratio], color=colors, 
                    edgecolor='black', height=0.5)
            ax4.axvline(x=1, color='gray', linestyle='--', linewidth=2, label='Equal (ratio=1)')
            ax4.set_xlim(0, max(2, ratio * 1.2))
            ax4.set_xlabel('Ratio', fontsize=12)
            ax4.set_title('Basin Width Ratio', fontsize=14, fontweight='bold')
            
            # Add interpretation
            if ratio > 1.5:
                interp = "Knowledge significantly\nflattens landscape"
                color = 'green'
            elif ratio > 1.1:
                interp = "Knowledge moderately\nflattens landscape"
                color = 'orange'
            else:
                interp = "Similar flatness"
                color = 'gray'
            
            ax4.text(ratio + 0.05, 0, f'{ratio:.2f}x\n{interp}', 
                    va='center', fontsize=11, fontweight='bold', color=color)
            ax4.grid(True, alpha=0.3, axis='x')
            
            plt.tight_layout()
            fig.savefig(plots_dir / "m2_loss_landscape.png", dpi=150, bbox_inches='tight', facecolor='white')
            fig.savefig(plots_dir / "m2_loss_landscape.pdf", dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            
            # ================================================================
            # PLOT 2: Heatmap of Loss Landscape
            # ================================================================
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            surface_k = results.get("surface_with_knowledge")
            surface_no_k = results.get("surface_without_knowledge")
            
            if surface_k and surface_no_k:
                plane_alphas = np.array(surface_k["alphas"])
                grid_k = np.array(surface_k["losses"])
                grid_no_k = np.array(surface_no_k["losses"])
                
                for data, title, ax in [
                    (grid_k, "With Knowledge (INP)", axes[0]),
                    (grid_no_k, "Without Knowledge (NP)", axes[1]),
                ]:
                    sns.heatmap(
                        data,
                        ax=ax,
                        cmap="rocket_r",
                        xticklabels=5,
                        yticklabels=5,
                        cbar_kws={"label": "Loss"},
                    )
                    ax.set_xlabel("α (direction A)", fontsize=12)
                    ax.set_ylabel("α (direction B)", fontsize=12)
                    ax.set_title(
                        f"Loss Landscape Heatmap: {title}",
                        fontsize=14,
                        fontweight="bold",
                    )
            else:
                # Fallback: show 1D profiles as a heatmap (directions x alpha)
                for data, title, ax in [
                    (losses_k, "With Knowledge (INP)", axes[0]),
                    (losses_no_k, "Without Knowledge (NP)", axes[1]),
                ]:
                    sns.heatmap(
                        data,
                        ax=ax,
                        cmap="rocket_r",
                        xticklabels=10,
                        yticklabels=True,
                        cbar_kws={"label": "Loss"},
                    )
                    ax.set_xlabel("α index", fontsize=12)
                    ax.set_ylabel("Direction", fontsize=12)
                    ax.set_title(
                        f"Loss Landscape Heatmap: {title}",
                        fontsize=14,
                        fontweight="bold",
                    )
            
            plt.tight_layout()
            fig.savefig(plots_dir / "m2_heatmap.png", dpi=150, bbox_inches='tight', facecolor='white')
            fig.savefig(plots_dir / "m2_heatmap.pdf", dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            
            # ================================================================
            # PLOT 3: 3D Surface Plot (if possible)
            # ================================================================
            try:
                from mpl_toolkits.mplot3d import Axes3D
                
                fig = plt.figure(figsize=(14, 6))

                if surface_k and surface_no_k:
                    plane_alphas = np.array(surface_k["alphas"])
                    grid_k = np.array(surface_k["losses"])
                    grid_no_k = np.array(surface_no_k["losses"])
                    
                    for idx, (data, title) in enumerate(
                        [
                            (grid_k, "With Knowledge (INP)"),
                            (grid_no_k, "Without Knowledge (NP)"),
                        ]
                    ):
                        ax = fig.add_subplot(1, 2, idx + 1, projection="3d")
                        X, Y = np.meshgrid(plane_alphas, plane_alphas)
                        ax.plot_surface(X, Y, data, cmap="rocket_r", alpha=0.85)
                        ax.set_xlabel("α (direction A)", fontsize=11)
                        ax.set_ylabel("α (direction B)", fontsize=11)
                        ax.set_zlabel("Loss", fontsize=11)
                        ax.set_title(title, fontsize=13, fontweight="bold")
                else:
                    for idx, (data, title) in enumerate(
                        [
                            (losses_k, "With Knowledge (INP)"),
                            (losses_no_k, "Without Knowledge (NP)"),
                        ]
                    ):
                        ax = fig.add_subplot(1, 2, idx + 1, projection="3d")
                        X, Y = np.meshgrid(alphas, np.arange(data.shape[0]))
                        ax.plot_surface(X, Y, data, cmap="rocket_r", alpha=0.8)
                        ax.set_xlabel("α", fontsize=11)
                        ax.set_ylabel("Direction", fontsize=11)
                        ax.set_zlabel("Loss", fontsize=11)
                        ax.set_title(title, fontsize=13, fontweight="bold")
                    
                plt.tight_layout()
                fig.savefig(plots_dir / "m2_3d_surface.png", dpi=150, bbox_inches='tight', facecolor='white')
                plt.close(fig)
            except Exception:
                pass
            
            print(f"  M2 visualizations saved to {plots_dir}")
            
        except ImportError as e:
            print(f"Visualization libraries not available: {e}")


def compare_landscapes(
    model_inp: nn.Module,
    model_np: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    config: ExperimentConfig,
    **kwargs
) -> Dict[str, Any]:
    """
    Compare loss landscapes between an INP model and a baseline NP model.
    
    This is useful when you have two separately trained models to compare.
    
    Args:
        model_inp: Trained INP model (with knowledge)
        model_np: Trained NP model (without knowledge)
        dataloader: Evaluation data
        config: Experiment configuration
        **kwargs: Additional arguments for LossLandscapeExperiment
        
    Returns:
        Comparison results dictionary
    """
    results = {}
    
    # Analyze INP
    print("Analyzing INP model...")
    exp_inp = LossLandscapeExperiment(model_inp, config, **kwargs)
    # Override to always use knowledge
    results["inp"] = exp_inp.run(dataloader)
    
    # Analyze NP
    print("Analyzing NP model...")
    exp_np = LossLandscapeExperiment(model_np, config, **kwargs)
    results["np"] = exp_np.run(dataloader)
    
    # Detailed comparison
    inp_metrics = results["inp"]["aggregated_with_knowledge"]
    np_metrics = results["np"]["aggregated_without_knowledge"]
    
    results["comparison"] = {
        "basin_width_inp": inp_metrics.get("basin_width_mean", 0),
        "basin_width_np": np_metrics.get("basin_width_mean", 0),
        "basin_width_ratio": inp_metrics.get("basin_width_mean", 0) / max(np_metrics.get("basin_width_mean", 1e-8), 1e-8),
        "curvature_inp": inp_metrics.get("curvature_mean", 0),
        "curvature_np": np_metrics.get("curvature_mean", 0),
    }
    
    return results
