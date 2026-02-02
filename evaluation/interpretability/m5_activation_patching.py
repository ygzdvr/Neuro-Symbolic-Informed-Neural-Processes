"""
M5: Causal Activation Patching (Interchange Intervention)

This module implements causal interventions to prove that the knowledge channel
has genuine causal efficacy on predictions, not just correlation.

Key Insight: Correlation ≠ Causation. Even if I(Z; K) is high, we need to prove
that changing K actually changes predictions in the expected direction.

Intervention Design:
    1. Task A: Context_A + Knowledge_A → Prediction_A
    2. Patched: Context_A + Knowledge_B → Prediction_Patched
    3. Task B: Context_B + Knowledge_B → Prediction_B

If Knowledge_B causally influences predictions:
    - Prediction_Patched should shift TOWARD Prediction_B
    - The shift magnitude quantifies "causal efficacy"

Causal Efficacy Metrics:
    1. Direct Effect: ||Pred_Patched - Pred_A|| (how much did prediction change?)
    2. Alignment Effect: cosine(Pred_Patched - Pred_A, Pred_B - Pred_A)
       (did it shift in the RIGHT direction?)
    3. Transfer Ratio: ||Pred_Patched - Pred_A|| / ||Pred_B - Pred_A||
       (what fraction of the "ideal" shift was achieved?)
    4. Ground Truth Shift: MSE(Pred_Patched, y_B) vs MSE(Pred_A, y_B)
       (does patching actually help predict Task B's data?)

For sinusoids with parameters (a, b, c):
    - Task A: function f_A(x) = a1*sin(b1*x + c1)
    - Task B: function f_B(x) = a2*sin(b2*x + c2)
    - If we patch knowledge (a2,b2,c2) into context from f_A,
      the model should try to predict f_B despite seeing f_A data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Optional, Callable
import numpy as np
from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass

from .base import InterpretabilityExperiment, ExperimentConfig


@dataclass
class PatchingResult:
    """Results from a single patching experiment."""
    pred_original: torch.Tensor      # Prediction with original knowledge
    pred_patched: torch.Tensor       # Prediction with patched knowledge
    pred_donor: torch.Tensor         # Prediction on donor task (reference)
    y_original: torch.Tensor         # Ground truth for original task
    y_donor: torch.Tensor            # Ground truth for donor task
    knowledge_original: torch.Tensor
    knowledge_donor: torch.Tensor


class KnowledgePatcher:
    """
    Utility class to perform clean knowledge interventions via hooks.
    
    This intercepts the knowledge encoding and replaces it with a 
    pre-computed override, enabling causal intervention experiments.
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.knowledge_override: Optional[torch.Tensor] = None
        self.hook_handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self._original_forward = None
    
    def _knowledge_hook(self, module: nn.Module, args: Tuple, output: torch.Tensor) -> torch.Tensor:
        """Hook that replaces knowledge encoding with override."""
        if self.knowledge_override is not None:
            # Replace the output with our override
            # Ensure shape matches
            if output.shape != self.knowledge_override.shape:
                # Broadcast if needed
                if self.knowledge_override.dim() == 2:
                    return self.knowledge_override.unsqueeze(1)
                return self.knowledge_override
            return self.knowledge_override
        return output
    
    def enable_patching(self, knowledge_override: torch.Tensor):
        """Enable knowledge patching with the given override tensor."""
        self.knowledge_override = knowledge_override
        
        # Register hook on the knowledge encoder
        if self.model.latent_encoder.knowledge_encoder is not None:
            self.hook_handle = self.model.latent_encoder.knowledge_encoder.register_forward_hook(
                self._knowledge_hook
            )
    
    def disable_patching(self):
        """Disable patching and restore normal forward pass."""
        self.knowledge_override = None
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.disable_patching()


class ActivationPatchingExperiment(InterpretabilityExperiment):
    """
    M5: Causal Activation Patching Experiment
    
    Tests whether the knowledge channel has genuine causal efficacy by
    performing interchange interventions.
    
    Experiment Design:
        For each pair of tasks (A, B):
        1. Run model normally on Task A → get Pred_A
        2. Run model on Task A with Knowledge_B patched in → get Pred_Patched
        3. Run model normally on Task B → get Pred_B (reference)
        4. Measure causal effect metrics
    
    Key Metrics:
        - Direct Effect: How much did patching change the prediction?
        - Alignment: Did the change go in the right direction?
        - Transfer Ratio: What fraction of the ideal shift was achieved?
        - Functional Shift: Does the patched prediction match donor ground truth better?
    """
    
    name = "m5_activation_patching"
    description = "Causal intervention analysis via knowledge patching"
    
    def __init__(
        self,
        model: nn.Module,
        config: ExperimentConfig,
        num_pairs: int = 100,
        num_z_samples: int = 32,
        max_batch_size: int = 16,
        max_target_points: Optional[int] = None,
    ):
        """
        Args:
            model: Trained INP model
            config: Experiment configuration
            num_pairs: Number of task pairs to test
            num_z_samples: Number of z samples for prediction
            max_batch_size: Cap batch size to reduce memory usage
            max_target_points: Optional cap on number of target points
        """
        super().__init__(model, config)
        self.num_pairs = num_pairs
        self.num_z_samples = num_z_samples
        self.max_batch_size = max_batch_size
        self.max_target_points = max_target_points
    
    def _get_prediction(
        self,
        x_context: torch.Tensor,
        y_context: torch.Tensor,
        x_target: torch.Tensor,
        knowledge: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get model prediction (mean and std) for given inputs.
        
        Returns:
            Tuple of (prediction_mean, prediction_std)
        """
        # Encode
        x_context_enc = self.model.x_encoder(x_context)
        x_target_enc = self.model.x_encoder(x_target)
        R = self.model.encode_globally(x_context_enc, y_context, x_target_enc)
        
        # Sample latent
        q_z = self.model.infer_latent_dist(R, knowledge, x_context.shape[1])
        z_samples = q_z.rsample([self.num_z_samples])
        
        # Decode
        R_target = z_samples.expand(-1, -1, x_target.shape[1], -1)
        p_y_stats = self.model.decoder(x_target_enc, R_target)
        p_y_loc, p_y_scale = p_y_stats.split(self.model.config.output_dim, dim=-1)
        p_y_scale = 0.1 + 0.9 * F.softplus(p_y_scale)
        
        # Average over z samples
        pred_mean = p_y_loc.mean(dim=0)  # [batch, targets, output_dim]
        pred_std = p_y_scale.mean(dim=0)
        
        return pred_mean, pred_std
    
    def _get_knowledge_embedding(self, knowledge: torch.Tensor) -> torch.Tensor:
        """Get the knowledge embedding from the encoder."""
        if self.model.latent_encoder.knowledge_encoder is not None:
            k = self.model.latent_encoder.knowledge_encoder(knowledge)
            return k
        return knowledge

    def _slice_knowledge(self, knowledge: Any, batch_size: int) -> Any:
        """Slice knowledge to match batch size across types."""
        if isinstance(knowledge, torch.Tensor):
            return knowledge[:batch_size].to(self.device)
        if isinstance(knowledge, (list, tuple)):
            return knowledge[:batch_size]
        return knowledge

    def _align_donor_ground_truth(
        self,
        x_donor: torch.Tensor,
        y_donor: torch.Tensor,
        x_target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Align donor ground-truth y to the target x grid.

        If x grids match, returns y_donor directly. Otherwise performs 1D linear
        interpolation per batch and output dimension.
        """
        if x_donor.shape == x_target.shape and torch.allclose(x_donor, x_target, atol=1e-6, rtol=1e-5):
            return y_donor

        # Only support 1D x for interpolation
        if x_donor.shape[-1] != 1 or x_target.shape[-1] != 1:
            # Fallback: truncate to target length if possible
            if y_donor.shape[1] >= x_target.shape[1]:
                return y_donor[:, :x_target.shape[1]]
            return y_donor

        # Interpolate on CPU for stability
        x_src = x_donor.detach().cpu().numpy()
        y_src = y_donor.detach().cpu().numpy()
        x_tgt = x_target.detach().cpu().numpy()
        batch_size, n_tgt, _ = x_tgt.shape
        out_dim = y_src.shape[-1]
        y_interp = np.zeros((batch_size, n_tgt, out_dim), dtype=y_src.dtype)

        for b in range(batch_size):
            x_src_b = x_src[b, :, 0]
            x_tgt_b = x_tgt[b, :, 0]
            order = np.argsort(x_src_b)
            x_src_b = x_src_b[order]
            for d in range(out_dim):
                y_src_b = y_src[b, :, d][order]
                y_interp[b, :, d] = np.interp(x_tgt_b, x_src_b, y_src_b)

        return torch.from_numpy(y_interp).to(self.device)
    
    def _compute_patching_metrics(self, result: PatchingResult) -> Dict[str, float]:
        """
        Compute causal effect metrics from a patching result.
        
        Metrics:
            1. direct_effect: ||Pred_Patched - Pred_Original||
            2. donor_distance: ||Pred_Donor - Pred_Original|| (reference)
            3. transfer_ratio: direct_effect / donor_distance
            4. alignment: cosine similarity of shift direction
            5. mse_original_to_donor_gt: How well original predicts donor's y
            6. mse_patched_to_donor_gt: How well patched predicts donor's y
            7. mse_improvement: Reduction in MSE to donor ground truth
        """
        pred_orig = result.pred_original.squeeze(-1)  # [batch, targets]
        pred_patch = result.pred_patched.squeeze(-1)
        pred_donor = result.pred_donor.squeeze(-1)
        y_donor = result.y_donor.squeeze(-1)
        y_orig = result.y_original.squeeze(-1)
        
        # Direct effect: how much did prediction change?
        direct_effect = torch.norm(pred_patch - pred_orig, dim=-1).mean().item()
        
        # Donor distance: how far is donor prediction from original?
        donor_distance = torch.norm(pred_donor - pred_orig, dim=-1).mean().item()
        
        # Transfer ratio: what fraction of ideal shift was achieved?
        transfer_ratio = direct_effect / max(donor_distance, 1e-8)
        
        # Alignment: did the shift go in the right direction?
        shift_actual = (pred_patch - pred_orig).reshape(pred_orig.size(0), -1)
        shift_ideal = (pred_donor - pred_orig).reshape(pred_orig.size(0), -1)
        
        # Compute per-sample alignment then average
        dot = (shift_actual * shift_ideal).sum(dim=-1)
        norm_actual = shift_actual.norm(dim=-1)
        norm_ideal = shift_ideal.norm(dim=-1)
        valid_mask = (norm_actual > 1e-8) & (norm_ideal > 1e-8)
        
        if valid_mask.any():
            alignment = (dot[valid_mask] / (norm_actual[valid_mask] * norm_ideal[valid_mask])).mean().item()
        else:
            alignment = 0.0
        
        # MSE to donor ground truth
        mse_orig_to_donor = F.mse_loss(pred_orig, y_donor).item()
        mse_patched_to_donor = F.mse_loss(pred_patch, y_donor).item()
        mse_improvement = mse_orig_to_donor - mse_patched_to_donor
        
        # Also check: does patching hurt prediction of original task?
        mse_orig_to_orig_gt = F.mse_loss(pred_orig, y_orig).item()
        mse_patched_to_orig_gt = F.mse_loss(pred_patch, y_orig).item()
        mse_degradation = mse_patched_to_orig_gt - mse_orig_to_orig_gt
        
        return {
            "direct_effect": direct_effect,
            "donor_distance": donor_distance,
            "transfer_ratio": transfer_ratio,
            "alignment": alignment,
            "mse_original_to_donor_gt": mse_orig_to_donor,
            "mse_patched_to_donor_gt": mse_patched_to_donor,
            "mse_improvement": mse_improvement,
            "mse_degradation_on_original": mse_degradation,
        }
    
    def _run_single_patching(
        self,
        batch_A: Tuple,
        batch_B: Tuple,
    ) -> Dict[str, float]:
        """
        Run patching experiment for a single pair of tasks.
        
        Args:
            batch_A: Original task batch (context, target, knowledge, ids)
            batch_B: Donor task batch (provides knowledge to patch)
            
        Returns:
            Dictionary of causal effect metrics
        """
        context_A, target_A, knowledge_A, _ = batch_A
        context_B, target_B, knowledge_B, _ = batch_B
        
        x_context_A, y_context_A = context_A
        x_target_A, y_target_A = target_A
        x_context_B, y_context_B = context_B
        x_target_B, y_target_B = target_B
        
        # Ensure matching sizes by taking minimum and applying caps
        batch_size = min(x_context_A.shape[0], x_context_B.shape[0])
        if self.max_batch_size is not None:
            batch_size = min(batch_size, self.max_batch_size)

        n_targets = min(x_target_A.shape[1], x_target_B.shape[1])
        if self.max_target_points is not None:
            n_targets = min(n_targets, self.max_target_points)

        n_context = min(x_context_A.shape[1], x_context_B.shape[1])
        
        # Slice to match
        x_context_A = x_context_A[:batch_size, :n_context]
        y_context_A = y_context_A[:batch_size, :n_context]
        x_target_A = x_target_A[:batch_size, :n_targets]
        y_target_A = y_target_A[:batch_size, :n_targets]
        
        x_context_B = x_context_B[:batch_size, :n_context]
        y_context_B = y_context_B[:batch_size, :n_context]
        x_target_B = x_target_B[:batch_size, :n_targets]
        y_target_B = y_target_B[:batch_size, :n_targets]
        
        # Move to device
        x_context_A = x_context_A.to(self.device)
        y_context_A = y_context_A.to(self.device)
        x_target_A = x_target_A.to(self.device)
        y_target_A = y_target_A.to(self.device)
        
        x_context_B = x_context_B.to(self.device)
        y_context_B = y_context_B.to(self.device)
        x_target_B = x_target_B.to(self.device)
        y_target_B = y_target_B.to(self.device)
        
        knowledge_A = self._slice_knowledge(knowledge_A, batch_size)
        knowledge_B = self._slice_knowledge(knowledge_B, batch_size)
        
        # 1. Original prediction on Task A
        pred_A_mean, _ = self._get_prediction(
            x_context_A, y_context_A, x_target_A, knowledge_A
        )
        
        # 2. Get knowledge embedding from Task B
        k_B = self._get_knowledge_embedding(knowledge_B)
        
        # 3. Patched prediction: Context A + Knowledge B
        patcher = KnowledgePatcher(self.model)
        patcher.enable_patching(k_B)
        try:
            pred_patched_mean, _ = self._get_prediction(
                x_context_A, y_context_A, x_target_A, knowledge_A  # knowledge_A will be overridden
            )
        finally:
            patcher.disable_patching()
        
        # 4. Reference: prediction on Task B (what we'd expect if patching works perfectly)
        # Note: Use x_target_A for pred_B to ensure same target locations
        pred_B_on_A_targets, _ = self._get_prediction(
            x_context_B, y_context_B, x_target_A, knowledge_B
        )

        # Align donor ground truth to x_target_A
        y_donor_on_A_targets = self._align_donor_ground_truth(
            x_target_B, y_target_B, x_target_A
        )
        
        # Create result object
        # Use pred_B_on_A_targets as the "ideal" donor prediction
        # since it's evaluated at the same x locations (x_target_A) as pred_original and pred_patched
        result = PatchingResult(
            pred_original=pred_A_mean,
            pred_patched=pred_patched_mean,
            pred_donor=pred_B_on_A_targets,
            y_original=y_target_A,
            y_donor=y_donor_on_A_targets,
            knowledge_original=knowledge_A,
            knowledge_donor=knowledge_B,
        )
        
        return self._compute_patching_metrics(result)
    
    def run(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, Any]:
        """
        Run the causal activation patching experiment.
        
        Creates pairs of tasks from the dataloader and tests knowledge patching
        on each pair.
        
        Returns:
            Dictionary containing:
                - individual_results: Metrics for each pair
                - aggregated: Mean and std of each metric
                - interpretation: Analysis of causal efficacy
        """
        self.model.eval()

        if self.model.latent_encoder.knowledge_encoder is None:
            raise ValueError("M5 requires a knowledge encoder; model has none.")

        prev_dropout = None
        if hasattr(self.model.latent_encoder, "knowledge_dropout"):
            prev_dropout = self.model.latent_encoder.knowledge_dropout
            self.model.latent_encoder.knowledge_dropout = 0.0

        try:
            results = {
                "config": {
                    "num_pairs": self.num_pairs,
                    "num_z_samples": self.num_z_samples,
                    "max_batch_size": self.max_batch_size,
                    "max_target_points": self.max_target_points,
                    "knowledge_dropout_disabled": True,
                },
                "individual_results": [],
            }
            
            # Collect batches
            batches = []
            for batch in dataloader:
                batches.append(batch)
                if len(batches) >= self.num_pairs * 2:
                    break
            
            if len(batches) < 2:
                raise ValueError("Need at least 2 batches for patching experiment")
            
            print(f"\nRunning patching experiments on {min(self.num_pairs, len(batches)//2)} pairs...")
            
            # Run pairwise patching
            with torch.no_grad():
                for i in tqdm(range(0, min(len(batches) - 1, self.num_pairs * 2), 2)):
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    batch_A = batches[i]
                    batch_B = batches[i + 1]
                    
                    metrics = self._run_single_patching(batch_A, batch_B)
                    results["individual_results"].append(metrics)
                    
                    # Also test reverse direction
                    metrics_reverse = self._run_single_patching(batch_B, batch_A)
                    results["individual_results"].append(metrics_reverse)
            
            # Aggregate results
            aggregated = {}
            if results["individual_results"]:
                keys = results["individual_results"][0].keys()
                for key in keys:
                    values = [r[key] for r in results["individual_results"] if not np.isnan(r[key])]
                    if values:
                        aggregated[key] = {
                            "mean": float(np.mean(values)),
                            "std": float(np.std(values)),
                            "min": float(np.min(values)),
                            "max": float(np.max(values)),
                        }
            
            results["aggregated"] = aggregated
            
            # Interpretation
            transfer_ratio = aggregated.get("transfer_ratio", {}).get("mean", 0)
            alignment = aggregated.get("alignment", {}).get("mean", 0)
            mse_improvement = aggregated.get("mse_improvement", {}).get("mean", 0)
            
            interpretation_parts = []
            
            # Transfer ratio interpretation
            if transfer_ratio > 0.5:
                interpretation_parts.append(
                    f"Strong causal efficacy: {transfer_ratio:.1%} of ideal shift achieved"
                )
            elif transfer_ratio > 0.2:
                interpretation_parts.append(
                    f"Moderate causal efficacy: {transfer_ratio:.1%} of ideal shift achieved"
                )
            else:
                interpretation_parts.append(
                    f"Weak causal efficacy: only {transfer_ratio:.1%} of ideal shift achieved"
                )
            
            # Alignment interpretation
            if alignment > 0.5:
                interpretation_parts.append(
                    f"Shift direction is correct (alignment={alignment:.2f})"
                )
            elif alignment > 0:
                interpretation_parts.append(
                    f"Shift is partially aligned (alignment={alignment:.2f})"
                )
            else:
                interpretation_parts.append(
                    f"WARNING: Shift is misaligned (alignment={alignment:.2f})"
                )
            
            # MSE improvement interpretation
            if mse_improvement > 0:
                interpretation_parts.append(
                    f"Patching improves prediction of donor task (MSE reduction: {mse_improvement:.4f})"
                )
            else:
                interpretation_parts.append(
                    f"Patching does not improve donor prediction"
                )
            
            results["interpretation"] = ". ".join(interpretation_parts)
            
            # Causal efficacy score (composite)
            causal_efficacy = (
                0.4 * min(transfer_ratio, 1.0) +  # Transfer ratio (capped at 1)
                0.4 * max(alignment, 0) +          # Alignment (only positive counts)
                0.2 * (1 if mse_improvement > 0 else 0)  # MSE improvement (binary)
            )
            results["causal_efficacy_score"] = float(causal_efficacy)
            
            self.results = results
            self.save_results(results)
            
            # Generate visualization
            self._save_visualization(results)
            
            return results
        finally:
            if prev_dropout is not None:
                self.model.latent_encoder.knowledge_dropout = prev_dropout
    
    def _save_visualization(self, results: Dict[str, Any]):
        """Generate comprehensive visualizations for M5."""
        try:
            from .enhanced_viz import viz_m5_activation_patching
            viz_m5_activation_patching(results, self.output_dir)
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
            
            individual = results.get("individual_results", [])
            agg = results.get("aggregated", {})
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 12))
            
            # Top-left: Transfer ratio distribution
            ax1 = axes[0, 0]
            transfer_ratios = [r["transfer_ratio"] for r in individual]
            ax1.hist(transfer_ratios, bins=25, color=palette[0],
                    edgecolor='black', alpha=0.7)
            ax1.axvline(x=np.mean(transfer_ratios), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(transfer_ratios):.3f}')
            ax1.axvline(x=0.5, color='green', linestyle=':', linewidth=2, label='Strong threshold')
            ax1.set_xlabel('Transfer Ratio', fontsize=12)
            ax1.set_ylabel('Count', fontsize=12)
            ax1.set_title('Distribution of Transfer Ratios', fontsize=14, fontweight='bold')
            ax1.legend(fontsize=11)
            ax1.grid(True, alpha=0.3)
            
            # Top-right: Alignment distribution
            ax2 = axes[0, 1]
            alignments = [r["alignment"] for r in individual]
            ax2.hist(alignments, bins=25, color=palette[2],
                    edgecolor='black', alpha=0.7)
            ax2.axvline(x=0, color='gray', linestyle='--', linewidth=2)
            ax2.axvline(x=np.mean(alignments), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(alignments):.3f}')
            ax2.set_xlabel('Alignment', fontsize=12)
            ax2.set_ylabel('Count', fontsize=12)
            ax2.set_title('Shift Direction Alignment', fontsize=14, fontweight='bold')
            ax2.legend(fontsize=11)
            ax2.grid(True, alpha=0.3)
            
            # Bottom-left: MSE scatter
            ax3 = axes[1, 0]
            mse_orig = [r["mse_original_to_donor_gt"] for r in individual]
            mse_patched = [r["mse_patched_to_donor_gt"] for r in individual]
            ax3.scatter(mse_orig, mse_patched, alpha=0.5, color=palette[0],
                       edgecolors='black', s=50)
            max_val = max(max(mse_orig), max(mse_patched)) * 1.1
            ax3.plot([0, max_val], [0, max_val], 'k--', linewidth=2, label='No change')
            ax3.fill_between([0, max_val], [0, 0], [0, max_val], alpha=0.1, color='green')
            improvements = sum(1 for o, p in zip(mse_orig, mse_patched) if p < o)
            ax3.text(0.95, 0.05, f"Improvements: {improvements}/{len(mse_orig)} ({improvements/len(mse_orig):.1%})",
                    transform=ax3.transAxes, ha='right', fontsize=11,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            ax3.set_xlabel('MSE (Original → Donor GT)', fontsize=12)
            ax3.set_ylabel('MSE (Patched → Donor GT)', fontsize=12)
            ax3.set_title('MSE to Donor Ground Truth', fontsize=14, fontweight='bold')
            ax3.legend(fontsize=11)
            ax3.grid(True, alpha=0.3)
            
            # Bottom-right: Summary metrics
            ax4 = axes[1, 1]
            metrics = ['Transfer\nRatio', 'Alignment', 'MSE\nImprovement', 'Causal\nEfficacy']
            values = [
                agg.get("transfer_ratio", {}).get("mean", 0),
                agg.get("alignment", {}).get("mean", 0),
                agg.get("mse_improvement", {}).get("mean", 0),
                results.get("causal_efficacy_score", 0),
            ]
            colors = [palette[i] for i in [0, 2, 4, 6]]
            bars = ax4.bar(metrics, values, color=colors, edgecolor='black')
            ax4.axhline(y=0, color='gray', linestyle='-', linewidth=1)
            ax4.set_ylabel('Value', fontsize=12)
            ax4.set_title('Summary Causal Effect Metrics', fontsize=14, fontweight='bold')
            ax4.grid(True, alpha=0.3, axis='y')
            
            for bar, val in zip(bars, values):
                ax4.annotate(f'{val:.3f}',
                            xy=(bar.get_x() + bar.get_width() / 2, val),
                            xytext=(0, 5 if val >= 0 else -15),
                            textcoords="offset points",
                            ha='center', fontsize=10, fontweight='bold')
            
            plt.tight_layout()
            fig.suptitle(f'M5: Causal Activation Patching\nCausal Efficacy Score: {results.get("causal_efficacy_score", 0):.3f}',
                        fontsize=16, fontweight='bold', y=1.02)
            
            fig.savefig(plots_dir / "m5_activation_patching.png", dpi=150, bbox_inches='tight', facecolor='white')
            fig.savefig(plots_dir / "m5_activation_patching.pdf", dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            
            print(f"  M5 visualizations saved to {plots_dir}")
            
        except ImportError as e:
            print(f"Visualization libraries not available: {e}")


def targeted_patching_experiment(
    model: nn.Module,
    task_sine: Dict,
    task_linear: Dict,
    device: str,
) -> Dict[str, float]:
    """
    Specific experiment: Patch "linear" knowledge into a "sine" task.
    
    This is the canonical test case mentioned in the paper:
    - Task A: Sinusoidal data + "sine" knowledge
    - Task B: Linear data + "linear" knowledge
    - Intervention: Sinusoidal context + "linear" knowledge
    
    If knowledge has causal efficacy, the output should shift toward linear.
    
    Args:
        model: Trained INP model
        task_sine: Dictionary with 'context', 'knowledge', 'target_x', 'target_y'
        task_linear: Dictionary with 'context', 'knowledge', 'target_x', 'target_y'
        device: Computation device
        
    Returns:
        Dictionary of causal effect metrics
    """
    model.eval()
    
    # This is a simplified version - the full experiment class handles this more robustly
    config = ExperimentConfig(device=device)
    exp = ActivationPatchingExperiment(model, config)
    
    # Convert to batch format expected by the experiment
    # (This would need adaptation based on actual data format)
    
    return {"note": "Use ActivationPatchingExperiment class for full analysis"}
