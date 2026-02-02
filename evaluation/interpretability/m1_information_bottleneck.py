"""
M1: Information Bottleneck Analysis using Mutual Information Neural Estimation (MINE)

This module implements the Information Bottleneck analysis for INPs, measuring:
- I(Z; D): Mutual information between latent code and data context
- I(Z; K): Mutual information between latent code and knowledge

Hypothesis: In successful INPs, I(Z; K) should rise early (establishing prior from knowledge),
while I(Z; D) plateaus at a lower level than in standard NPs (compression via knowledge).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
from tqdm import tqdm

from .base import InterpretabilityExperiment, ExperimentConfig, extract_intermediate_representations


class MINENetwork(nn.Module):
    """
    Statistics network T(x, z) for estimating Mutual Information.
    
    MINE uses a neural network to estimate the KL divergence between
    the joint distribution P(X,Z) and the product of marginals P(X)P(Z).
    """
    
    def __init__(self, input_dim: int, z_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + z_dim, hidden_dim),
            nn.ReLU(),  # ReLU instead of ELU for more stability
            nn.LayerNorm(hidden_dim),  # Add LayerNorm for stability
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize weights with smaller values
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Compute T(x, z) for MINE estimation.
        
        Args:
            x: Input tensor [batch, input_dim] (Data or Knowledge embedding)
            z: Latent tensor [batch, z_dim]
            
        Returns:
            T(x, z) values [batch, 1]
        """
        # Normalize inputs for stability
        x = x / (x.norm(dim=-1, keepdim=True) + 1e-8)
        z = z / (z.norm(dim=-1, keepdim=True) + 1e-8)
        
        combined = torch.cat([x, z], dim=-1)
        output = self.net(combined)
        
        # Clamp output to prevent extreme values that cause exp() overflow/underflow
        output = output.clamp(min=-10, max=10)
        
        return output


class MINEEstimator:
    """
    Estimates mutual information using MINE with exponential moving average.
    
    The MINE objective is:
        I(X; Z) >= E_joint[T(x,z)] - log(E_marginal[exp(T(x,z))])
    
    Using EMA helps stabilize training.
    """
    
    def __init__(
        self,
        input_dim: int,
        z_dim: int,
        hidden_dim: int = 128,
        lr: float = 1e-4,
        ema_decay: float = 0.99,
        device: str = "cpu"
    ):
        self.device = torch.device(device)
        self.network = MINENetwork(input_dim, z_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, weight_decay=1e-5)
        self.ema_decay = ema_decay
        self.ema_et = None  # Exponential moving average of exp(T) for marginal
        self.max_grad_norm = 1.0  # Gradient clipping
        self.last_mi_estimate = None
        self.last_mi_lb = None
    
    def _log_mean_exp(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute log(mean(exp(x))) in a numerically stable way using log-sum-exp trick.
        
        log(mean(exp(x))) = log(sum(exp(x))/n) = log(sum(exp(x))) - log(n)
                          = log_sum_exp(x) - log(n)
        """
        n = x.size(0)
        # Use torch.logsumexp for numerical stability
        return torch.logsumexp(x, dim=0) - np.log(n)
    
    def estimate(self, x: torch.Tensor, z: torch.Tensor) -> Tuple[float, float]:
        """
        Perform one step of MINE optimization and return MI estimate.
        
        Args:
            x: Input samples from joint distribution [batch, input_dim]
            z: Latent samples from joint distribution [batch, z_dim]
            
        Returns:
            Tuple of (MI lower bound estimate, loss value)
        """
        # Reset cached values for this step
        self.last_mi_estimate = None
        self.last_mi_lb = None

        x = x.to(self.device)
        z = z.to(self.device)
        
        # Check for NaN/Inf in inputs
        if torch.isnan(x).any() or torch.isnan(z).any() or \
           torch.isinf(x).any() or torch.isinf(z).any():
            return 0.0, 0.0
        
        # Joint distribution samples (x_i, z_i)
        t_joint = self.network(x, z)
        
        # Marginal distribution samples: shuffle z to break correlation
        # (x_i, z_j) approximates P(x)P(z)
        z_shuffled = z[torch.randperm(z.size(0))]
        t_marginal = self.network(x, z_shuffled)
        
        # Use numerically stable log-mean-exp for the marginal term
        # MI = E_joint[T] - log(E_marginal[exp(T)])
        mean_t_joint = torch.mean(t_joint)
        log_mean_exp_marginal = self._log_mean_exp(t_marginal.squeeze())
        
        # MI estimate (Donsker-Varadhan representation)
        mi_estimate = mean_t_joint - log_mean_exp_marginal
        
        # Clamp MI estimate to reasonable range
        # MI should be non-negative, but MINE can give small negatives
        mi_estimate_clamped = mi_estimate.clamp(min=-5, max=50)
        
        # For EMA-based loss (more stable gradients)
        et = torch.exp(t_marginal - t_marginal.max()).mean() * torch.exp(t_marginal.max())
        if self.ema_et is None:
            self.ema_et = et.detach().clamp(min=1e-6, max=1e6)
        else:
            self.ema_et = (self.ema_decay * self.ema_et + 
                          (1 - self.ema_decay) * et.detach()).clamp(min=1e-6, max=1e6)
        
        # MINE loss with EMA for stability
        mi_lb = mean_t_joint - torch.log(self.ema_et + 1e-8)
        
        # Minimize negative MI (maximize MI lower bound)
        loss = -mi_lb.clamp(min=-50, max=50)
        
        # Check for NaN before backward
        if torch.isnan(loss) or torch.isinf(loss):
            return 0.0, 0.0
        
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        
        self.optimizer.step()
        
        # Return clamped estimate (ensure non-negative for MI)
        mi_value = max(0.0, mi_estimate_clamped.item())
        mi_lb_value = max(0.0, mi_lb.clamp(min=-5, max=50).item())
        self.last_mi_estimate = mi_value
        self.last_mi_lb = mi_lb_value
        
        return mi_value, loss.item()


class InformationBottleneckExperiment(InterpretabilityExperiment):
    """
    M1: Information Bottleneck Analysis
    
    Tracks the information flow from Data (D) and Knowledge (K) to the latent code Z.
    Uses MINE to estimate I(Z; D) and I(Z; K) during training or on a trained model.
    
    Key metrics:
        - I(Z; D): How much the latent relies on context data
        - I(Z; K): How much the latent relies on knowledge
        - I(Z; K) / (I(Z; D) + I(Z; K)): Knowledge reliance ratio
    
    Expected behavior for successful INPs:
        - I(Z; K) rises early, establishing knowledge-based prior
        - I(Z; D) stays lower than in standard NPs (compression)
        - Ratio increases, indicating knowledge substitution
    """
    
    name = "m1_information_bottleneck"
    description = "Mutual Information Neural Estimation for Data and Knowledge flows"
    
    def __init__(
        self,
        model: nn.Module,
        config: ExperimentConfig,
        mine_hidden_dim: int = 128,
        mine_lr: float = 1e-4,
        mine_train_steps: int = 100,
    ):
        super().__init__(model, config)
        
        # MINE hyperparameters
        self.mine_hidden_dim = mine_hidden_dim
        self.mine_lr = mine_lr
        self.mine_train_steps = mine_train_steps
        
        # These will be initialized when we know the dimensions
        self.mine_data: Optional[MINEEstimator] = None
        self.mine_knowledge: Optional[MINEEstimator] = None
        
        # Track estimates over batches
        self.mi_data_history: List[float] = []
        self.mi_knowledge_history: List[float] = []
    
    def _initialize_estimators(self, context_dim: int, knowledge_dim: int, z_dim: int):
        """Initialize MINE estimators with correct dimensions."""
        self.mine_data = MINEEstimator(
            input_dim=context_dim,
            z_dim=z_dim,
            hidden_dim=self.mine_hidden_dim,
            lr=self.mine_lr,
            device=self.config.device
        )
        self.mine_knowledge = MINEEstimator(
            input_dim=knowledge_dim,
            z_dim=z_dim,
            hidden_dim=self.mine_hidden_dim,
            lr=self.mine_lr,
            device=self.config.device
        )
    
    def _flatten_context(
        self,
        x_context: torch.Tensor,
        y_context: torch.Tensor
    ) -> torch.Tensor:
        """Flatten context pairs into a single vector per batch element."""
        # x_context: [batch, num_points, x_dim]
        # y_context: [batch, num_points, y_dim]
        xy = torch.cat([x_context, y_context], dim=-1)  # [batch, num_points, x_dim+y_dim]
        return xy.view(xy.size(0), -1)  # [batch, num_points * (x_dim+y_dim)]
    
    def _get_context_representation(
        self,
        x_context: torch.Tensor,
        y_context: torch.Tensor,
        use_encoder: bool = True
    ) -> torch.Tensor:
        """
        Get context representation for MI estimation.
        
        If use_encoder=True, uses the model's XYEncoder output.
        Otherwise, flattens raw (x, y) pairs.
        """
        if use_encoder:
            # Use the encoded representation R
            x_enc = self.model.x_encoder(x_context)
            R = self.model.encode_globally(x_enc, y_context, x_enc)
            return R.squeeze(1)  # [batch, hidden_dim]
        else:
            # Use raw flattened context
            return self._flatten_context(x_context, y_context)
    
    def _get_knowledge_embedding(self, knowledge: torch.Tensor) -> torch.Tensor:
        """Get knowledge embedding from the model's knowledge encoder."""
        if self.model.latent_encoder.knowledge_encoder is not None:
            k = self.model.latent_encoder.knowledge_encoder(knowledge)
            if k.dim() == 3:
                k = k.squeeze(1)  # [batch, knowledge_dim]
            return k
        else:
            # Fallback: flatten raw knowledge
            return knowledge.view(knowledge.size(0), -1)

    def _compute_stable_summary(
        self,
        values: List[Optional[float]],
        stable_frac: float = 0.7,
        smooth_frac: float = 0.1,
        trim_frac: float = 0.1,
    ) -> Dict[str, Any]:
        """Compute a robust summary using a peak-plateau window and trimmed mean."""
        cleaned = np.array([v for v in values if v is not None], dtype=float)
        if cleaned.size == 0:
            return {
                "mean": 0.0,
                "median": 0.0,
                "peak_raw": 0.0,
                "peak_smoothed": 0.0,
                "window_start": None,
                "window_end": None,
                "window_size": 0,
                "num_points": 0,
                "method": "peak_plateau_trimmed",
                "stable_frac": stable_frac,
                "smooth_frac": smooth_frac,
                "trim_frac": trim_frac,
            }

        window = max(1, int(round(cleaned.size * smooth_frac)))
        if window <= 1:
            smoothed = cleaned.copy()
        else:
            kernel = np.ones(window, dtype=float) / window
            smoothed = np.convolve(cleaned, kernel, mode="valid")

        peak_smoothed = float(np.max(smoothed))
        peak_raw = float(np.max(cleaned))
        stable_vals = cleaned
        window_start = 0
        window_end = cleaned.size

        if peak_smoothed > 0:
            threshold = peak_smoothed * stable_frac
            mask = smoothed >= threshold
            if mask.any():
                peak_idx = int(np.argmax(smoothed))
                idx = np.where(mask)[0]
                split_points = np.where(np.diff(idx) != 1)[0] + 1
                segments = np.split(idx, split_points)
                chosen = None
                for seg in segments:
                    if seg[0] <= peak_idx <= seg[-1]:
                        chosen = seg
                        break
                if chosen is None:
                    chosen = max(segments, key=len)

                window_start = int(chosen[0])
                window_end = int(min(cleaned.size, chosen[-1] + window))
                stable_vals = cleaned[window_start:window_end]

        if stable_vals.size == 0:
            stable_vals = cleaned

        if stable_vals.size >= 5:
            sorted_vals = np.sort(stable_vals)
            trim = max(1, int(round(sorted_vals.size * trim_frac)))
            if sorted_vals.size > 2 * trim:
                trimmed = sorted_vals[trim:-trim]
            else:
                trimmed = sorted_vals
        else:
            trimmed = stable_vals

        return {
            "mean": float(np.mean(trimmed)),
            "median": float(np.median(trimmed)),
            "peak_raw": peak_raw,
            "peak_smoothed": peak_smoothed,
            "window_start": window_start,
            "window_end": window_end,
            "window_size": window,
            "num_points": int(trimmed.size),
            "method": "peak_plateau_trimmed",
            "stable_frac": stable_frac,
            "smooth_frac": smooth_frac,
            "trim_frac": trim_frac,
        }
    
    def run(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, Any]:
        """
        Run the Information Bottleneck experiment.
        
        Args:
            dataloader: DataLoader providing batches
            
        Returns:
            Dictionary containing:
                - mi_data_mean: Mean I(Z; D) estimate
                - mi_knowledge_mean: Mean I(Z; K) estimate
                - knowledge_reliance: I(Z;K) / (I(Z;D) + I(Z;K))
                - mi_data_history: Per-batch I(Z; D) estimates
                - mi_knowledge_history: Per-batch I(Z; K) estimates
        """
        self.model.eval()
        original_knowledge_dropout = self.model.latent_encoder.knowledge_dropout
        # Disable dropout for stable MI estimation (restore at end)
        self.model.latent_encoder.knowledge_dropout = 0.0
        
        mi_data_estimates = []
        mi_knowledge_estimates = []
        mi_data_lb_estimates = []
        mi_knowledge_lb_estimates = []
        
        first_batch = True
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="MI Estimation")):
                if batch_idx >= self.config.num_batches:
                    break
                
                context, target, knowledge, _ = batch
                x_context, y_context = context
                x_target, y_target = target
                
                x_context = x_context.to(self.device)
                y_context = y_context.to(self.device)
                x_target = x_target.to(self.device)
                y_target = y_target.to(self.device)
                
                # Handle different knowledge types
                if isinstance(knowledge, torch.Tensor):
                    knowledge = knowledge.to(self.device)
                
                # Get representations
                context_repr = self._get_context_representation(
                    x_context, y_context, use_encoder=True
                )
                
                if isinstance(knowledge, torch.Tensor):
                    knowledge_repr = self._get_knowledge_embedding(knowledge)
                else:
                    # Skip knowledge MI for non-tensor knowledge (e.g., text)
                    continue
                
                # Get latent samples
                x_enc = self.model.x_encoder(x_context)
                x_target_enc = self.model.x_encoder(x_target)
                R = self.model.encode_globally(x_enc, y_context, x_target_enc)
                z_samples, q_zCc, _ = self.model.sample_latent(
                    R, x_enc, x_target_enc, None, knowledge
                )
                
                # Use mean of z for MI estimation (more stable than samples)
                # q_zCc is Independent(Normal(...)), so access base_dist.loc
                z = q_zCc.base_dist.loc.squeeze(1)  # [batch, z_dim]
                
                # Initialize MINE estimators on first batch
                if first_batch:
                    context_dim = context_repr.shape[-1]
                    knowledge_dim = knowledge_repr.shape[-1]
                    z_dim = z.shape[-1]
                    self._initialize_estimators(context_dim, knowledge_dim, z_dim)
                    first_batch = False
        
        # Now train MINE estimators
        print(f"\nTraining MINE estimators for {self.mine_train_steps} steps...")
        
        for step in tqdm(range(self.mine_train_steps), desc="MINE Training"):
            step_mi_data = []
            step_mi_knowledge = []
            step_mi_data_lb = []
            step_mi_knowledge_lb = []
            
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= self.config.num_batches:
                    break
                
                context, target, knowledge, _ = batch
                x_context, y_context = context
                x_target, y_target = target
                
                x_context = x_context.to(self.device)
                y_context = y_context.to(self.device)
                x_target = x_target.to(self.device)
                
                if isinstance(knowledge, torch.Tensor):
                    knowledge = knowledge.to(self.device)
                else:
                    continue
                
                with torch.no_grad():
                    context_repr = self._get_context_representation(
                        x_context, y_context, use_encoder=True
                    )
                    knowledge_repr = self._get_knowledge_embedding(knowledge)
                    
                    x_enc = self.model.x_encoder(x_context)
                    x_target_enc = self.model.x_encoder(x_target)
                    R = self.model.encode_globally(x_enc, y_context, x_target_enc)
                    z_samples, q_zCc, _ = self.model.sample_latent(
                        R, x_enc, x_target_enc, None, knowledge
                    )
                    z = q_zCc.base_dist.loc.squeeze(1)
                
                # Estimate MI (this updates MINE networks)
                mi_d, _ = self.mine_data.estimate(context_repr, z)
                mi_k, _ = self.mine_knowledge.estimate(knowledge_repr, z)
                mi_d_lb = self.mine_data.last_mi_lb
                mi_k_lb = self.mine_knowledge.last_mi_lb
                
                step_mi_data.append(mi_d)
                step_mi_knowledge.append(mi_k)
                if mi_d_lb is not None:
                    step_mi_data_lb.append(mi_d_lb)
                if mi_k_lb is not None:
                    step_mi_knowledge_lb.append(mi_k_lb)
            
            # Store mean estimates for this training step (filter out zeros from failed batches)
            valid_step_data = [v for v in step_mi_data if v > 0]
            valid_step_knowledge = [v for v in step_mi_knowledge if v > 0]
            
            if valid_step_data:
                mi_data_estimates.append(np.mean(valid_step_data))
            else:
                mi_data_estimates.append(0.0)
                
            if valid_step_knowledge:
                mi_knowledge_estimates.append(np.mean(valid_step_knowledge))
            else:
                mi_knowledge_estimates.append(0.0)

            valid_step_data_lb = [v for v in step_mi_data_lb if v > 0]
            valid_step_knowledge_lb = [v for v in step_mi_knowledge_lb if v > 0]

            if valid_step_data_lb:
                mi_data_lb_estimates.append(np.mean(valid_step_data_lb))
            else:
                mi_data_lb_estimates.append(0.0)

            if valid_step_knowledge_lb:
                mi_knowledge_lb_estimates.append(np.mean(valid_step_knowledge_lb))
            else:
                mi_knowledge_lb_estimates.append(0.0)
        
        # Compute final statistics
        # Use last 20% of estimates (after MINE has converged)
        converged_start = int(0.8 * len(mi_data_estimates))
        
        # Filter out NaN, Inf, and non-positive values (MI should be >= 0)
        def is_valid_mi(v):
            return (v is not None and 
                    not np.isnan(v) and 
                    not np.isinf(v) and 
                    v >= 0)
        
        mi_data_valid = [v for v in mi_data_estimates[converged_start:] if is_valid_mi(v)]
        mi_knowledge_valid = [v for v in mi_knowledge_estimates[converged_start:] if is_valid_mi(v)]
        
        mi_data_last20 = float(np.mean(mi_data_valid)) if mi_data_valid else 0.0
        mi_knowledge_last20 = float(np.mean(mi_knowledge_valid)) if mi_knowledge_valid else 0.0

        stable_data_summary = self._compute_stable_summary(mi_data_estimates)
        stable_knowledge_summary = self._compute_stable_summary(mi_knowledge_estimates)
        stable_data_lb_summary = self._compute_stable_summary(mi_data_lb_estimates)
        stable_knowledge_lb_summary = self._compute_stable_summary(mi_knowledge_lb_estimates)

        mi_data_final = (
            stable_data_summary["mean"]
            if stable_data_summary["num_points"] > 0
            else mi_data_last20
        )
        mi_knowledge_final = (
            stable_knowledge_summary["mean"]
            if stable_knowledge_summary["num_points"] > 0
            else mi_knowledge_last20
        )

        mi_data_lb_final = (
            stable_data_lb_summary["mean"]
            if stable_data_lb_summary["num_points"] > 0
            else 0.0
        )
        mi_knowledge_lb_final = (
            stable_knowledge_lb_summary["mean"]
            if stable_knowledge_lb_summary["num_points"] > 0
            else 0.0
        )
        
        # Handle NaN/Inf in final values
        if not is_valid_mi(mi_data_final):
            mi_data_final = 0.0
        if not is_valid_mi(mi_knowledge_final):
            mi_knowledge_final = 0.0
        
        # Compute knowledge reliance ratio with safety
        total_mi = mi_data_final + mi_knowledge_final
        if total_mi > 1e-8:
            knowledge_reliance = mi_knowledge_final / total_mi
            data_contribution = mi_data_final / total_mi
        else:
            knowledge_reliance = 0.5  # Default to balanced if no info
            data_contribution = 0.5
        
        # Clean history for JSON serialization (replace invalid values with None)
        def clean_value(v):
            if is_valid_mi(v):
                return float(v)
            return None
            
        mi_data_history_clean = [clean_value(v) for v in mi_data_estimates]
        mi_knowledge_history_clean = [clean_value(v) for v in mi_knowledge_estimates]
        mi_data_lb_history_clean = [clean_value(v) for v in mi_data_lb_estimates]
        mi_knowledge_lb_history_clean = [clean_value(v) for v in mi_knowledge_lb_estimates]
        
        results = {
            "mi_data_mean": float(mi_data_final),
            "mi_knowledge_mean": float(mi_knowledge_final),
            "knowledge_reliance_ratio": float(knowledge_reliance),
            "mi_data_mean_last20": float(mi_data_last20),
            "mi_knowledge_mean_last20": float(mi_knowledge_last20),
            "mi_data_lb_mean": float(mi_data_lb_final),
            "mi_knowledge_lb_mean": float(mi_knowledge_lb_final),
            "mi_data_history": mi_data_history_clean,
            "mi_knowledge_history": mi_knowledge_history_clean,
            "mi_data_lb_history": mi_data_lb_history_clean,
            "mi_knowledge_lb_history": mi_knowledge_lb_history_clean,
            "analysis": {
                "total_mutual_information": float(total_mi),
                "data_contribution": float(data_contribution),
                "knowledge_contribution": float(knowledge_reliance),
                "num_valid_data_estimates": len(mi_data_valid),
                "num_valid_knowledge_estimates": len(mi_knowledge_valid),
                "stable_summary": {
                    "data": stable_data_summary,
                    "knowledge": stable_knowledge_summary,
                },
                "stable_summary_lb": {
                    "data": stable_data_lb_summary,
                    "knowledge": stable_knowledge_lb_summary,
                },
                "last20_summary": {
                    "mi_data_mean": mi_data_last20,
                    "mi_knowledge_mean": mi_knowledge_last20,
                },
            }
        }
        
        # Interpretation
        if len(mi_data_valid) == 0 and len(mi_knowledge_valid) == 0:
            results["interpretation"] = "MINE estimation failed - all values NaN"
        elif knowledge_reliance > 0.5:
            results["interpretation"] = "Model relies more on knowledge than context data"
        elif knowledge_reliance < 0.2:
            results["interpretation"] = "Model primarily relies on context data"
        else:
            results["interpretation"] = "Model balances knowledge and context data"
        
        self.results = results
        self.save_results(results)
        
        # Generate comprehensive visualizations
        self._save_visualizations(results)
        
        # Restore model dropout setting
        self.model.latent_encoder.knowledge_dropout = original_knowledge_dropout
        
        return results
    
    def _save_visualizations(self, results: Dict[str, Any]):
        """Generate comprehensive visualizations for M1."""
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
            
            # ================================================================
            # PLOT 1: MI Training Curves
            # ================================================================
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Convert None to NaN for plotting (matplotlib handles NaN gracefully)
            mi_data_raw = results["mi_data_history"]
            mi_knowledge_raw = results["mi_knowledge_history"]
            mi_data = [v if v is not None else np.nan for v in mi_data_raw]
            mi_knowledge = [v if v is not None else np.nan for v in mi_knowledge_raw]
            steps = list(range(len(mi_data)))
            
            # Top-left: Both MI curves
            ax1 = axes[0, 0]
            ax1.plot(steps, mi_data, 'b-', linewidth=2.5, label='I(Z; Data)', alpha=0.8)
            ax1.plot(steps, mi_knowledge, 'r-', linewidth=2.5, label='I(Z; Knowledge)', alpha=0.8)
            ax1.fill_between(steps, mi_data, alpha=0.3, color='blue')
            ax1.fill_between(steps, mi_knowledge, alpha=0.3, color='red')
            ax1.set_xlabel('MINE Training Step', fontsize=12)
            ax1.set_ylabel('Mutual Information (nats)', fontsize=12)
            ax1.set_title('Mutual Information Estimation During MINE Training', fontsize=14, fontweight='bold')
            ax1.legend(fontsize=11, loc='best')
            ax1.grid(True, alpha=0.3)
            
            # Top-right: MI Convergence (last 50%)
            ax2 = axes[0, 1]
            converge_start = len(mi_data) // 2
            ax2.plot(steps[converge_start:], mi_data[converge_start:], 'b-', linewidth=2, label='I(Z; Data)')
            ax2.plot(steps[converge_start:], mi_knowledge[converge_start:], 'r-', linewidth=2, label='I(Z; Knowledge)')
            stable_data_mean = results.get("mi_data_mean", 0.0)
            stable_knowledge_mean = results.get("mi_knowledge_mean", 0.0)
            last20_data_mean = results.get("mi_data_mean_last20", stable_data_mean)
            last20_knowledge_mean = results.get("mi_knowledge_mean_last20", stable_knowledge_mean)
            ax2.axhline(
                y=stable_data_mean,
                color='blue',
                linestyle='--',
                alpha=0.7,
                label=f'Data Stable Mean: {stable_data_mean:.3f}'
            )
            ax2.axhline(
                y=stable_knowledge_mean,
                color='red',
                linestyle='--',
                alpha=0.7,
                label=f'Knowledge Stable Mean: {stable_knowledge_mean:.3f}'
            )
            ax2.axhline(
                y=last20_data_mean,
                color='blue',
                linestyle=':',
                alpha=0.5,
                label=f'Data Last20 Mean: {last20_data_mean:.3f}'
            )
            ax2.axhline(
                y=last20_knowledge_mean,
                color='red',
                linestyle=':',
                alpha=0.5,
                label=f'Knowledge Last20 Mean: {last20_knowledge_mean:.3f}'
            )
            ax2.set_xlabel('MINE Training Step', fontsize=12)
            ax2.set_ylabel('Mutual Information (nats)', fontsize=12)
            ax2.set_title('Converged MI Estimates (Stable vs Last20)', fontsize=14, fontweight='bold')
            ax2.legend(fontsize=10, loc='best')
            ax2.grid(True, alpha=0.3)
            
            # Bottom-left: Final MI Comparison Bar Chart
            ax3 = axes[1, 0]
            categories = ['I(Z; Data)', 'I(Z; Knowledge)']
            values = [stable_data_mean, stable_knowledge_mean]
            colors = sns.color_palette("rocket", n_colors=2)
            bars = ax3.bar(categories, values, color=colors, edgecolor='black', linewidth=1.5)
            for bar, val in zip(bars, values):
                ax3.annotate(f'{val:.3f}',
                            xy=(bar.get_x() + bar.get_width() / 2, val),
                            xytext=(0, 5), textcoords="offset points",
                            ha='center', va='bottom', fontsize=12, fontweight='bold')
            ax3.set_ylabel('Mutual Information (nats)', fontsize=12)
            ax3.set_title('Stable Mutual Information Estimates', fontsize=14, fontweight='bold')
            ax3.grid(True, alpha=0.3, axis='y')
            
            # Bottom-right: Knowledge Reliance Pie Chart
            ax4 = axes[1, 1]
            reliance = results["knowledge_reliance_ratio"]
            sizes = [reliance, 1 - reliance]
            labels = [f'Knowledge\n{reliance:.1%}', f'Data\n{1-reliance:.1%}']
            colors_pie = [palette[0], palette[3]]
            wedges, texts, autotexts = ax4.pie(sizes, labels=labels, colors=colors_pie, 
                                               autopct='', startangle=90,
                                               wedgeprops=dict(width=0.6, edgecolor='white', linewidth=2))
            ax4.set_title('Knowledge Reliance Ratio', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            fig.savefig(plots_dir / "m1_mi_analysis.png", dpi=150, bbox_inches='tight', facecolor='white')
            fig.savefig(plots_dir / "m1_mi_analysis.pdf", dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            
            # ================================================================
            # PLOT 2: MI Heatmap Over Training
            # ================================================================
            fig, ax = plt.subplots(figsize=(12, 4))
            
            # Create heatmap data
            window = max(1, len(mi_data) // 20)
            mi_data_smooth = np.convolve(mi_data, np.ones(window)/window, mode='valid')
            mi_knowledge_smooth = np.convolve(mi_knowledge, np.ones(window)/window, mode='valid')
            
            heatmap_data = np.vstack([mi_data_smooth, mi_knowledge_smooth])
            
            sns.heatmap(heatmap_data, ax=ax, cmap='rocket', 
                       xticklabels=max(1, len(mi_data_smooth)//10),
                       yticklabels=['I(Z; Data)', 'I(Z; Knowledge)'],
                       cbar_kws={'label': 'Mutual Information (nats)'})
            ax.set_xlabel('MINE Training Step', fontsize=12)
            ax.set_title('Mutual Information Evolution Heatmap', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            fig.savefig(plots_dir / "m1_mi_heatmap.png", dpi=150, bbox_inches='tight', facecolor='white')
            fig.savefig(plots_dir / "m1_mi_heatmap.pdf", dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            
            # ================================================================
            # PLOT 3: Information Flow Diagram
            # ================================================================
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Create Sankey-like diagram
            from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
            
            # Boxes
            box_style = dict(boxstyle="round,pad=0.3", facecolor='lightblue', edgecolor='navy', linewidth=2)
            
            # Data box
            ax.text(0.1, 0.7, 'Data\nContext\n$\\mathcal{D}_C$', fontsize=14, ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor='#3498db', edgecolor='#2c3e50', linewidth=2),
                   color='white', fontweight='bold')
            
            # Knowledge box
            ax.text(0.1, 0.3, 'Knowledge\n$\\mathcal{K}$', fontsize=14, ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor='#e74c3c', edgecolor='#c0392b', linewidth=2),
                   color='white', fontweight='bold')
            
            # Latent box
            ax.text(0.5, 0.5, f'Latent\n$z$', fontsize=14, ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor='#9b59b6', edgecolor='#8e44ad', linewidth=2),
                   color='white', fontweight='bold')
            
            # Output box
            ax.text(0.9, 0.5, 'Prediction\n$\\hat{y}$', fontsize=14, ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor='#2ecc71', edgecolor='#27ae60', linewidth=2),
                   color='white', fontweight='bold')
            
            # Arrows with MI values
            ax.annotate('', xy=(0.35, 0.55), xytext=(0.2, 0.65),
                       arrowprops=dict(arrowstyle='->', color='#3498db', lw=3))
            ax.text(0.27, 0.62, f'I(Z;D)={results["mi_data_mean"]:.2f}', fontsize=11, 
                   color='#3498db', fontweight='bold')
            
            ax.annotate('', xy=(0.35, 0.45), xytext=(0.2, 0.35),
                       arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=3))
            ax.text(0.27, 0.38, f'I(Z;K)={results["mi_knowledge_mean"]:.2f}', fontsize=11,
                   color='#e74c3c', fontweight='bold')
            
            ax.annotate('', xy=(0.75, 0.5), xytext=(0.65, 0.5),
                       arrowprops=dict(arrowstyle='->', color='#9b59b6', lw=3))
            
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            ax.set_title('Information Flow Diagram', fontsize=16, fontweight='bold', pad=20)
            
            # Add interpretation box
            interp_text = f"Knowledge Reliance: {results['knowledge_reliance_ratio']:.1%}\n{results['interpretation']}"
            ax.text(0.5, 0.05, interp_text, fontsize=11, ha='center', va='bottom',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor='#f0f0f0', edgecolor='gray'),
                   style='italic')
            
            plt.tight_layout()
            fig.savefig(plots_dir / "m1_information_flow.png", dpi=150, bbox_inches='tight', facecolor='white')
            fig.savefig(plots_dir / "m1_information_flow.pdf", dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            
            print(f"  M1 visualizations saved to {plots_dir}")
            
        except ImportError as e:
            print(f"Visualization libraries not available: {e}")


class TrainingMITracker:
    """
    Utility class to track mutual information during INP training.
    
    Usage:
        tracker = TrainingMITracker(model, device)
        
        # During training loop:
        for epoch in range(epochs):
            for batch in dataloader:
                # ... normal training ...
                
                # Track MI periodically
                if step % 100 == 0:
                    mi_d, mi_k = tracker.track_step(
                        context_repr, knowledge_repr, z
                    )
                    wandb.log({"mi_data": mi_d, "mi_knowledge": mi_k})
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        hidden_dim: int = 128,
        lr: float = 1e-4
    ):
        self.model = model
        self.device = torch.device(device)
        self.hidden_dim = hidden_dim
        self.lr = lr
        
        self.mine_data: Optional[MINEEstimator] = None
        self.mine_knowledge: Optional[MINEEstimator] = None
        self.initialized = False
        
        self.history = {
            "mi_data": [],
            "mi_knowledge": [],
            "steps": []
        }
        self.step_count = 0
    
    def _initialize(self, context_dim: int, knowledge_dim: int, z_dim: int):
        """Lazily initialize MINE estimators."""
        self.mine_data = MINEEstimator(
            context_dim, z_dim, self.hidden_dim, self.lr, str(self.device)
        )
        self.mine_knowledge = MINEEstimator(
            knowledge_dim, z_dim, self.hidden_dim, self.lr, str(self.device)
        )
        self.initialized = True
    
    def track_step(
        self,
        context_repr: torch.Tensor,
        knowledge_repr: torch.Tensor,
        z: torch.Tensor
    ) -> Tuple[float, float]:
        """
        Track MI for one step.
        
        Args:
            context_repr: Encoded context [batch, context_dim]
            knowledge_repr: Encoded knowledge [batch, knowledge_dim]
            z: Latent code [batch, z_dim]
            
        Returns:
            Tuple of (I(Z;D), I(Z;K)) estimates
        """
        if not self.initialized:
            self._initialize(
                context_repr.shape[-1],
                knowledge_repr.shape[-1],
                z.shape[-1]
            )
        
        # Detach to avoid affecting training gradients
        context_repr = context_repr.detach()
        knowledge_repr = knowledge_repr.detach()
        z = z.detach()
        
        mi_d, _ = self.mine_data.estimate(context_repr, z)
        mi_k, _ = self.mine_knowledge.estimate(knowledge_repr, z)
        
        self.history["mi_data"].append(mi_d)
        self.history["mi_knowledge"].append(mi_k)
        self.history["steps"].append(self.step_count)
        self.step_count += 1
        
        return mi_d, mi_k
    
    def get_history(self) -> Dict[str, List[float]]:
        """Return the tracking history."""
        return self.history
