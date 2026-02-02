"""
M7: Linear Probing of Latent Representations

This module tests whether INP latents explicitly encode ground-truth generative
parameters in a linearly decodable way.

Hypothesis: If INP truly "disentangles" task structure using knowledge, the latent
variable z should explicitly encode ground-truth parameters (amplitude, frequency, phase).

Test:
    - Train a simple linear probe: f_probe(z) = Wz + b
    - Predict ground-truth parameters θ_GT from z
    - Measure R² score for parameter recovery

Expected Results:
    - INP latents: High R² with LINEAR probe (disentangled representation)
    - NP latents: Low R² with linear probe, may need MLP (entangled representation)
    
This is a key test of representation quality from the disentanglement literature.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from .base import InterpretabilityExperiment, ExperimentConfig


class LinearProbe(nn.Module):
    """Simple linear probe for parameter prediction."""
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class MLPProbe(nn.Module):
    """MLP probe for comparison with linear probe."""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LinearProbingExperiment(InterpretabilityExperiment):
    """
    M7: Linear Probing Analysis
    
    Tests whether latent representations linearly encode ground-truth parameters.
    
    Procedure:
        1. Collect latent codes z from the model
        2. Extract ground-truth parameters from knowledge (for sinusoids: a, b, c)
        3. Train linear probe: z → θ_GT
        4. Evaluate R² score
        5. Compare with MLP probe to test linearity hypothesis
        6. Compare INP (with knowledge) vs NP (without knowledge)
    
    Key Metrics:
        - R² (linear): How linearly decodable are the parameters?
        - R² (MLP): Upper bound on decodability
        - Linear gap: R²(MLP) - R²(linear), measures non-linearity
        - Per-parameter R²: Which parameters are best encoded?
    """
    
    name = "m7_linear_probing"
    description = "Linear probing of latent representations for parameter recovery"
    
    def __init__(
        self,
        model: nn.Module,
        config: ExperimentConfig,
        probe_epochs: int = 100,
        probe_lr: float = 1e-3,
        test_fraction: float = 0.2,
        use_sklearn: bool = True,
    ):
        """
        Args:
            model: Trained INP model
            config: Experiment configuration
            probe_epochs: Number of epochs for PyTorch probe training
            probe_lr: Learning rate for probe training
            test_fraction: Fraction of data for testing
            use_sklearn: If True, use sklearn for faster/more robust fitting
        """
        super().__init__(model, config)
        self.probe_epochs = probe_epochs
        self.probe_lr = probe_lr
        self.test_fraction = test_fraction
        self.use_sklearn = use_sklearn
    
    def _disable_knowledge_dropout(self) -> Optional[float]:
        latent_encoder = getattr(self.model, "latent_encoder", None)
        if latent_encoder is None or not hasattr(latent_encoder, "knowledge_dropout"):
            return None
        original = float(latent_encoder.knowledge_dropout)
        latent_encoder.knowledge_dropout = 0.0
        return original

    def _can_extract_full_params(self, dataset) -> bool:
        return (
            hasattr(dataset, "data")
            and hasattr(dataset, "knowledge")
            and hasattr(dataset.data, "columns")
            and "curve_id" in dataset.data.columns
            and hasattr(dataset.knowledge, "columns")
            and "curve_id" in dataset.knowledge.columns
        )

    def _get_full_params_from_dataset(self, dataset, idxs: List[int]) -> Optional[np.ndarray]:
        try:
            curve_ids = dataset.data.iloc[idxs]["curve_id"].values
            knowledge_df = dataset.knowledge.set_index("curve_id")
            params = knowledge_df.loc[curve_ids].values
            return np.asarray(params)
        except Exception:
            return None

    def _collect_latents_and_params(
        self,
        dataloader: torch.utils.data.DataLoader,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        """
        Collect aligned latent codes (with/without knowledge) and parameters.
        
        Returns:
            Tuple of (latents_with_k [N, latent_dim],
                     latents_without_k [N, latent_dim],
                     params [N, num_params],
                     params_source ["dataset" or "knowledge"])
        """
        latents_k = []
        latents_no_k = []
        params = []
        
        dataset = getattr(dataloader, "dataset", None)
        use_full_params = self._can_extract_full_params(dataset) if dataset is not None else False
        params_source = "dataset" if use_full_params else "knowledge"
        sample_idx = 0
        
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Collecting latents"):
                context, target, knowledge, _ = batch
                x_context, y_context = context
                x_target, y_target = target
                
                x_context = x_context.to(self.device)
                y_context = y_context.to(self.device)
                x_target = x_target.to(self.device)
                
                batch_size = x_context.shape[0]
                idxs = list(range(sample_idx, sample_idx + batch_size))
                sample_idx += batch_size
                
                if isinstance(knowledge, torch.Tensor):
                    knowledge_tensor = knowledge.to(self.device)
                else:
                    continue  # Skip non-tensor knowledge
                
                # Encode context once
                x_enc = self.model.x_encoder(x_context)
                x_target_enc = self.model.x_encoder(x_target)
                R = self.model.encode_globally(x_enc, y_context, x_target_enc)
                
                # Latents with knowledge
                q_z_k = self.model.infer_latent_dist(R, knowledge_tensor, x_context.shape[1])
                mu_z_k = q_z_k.base_dist.loc.squeeze(1)
                latents_k.append(mu_z_k.cpu().numpy())
                
                # Latents without knowledge
                q_z_no_k = self.model.infer_latent_dist(R, None, x_context.shape[1])
                mu_z_no_k = q_z_no_k.base_dist.loc.squeeze(1)
                latents_no_k.append(mu_z_no_k.cpu().numpy())
                
                # Ground-truth params
                if use_full_params and dataset is not None:
                    gt_params = self._get_full_params_from_dataset(dataset, idxs)
                    if gt_params is None:
                        use_full_params = False
                        params_source = "knowledge"
                
                if not use_full_params:
                    if knowledge_tensor.dim() == 3:
                        gt_params = knowledge_tensor[:, :, -1].cpu().numpy()  # [batch, 3]
                    elif knowledge_tensor.dim() == 2:
                        gt_params = knowledge_tensor.cpu().numpy()
                    else:
                        gt_params = knowledge_tensor.reshape(knowledge_tensor.size(0), -1).cpu().numpy()
                
                params.append(gt_params)
        
        latents_k = np.concatenate(latents_k, axis=0)
        latents_no_k = np.concatenate(latents_no_k, axis=0)
        params = np.concatenate(params, axis=0)
        
        if params.ndim == 1:
            params = params.reshape(-1, 1)
        
        return latents_k, latents_no_k, params, params_source
    
    def _train_sklearn_probe(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        probe_type: str = "linear",
    ) -> Dict[str, float]:
        """
        Train probe using sklearn and evaluate.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            probe_type: "linear" or "mlp"
            
        Returns:
            Dictionary with R² scores
        """
        # Ensure y arrays are 2D
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)
        if y_test.ndim == 1:
            y_test = y_test.reshape(-1, 1)
        
        # Standardize features
        scaler_X = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        
        # Standardize targets
        scaler_y = StandardScaler()
        y_train_scaled = scaler_y.fit_transform(y_train)
        y_test_scaled = scaler_y.transform(y_test)
        
        results = {}
        
        if probe_type == "linear":
            # Train Ridge regression (regularized linear)
            model = Ridge(alpha=1.0)
            model.fit(X_train_scaled, y_train_scaled)
            y_pred_scaled = model.predict(X_test_scaled)
            
        else:  # MLP
            model = MLPRegressor(
                hidden_layer_sizes=(64, 64),
                max_iter=500,
                early_stopping=True,
                random_state=42,
            )
            model.fit(X_train_scaled, y_train_scaled)
            y_pred_scaled = model.predict(X_test_scaled)
        
        # Ensure predictions are 2D for inverse_transform
        if y_pred_scaled.ndim == 1:
            y_pred_scaled = y_pred_scaled.reshape(-1, 1)
        
        # Inverse transform predictions
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        
        # Compute overall R²
        results["r2_overall"] = float(r2_score(y_test, y_pred))
        
        # Per-parameter R²
        for i in range(y_test.shape[1]):
            results[f"r2_param_{i}"] = float(r2_score(y_test[:, i], y_pred[:, i]))
        
        # MSE
        results["mse"] = float(np.mean((y_test - y_pred) ** 2))
        
        return results
    
    def _train_pytorch_probe(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        probe_type: str = "linear",
    ) -> Dict[str, float]:
        """
        Train probe using PyTorch and evaluate.
        """
        # Ensure y arrays are 2D
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)
        if y_test.ndim == 1:
            y_test = y_test.reshape(-1, 1)
        
        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).to(self.device)
        X_test_t = torch.FloatTensor(X_test).to(self.device)
        y_test_t = torch.FloatTensor(y_test).to(self.device)
        
        # Normalize
        X_mean, X_std = X_train_t.mean(0), X_train_t.std(0) + 1e-8
        y_mean, y_std = y_train_t.mean(0), y_train_t.std(0) + 1e-8
        
        X_train_n = (X_train_t - X_mean) / X_std
        X_test_n = (X_test_t - X_mean) / X_std
        y_train_n = (y_train_t - y_mean) / y_std
        
        # Create probe
        input_dim = X_train.shape[1]
        output_dim = y_train.shape[1]
        
        if probe_type == "linear":
            probe = LinearProbe(input_dim, output_dim).to(self.device)
        else:
            probe = MLPProbe(input_dim, output_dim).to(self.device)
        
        optimizer = optim.Adam(probe.parameters(), lr=self.probe_lr)
        
        # Create data loader
        dataset = TensorDataset(X_train_n, y_train_n)
        loader = DataLoader(dataset, batch_size=256, shuffle=True)
        
        # Train
        probe.train()
        for epoch in range(self.probe_epochs):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                pred = probe(batch_X)
                loss = F.mse_loss(pred, batch_y)
                loss.backward()
                optimizer.step()
        
        # Evaluate
        probe.eval()
        with torch.no_grad():
            y_pred_n = probe(X_test_n)
            y_pred = y_pred_n * y_std + y_mean
            y_pred = y_pred.cpu().numpy()
        
        results = {}
        results["r2_overall"] = float(r2_score(y_test, y_pred))
        
        for i in range(y_test.shape[1]):
            results[f"r2_param_{i}"] = float(r2_score(y_test[:, i], y_pred[:, i]))
        
        results["mse"] = float(np.mean((y_test - y_pred) ** 2))
        
        return results
    
    def run(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, Any]:
        """
        Run the linear probing analysis.
        
        Procedure:
            1. Collect latents WITH knowledge (INP)
            2. Collect latents WITHOUT knowledge (NP baseline)
            3. Train linear and MLP probes on each
            4. Compare R² scores
        
        Returns:
            Dictionary containing R² scores and analysis
        """
        results = {
            "config": {
                "probe_epochs": self.probe_epochs,
                "test_fraction": self.test_fraction,
                "use_sklearn": self.use_sklearn,
            }
        }
        
        original_dropout = self._disable_knowledge_dropout()
        
        try:
            ordered_loader = DataLoader(
                dataloader.dataset,
                batch_size=dataloader.batch_size or self.config.batch_size,
                shuffle=False,
                collate_fn=dataloader.collate_fn,
                drop_last=False,
            )
            
            # Collect latents WITH/WITHOUT knowledge in aligned order
            print("\nCollecting aligned latents with/without knowledge...")
            latents_k, latents_no_k, params, params_source = self._collect_latents_and_params(ordered_loader)
            print(f"Collected {latents_k.shape[0]} samples, latent_dim={latents_k.shape[1]}, params={params.shape[1]}")
        
        finally:
            if original_dropout is not None:
                self.model.latent_encoder.knowledge_dropout = original_dropout
        
        results["config"]["params_source"] = params_source
        results["config"]["knowledge_dropout_original"] = original_dropout
        results["config"]["knowledge_dropout_set_to"] = 0.0 if original_dropout is not None else None
        results["config"]["ordered_loader"] = True
        
        # Filter masked-knowledge samples only when params come from knowledge
        filtered_masked = False
        if params_source != "dataset":
            valid_mask = np.any(params != 0, axis=1)
            if np.any(valid_mask) and not np.all(valid_mask):
                latents_k = latents_k[valid_mask]
                latents_no_k = latents_no_k[valid_mask]
                params = params[valid_mask]
                filtered_masked = True
        
        results["config"]["filtered_masked_params"] = filtered_masked
        print(f"After filtering: {latents_k.shape[0]} valid samples")
        
        if latents_k.shape[0] < 100:
            results["error"] = "Not enough valid samples for probing"
            return results
        
        # Train/test split
        n = latents_k.shape[0]
        n_test = int(n * self.test_fraction)
        indices = np.random.permutation(n)
        train_idx, test_idx = indices[n_test:], indices[:n_test]
        
        X_train_k, X_test_k = latents_k[train_idx], latents_k[test_idx]
        X_train_no_k, X_test_no_k = latents_no_k[train_idx], latents_no_k[test_idx]
        y_train, y_test = params[train_idx], params[test_idx]
        
        print(f"\nTrain: {len(train_idx)}, Test: {len(test_idx)}")
        
        # Train probes
        train_func = self._train_sklearn_probe if self.use_sklearn else self._train_pytorch_probe
        
        print("\nTraining probes on INP latents (with knowledge)...")
        results["with_knowledge"] = {
            "linear": train_func(X_train_k, y_train, X_test_k, y_test, "linear"),
            "mlp": train_func(X_train_k, y_train, X_test_k, y_test, "mlp"),
        }
        
        print("Training probes on NP latents (without knowledge)...")
        results["without_knowledge"] = {
            "linear": train_func(X_train_no_k, y_train, X_test_no_k, y_test, "linear"),
            "mlp": train_func(X_train_no_k, y_train, X_test_no_k, y_test, "mlp"),
        }
        
        # Analysis
        r2_k_linear = results["with_knowledge"]["linear"]["r2_overall"]
        r2_k_mlp = results["with_knowledge"]["mlp"]["r2_overall"]
        r2_no_k_linear = results["without_knowledge"]["linear"]["r2_overall"]
        r2_no_k_mlp = results["without_knowledge"]["mlp"]["r2_overall"]
        
        results["comparison"] = {
            "r2_linear_with_k": r2_k_linear,
            "r2_linear_without_k": r2_no_k_linear,
            "r2_mlp_with_k": r2_k_mlp,
            "r2_mlp_without_k": r2_no_k_mlp,
            "linear_gap_with_k": r2_k_mlp - r2_k_linear,
            "linear_gap_without_k": r2_no_k_mlp - r2_no_k_linear,
            "knowledge_benefit_linear": r2_k_linear - r2_no_k_linear,
            "knowledge_benefit_mlp": r2_k_mlp - r2_no_k_mlp,
        }
        
        # Parameter names for sinusoids
        n_params = params.shape[1]
        if n_params == 3:
            results["parameter_names"] = ["a (amplitude)", "b (frequency)", "c (phase)"]
        elif n_params == 1:
            results["parameter_names"] = ["b (frequency)"]  # For dist_shift models
        else:
            results["parameter_names"] = [f"param_{i}" for i in range(n_params)]
        
        results["per_parameter"] = {}
        for i, name in enumerate(results["parameter_names"]):
            results["per_parameter"][name] = {
                "r2_linear_with_k": results["with_knowledge"]["linear"][f"r2_param_{i}"],
                "r2_linear_without_k": results["without_knowledge"]["linear"][f"r2_param_{i}"],
            }
        
        # Interpretation
        interpretation_parts = []
        
        if r2_k_linear > 0.8:
            interpretation_parts.append(
                f"Strong disentanglement: Linear probe achieves R²={r2_k_linear:.3f}"
            )
        elif r2_k_linear > 0.5:
            interpretation_parts.append(
                f"Moderate disentanglement: Linear probe achieves R²={r2_k_linear:.3f}"
            )
        else:
            interpretation_parts.append(
                f"Weak disentanglement: Linear probe only achieves R²={r2_k_linear:.3f}"
            )
        
        linear_gap = r2_k_mlp - r2_k_linear
        if linear_gap < 0.1:
            interpretation_parts.append(
                "Representation is highly linear (MLP adds little)"
            )
        else:
            interpretation_parts.append(
                f"Representation has non-linear structure (MLP improves by {linear_gap:.3f})"
            )
        
        knowledge_benefit = r2_k_linear - r2_no_k_linear
        if knowledge_benefit > 0.2:
            interpretation_parts.append(
                f"Knowledge significantly improves linear decodability (+{knowledge_benefit:.3f})"
            )
        elif knowledge_benefit > 0:
            interpretation_parts.append(
                f"Knowledge modestly improves linear decodability (+{knowledge_benefit:.3f})"
            )
        else:
            interpretation_parts.append(
                "Knowledge does not improve linear decodability"
            )
        
        results["interpretation"] = ". ".join(interpretation_parts)
        
        self.results = results
        self.save_results(results)
        
        # Generate visualization
        self._save_visualization(results)
        
        return results
    
    def _save_visualization(self, results: Dict[str, Any]):
        """Generate comprehensive visualizations for M7."""
        try:
            from .enhanced_viz import viz_m7_linear_probing
            viz_m7_linear_probing(results, self.output_dir)
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
            
            comp = results.get("comparison", {})
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 12))
            
            # Top-left: Linear vs MLP comparison
            ax1 = axes[0, 0]
            categories = ["With Knowledge\n(INP)", "Without Knowledge\n(NP)"]
            linear_scores = [comp.get("r2_linear_with_k", 0), comp.get("r2_linear_without_k", 0)]
            mlp_scores = [comp.get("r2_mlp_with_k", 0), comp.get("r2_mlp_without_k", 0)]
            
            x = np.arange(len(categories))
            width = 0.35
            
            bars1 = ax1.bar(x - width/2, linear_scores, width, label='Linear Probe',
                           color=palette[0], edgecolor='black')
            bars2 = ax1.bar(x + width/2, mlp_scores, width, label='MLP Probe',
                           color=palette[4], edgecolor='black')
            
            ax1.set_ylabel('R² Score', fontsize=12)
            ax1.set_title('Parameter Recovery: Linear vs MLP', fontsize=14, fontweight='bold')
            ax1.set_xticks(x)
            ax1.set_xticklabels(categories)
            ax1.legend(fontsize=11)
            ax1.set_ylim(0, 1.15)
            ax1.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
            ax1.grid(True, alpha=0.3, axis='y')
            
            for bars in [bars1, bars2]:
                for bar in bars:
                    ax1.annotate(f'{bar.get_height():.3f}',
                                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                                xytext=(0, 3), textcoords="offset points",
                                ha='center', fontsize=10, fontweight='bold')
            
            # Top-right: Per-parameter breakdown
            ax2 = axes[0, 1]
            if "per_parameter" in results:
                params = list(results["per_parameter"].keys())
                r2_k = [results["per_parameter"][p]["r2_linear_with_k"] for p in params]
                r2_no_k = [results["per_parameter"][p]["r2_linear_without_k"] for p in params]
                
                x = np.arange(len(params))
                bars1 = ax2.bar(x - width/2, r2_k, width, label='With Knowledge',
                               color=palette[0], edgecolor='black')
                bars2 = ax2.bar(x + width/2, r2_no_k, width, label='Without Knowledge',
                               color=palette[5], edgecolor='black')
                
                ax2.set_ylabel('R² Score (Linear)', fontsize=12)
                ax2.set_title('Per-Parameter Linear Recovery', fontsize=14, fontweight='bold')
                ax2.set_xticks(x)
                ax2.set_xticklabels([p.split()[0] for p in params])
                ax2.legend(fontsize=10)
                ax2.set_ylim(0, 1.15)
                ax2.grid(True, alpha=0.3, axis='y')
            
            # Bottom-left: Knowledge benefit
            ax3 = axes[1, 0]
            metrics = ['Linear R²\nBenefit', 'MLP R²\nBenefit', 'Linear Gap\nReduction']
            benefits = [
                comp.get("knowledge_benefit_linear", 0),
                comp.get("knowledge_benefit_mlp", 0),
                comp.get("linear_gap_without_k", 0) - comp.get("linear_gap_with_k", 0)
            ]
            colors = ['green' if b > 0 else 'red' for b in benefits]
            bars = ax3.bar(metrics, benefits, color=colors, edgecolor='black')
            ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
            ax3.set_ylabel('Improvement', fontsize=12)
            ax3.set_title('Knowledge Benefit', fontsize=14, fontweight='bold')
            ax3.grid(True, alpha=0.3, axis='y')
            
            for bar, val in zip(bars, benefits):
                ax3.annotate(f'{val:+.3f}',
                            xy=(bar.get_x() + bar.get_width() / 2, val),
                            xytext=(0, 5 if val >= 0 else -15),
                            textcoords="offset points",
                            ha='center', fontsize=11, fontweight='bold')
            
            # Bottom-right: Heatmap
            ax4 = axes[1, 1]
            if "per_parameter" in results:
                params = list(results["per_parameter"].keys())
                heatmap_data = np.array([
                    [results["per_parameter"][p]["r2_linear_with_k"] for p in params],
                    [results["per_parameter"][p]["r2_linear_without_k"] for p in params]
                ])
                
                sns.heatmap(heatmap_data, ax=ax4, cmap='rocket', vmin=0, vmax=1,
                           xticklabels=[p.split()[0] for p in params],
                           yticklabels=['With K', 'Without K'],
                           annot=True, fmt='.3f', cbar_kws={'label': 'R²'},
                           linewidths=1, linecolor='white')
                ax4.set_title('Linear Probe R² Heatmap', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            fig.savefig(plots_dir / "m7_linear_probing.png", dpi=150, bbox_inches='tight', facecolor='white')
            fig.savefig(plots_dir / "m7_linear_probing.pdf", dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            
            print(f"  M7 visualizations saved to {plots_dir}")
            
        except ImportError as e:
            print(f"Visualization libraries not available: {e}")
