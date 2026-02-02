"""
M6: Knowledge Saliency & Attribution via Integrated Gradients

This module analyzes which parts of the knowledge input drive predictions
using Integrated Gradients (IG) attribution.

For text knowledge: Which words/tokens are most important?
    - Validates that INP attends to semantic content ("increasing", "periodic")
    - Not just stop words ("the", "is")

For numeric knowledge: Which parameters matter most?
    - For sinusoids (a, b, c): Which parameter dominates predictions?

Integrated Gradients Formula:
    IG_i(x) = (x_i - x'_i) × ∫_{α=0}^{1} ∂F(x' + α(x - x')) / ∂x_i dα

where x' is a baseline (typically zero embedding).

Properties:
    - Completeness: Σ IG_i = F(x) - F(x')
    - Sensitivity: If feature differs from baseline and affects output, attribution ≠ 0
    - Implementation Invariance: Same attributions regardless of network architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
from tqdm import tqdm
from pathlib import Path

from .base import InterpretabilityExperiment, ExperimentConfig


def integrated_gradients(
    forward_func,
    inputs: torch.Tensor,
    baseline: Optional[torch.Tensor] = None,
    n_steps: int = 50,
    return_convergence_delta: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, float]]:
    """
    Compute Integrated Gradients attribution.
    
    This is a standalone implementation that doesn't require Captum.
    
    Args:
        forward_func: Function that takes inputs and returns a scalar
        inputs: Input tensor to attribute [batch, ...]
        baseline: Baseline tensor (default: zeros)
        n_steps: Number of integration steps
        return_convergence_delta: If True, return the convergence delta
        
    Returns:
        Attribution tensor with same shape as inputs
        Optionally, convergence delta (should be close to 0)
    """
    if baseline is None:
        baseline = torch.zeros_like(inputs)
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1 for integrated gradients.")
    
    inputs = inputs.clone().detach()
    baseline = baseline.clone().detach()
    
    # Generate interpolation points
    alphas = torch.linspace(0, 1, n_steps, device=inputs.device)
    
    # Accumulate gradients along path
    integrated_grads = torch.zeros_like(inputs)
    
    for alpha in alphas:
        # Interpolated input
        interpolated = baseline + alpha * (inputs - baseline)
        interpolated = interpolated.clone().detach().requires_grad_(True)
        
        # Forward pass
        output = forward_func(interpolated)
        if output is None:
            continue
        if output.dim() > 0:
            output = output.sum()
        
        grads = torch.autograd.grad(
            outputs=output,
            inputs=interpolated,
            retain_graph=False,
            allow_unused=True,
        )[0]
        
        if grads is not None:
            integrated_grads += grads
    
    # Average and scale by (input - baseline)
    integrated_grads = integrated_grads / n_steps
    attributions = (inputs - baseline) * integrated_grads
    
    if return_convergence_delta:
        # Completeness check: sum of attributions should equal F(x) - F(baseline)
        with torch.no_grad():
            f_input = forward_func(inputs)
            f_baseline = forward_func(baseline)
            if f_input.dim() > 0:
                f_input = f_input.sum()
            if f_baseline.dim() > 0:
                f_baseline = f_baseline.sum()
            expected_diff = (f_input - f_baseline).item()
            actual_diff = attributions.sum().item()
            delta = abs(expected_diff - actual_diff)
        return attributions, delta
    
    return attributions


class KnowledgeSaliencyExperiment(InterpretabilityExperiment):
    """
    M6: Knowledge Saliency & Attribution Analysis
    
    Uses Integrated Gradients to determine which parts of the knowledge
    input are most important for predictions.
    
    Supports:
        - Numeric knowledge (SetEmbedding): Feature-level attribution
        - Text knowledge (RoBERTa): Token-level attribution
        
    Key outputs:
        - Per-feature/token importance scores
        - Aggregated importance across dataset
        - Visualization of knowledge saliency
    """
    
    name = "m6_knowledge_saliency"
    description = "Integrated Gradients attribution for knowledge inputs"
    
    def __init__(
        self,
        model: nn.Module,
        config: ExperimentConfig,
        n_steps: int = 50,
        use_captum: bool = False,
    ):
        """
        Args:
            model: Trained INP model
            config: Experiment configuration
            n_steps: Number of integration steps for IG
            use_captum: If True and available, use Captum library
        """
        super().__init__(model, config)
        self.n_steps = n_steps
        self.use_captum = use_captum
        self.text_tokenizer = None
        
        # Check if using text encoder
        self.is_text_knowledge = False
        if hasattr(self.model.latent_encoder, 'knowledge_encoder') and \
           self.model.latent_encoder.knowledge_encoder is not None:
            encoder = self.model.latent_encoder.knowledge_encoder
            if hasattr(encoder, 'text_encoder') and hasattr(encoder.text_encoder, 'tokenizer'):
                self.is_text_knowledge = True
                self.text_tokenizer = encoder.text_encoder.tokenizer

    def _get_text_components(self):
        if not hasattr(self.model, "latent_encoder") or self.model.latent_encoder is None:
            raise ValueError("Model has no latent encoder for text attribution.")
        encoder = self.model.latent_encoder.knowledge_encoder
        if encoder is None or not hasattr(encoder, "text_encoder"):
            raise ValueError("Knowledge encoder does not support text attribution.")
        text_encoder = encoder.text_encoder
        if not hasattr(text_encoder, "tokenizer") or not hasattr(text_encoder, "llm"):
            raise ValueError("Text encoder missing tokenizer/llm.")
        return encoder, text_encoder, text_encoder.tokenizer

    def _compute_text_attribution(
        self,
        x_context: torch.Tensor,
        y_context: torch.Tensor,
        x_target: torch.Tensor,
        knowledge_text: List[str],
    ) -> Tuple[torch.Tensor, float, List[List[str]]]:
        """
        Compute token-level attribution for text knowledge using IG on input embeddings.
        """
        encoder, text_encoder, tokenizer = self._get_text_components()

        tokenized = tokenizer.batch_encode_plus(
            knowledge_text,
            return_tensors="pt",
            return_token_type_ids=True,
            padding=True,
            truncation=True,
        )
        input_ids = tokenized["input_ids"].to(self.device)
        attention_mask = tokenized["attention_mask"].to(self.device)
        token_type_ids = tokenized.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(self.device)

        embed_layer = text_encoder.llm.get_input_embeddings()
        input_embeds = embed_layer(input_ids)
        baseline = torch.zeros_like(input_embeds)

        x_context_enc = self.model.x_encoder(x_context)
        x_target_enc = self.model.x_encoder(x_target)
        R = self.model.encode_globally(x_context_enc, y_context, x_target_enc)
        config = self.model.config
        latent_encoder = self.model.latent_encoder

        def forward_from_embeds(embeds: torch.Tensor) -> torch.Tensor:
            llm_output = text_encoder.llm(
                inputs_embeds=embeds,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            hidden_state = llm_output[0]
            cls = hidden_state[:, 0]
            k_embed = encoder.knowledge_extractor(cls)
            if k_embed.dim() == 2:
                k_embed = k_embed.unsqueeze(1)

            if config.knowledge_merge == "sum":
                encoder_input = F.relu(R + k_embed)
            elif config.knowledge_merge == "concat":
                encoder_input = torch.cat([R, k_embed], dim=-1)
            elif config.knowledge_merge == "mlp":
                encoder_input = latent_encoder.knowledge_merger(torch.cat([R, k_embed], dim=-1))
            else:
                encoder_input = F.relu(R + k_embed)

            q_z_stats = latent_encoder.encoder(encoder_input)
            q_z_loc, _ = q_z_stats.split(config.hidden_dim, dim=-1)
            z = q_z_loc.unsqueeze(0)

            R_target = z.expand(-1, -1, x_target.shape[1], -1)
            p_y_stats = self.model.decoder(x_target_enc, R_target)
            p_y_loc, _ = p_y_stats.split(config.output_dim, dim=-1)
            return p_y_loc.mean()

        attributions, delta = integrated_gradients(
            forward_from_embeds,
            input_embeds,
            baseline=baseline,
            n_steps=self.n_steps,
            return_convergence_delta=True,
        )

        token_attributions = attributions.abs().sum(dim=-1)
        token_attributions = token_attributions * attention_mask
        tokens = [tokenizer.convert_ids_to_tokens(ids) for ids in input_ids]
        return token_attributions, delta, tokens

    def _accumulate_token_importance(
        self,
        results: Dict[str, Any],
        tokens: List[List[str]],
        token_attributions: torch.Tensor,
    ) -> None:
        if "token_importance" not in results:
            results["token_importance"] = {}
            results["token_importance_counts"] = {}
        special_tokens = set(self.text_tokenizer.all_special_tokens) if self.text_tokenizer else set()
        for token_list, scores in zip(tokens, token_attributions):
            for token, score in zip(token_list, scores):
                if token in special_tokens:
                    continue
                score_val = float(score.item())
                results["token_importance"][token] = results["token_importance"].get(token, 0.0) + score_val
                results["token_importance_counts"][token] = results["token_importance_counts"].get(token, 0) + 1

    def _disable_knowledge_dropout(self) -> Optional[float]:
        latent_encoder = getattr(self.model, "latent_encoder", None)
        if latent_encoder is None or not hasattr(latent_encoder, "knowledge_dropout"):
            return None
        original = float(latent_encoder.knowledge_dropout)
        latent_encoder.knowledge_dropout = 0.0
        return original
    
    def _create_forward_func(
        self,
        x_context: torch.Tensor,
        y_context: torch.Tensor,
        x_target: torch.Tensor,
        output_type: str = "mean_prediction",
    ):
        """
        Create a forward function that takes knowledge embedding and returns scalar.
        
        This function will be used by Integrated Gradients.
        """
        def forward_from_knowledge_embedding(k_embed: torch.Tensor) -> torch.Tensor:
            """
            Forward pass using pre-computed knowledge embedding.
            
            Args:
                k_embed: Knowledge embedding [batch, 1, knowledge_dim] or [batch, knowledge_dim]
            """
            # Ensure correct shape
            if k_embed.dim() == 2:
                k_embed = k_embed.unsqueeze(1)
            
            # Encode context
            x_context_enc = self.model.x_encoder(x_context)
            x_target_enc = self.model.x_encoder(x_target)
            R = self.model.encode_globally(x_context_enc, y_context, x_target_enc)
            
            # Manually merge knowledge with representation
            # This bypasses the knowledge encoder to allow attribution
            config = self.model.config
            
            if config.knowledge_merge == "sum":
                encoder_input = F.relu(R + k_embed)
            elif config.knowledge_merge == "concat":
                encoder_input = torch.cat([R, k_embed], dim=-1)
            elif config.knowledge_merge == "mlp":
                encoder_input = self.model.latent_encoder.knowledge_merger(
                    torch.cat([R, k_embed], dim=-1)
                )
            else:
                encoder_input = F.relu(R + k_embed)
            
            # Get latent distribution
            q_z_stats = self.model.latent_encoder.encoder(encoder_input)
            q_z_loc, q_z_scale = q_z_stats.split(config.hidden_dim, dim=-1)
            
            # Use mean of latent (deterministic for attribution)
            z = q_z_loc.unsqueeze(0)  # [1, batch, 1, hidden_dim]
            
            # Decode
            R_target = z.expand(-1, -1, x_target.shape[1], -1)
            p_y_stats = self.model.decoder(x_target_enc, R_target)
            p_y_loc, _ = p_y_stats.split(config.output_dim, dim=-1)
            
            # Return scalar output
            if output_type == "mean_prediction":
                return p_y_loc.mean()
            elif output_type == "sum_prediction":
                return p_y_loc.sum()
            else:
                return p_y_loc.mean()
        
        return forward_from_knowledge_embedding
    
    def _compute_attribution_for_batch(
        self,
        x_context: torch.Tensor,
        y_context: torch.Tensor,
        x_target: torch.Tensor,
        knowledge: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        """
        Compute Integrated Gradients attribution for a batch.
        
        Returns:
            Tuple of (attributions, convergence_delta)
        """
        # Get knowledge embedding
        k_embed = self.model.latent_encoder.knowledge_encoder(knowledge)
        if k_embed.dim() == 3:
            k_embed = k_embed.squeeze(1)
        
        # Create forward function
        forward_func = self._create_forward_func(
            x_context, y_context, x_target, output_type="mean_prediction"
        )
        
        # Compute attributions
        baseline = torch.zeros_like(k_embed)
        
        attributions, delta = integrated_gradients(
            forward_func,
            k_embed,
            baseline=baseline,
            n_steps=self.n_steps,
            return_convergence_delta=True,
        )
        
        return attributions, delta
    
    def _compute_raw_knowledge_attribution(
        self,
        x_context: torch.Tensor,
        y_context: torch.Tensor,
        x_target: torch.Tensor,
        knowledge: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        """
        Compute attribution w.r.t. raw knowledge input (before encoding).
        
        This gives per-feature importance for numeric knowledge.
        """
        knowledge = knowledge.clone().detach().requires_grad_(True)
        
        def forward_from_raw_knowledge(k: torch.Tensor) -> torch.Tensor:
            # Encode context
            x_context_enc = self.model.x_encoder(x_context)
            x_target_enc = self.model.x_encoder(x_target)
            R = self.model.encode_globally(x_context_enc, y_context, x_target_enc)
            
            # Full forward through knowledge encoder
            q_z = self.model.infer_latent_dist(R, k, x_context.shape[1])
            z = q_z.base_dist.loc.unsqueeze(0)
            
            # Decode
            R_target = z.expand(-1, -1, x_target.shape[1], -1)
            p_y_stats = self.model.decoder(x_target_enc, R_target)
            p_y_loc, _ = p_y_stats.split(self.model.config.output_dim, dim=-1)
            
            return p_y_loc.mean()
        
        baseline = torch.zeros_like(knowledge)
        
        attributions, delta = integrated_gradients(
            forward_from_raw_knowledge,
            knowledge,
            baseline=baseline,
            n_steps=self.n_steps,
            return_convergence_delta=True,
        )
        
        return attributions, delta
    
    def run(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, Any]:
        """
        Run the Knowledge Saliency analysis.
        
        Computes Integrated Gradients attributions for knowledge inputs
        across the dataset.
        
        Returns:
            Dictionary containing:
                - per_sample_attributions: Attributions for each sample
                - aggregated_importance: Mean importance per feature/dimension
                - convergence_deltas: IG completeness check values
        """
        self.model.eval()
        if not hasattr(self.model, "latent_encoder") or self.model.latent_encoder.knowledge_encoder is None:
            raise ValueError("M6 requires a model with a knowledge encoder.")
        
        original_dropout = self._disable_knowledge_dropout()
        
        results = {
            "config": {
                "n_steps": self.n_steps,
                "is_text_knowledge": self.is_text_knowledge,
                "knowledge_dropout_original": original_dropout,
                "knowledge_dropout_set_to": 0.0 if original_dropout is not None else None,
            },
            "raw_attributions": [],
            "embedding_attributions": [],
            "convergence_deltas": [],
            "knowledge_values": [],
            "text_attributions": [],
            "text_tokens": [],
            "text_samples": [],
            "token_importance": {},
            "token_importance_counts": {},
        }
        
        num_batches = min(self.config.num_batches, len(dataloader))
        
        print(f"\nComputing knowledge attributions over {num_batches} batches...")
        
        try:
            for batch_idx, batch in enumerate(tqdm(dataloader, total=num_batches)):
                if batch_idx >= num_batches:
                    break
                
                context, target, knowledge, _ = batch
                x_context, y_context = context
                x_target, y_target = target
                
                x_context = x_context.to(self.device)
                y_context = y_context.to(self.device)
                x_target = x_target.to(self.device)
                
                if not isinstance(knowledge, torch.Tensor):
                    if not self.is_text_knowledge:
                        print("Non-tensor knowledge detected but no text encoder available.")
                        continue
                    try:
                        token_attr, delta, tokens = self._compute_text_attribution(
                            x_context, y_context, x_target, list(knowledge)
                        )
                        results["text_attributions"].append(token_attr.detach().cpu())
                        results["text_tokens"].append(tokens)
                        results["text_samples"].append(list(knowledge))
                        results["convergence_deltas"].append(delta)
                        self._accumulate_token_importance(results, tokens, token_attr)
                    except Exception as e:
                        print(f"Text attribution failed for batch {batch_idx}: {e}")
                    continue
                
                knowledge = knowledge.to(self.device)
                
                # Compute raw knowledge attribution
                try:
                    raw_attr, delta = self._compute_raw_knowledge_attribution(
                        x_context, y_context, x_target, knowledge
                    )
                    
                    results["raw_attributions"].append(raw_attr.detach().cpu())
                    results["convergence_deltas"].append(delta)
                    results["knowledge_values"].append(knowledge.detach().cpu())
                    
                except Exception as e:
                    print(f"Attribution failed for batch {batch_idx}: {e}")
                    continue
        finally:
            if original_dropout is not None:
                self.model.latent_encoder.knowledge_dropout = original_dropout
        
        # Aggregate results
        if results["raw_attributions"]:
            all_attrs = torch.cat(results["raw_attributions"], dim=0)
            all_knowledge = torch.cat(results["knowledge_values"], dim=0)
            
            # Handle different knowledge shapes
            # For sinusoids: [batch, 3, 4] where 3 = (a,b,c) and 4 = (indicator + value)
            if all_attrs.dim() == 3:
                # Sum over the last dimension (indicator + value)
                feature_importance = all_attrs.abs().sum(dim=-1)  # [N, 3]
                
                # Mean importance per feature
                mean_importance = feature_importance.mean(dim=0)  # [3]
                std_importance = feature_importance.std(dim=0)
                
                # Normalize to get relative importance
                total = mean_importance.sum()
                relative_importance = mean_importance / (total + 1e-8)
                
                results["aggregated"] = {
                    "mean_importance": mean_importance.tolist(),
                    "std_importance": std_importance.tolist(),
                    "relative_importance": relative_importance.tolist(),
                    "num_features": len(mean_importance),
                }
                
                # For sinusoids, label the features
                if len(mean_importance) == 3:
                    results["feature_labels"] = ["a (amplitude)", "b (frequency)", "c (phase)"]
                    results["feature_importance"] = {
                        "a (amplitude)": float(relative_importance[0]),
                        "b (frequency)": float(relative_importance[1]),
                        "c (phase)": float(relative_importance[2]),
                    }
                
                # Value-only attribution (exclude indicator columns)
                if all_attrs.size(-1) >= 2:
                    indicator_attr = all_attrs[..., :-1].abs().sum(dim=-1)
                    value_attr = all_attrs[..., -1].abs()
                    
                    mean_value = value_attr.mean(dim=0)
                    std_value = value_attr.std(dim=0)
                    rel_value = mean_value / (mean_value.sum() + 1e-8)
                    
                    results["aggregated_value_only"] = {
                        "mean_importance": mean_value.tolist(),
                        "std_importance": std_value.tolist(),
                        "relative_importance": rel_value.tolist(),
                        "num_features": len(mean_value),
                    }
                    
                    if len(mean_value) == 3:
                        results["feature_importance_value_only"] = {
                            "a (amplitude)": float(rel_value[0]),
                            "b (frequency)": float(rel_value[1]),
                            "c (phase)": float(rel_value[2]),
                        }
                    
                    indicator_total = float(indicator_attr.sum().item())
                    value_total = float(value_attr.sum().item())
                    total = indicator_total + value_total + 1e-8
                    results["indicator_vs_value"] = {
                        "indicator": indicator_total,
                        "value": value_total,
                        "indicator_fraction": indicator_total / total,
                        "value_fraction": value_total / total,
                    }
            else:
                # Flatten and aggregate
                feature_importance = all_attrs.abs()
                if feature_importance.dim() > 2:
                    feature_importance = feature_importance.reshape(feature_importance.size(0), -1)
                
                mean_importance = feature_importance.mean(dim=0)
                relative_importance = mean_importance / (mean_importance.sum() + 1e-8)
                
                results["aggregated"] = {
                    "mean_importance": mean_importance.tolist(),
                    "relative_importance": relative_importance.tolist(),
                    "num_features": len(mean_importance),
                }
        
        # Convergence check
        deltas = results["convergence_deltas"]
        if deltas:
            results["convergence_summary"] = {
                "mean_delta": float(np.mean(deltas)),
                "max_delta": float(np.max(deltas)),
                "converged": float(np.mean(deltas)) < 0.1,
            }
        
        # Interpretation
        if results.get("token_importance"):
            mean_scores = {}
            for token, total in results["token_importance"].items():
                count = results["token_importance_counts"].get(token, 1)
                mean_scores[token] = total / max(count, 1)
            sorted_tokens = sorted(mean_scores.items(), key=lambda x: x[1], reverse=True)
            results["token_importance_mean"] = [(t, float(s)) for t, s in sorted_tokens[:20]]
            top_tokens = results["token_importance_mean"][:5]
            results["interpretation"] = (
                "Most important tokens: " +
                ", ".join([f"{t}: {s:.3f}" for t, s in top_tokens])
            )
        elif "feature_importance_value_only" in results:
            imp = results["feature_importance_value_only"]
            sorted_features = sorted(imp.items(), key=lambda x: x[1], reverse=True)
            top_feature = sorted_features[0][0]
            top_importance = sorted_features[0][1]
            
            results["interpretation"] = (
                f"Most important knowledge value: {top_feature} ({top_importance:.1%}). "
                f"Feature ranking: {', '.join([f'{k}: {v:.1%}' for k, v in sorted_features])}"
            )
        elif "feature_importance" in results:
            imp = results["feature_importance"]
            sorted_features = sorted(imp.items(), key=lambda x: x[1], reverse=True)
            top_feature = sorted_features[0][0]
            top_importance = sorted_features[0][1]
            
            results["interpretation"] = (
                f"Most important knowledge feature: {top_feature} ({top_importance:.1%}). "
                f"Feature ranking: {', '.join([f'{k}: {v:.1%}' for k, v in sorted_features])}"
            )
        elif "aggregated" in results:
            rel_imp = results["aggregated"]["relative_importance"]
            max_idx = np.argmax(rel_imp)
            results["interpretation"] = (
                f"Feature {max_idx} has highest importance ({rel_imp[max_idx]:.1%})"
            )
        else:
            results["interpretation"] = "Unable to compute attributions"
        
        # Clean up for JSON serialization
        results["raw_attributions"] = [a.tolist() for a in results["raw_attributions"][:10]]
        results["knowledge_values"] = [k.tolist() for k in results["knowledge_values"][:10]]
        results["text_attributions"] = [a.tolist() for a in results["text_attributions"][:5]]
        results["text_tokens"] = results["text_tokens"][:5]
        results["text_samples"] = results["text_samples"][:5]
        if "token_importance" in results:
            results.pop("token_importance", None)
            results.pop("token_importance_counts", None)
        
        self.results = results
        self.save_results(results)
        
        # Generate visualization
        self._save_visualization(results)
        
        return results
    
    def _save_visualization(self, results: Dict[str, Any]):
        """Generate comprehensive visualizations for M6."""
        try:
            from .enhanced_viz import viz_m6_knowledge_saliency
            viz_m6_knowledge_saliency(results, self.output_dir)
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
            
            if "aggregated" not in results and "aggregated_value_only" not in results:
                print("No aggregated results to visualize")
                return
            
            plots_dir = self.output_dir / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            # Left: Feature importance bar chart
            ax1 = axes[0]
            aggregated = results.get("aggregated_value_only") or results.get("aggregated", {})
            if "feature_labels" in results:
                labels = results["feature_labels"]
            else:
                n = aggregated.get("num_features", 10)
                labels = [f"Feature {i}" for i in range(min(n, 20))]
            
            importance = aggregated.get("relative_importance", [])[:len(labels)]
            title_suffix = " (Value-only)" if "aggregated_value_only" in results else ""
            
            if importance:
                colors = sns.color_palette("rocket", n_colors=len(importance))
                bars = ax1.barh(labels, importance, color=colors, edgecolor='black')
                ax1.set_xlabel("Relative Importance", fontsize=12)
                ax1.set_title(f"Knowledge Feature Importance{title_suffix}\n(Integrated Gradients)", fontsize=14, fontweight='bold')
                ax1.grid(True, alpha=0.3, axis='x')
                
                for bar, val in zip(bars, importance):
                    ax1.annotate(f'{val:.1%}',
                                xy=(val, bar.get_y() + bar.get_height()/2),
                                xytext=(5, 0), textcoords="offset points",
                                ha='left', va='center', fontsize=11, fontweight='bold')
            
            # Right: Pie chart
            ax2 = axes[1]
            if importance and len(importance) <= 10:
                colors = sns.color_palette("rocket", n_colors=len(importance))
                wedges, texts, autotexts = ax2.pie(importance, labels=labels, colors=colors,
                                                   autopct='%1.1f%%', startangle=90,
                                                   wedgeprops=dict(edgecolor='white', linewidth=2))
                ax2.set_title("Feature Importance Distribution", fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            fig.savefig(plots_dir / "m6_feature_importance.png", dpi=150, bbox_inches='tight', facecolor='white')
            fig.savefig(plots_dir / "m6_feature_importance.pdf", dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            
            # Convergence plot
            if results.get("convergence_deltas"):
                fig, ax = plt.subplots(figsize=(10, 6))
                deltas = results["convergence_deltas"]
                ax.hist(deltas, bins=25, color=palette[0],
                       edgecolor='black', alpha=0.7)
                ax.axvline(x=np.mean(deltas), color='red', linestyle='--', linewidth=2,
                          label=f'Mean: {np.mean(deltas):.4f}')
                ax.axvline(x=0.1, color='green', linestyle=':', linewidth=2,
                          label='Convergence threshold')
                ax.set_xlabel('Convergence Delta', fontsize=12)
                ax.set_ylabel('Count', fontsize=12)
                ax.set_title('Integrated Gradients Convergence Check\n(Smaller = Better)', 
                            fontsize=14, fontweight='bold')
                ax.legend(fontsize=11)
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                fig.savefig(plots_dir / "m6_convergence.png", dpi=150, bbox_inches='tight', facecolor='white')
                fig.savefig(plots_dir / "m6_convergence.pdf", dpi=150, bbox_inches='tight', facecolor='white')
                plt.close(fig)
            
            print(f"  M6 visualizations saved to {plots_dir}")
            
        except ImportError as e:
            print(f"Visualization libraries not available: {e}")


def visualize_text_attribution(
    model: nn.Module,
    text: str,
    attributions: torch.Tensor,
    tokenizer,
) -> str:
    """
    Create HTML visualization of token attributions for text knowledge.
    
    Args:
        model: The INP model
        text: Input text
        attributions: Token-level attributions
        tokenizer: Tokenizer used by the model
        
    Returns:
        HTML string with colored tokens
    """
    # Tokenize
    tokens = tokenizer.tokenize(text)
    
    # Normalize attributions to [0, 1]
    attr_normalized = attributions.abs()
    attr_normalized = attr_normalized / (attr_normalized.max() + 1e-8)
    
    # Create HTML
    html_parts = []
    for token, attr in zip(tokens, attr_normalized):
        # Color intensity based on attribution
        intensity = int(255 * (1 - attr.item()))
        color = f"rgb(255, {intensity}, {intensity})"
        html_parts.append(f'<span style="background-color: {color}">{token}</span>')
    
    return " ".join(html_parts)
