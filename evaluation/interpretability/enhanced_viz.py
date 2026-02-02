"""
Enhanced visualization functions for all interpretability experiments.

Call these functions from each experiment's _save_visualization method
or import and use directly.

All functions use consistent seaborn styling:
    sns.set(font_scale=1.3)
    sns.set_style("whitegrid")
    sns.set_palette("rocket", n_colors=10)
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import warnings


def setup_style():
    """Setup seaborn and matplotlib styling.
    
    Returns:
        plt: matplotlib.pyplot module
        sns: seaborn module
        palette: Color palette with 10 colors (use palette[0], palette[5], etc.)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    sns.set(font_scale=1.3)
    sns.set_style("whitegrid")
    
    # Create palette with enough colors to avoid IndexError
    palette = sns.color_palette("rocket", n_colors=10)
    sns.set_palette(palette)
    
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['savefig.facecolor'] = 'white'
    plt.rcParams['savefig.dpi'] = 150
    plt.rcParams['savefig.bbox'] = 'tight'
    
    return plt, sns, palette


def save_all_formats(fig, plots_dir: Path, name: str):
    """Save figure in PNG and PDF formats."""
    fig.savefig(plots_dir / f"{name}.png", dpi=150, bbox_inches='tight', facecolor='white')
    fig.savefig(plots_dir / f"{name}.pdf", dpi=150, bbox_inches='tight', facecolor='white')


# ============================================================================
# M4: Gradient Alignment Visualizations
# ============================================================================
def viz_m4_gradient_alignment(results: Dict[str, Any], output_dir: Path):
    """Generate comprehensive visualizations for M4 Gradient Alignment."""
    try:
        plt, sns, palette = setup_style()
        
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        agg_k = results.get("with_knowledge_aggregated", {})
        agg_rand = results.get("with_random_knowledge_aggregated", {})
        metric_type = results.get("config", {}).get("metric_type", "loss_balance_proxy")
        score_label = (
            "Loss Balance Score"
            if metric_type == "loss_balance_proxy"
            else "Gradient Alignment Score (GAS)"
        )
        
        # ================================================================
        # PLOT 1: Module-wise Gradient Alignment (4-panel)
        # ================================================================
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Top-left: Bar chart comparison
        ax1 = axes[0, 0]
        candidate_modules = ["all", "latent_encoder", "xy_encoder", "decoder", "x_encoder", "knowledge_encoder"]
        modules = [
            m for m in candidate_modules
            if (f"{m}_alignment" in agg_k) or (f"{m}_alignment" in agg_rand)
        ]
        if not modules:
            modules = ["all"]
        x_pos = np.arange(len(modules))
        width = 0.35
        
        means_k = [agg_k.get(f"{m}_alignment", {}).get("mean", 0) for m in modules]
        stds_k = [agg_k.get(f"{m}_alignment", {}).get("std", 0) for m in modules]
        means_rand = [agg_rand.get(f"{m}_alignment", {}).get("mean", 0) for m in modules]
        stds_rand = [agg_rand.get(f"{m}_alignment", {}).get("std", 0) for m in modules]
        
        bars1 = ax1.bar(x_pos - width/2, means_k, width, yerr=stds_k, 
                       label='With Knowledge', color=palette[0], 
                       edgecolor='black', capsize=4)
        bars2 = ax1.bar(x_pos + width/2, means_rand, width, yerr=stds_rand,
                       label='Random Knowledge', color=palette[5],
                       edgecolor='black', alpha=0.7, capsize=4)
        
        ax1.set_ylabel(score_label, fontsize=12)
        ax1.set_title('Module-wise Loss Balance', fontsize=14, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(modules, rotation=15)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_ylim(0, 1)

        # Add interpretation zones (proxy thresholds)
        ax1.axhspan(0.7, 1.0, alpha=0.08, color='green')
        ax1.axhspan(0.4, 0.7, alpha=0.08, color='orange')
        ax1.axhspan(0.0, 0.4, alpha=0.08, color='red')
        
        # Top-right: Time series
        ax2 = axes[0, 1]
        if "all_alignment" in results.get("with_knowledge", {}):
            align_k = results["with_knowledge"]["all_alignment"]
            align_rand = results["with_random_knowledge"]["all_alignment"]
            
            ax2.plot(align_k, color=palette[0], alpha=0.6, linewidth=1)
            ax2.plot(align_rand, color=palette[5], alpha=0.6, linewidth=1, linestyle='--')
            
            # Smoothed lines
            window = max(5, len(align_k) // 10)
            if window > 1 and len(align_k) > window:
                smooth_k = np.convolve(align_k, np.ones(window)/window, mode='valid')
                smooth_rand = np.convolve(align_rand, np.ones(window)/window, mode='valid')
                ax2.plot(range(window//2, window//2 + len(smooth_k)), smooth_k,
                        color=palette[0], linewidth=3, label='With K (smoothed)')
                ax2.plot(range(window//2, window//2 + len(smooth_rand)), smooth_rand,
                        color=palette[5], linewidth=3, linestyle='--', label='Random K (smoothed)')
        
        ax2.set_xlabel('Batch', fontsize=12)
        ax2.set_ylabel(score_label, fontsize=12)
        ax2.set_title('Loss Balance Over Batches', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.set_ylim(0, 1)
        ax2.axhline(y=0.7, color='green', linestyle=':', alpha=0.6)
        ax2.axhline(y=0.4, color='orange', linestyle=':', alpha=0.6)
        ax2.grid(True, alpha=0.3)
        
        # Bottom-left: Loss magnitude comparison
        ax3 = axes[1, 0]
        nll_k = agg_k.get("nll_values", {}).get("mean", None)
        kl_k = agg_k.get("kl_scaled_values", {}).get("mean", None)
        nll_rand = agg_rand.get("nll_values", {}).get("mean", None)
        kl_rand = agg_rand.get("kl_scaled_values", {}).get("mean", None)
        
        if nll_k is not None and kl_k is not None and nll_rand is not None and kl_rand is not None:
            categories = ["NLL", "beta*KL"]
            x = np.arange(len(categories))
            width = 0.35
            
            vals_k = [abs(nll_k), abs(kl_k)]
            vals_rand = [abs(nll_rand), abs(kl_rand)]
            
            ax3.bar(x - width/2, vals_k, width, label="With Knowledge",
                    color=palette[2], edgecolor='black')
            ax3.bar(x + width/2, vals_rand, width, label="Random Knowledge",
                    color=palette[4], edgecolor='black', alpha=0.7)
            
            ax3.set_xticks(x)
            ax3.set_xticklabels(categories)
            ax3.set_ylabel('Loss Magnitude (abs)', fontsize=12)
            ax3.set_title('Loss Magnitudes', fontsize=14, fontweight='bold')
            ax3.legend(fontsize=11)
            ax3.grid(True, alpha=0.3, axis='y')
        else:
            ax3.axis('off')
        
        # Bottom-right: Heatmap of alignment
        ax4 = axes[1, 1]
        heatmap_data = np.array([[agg_k.get(f"{m}_alignment", {}).get("mean", 0) for m in modules],
                                 [agg_rand.get(f"{m}_alignment", {}).get("mean", 0) for m in modules]])
        
        sns.heatmap(heatmap_data, ax=ax4, cmap='rocket', vmin=0, vmax=1,
                   xticklabels=modules, yticklabels=['With Knowledge', 'Random Knowledge'],
                   annot=True, fmt='.3f', cbar_kws={'label': score_label},
                   linewidths=1, linecolor='white')
        ax4.set_title('Loss Balance Heatmap', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        save_all_formats(fig, plots_dir, "m4_gradient_alignment")
        plt.close(fig)
        
        # ================================================================
        # PLOT 2: Alignment Distribution
        # ================================================================
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if "all_alignment" in results.get("with_knowledge", {}):
            align_k = results["with_knowledge"]["all_alignment"]
            align_rand = results["with_random_knowledge"]["all_alignment"]
            
            ax.hist(align_k, bins=30, alpha=0.6, color=palette[0],
                   label='With Knowledge', edgecolor='black')
            ax.hist(align_rand, bins=30, alpha=0.6, color=palette[5],
                   label='Random Knowledge', edgecolor='black')
        
        ax.axvline(x=np.mean(align_k), color=palette[0], 
                  linestyle='-', linewidth=2, label=f'Mean K: {np.mean(align_k):.3f}')
        ax.axvline(x=np.mean(align_rand), color=palette[5],
                  linestyle='-', linewidth=2, label=f'Mean Rand: {np.mean(align_rand):.3f}')
        ax.set_xlabel(score_label, fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Distribution of Loss Balance Scores', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_all_formats(fig, plots_dir, "m4_alignment_distribution")
        plt.close(fig)
        
        print(f"  M4 visualizations saved to {plots_dir}")
        
    except ImportError as e:
        print(f"Visualization libraries not available: {e}")


# ============================================================================
# M5: Activation Patching Visualizations  
# ============================================================================
def viz_m5_activation_patching(results: Dict[str, Any], output_dir: Path):
    """Generate comprehensive visualizations for M5 Activation Patching."""
    try:
        plt, sns, palette = setup_style()
        
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        individual = results.get("individual_results", [])
        agg = results.get("aggregated", {})
        
        # ================================================================
        # PLOT 1: Causal Effect Metrics (2x2)
        # ================================================================
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Top-left: Transfer ratio distribution
        ax1 = axes[0, 0]
        transfer_ratios = [r["transfer_ratio"] for r in individual]
        ax1.hist(transfer_ratios, bins=25, color=palette[0],
                edgecolor='black', alpha=0.7)
        ax1.axvline(x=np.mean(transfer_ratios), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(transfer_ratios):.3f}')
        ax1.axvline(x=0.5, color='green', linestyle=':', linewidth=2, label='Strong threshold (0.5)')
        ax1.set_xlabel('Transfer Ratio', fontsize=12)
        ax1.set_ylabel('Count', fontsize=12)
        ax1.set_title('Distribution of Transfer Ratios', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Top-right: Alignment distribution
        ax2 = axes[0, 1]
        alignments = [r["alignment"] for r in individual]
        colors = ['green' if a > 0 else 'red' for a in alignments]
        ax2.hist(alignments, bins=25, color=palette[2],
                edgecolor='black', alpha=0.7)
        ax2.axvline(x=0, color='gray', linestyle='--', linewidth=2)
        ax2.axvline(x=np.mean(alignments), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(alignments):.3f}')
        ax2.set_xlabel('Alignment (cosine similarity)', fontsize=12)
        ax2.set_ylabel('Count', fontsize=12)
        ax2.set_title('Shift Direction Alignment', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        # Bottom-left: MSE comparison scatter
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
        save_all_formats(fig, plots_dir, "m5_activation_patching")
        plt.close(fig)
        
        # ================================================================
        # PLOT 2: Heatmap of metrics per sample
        # ================================================================
        fig, ax = plt.subplots(figsize=(14, 6))
        
        n_samples = min(50, len(individual))
        metrics_matrix = np.array([
            [r["transfer_ratio"] for r in individual[:n_samples]],
            [r["alignment"] for r in individual[:n_samples]],
            [r["mse_improvement"] for r in individual[:n_samples]]
        ])
        
        sns.heatmap(metrics_matrix, ax=ax, cmap='rocket', center=0,
                   yticklabels=['Transfer Ratio', 'Alignment', 'MSE Improvement'],
                   cbar_kws={'label': 'Value'}, xticklabels=5)
        ax.set_xlabel('Sample Pair Index', fontsize=12)
        ax.set_title('Causal Effect Metrics per Sample Pair', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        save_all_formats(fig, plots_dir, "m5_metrics_heatmap")
        plt.close(fig)
        
        print(f"  M5 visualizations saved to {plots_dir}")
        
    except ImportError as e:
        print(f"Visualization libraries not available: {e}")


# ============================================================================
# M6: Knowledge Saliency Visualizations
# ============================================================================
def viz_m6_knowledge_saliency(results: Dict[str, Any], output_dir: Path):
    """Generate comprehensive visualizations for M6 Knowledge Saliency."""
    try:
        plt, sns, palette = setup_style()
        
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # ================================================================
        # PLOT 1: Feature Importance
        # ================================================================
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
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
        
        # Pie chart
        ax2 = axes[1]
        if importance and len(importance) <= 10:
            colors = sns.color_palette("rocket", n_colors=len(importance))
            wedges, texts, autotexts = ax2.pie(importance, labels=labels, colors=colors,
                                               autopct='%1.1f%%', startangle=90,
                                               wedgeprops=dict(edgecolor='white', linewidth=2))
            ax2.set_title("Feature Importance Distribution", fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        save_all_formats(fig, plots_dir, "m6_feature_importance")
        plt.close(fig)
        
        # ================================================================
        # PLOT 2: Convergence Check
        # ================================================================
        if results.get("convergence_deltas"):
            fig, ax = plt.subplots(figsize=(10, 6))
            
            deltas = results["convergence_deltas"]
            ax.hist(deltas, bins=25, color=palette[0],
                   edgecolor='black', alpha=0.7)
            ax.axvline(x=np.mean(deltas), color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {np.mean(deltas):.4f}')
            ax.axvline(x=0.1, color='green', linestyle=':', linewidth=2,
                      label='Convergence threshold (0.1)')
            
            ax.set_xlabel('Convergence Delta', fontsize=12)
            ax.set_ylabel('Count', fontsize=12)
            ax.set_title('Integrated Gradients Convergence Check\n(Smaller = Better Completeness)', 
                        fontsize=14, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            save_all_formats(fig, plots_dir, "m6_convergence")
            plt.close(fig)
        
        print(f"  M6 visualizations saved to {plots_dir}")
        
    except ImportError as e:
        print(f"Visualization libraries not available: {e}")


# ============================================================================
# M7: Linear Probing Visualizations
# ============================================================================
def viz_m7_linear_probing(results: Dict[str, Any], output_dir: Path):
    """Generate comprehensive visualizations for M7 Linear Probing."""
    try:
        plt, sns, palette = setup_style()
        
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # ================================================================
        # PLOT 1: Main Results (2x2)
        # ================================================================
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        comp = results.get("comparison", {})
        
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
        save_all_formats(fig, plots_dir, "m7_linear_probing")
        plt.close(fig)
        
        print(f"  M7 visualizations saved to {plots_dir}")
        
    except ImportError as e:
        print(f"Visualization libraries not available: {e}")


# ============================================================================
# M8: Uncertainty Decomposition Visualizations
# ============================================================================
def viz_m8_uncertainty(results: Dict[str, Any], output_dir: Path):
    """Generate comprehensive visualizations for M8 Uncertainty Decomposition."""
    try:
        plt, sns, palette = setup_style()
        
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        agg_k = results.get("with_knowledge_aggregated", {})
        agg_no_k = results.get("without_knowledge_aggregated", {})
        
        n_values_k = sorted([int(k) for k in agg_k.keys()])
        n_values_no_k = sorted([int(k) for k in agg_no_k.keys()])
        
        # ================================================================
        # PLOT 1: Main Uncertainty Analysis (2x2)
        # ================================================================
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
        
        # Top-right: Total, Aleatoric, Epistemic stacked
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
        heatmap_data = []
        n_common = sorted(set(n_values_k) & set(n_values_no_k))
        
        if n_common:
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
        save_all_formats(fig, plots_dir, "m8_uncertainty")
        plt.close(fig)
        
        print(f"  M8 visualizations saved to {plots_dir}")
        
    except ImportError as e:
        print(f"Visualization libraries not available: {e}")


# ============================================================================
# M9: Spectral Analysis Visualizations
# ============================================================================
def viz_m9_spectral(results: Dict[str, Any], output_dir: Path):
    """Generate comprehensive visualizations for M9 Spectral Analysis."""
    try:
        plt, sns, palette = setup_style()
        
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        layers = results.get("layers", [])
        
        # ================================================================
        # PLOT 1: Alpha Distribution and Analysis
        # ================================================================
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
        
        # Bottom-right: Goldilocks analysis pie
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
        save_all_formats(fig, plots_dir, "m9_spectral")
        plt.close(fig)
        
        # ================================================================
        # PLOT 2: Layer-wise heatmap
        # ================================================================
        if len(layers) > 0:
            fig, ax = plt.subplots(figsize=(14, max(4, len(layers) * 0.3)))
            
            layer_names = [l.get("name", f"layer_{i}")[-30:] for i, l in enumerate(layers[:50])]
            metrics = np.array([[l.get("alpha", 0), l.get("stable_rank", 0), 
                               np.log10(l.get("spectral_norm", 1) + 1e-10)] 
                              for l in layers[:50]])
            
            if metrics.shape[0] > 0:
                sns.heatmap(metrics, ax=ax, cmap='rocket',
                           yticklabels=layer_names,
                           xticklabels=['α', 'Stable Rank', 'log₁₀(Spectral Norm)'],
                           cbar_kws={'label': 'Value'})
                ax.set_title('Layer-wise Spectral Metrics', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            save_all_formats(fig, plots_dir, "m9_layer_heatmap")
            plt.close(fig)
        
        print(f"  M9 visualizations saved to {plots_dir}")
        
    except ImportError as e:
        print(f"Visualization libraries not available: {e}")


# ============================================================================
# M10: CKA Similarity Visualizations
# ============================================================================
def viz_m10_cka(results: Dict[str, Any], output_dir: Path):
    """Generate comprehensive visualizations for M10 CKA Similarity."""
    try:
        plt, sns, palette = setup_style()
        
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        cka_scores = results.get("cka_scores", {})
        
        # ================================================================
        # PLOT 1: Main CKA Analysis
        # ================================================================
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
        
        # Right: Hook layers heatmap
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
        save_all_formats(fig, plots_dir, "m10_cka_similarity")
        plt.close(fig)
        
        # ================================================================
        # PLOT 2: CKA Heatmap Grid
        # ================================================================
        if len(cka_scores) > 0:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            all_keys = sorted([k for k in cka_scores.keys() if not np.isnan(cka_scores[k])])[:30]
            
            if len(all_keys) > 1:
                # Create a matrix (even though CKA is just between two conditions)
                values = [cka_scores[k] for k in all_keys]
                short_names = [k.replace("hook_", "")[-20:] for k in all_keys]
                
                # Single column heatmap
                heatmap_data = np.array(values).reshape(-1, 1)
                sns.heatmap(heatmap_data, ax=ax, cmap='rocket', vmin=0, vmax=1,
                           yticklabels=short_names, xticklabels=['CKA'],
                           annot=True, fmt='.3f', cbar_kws={'label': 'CKA Score'},
                           linewidths=0.5, linecolor='white')
                ax.set_title('CKA Similarity Heatmap\n(With K vs Without K)', 
                            fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            save_all_formats(fig, plots_dir, "m10_cka_heatmap")
            plt.close(fig)
        
        print(f"  M10 visualizations saved to {plots_dir}")
        
    except ImportError as e:
        print(f"Visualization libraries not available: {e}")
