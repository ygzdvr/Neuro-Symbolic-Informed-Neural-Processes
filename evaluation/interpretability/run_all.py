#!/usr/bin/env python
"""
Run all interpretability experiments (M1-M10) on a trained INP model.

Usage:
    # Run all experiments
    python run_all.py --model-path saves/project/run_0/model_best.pt \
                      --config-path saves/project/run_0/config.toml

    # Run specific experiments
    python run_all.py --model-path saves/project/run_0/model_best.pt \
                      --config-path saves/project/run_0/config.toml \
                      --experiments M1 M3 M7

Available experiments:
    M1:  Information Bottleneck (MINE)
    M2:  Loss Landscape Visualization
    M3:  Effective Dimensionality (SVD)
    M4:  Gradient Alignment Score
    M5:  Causal Activation Patching
    M6:  Knowledge Saliency (Integrated Gradients)
    M7:  Linear Probing
    M8:  Uncertainty Decomposition
    M9:  Spectral Analysis (WeightWatcher)
    M10: CKA Similarity
"""

import argparse
import sys
import os
import toml
import torch
import numpy as np
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))))

from config import Config
from models.inp import INP
from dataset.utils import setup_dataloaders
from evaluation.interpretability.base import ExperimentConfig

# Import all experiments
from evaluation.interpretability.m1_information_bottleneck import InformationBottleneckExperiment
from evaluation.interpretability.m2_loss_landscape import LossLandscapeExperiment
from evaluation.interpretability.m3_effective_dimensionality import EffectiveDimensionalityExperiment
from evaluation.interpretability.m4_gradient_alignment import GradientAlignmentExperiment
from evaluation.interpretability.m5_activation_patching import ActivationPatchingExperiment
from evaluation.interpretability.m6_knowledge_saliency import KnowledgeSaliencyExperiment
from evaluation.interpretability.m7_linear_probing import LinearProbingExperiment
from evaluation.interpretability.m8_uncertainty_decomposition import UncertaintyDecompositionExperiment
from evaluation.interpretability.m9_spectral_analysis import SpectralAnalysisExperiment
from evaluation.interpretability.m10_cka_similarity import CKASimilarityExperiment


EXPERIMENTS = {
    "M1": ("Information Bottleneck (MINE)", InformationBottleneckExperiment),
    "M2": ("Loss Landscape Visualization", LossLandscapeExperiment),
    "M3": ("Effective Dimensionality (SVD)", EffectiveDimensionalityExperiment),
    "M4": ("Gradient Alignment Score", GradientAlignmentExperiment),
    "M5": ("Causal Activation Patching", ActivationPatchingExperiment),
    "M6": ("Knowledge Saliency (IG)", KnowledgeSaliencyExperiment),
    "M7": ("Linear Probing", LinearProbingExperiment),
    "M8": ("Uncertainty Decomposition", UncertaintyDecompositionExperiment),
    "M9": ("Spectral Analysis (HTSR)", SpectralAnalysisExperiment),
    "M10": ("CKA Similarity", CKASimilarityExperiment),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run INP interpretability experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--config-path", type=str, required=True,
                        help="Path to model config (.toml)")
    parser.add_argument("--output-dir", type=str, default="./interpretability_results",
                        help="Directory to save results")
    parser.add_argument("--experiments", type=str, nargs="+", 
                        default=list(EXPERIMENTS.keys()),
                        help="Which experiments to run (default: all)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode with reduced iterations")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Validate experiment names
    for exp in args.experiments:
        if exp.upper() not in EXPERIMENTS:
            print(f"Unknown experiment: {exp}")
            print(f"Available: {', '.join(EXPERIMENTS.keys())}")
            return
    
    args.experiments = [exp.upper() for exp in args.experiments]
    
    # Load config
    print(f"Loading config from {args.config_path}")
    config_dict = toml.load(args.config_path)
    config = Config(**config_dict)
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config.device = device
    print(f"Using device: {device}")
    
    # Setup dataloaders
    print("Setting up dataloaders...")
    train_dataloader, val_dataloader, _, extras = setup_dataloaders(config)
    for k, v in extras.items():
        config.__dict__[k] = v
    
    # Load model
    print(f"Loading model from {args.model_path}")
    model = INP(config)
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    
    # Check if model uses knowledge
    uses_knowledge = getattr(config, 'use_knowledge', True)
    if not uses_knowledge:
        print("\n" + "=" * 70)
        print("WARNING: This is a Neural Process (NP) baseline without knowledge.")
        print("Many interpretability experiments are designed for INP models with knowledge.")
        print("Experiments requiring knowledge will be skipped.")
        print("=" * 70)
    
    # Experiments that require knowledge integration
    KNOWLEDGE_REQUIRED_EXPERIMENTS = {"M1", "M5", "M6", "M10"}
    # Experiments that compare with/without knowledge (can run but limited)
    KNOWLEDGE_COMPARISON_EXPERIMENTS = {"M3", "M7", "M8"}
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup experiment config
    exp_config = ExperimentConfig(
        model_path=args.model_path,
        config_path=args.config_path,
        dataset=config.dataset,
        knowledge_type=getattr(config, 'knowledge_type', 'abc2'),
        uses_knowledge=uses_knowledge,
        device=device,
        batch_size=config.batch_size,
        num_batches=10 if args.quick else 50,
        seed=args.seed,
        output_dir=str(output_dir),
    )
    
    # Run experiments
    all_results = {}
    
    print("\n" + "=" * 70)
    print(f"Running {len(args.experiments)} experiments")
    print("=" * 70)
    
    for exp_id in args.experiments:
        name, ExperimentClass = EXPERIMENTS[exp_id]
        
        print(f"\n{'=' * 70}")
        print(f"{exp_id}: {name}")
        print("=" * 70)
        
        # Skip knowledge-requiring experiments for NP baselines
        if not uses_knowledge:
            if exp_id in KNOWLEDGE_REQUIRED_EXPERIMENTS:
                print(f"\nSkipping {exp_id}: Requires knowledge integration (this is an NP baseline)")
                all_results[exp_id] = {
                    "name": name,
                    "status": "skipped",
                    "reason": "NP baseline without knowledge",
                }
                continue
            elif exp_id in KNOWLEDGE_COMPARISON_EXPERIMENTS:
                print(f"\nNote: {exp_id} will run in limited mode (no knowledge comparison)")
        
        try:
            # Create experiment with appropriate kwargs
            if exp_id == "M1":
                kwargs = {"mine_train_steps": 30 if args.quick else 100}
            elif exp_id == "M2":
                kwargs = {
                    "num_directions": 2 if args.quick else 5,
                    "num_eval_batches": exp_config.num_batches,
                    "loss_mode": "elbo",
                    "beta": getattr(config, "beta", 1.0),
                }
            elif exp_id == "M3":
                kwargs = {"max_samples": 500 if args.quick else 2000}
            elif exp_id == "M5":
                kwargs = {"num_pairs": 20 if args.quick else 50}
            elif exp_id == "M7":
                kwargs = {"probe_epochs": 50 if args.quick else 100}
            elif exp_id == "M8":
                kwargs = {"num_tasks": 30 if args.quick else 100}
            elif exp_id == "M10":
                kwargs = {"max_samples": 200 if args.quick else 500}
            else:
                kwargs = {}
            
            experiment = ExperimentClass(model, exp_config, **kwargs)
            results = experiment.run(val_dataloader)
            
            all_results[exp_id] = {
                "name": name,
                "interpretation": results.get("interpretation", "N/A"),
                "status": "success",
            }
            
            # Print summary
            print(f"\n{exp_id} Summary:")
            print(f"  Interpretation: {results.get('interpretation', 'N/A')}")
            
        except Exception as e:
            print(f"\n{exp_id} FAILED: {e}")
            all_results[exp_id] = {
                "name": name,
                "status": "failed",
                "error": str(e),
            }
    
    # Save summary
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Create summary visualization
    _create_summary_visualization(all_results, output_dir)
    
    # Print final summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    for exp_id, result in all_results.items():
        if result["status"] == "success":
            status = "✓"
        elif result["status"] == "skipped":
            status = "○"
        else:
            status = "✗"
        print(f"{status} {exp_id}: {result['name']}")
        if result["status"] == "success":
            print(f"    {result['interpretation'][:80]}...")
        elif result["status"] == "skipped":
            print(f"    Skipped: {result.get('reason', 'Unknown reason')}")
        else:
            print(f"    Error: {result.get('error', 'Unknown')}")
    
    print(f"\nAll results saved to: {output_dir}")
    print(f"Plots saved in: {output_dir}/plots/")
    
    return all_results


def _create_summary_visualization(all_results: dict, output_dir: Path):
    """Create comprehensive summary visualization of all experiments."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set styling
        sns.set(font_scale=1.3)
        sns.set_style("whitegrid")
        
        # Create palette with enough colors
        palette = sns.color_palette("rocket", n_colors=10)
        sns.set_palette(palette)
        
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Create summary dashboard
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        fig.suptitle('INP Mechanistic Interpretability Suite\nExperiment Summary', 
                    fontsize=18, fontweight='bold', y=1.02)
        
        # Top-left: Experiment status overview
        ax1 = axes[0, 0]
        exp_ids = list(all_results.keys())
        statuses = [1 if r.get("status") == "success" else 0 for r in all_results.values()]
        colors = ['green' if s == 1 else 'red' for s in statuses]
        
        bars = ax1.barh(exp_ids, statuses, color=colors, edgecolor='black')
        ax1.set_xlim(0, 1.1)
        ax1.set_xlabel('Status (1=Success, 0=Failed)', fontsize=12)
        ax1.set_title('Experiment Completion Status', fontsize=14, fontweight='bold')
        ax1.invert_yaxis()
        
        for bar, status in zip(bars, statuses):
            label = '✓' if status == 1 else '✗'
            ax1.annotate(label,
                        xy=(0.05, bar.get_y() + bar.get_height()/2),
                        fontsize=14, fontweight='bold',
                        color='white' if status == 1 else 'white',
                        ha='left', va='center')
        
        # Top-right: Experiment categories
        ax2 = axes[0, 1]
        categories = {
            'Information\nDynamics': ['M1'],
            'Optimization\nGeometry': ['M2'],
            'Latent\nStructure': ['M3', 'M7'],
            'Learning\nDynamics': ['M4'],
            'Causal\nReliance': ['M5', 'M6'],
            'Uncertainty': ['M8'],
            'Generalization': ['M9'],
            'Representation': ['M10'],
        }
        
        cat_names = list(categories.keys())
        cat_counts = [len(v) for v in categories.values()]
        cat_success = []
        for exps in categories.values():
            success = sum(1 for e in exps if all_results.get(e, {}).get("status") == "success")
            cat_success.append(success)
        
        x = np.arange(len(cat_names))
        width = 0.4
        
        bars1 = ax2.bar(x - width/2, cat_counts, width, label='Total', 
                       color=palette[2], alpha=0.5, edgecolor='black')
        bars2 = ax2.bar(x + width/2, cat_success, width, label='Completed',
                       color=palette[0], edgecolor='black')
        
        ax2.set_ylabel('Count', fontsize=12)
        ax2.set_title('Experiments by Category', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(cat_names, fontsize=9, rotation=15, ha='right')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Bottom-left: Key insights summary
        ax3 = axes[1, 0]
        ax3.axis('off')
        
        insights = []
        for exp_id, result in all_results.items():
            if result.get("status") == "success":
                interp = result.get("interpretation", "N/A")
                if len(interp) > 60:
                    interp = interp[:60] + "..."
                insights.append(f"{exp_id}: {interp}")
        
        insights_text = "\n\n".join(insights) if insights else "No successful experiments"
        
        ax3.text(0.05, 0.95, "Key Insights:", fontsize=14, fontweight='bold',
                transform=ax3.transAxes, va='top')
        ax3.text(0.05, 0.85, insights_text, fontsize=10,
                transform=ax3.transAxes, va='top', wrap=True,
                fontfamily='monospace')
        ax3.set_title('Experiment Interpretations', fontsize=14, fontweight='bold')
        
        # Bottom-right: Methodology overview
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        methods = [
            "M1:  MINE (Mutual Information)",
            "M2:  Filter-normalized 1D Loss Profiles",
            "M3:  SVD & Participation Ratio",
            "M4:  Cosine Similarity of Gradients",
            "M5:  Activation Patching Intervention",
            "M6:  Integrated Gradients Attribution",
            "M7:  Linear & MLP Probes",
            "M8:  Aleatoric/Epistemic Decomposition",
            "M9:  Power Law ESD Analysis",
            "M10: Centered Kernel Alignment",
        ]
        
        methods_text = "\n".join(methods)
        ax4.text(0.05, 0.95, "Methods Used:", fontsize=14, fontweight='bold',
                transform=ax4.transAxes, va='top')
        ax4.text(0.05, 0.85, methods_text, fontsize=10,
                transform=ax4.transAxes, va='top',
                fontfamily='monospace')
        ax4.set_title('Methodology Summary', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        fig.savefig(plots_dir / "summary_dashboard.png", dpi=150, bbox_inches='tight', facecolor='white')
        fig.savefig(plots_dir / "summary_dashboard.pdf", dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        print(f"\nSummary dashboard saved to {plots_dir}")
        
    except ImportError as e:
        print(f"Could not create summary visualization: {e}")


if __name__ == "__main__":
    main()
