#!/usr/bin/env python
"""
Run M3: Effective Dimensionality Analysis on a trained INP model.

Usage:
    python run_m3.py --model-path saves/project/run_0/model_best.pt \
                     --config-path saves/project/run_0/config.toml

This script analyzes:
    - Intrinsic dimensionality of latent codes using SVD
    - Participation ratio (Effective Dimensionality)
    - Cumulative explained variance curves
    - Comparison with knowledge disabled or NP baseline checkpoint
"""

import argparse
import sys
import os
import toml
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))))

from config import Config
from models.inp import INP
from dataset.utils import setup_dataloaders
from evaluation.interpretability.m3_effective_dimensionality import EffectiveDimensionalityExperiment
from evaluation.interpretability.base import ExperimentConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Run M3 Effective Dimensionality Analysis")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--config-path", type=str, required=True,
                        help="Path to model config (.toml)")
    parser.add_argument("--np-model-path", type=str, default=None,
                        help="Optional NP baseline checkpoint (.pt)")
    parser.add_argument("--np-config-path", type=str, default=None,
                        help="Optional NP baseline config (.toml)")
    parser.add_argument("--output-dir", type=str, default="./interpretability_results",
                        help="Directory to save results")
    parser.add_argument("--max-samples", type=int, default=2000,
                        help="Maximum number of latent samples to collect")
    parser.add_argument("--max-visual-samples", type=int, default=1000,
                        help="Max samples per condition for PCA/t-SNE plots")
    parser.add_argument("--variance-threshold", type=float, default=0.95,
                        help="Threshold for components-to-X% metric")
    parser.add_argument("--no-tsne", action="store_true",
                        help="Disable t-SNE projection plot")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    if (args.np_model_path is None) != (args.np_config_path is None):
        raise ValueError("Both --np-model-path and --np-config-path must be provided together.")
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load INP config
    print(f"Loading config from {args.config_path}")
    config_dict = toml.load(args.config_path)
    config = Config(**config_dict)

    np_config = None
    if args.np_config_path is not None:
        print(f"Loading NP config from {args.np_config_path}")
        np_config_dict = toml.load(args.np_config_path)
        np_config = Config(**np_config_dict)
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config.device = device
    print(f"Using device: {device}")
    
    # Setup dataloaders (shared for INP/NP comparison)
    print("Setting up dataloaders...")
    train_dataloader, val_dataloader, _, extras = setup_dataloaders(config)
    for k, v in extras.items():
        config.__dict__[k] = v
    if np_config is not None:
        for k, v in extras.items():
            np_config.__dict__[k] = v
    
    # Load INP model
    print(f"Loading model from {args.model_path}")
    model = INP(config)
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Load NP baseline model if provided
    np_model = None
    baseline_label = "Without Knowledge (Ablation)"
    baseline_info = {"type": "ablation_same_model"}
    if args.np_model_path is not None and np_config is not None:
        print(f"Loading NP model from {args.np_model_path}")
        np_model = INP(np_config)
        np_state_dict = torch.load(args.np_model_path, map_location=device)
        np_model.load_state_dict(np_state_dict)
        np_model.to(device)
        np_model.eval()
        baseline_label = "NP Baseline"
        baseline_info = {
            "type": "np_checkpoint",
            "model_path": args.np_model_path,
            "config_path": args.np_config_path,
        }
    
    # Setup experiment config
    exp_config = ExperimentConfig(
        model_path=args.model_path,
        config_path=args.config_path,
        dataset=config.dataset,
        knowledge_type=getattr(config, 'knowledge_type', 'abc2'),
        device=device,
        batch_size=config.batch_size,
        seed=args.seed,
        output_dir=args.output_dir,
        method_configs={"baseline": baseline_info},
    )
    
    # Run experiment
    print("\n" + "=" * 60)
    print("M3: Effective Dimensionality Analysis")
    print("=" * 60)
    
    experiment = EffectiveDimensionalityExperiment(
        model=model,
        config=exp_config,
        variance_threshold=args.variance_threshold,
        max_samples=args.max_samples,
        baseline_model=np_model,
        baseline_label=baseline_label,
        baseline_info=baseline_info,
        max_visual_samples=args.max_visual_samples,
        compute_tsne=not args.no_tsne,
    )
    
    results = experiment.run(val_dataloader)
    
    # Print results
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    
    with_k = results["with_knowledge"]
    without_k = results["without_knowledge"]
    
    print(f"\nLatent dimension: {with_k['latent_dim']}")
    print(f"Samples analyzed: {with_k['num_samples']}")
    
    print("\nWith Knowledge (INP):")
    print(f"  Effective Dimensionality: {with_k['effective_dimensionality']:.2f}")
    print(f"  Components to {args.variance_threshold:.0%}:  {with_k['components_to_threshold']}")
    
    print("\nWithout Knowledge (NP baseline):")
    print(f"  Effective Dimensionality: {without_k['effective_dimensionality']:.2f}")
    print(f"  Components to {args.variance_threshold:.0%}:  {without_k['components_to_threshold']}")
    
    print("\nComparison:")
    print(f"  ED ratio (INP/NP):    {results['comparison']['ed_ratio']:.2%}")
    print(f"  ED reduction:         {results['comparison']['ed_reduction']:.2f}")
    print(f"  Components reduction: {results['comparison']['components_reduction']}")
    
    print(f"\nInterpretation: {results['interpretation']}")
    print(f"\nResults saved to: {experiment.output_dir}")
    
    return results


if __name__ == "__main__":
    main()
