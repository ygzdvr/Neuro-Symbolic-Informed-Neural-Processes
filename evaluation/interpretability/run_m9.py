#!/usr/bin/env python
"""
Run M9: Heavy-Tailed Self-Regularization Analysis on a trained INP model.

Usage:
    python run_m9.py --model-path saves/project/run_0/model_best.pt \
                     --config-path saves/project/run_0/config.toml

This script analyzes:
    - Power law exponent (α) of weight matrix spectra
    - Goldilocks zone analysis (α ∈ [2, 4])
    - Per-module spectral properties
    - Stable rank of weight matrices
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
from evaluation.interpretability.m9_spectral_analysis import SpectralAnalysisExperiment
from evaluation.interpretability.base import ExperimentConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Run M9 Spectral Analysis")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--config-path", type=str, required=True,
                        help="Path to model config (.toml)")
    parser.add_argument("--output-dir", type=str, default="./interpretability_results",
                        help="Directory to save results")
    parser.add_argument("--use-weightwatcher", action="store_true",
                        help="Use WeightWatcher library if available")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load config
    print(f"Loading config from {args.config_path}")
    config_dict = toml.load(args.config_path)
    config = Config(**config_dict)
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config.device = device
    print(f"Using device: {device}")
    
    # Setup dataloaders (needed for config extras)
    print("Setting up config...")
    train_dataloader, val_dataloader, _, extras = setup_dataloaders(config)
    for k, v in extras.items():
        config.__dict__[k] = v
    
    # Load model
    print(f"Loading model from {args.model_path}")
    model = INP(config)
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # Setup experiment config
    exp_config = ExperimentConfig(
        model_path=args.model_path,
        config_path=args.config_path,
        dataset=config.dataset,
        knowledge_type=getattr(config, 'knowledge_type', 'abc2'),
        device=device,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    
    # Run experiment
    print("\n" + "=" * 60)
    print("M9: Heavy-Tailed Self-Regularization (HTSR) Analysis")
    print("=" * 60)
    print("\nAnalyzing power law exponent (α) of weight matrix spectra.")
    print("α ∈ [2, 4] = Goldilocks zone (good generalization)")
    
    experiment = SpectralAnalysisExperiment(
        model=model,
        config=exp_config,
        use_weightwatcher=args.use_weightwatcher,
    )
    
    results = experiment.run()
    
    # Print results
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    
    if "summary" in results:
        summary = results["summary"]
        print(f"\nOverall Summary:")
        print(f"  Mean α:  {summary['mean_alpha']:.3f} ± {summary['std_alpha']:.3f}")
        print(f"  Range:   [{summary['min_alpha']:.3f}, {summary['max_alpha']:.3f}]")
    
    if "goldilocks_analysis" in results:
        ga = results["goldilocks_analysis"]
        print(f"\nGoldilocks Analysis:")
        print(f"  Layers in [2,4]:  {ga['num_in_goldilocks']} ({ga['fraction_in_goldilocks']:.1%})")
        print(f"  Overfit (α<2):    {ga['num_overfit']}")
        print(f"  Underfit (α>6):   {ga['num_underfit']}")
    
    if "by_module" in results:
        print("\nPer-Module Analysis:")
        for module, data in results["by_module"].items():
            if not np.isnan(data.get("mean_alpha", float('nan'))):
                print(f"  {module:20s}: α = {data['mean_alpha']:.3f} ± {data.get('std_alpha', 0):.3f}")
    
    print(f"\nInterpretation: {results['interpretation']}")
    print(f"\nResults saved to: {experiment.output_dir}")
    
    return results


if __name__ == "__main__":
    main()
