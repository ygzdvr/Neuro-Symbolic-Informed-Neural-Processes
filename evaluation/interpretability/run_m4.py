#!/usr/bin/env python
"""
Run M4: Gradient Alignment Score Analysis on a trained INP model.

Usage:
    python run_m4.py --model-path saves/project/run_0/model_best.pt \
                     --config-path saves/project/run_0/config.toml

This script analyzes:
    - Loss balance between NLL and beta * KL loss terms (proxy)
    - Comparison with randomized knowledge baseline
    - Detection of prior-data conflict
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
from evaluation.interpretability.m4_gradient_alignment import GradientAlignmentExperiment
from evaluation.interpretability.base import ExperimentConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Run M4 Gradient Alignment Analysis")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--config-path", type=str, required=True,
                        help="Path to model config (.toml)")
    parser.add_argument("--output-dir", type=str, default="./interpretability_results",
                        help="Directory to save results")
    parser.add_argument("--num-batches", type=int, default=50,
                        help="Number of batches to analyze")
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
    
    # Setup experiment config
    exp_config = ExperimentConfig(
        model_path=args.model_path,
        config_path=args.config_path,
        dataset=config.dataset,
        knowledge_type=getattr(config, 'knowledge_type', 'abc2'),
        device=device,
        batch_size=config.batch_size,
        num_batches=args.num_batches,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    
    # Run experiment
    print("\n" + "=" * 60)
    print("M4: Gradient Alignment Score Analysis")
    print("=" * 60)
    
    experiment = GradientAlignmentExperiment(
        model=model,
        config=exp_config,
        analyze_modules=True,
        beta=getattr(config, 'beta', 1.0),
    )
    
    results = experiment.run(val_dataloader)
    
    # Print results
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    
    agg_k = results["with_knowledge_aggregated"]
    agg_rand = results["with_random_knowledge_aggregated"]
    
    print("\nWith Knowledge:")
    for module in ["all", "latent_encoder", "xy_encoder", "decoder"]:
        key = f"{module}_alignment"
        if key in agg_k:
            print(f"  {module:20s}: Balance = {agg_k[key]['mean']:+.4f} ± {agg_k[key]['std']:.4f}")
    
    print("\nWith Random Knowledge (baseline):")
    for module in ["all", "latent_encoder", "xy_encoder", "decoder"]:
        key = f"{module}_alignment"
        if key in agg_rand:
            print(f"  {module:20s}: Balance = {agg_rand[key]['mean']:+.4f} ± {agg_rand[key]['std']:.4f}")
    
    print("\nComparison:")
    for key, value in results["comparison"].items():
        print(f"  {key}: {value:+.4f}")
    
    print(f"\nInterpretation: {results['interpretation']}")
    print(f"\nResults saved to: {experiment.output_dir}")
    
    return results


if __name__ == "__main__":
    main()
