#!/usr/bin/env python
"""
Run M7: Linear Probing Analysis on a trained INP model.

Usage:
    python run_m7.py --model-path saves/project/run_0/model_best.pt \
                     --config-path saves/project/run_0/config.toml

This script analyzes:
    - Whether latent codes linearly encode ground-truth parameters
    - R² scores for linear vs MLP probes
    - Per-parameter recovery quality (a, b, c for sinusoids)
    - Knowledge benefit for representation disentanglement
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
from evaluation.interpretability.m7_linear_probing import LinearProbingExperiment
from evaluation.interpretability.base import ExperimentConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Run M7 Linear Probing Analysis")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--config-path", type=str, required=True,
                        help="Path to model config (.toml)")
    parser.add_argument("--output-dir", type=str, default="./interpretability_results",
                        help="Directory to save results")
    parser.add_argument("--probe-epochs", type=int, default=100,
                        help="Number of epochs for probe training")
    parser.add_argument("--test-fraction", type=float, default=0.2,
                        help="Fraction of data for testing")
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
    model.eval()
    
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
    )
    
    # Run experiment
    print("\n" + "=" * 60)
    print("M7: Linear Probing of Latent Representations")
    print("=" * 60)
    print("\nTesting whether latent codes z linearly encode ground-truth")
    print("generative parameters (a, b, c for sinusoids).")
    
    experiment = LinearProbingExperiment(
        model=model,
        config=exp_config,
        probe_epochs=args.probe_epochs,
        test_fraction=args.test_fraction,
        use_sklearn=True,
    )
    
    results = experiment.run(val_dataloader)
    
    # Print results
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    
    comp = results["comparison"]
    
    print("\nOverall R² Scores:")
    print(f"                        Linear    MLP")
    print(f"  With Knowledge (INP):  {comp['r2_linear_with_k']:.3f}    {comp['r2_mlp_with_k']:.3f}")
    print(f"  Without Knowledge (NP): {comp['r2_linear_without_k']:.3f}    {comp['r2_mlp_without_k']:.3f}")
    
    print(f"\nKnowledge Benefit (Linear): {comp['knowledge_benefit_linear']:+.3f}")
    print(f"Linear Gap (MLP - Linear):")
    print(f"  With Knowledge:    {comp['linear_gap_with_k']:.3f}")
    print(f"  Without Knowledge: {comp['linear_gap_without_k']:.3f}")
    
    if "per_parameter" in results:
        print("\nPer-Parameter R² (Linear Probe):")
        for name, scores in results["per_parameter"].items():
            print(f"  {name:20s}: With K = {scores['r2_linear_with_k']:.3f}, Without K = {scores['r2_linear_without_k']:.3f}")
    
    print(f"\nInterpretation: {results['interpretation']}")
    print(f"\nResults saved to: {experiment.output_dir}")
    
    return results


if __name__ == "__main__":
    main()
