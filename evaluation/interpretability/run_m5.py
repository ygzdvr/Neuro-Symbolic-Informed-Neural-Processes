#!/usr/bin/env python
"""
Run M5: Causal Activation Patching on a trained INP model.

Usage:
    python run_m5.py --model-path saves/project/run_0/model_best.pt \
                     --config-path saves/project/run_0/config.toml

This script analyzes:
    - Causal efficacy of the knowledge channel via intervention
    - Transfer ratio: How much of the ideal shift is achieved?
    - Alignment: Does the shift go in the right direction?
    - MSE improvement: Does patching help predict donor task?
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
from evaluation.interpretability.m5_activation_patching import ActivationPatchingExperiment
from evaluation.interpretability.base import ExperimentConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Run M5 Causal Activation Patching")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--config-path", type=str, required=True,
                        help="Path to model config (.toml)")
    parser.add_argument("--output-dir", type=str, default="./interpretability_results",
                        help="Directory to save results")
    parser.add_argument("--num-pairs", type=int, default=50,
                        help="Number of task pairs to test")
    parser.add_argument("--num-z-samples", type=int, default=32,
                        help="Number of latent samples for prediction")
    parser.add_argument("--max-batch-size", type=int, default=16,
                        help="Cap batch size per patching pair")
    parser.add_argument("--max-target-points", type=int, default=None,
                        help="Optional cap on number of target points")
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
    print("M5: Causal Activation Patching")
    print("=" * 60)
    print("\nThis experiment tests whether the knowledge channel has")
    print("genuine causal efficacy by patching knowledge from one task")
    print("into another and measuring how predictions shift.")
    
    experiment = ActivationPatchingExperiment(
        model=model,
        config=exp_config,
        num_pairs=args.num_pairs,
        num_z_samples=args.num_z_samples,
        max_batch_size=args.max_batch_size,
        max_target_points=args.max_target_points,
    )
    
    results = experiment.run(val_dataloader)
    
    # Print results
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    
    agg = results["aggregated"]
    
    print("\nCausal Effect Metrics:")
    print(f"  Transfer Ratio:    {agg['transfer_ratio']['mean']:.3f} ± {agg['transfer_ratio']['std']:.3f}")
    print(f"  Alignment:         {agg['alignment']['mean']:.3f} ± {agg['alignment']['std']:.3f}")
    print(f"  Direct Effect:     {agg['direct_effect']['mean']:.4f} ± {agg['direct_effect']['std']:.4f}")
    print(f"  MSE Improvement:   {agg['mse_improvement']['mean']:.4f} ± {agg['mse_improvement']['std']:.4f}")
    
    print(f"\nCausal Efficacy Score: {results['causal_efficacy_score']:.3f}")
    
    print(f"\nInterpretation: {results['interpretation']}")
    print(f"\nResults saved to: {experiment.output_dir}")
    
    return results


if __name__ == "__main__":
    main()
