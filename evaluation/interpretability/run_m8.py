#!/usr/bin/env python
"""
Run M8: Epistemic Uncertainty Decomposition on a trained INP model.

Usage:
    python run_m8.py --model-path saves/project/run_0/model_best.pt \
                     --config-path saves/project/run_0/config.toml

This script analyzes:
    - Total, aleatoric, and epistemic uncertainty vs context size
    - Zero-shot epistemic gap between INP and NP
    - "Bit-value" of knowledge (uncertainty reduction from K)
    - Convergence rate of epistemic uncertainty
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
from evaluation.interpretability.m8_uncertainty_decomposition import UncertaintyDecompositionExperiment
from evaluation.interpretability.base import ExperimentConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Run M8 Uncertainty Decomposition")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--config-path", type=str, required=True,
                        help="Path to model config (.toml)")
    parser.add_argument("--output-dir", type=str, default="./interpretability_results",
                        help="Directory to save results")
    parser.add_argument("--num-tasks", type=int, default=100,
                        help="Number of tasks to analyze")
    parser.add_argument("--num-z-samples", type=int, default=50,
                        help="Number of z samples for uncertainty estimation")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"],
                        help="Device to use (default: auto)")
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
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
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
    try:
        state_dict = torch.load(args.model_path, map_location=device)
    except Exception as e:
        if device == "cuda" and "out of memory" in str(e).lower():
            print("CUDA OOM while loading checkpoint. Falling back to CPU.")
            device = "cpu"
            config.device = device
            state_dict = torch.load(args.model_path, map_location=device)
        else:
            raise
    model.load_state_dict(state_dict)
    try:
        model.to(device)
    except Exception as e:
        if device == "cuda" and "out of memory" in str(e).lower():
            print("CUDA OOM while moving model to GPU. Falling back to CPU.")
            device = "cpu"
            config.device = device
            model.to(device)
        else:
            raise
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
    print("M8: Epistemic Uncertainty Decomposition")
    print("=" * 60)
    print("\nDecomposing predictive uncertainty into aleatoric (noise)")
    print("and epistemic (model ignorance) components.")
    
    experiment = UncertaintyDecompositionExperiment(
        model=model,
        config=exp_config,
        context_sizes=[0, 1, 3, 5, 10, 20, 30],
        num_z_samples=args.num_z_samples,
        num_tasks=args.num_tasks,
    )
    
    results = experiment.run(val_dataloader)
    
    # Print results
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    
    print("\nEpistemic Uncertainty vs Context Size:")
    print(f"{'N':>5}  {'With K (INP)':>15}  {'Without K (NP)':>15}")
    print("-" * 40)
    
    agg_k = results["with_knowledge_aggregated"]
    agg_no_k = results["without_knowledge_aggregated"]
    
    for n in sorted(agg_k.keys()):
        ep_k = agg_k[n]["epistemic_mean"]
        ep_no_k = agg_no_k.get(n, {"epistemic_mean": float('nan')})["epistemic_mean"]
        print(f"{n:>5}  {ep_k:>15.3f}  {ep_no_k:>15.3f}")
    
    if "zero_shot_analysis" in results:
        zs = results["zero_shot_analysis"]
        print(f"\nZero-Shot Analysis (N=0):")
        print(f"  Epistemic with K:    {zs['epistemic_with_k']:.3f} nats")
        print(f"  Epistemic without K: {zs['epistemic_without_k']:.3f} nats")
        print(f"  Reduction:           {zs['epistemic_reduction']:.3f} nats")
        print(f"  Bit-value of K:      {zs['bit_value_of_knowledge']:.3f} bits")
    
    print(f"\nInterpretation: {results['interpretation']}")
    print(f"\nResults saved to: {experiment.output_dir}")
    
    return results


if __name__ == "__main__":
    main()
