#!/usr/bin/env python
"""
Run M2: Loss Landscape Visualization on a trained INP model.

Usage:
    python run_m2.py --model-path saves/project/run_0/model_best.pt \
                     --config-path saves/project/run_0/config.toml

This script analyzes:
    - 1D loss profiles along filter-normalized random directions
    - Basin width (flatness indicator)
    - Curvature at minimum
    - Comparison with/without knowledge conditioning
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
from evaluation.interpretability.m2_loss_landscape import LossLandscapeExperiment
from evaluation.interpretability.base import ExperimentConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Run M2 Loss Landscape Analysis")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--config-path", type=str, required=True,
                        help="Path to model config (.toml)")
    parser.add_argument("--output-dir", type=str, default="./interpretability_results",
                        help="Directory to save results")
    parser.add_argument("--num-directions", type=int, default=5,
                        help="Number of random directions to sample")
    parser.add_argument("--num-steps", type=int, default=51,
                        help="Number of points along each direction")
    parser.add_argument("--alpha-range", type=float, nargs=2, default=[-0.5, 0.5],
                        help="Range of perturbation magnitudes")
    parser.add_argument("--flatness-epsilon", type=float, default=0.1,
                        help="Absolute threshold for basin width calculation")
    parser.add_argument("--flatness-epsilon-ratio", type=float, default=0.1,
                        help="Relative threshold (fraction of loss at origin)")
    parser.add_argument("--num-eval-batches", type=int, default=10,
                        help="Number of batches for each loss estimate")
    parser.add_argument("--loss-mode", type=str, default="elbo",
                        choices=["nll", "elbo"],
                        help="Loss mode for landscape evaluation")
    parser.add_argument("--beta", type=float, default=None,
                        help="KL weight for ELBO (defaults to config beta)")
    parser.add_argument("--num-z-samples", type=int, default=None,
                        help="Number of z samples for loss (defaults to model train samples)")
    parser.add_argument("--compute-plane", type=str, default="true",
                        help="Compute true 2D loss surface (true/false)")
    parser.add_argument("--plane-steps", type=int, default=21,
                        help="Resolution for 2D surface per axis")
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
    print("M2: Loss Landscape Visualization")
    print("=" * 60)
    
    experiment = LossLandscapeExperiment(
        model=model,
        config=exp_config,
        alpha_range=tuple(args.alpha_range),
        num_steps=args.num_steps,
        num_directions=args.num_directions,
        use_filter_normalization=True,
        flatness_epsilon=args.flatness_epsilon,
        flatness_epsilon_ratio=args.flatness_epsilon_ratio,
        num_eval_batches=args.num_eval_batches,
        loss_mode=args.loss_mode,
        beta=args.beta if args.beta is not None else getattr(config, "beta", 1.0),
        num_z_samples=args.num_z_samples,
        compute_plane=str(args.compute_plane).lower() == "true",
        plane_num_steps=args.plane_steps,
    )
    
    results = experiment.run(val_dataloader)
    
    # Print results
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    
    agg_k = results["aggregated_with_knowledge"]
    agg_no_k = results["aggregated_without_knowledge"]
    
    print("\nWith Knowledge (INP):")
    print(f"  Basin width:    {agg_k.get('basin_width_mean', 0):.4f} ± {agg_k.get('basin_width_std', 0):.4f}")
    print(f"  Curvature:      {agg_k.get('curvature_mean', 0):.4f} ± {agg_k.get('curvature_std', 0):.4f}")
    print(f"  Barrier height: {agg_k.get('barrier_height_mean', 0):.4f} ± {agg_k.get('barrier_height_std', 0):.4f}")
    
    print("\nWithout Knowledge (NP baseline):")
    print(f"  Basin width:    {agg_no_k.get('basin_width_mean', 0):.4f} ± {agg_no_k.get('basin_width_std', 0):.4f}")
    print(f"  Curvature:      {agg_no_k.get('curvature_mean', 0):.4f} ± {agg_no_k.get('curvature_std', 0):.4f}")
    print(f"  Barrier height: {agg_no_k.get('barrier_height_mean', 0):.4f} ± {agg_no_k.get('barrier_height_std', 0):.4f}")
    
    print("\nComparison:")
    print(f"  Basin width ratio (INP/NP): {results['comparison']['basin_width_ratio']:.2f}")
    print(f"  Barrier reduction:          {results['comparison']['barrier_reduction']:.4f}")
    
    print(f"\nInterpretation: {results['interpretation']}")
    print(f"\nResults saved to: {experiment.output_dir}")
    
    return results


if __name__ == "__main__":
    main()
