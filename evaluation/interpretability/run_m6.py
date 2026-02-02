#!/usr/bin/env python
"""
Run M6: Knowledge Saliency & Attribution on a trained INP model.

Usage:
    python run_m6.py --model-path saves/project/run_0/model_best.pt \
                     --config-path saves/project/run_0/config.toml

This script analyzes:
    - Which parts of knowledge input drive predictions
    - Feature-level importance for numeric knowledge (a, b, c for sinusoids)
    - Token-level importance for text knowledge
    - Integrated Gradients attribution with completeness check
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
from evaluation.interpretability.m6_knowledge_saliency import KnowledgeSaliencyExperiment
from evaluation.interpretability.base import ExperimentConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Run M6 Knowledge Saliency Analysis")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--config-path", type=str, required=True,
                        help="Path to model config (.toml)")
    parser.add_argument("--output-dir", type=str, default="./interpretability_results",
                        help="Directory to save results")
    parser.add_argument("--num-batches", type=int, default=50,
                        help="Number of batches to analyze")
    parser.add_argument("--n-steps", type=int, default=50,
                        help="Number of integration steps for IG")
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
        num_batches=args.num_batches,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    
    # Run experiment
    print("\n" + "=" * 60)
    print("M6: Knowledge Saliency & Attribution")
    print("=" * 60)
    print("\nUsing Integrated Gradients to determine which parts of")
    print("the knowledge input are most important for predictions.")
    
    experiment = KnowledgeSaliencyExperiment(
        model=model,
        config=exp_config,
        n_steps=args.n_steps,
    )
    
    results = experiment.run(val_dataloader)
    
    # Print results
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    
    if "feature_importance" in results:
        print("\nKnowledge Feature Importance:")
        for feature, importance in results["feature_importance"].items():
            bar = "█" * int(importance * 20)
            print(f"  {feature:20s}: {importance:6.1%} {bar}")
    elif "aggregated" in results:
        print("\nFeature Importance (relative):")
        for i, imp in enumerate(results["aggregated"]["relative_importance"][:10]):
            bar = "█" * int(imp * 20)
            print(f"  Feature {i:2d}: {imp:6.1%} {bar}")
    
    if "convergence_summary" in results:
        conv = results["convergence_summary"]
        print(f"\nConvergence Check:")
        print(f"  Mean delta: {conv['mean_delta']:.6f}")
        print(f"  Converged:  {'Yes' if conv['converged'] else 'No'}")
    
    print(f"\nInterpretation: {results['interpretation']}")
    print(f"\nResults saved to: {experiment.output_dir}")
    
    return results


if __name__ == "__main__":
    main()
