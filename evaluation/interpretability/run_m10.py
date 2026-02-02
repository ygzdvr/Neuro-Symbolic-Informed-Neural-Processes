#!/usr/bin/env python
"""
Run M10: CKA Representation Similarity Analysis on a trained INP model.

Usage:
    python run_m10.py --model-path saves/project/run_0/model_best.pt \
                      --config-path saves/project/run_0/config.toml

This script analyzes:
    - CKA between representations with vs without knowledge
    - Layer-wise similarity patterns
    - Where knowledge affects processing most (info-fusion point)
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
from evaluation.interpretability.m10_cka_similarity import CKASimilarityExperiment
from evaluation.interpretability.base import ExperimentConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Run M10 CKA Similarity Analysis")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--config-path", type=str, required=True,
                        help="Path to model config (.toml)")
    parser.add_argument("--output-dir", type=str, default="./interpretability_results",
                        help="Directory to save results")
    parser.add_argument("--kernel", type=str, default="linear", choices=["linear", "rbf"],
                        help="Kernel for CKA computation")
    parser.add_argument("--max-samples", type=int, default=500,
                        help="Maximum samples for CKA computation")
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
    print("M10: CKA Representation Similarity Analysis")
    print("=" * 60)
    print("\nComparing internal representations WITH vs WITHOUT knowledge")
    print("to understand where knowledge affects processing.")
    
    experiment = CKASimilarityExperiment(
        model=model,
        config=exp_config,
        kernel=args.kernel,
        max_samples=args.max_samples,
    )
    
    results = experiment.run(val_dataloader)
    
    # Print results
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    
    print("\nKey Representation CKA Scores:")
    for key in ["R", "z_mean", "z_std", "pred_mean"]:
        if key in results["cka_scores"]:
            score = results["cka_scores"][key]
            bar = "█" * int(score * 20)
            interpretation = "Similar" if score > 0.7 else "Different" if score < 0.4 else "Moderate"
            print(f"  {key:12s}: {score:.3f} {bar} ({interpretation})")
    
    if "analysis" in results:
        analysis = results["analysis"]
        print(f"\nAnalysis:")
        print(f"  Mean CKA:       {analysis['mean_cka']:.3f} ± {analysis['std_cka']:.3f}")
        print(f"  Max CKA layer:  {analysis['max_cka_layer']} ({analysis['max_cka_score']:.3f})")
        print(f"  Min CKA layer:  {analysis['min_cka_layer']} ({analysis['min_cka_score']:.3f})")
    
    print(f"\nInterpretation: {results['interpretation']}")
    print(f"\nResults saved to: {experiment.output_dir}")
    
    return results


if __name__ == "__main__":
    main()
