#!/usr/bin/env python
"""
Run interpretability experiments across multiple models in a directory.

This script:
1. Discovers all model subdirectories in the given directory
2. Runs experiments sequentially: M1 for all models, then M2 for all models, etc.
3. Creates individual plots per model

Usage:
    python run_batch.py --models-dir saves/INPs_sinusoids --output-dir ./interpretability_results/batch

    # Run specific experiments only
    python run_batch.py --models-dir saves/INPs_sinusoids --experiments M1 M3 M7

    # Quick mode
    python run_batch.py --models-dir saves/INPs_sinusoids --quick
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
from typing import Dict, List, Any, Tuple
from tqdm import tqdm
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))))

from config import Config
from models.inp import INP
from models.nsinp import NSINP
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

# Experiments that require knowledge integration
KNOWLEDGE_REQUIRED_EXPERIMENTS = {"M1", "M5", "M6", "M10"}


def discover_models(models_dir: Path) -> List[Dict[str, Any]]:
    """
    Discover all model subdirectories in the given directory.
    
    Each subdirectory should contain:
    - model_best.pt: The trained model weights
    - config.toml: The model configuration
    
    Returns:
        List of dicts with model info: {name, model_path, config_path, uses_knowledge}
    """
    models = []
    
    for subdir in sorted(models_dir.iterdir()):
        if not subdir.is_dir():
            continue
            
        model_path = subdir / "model_best.pt"
        config_path = subdir / "config.toml"
        
        if not model_path.exists() or not config_path.exists():
            print(f"  Skipping {subdir.name}: missing model_best.pt or config.toml")
            continue
        
        # Load config to check if it uses knowledge
        try:
            config_dict = toml.load(config_path)
            uses_knowledge = config_dict.get('use_knowledge', True)
        except Exception as e:
            print(f"  Skipping {subdir.name}: could not load config - {e}")
            continue
        
        models.append({
            "name": subdir.name,
            "model_path": str(model_path),
            "config_path": str(config_path),
            "uses_knowledge": uses_knowledge,
        })
    
    return models


def load_model(model_info: Dict[str, Any], device: str) -> Tuple[INP, Config, Any]:
    """Load a model and its dataloader."""
    config_dict = toml.load(model_info["config_path"])
    config = Config(**config_dict)
    config.device = device

    # Setup dataloaders
    train_dataloader, val_dataloader, _, extras = setup_dataloaders(config)
    for k, v in extras.items():
        config.__dict__[k] = v

    # Load model based on model_type
    model_type = getattr(config, 'model_type', 'inp')
    if model_type == 'nsinp':
        model = NSINP(config)
    else:
        model = INP(config)
    state_dict = torch.load(model_info["model_path"], map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    return model, config, val_dataloader


def run_experiment_for_model(
    exp_id: str,
    model: INP,
    config: Config,
    dataloader: Any,
    model_info: Dict[str, Any],
    output_dir: Path,
    quick: bool = False,
) -> Dict[str, Any]:
    """Run a single experiment for a single model."""
    
    name, ExperimentClass = EXPERIMENTS[exp_id]
    model_name = model_info["name"]
    uses_knowledge = model_info["uses_knowledge"]
    
    # Skip knowledge-requiring experiments for NP baselines
    if not uses_knowledge and exp_id in KNOWLEDGE_REQUIRED_EXPERIMENTS:
        return {
            "model": model_name,
            "experiment": exp_id,
            "status": "skipped",
            "reason": "NP baseline without knowledge",
        }
    
    # Create experiment config
    exp_config = ExperimentConfig(
        model_path=model_info["model_path"],
        config_path=model_info["config_path"],
        dataset=config.dataset,
        knowledge_type=getattr(config, 'knowledge_type', 'abc2'),
        uses_knowledge=uses_knowledge,
        device=config.device,
        batch_size=config.batch_size,
        num_batches=10 if quick else 50,
        seed=42,
        output_dir=str(output_dir / model_name / exp_id.lower()),
    )
    
    # Create output directory
    exp_output_dir = output_dir / model_name / exp_id.lower()
    exp_output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Create experiment with appropriate kwargs
        if exp_id == "M1":
            kwargs = {"mine_train_steps": 30 if quick else 100}
        elif exp_id == "M2":
            kwargs = {"num_directions": 2 if quick else 5}
        elif exp_id == "M3":
            kwargs = {"max_samples": 500 if quick else 2000}
        elif exp_id == "M5":
            kwargs = {"num_pairs": 20 if quick else 50}
        elif exp_id == "M7":
            kwargs = {"probe_epochs": 50 if quick else 100}
        elif exp_id == "M8":
            kwargs = {"num_tasks": 30 if quick else 100}
        elif exp_id == "M10":
            kwargs = {"max_samples": 200 if quick else 500}
        else:
            kwargs = {}
        
        experiment = ExperimentClass(model, exp_config, **kwargs)
        results = experiment.run(dataloader)
        
        # Save individual results
        results_path = exp_output_dir / "results.json"
        with open(results_path, "w") as f:
            # Convert numpy arrays to lists for JSON serialization
            json.dump(_serialize_results(results), f, indent=2)
        
        return {
            "model": model_name,
            "experiment": exp_id,
            "status": "success",
            "results": results,
            "interpretation": results.get("interpretation", "N/A"),
            "output_dir": str(exp_output_dir),
        }
        
    except Exception as e:
        return {
            "model": model_name,
            "experiment": exp_id,
            "status": "failed",
            "error": str(e),
        }


def _serialize_results(obj):
    """Convert numpy arrays and other non-JSON types to JSON-serializable format."""
    if isinstance(obj, dict):
        return {k: _serialize_results(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_serialize_results(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    else:
        return obj


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run interpretability experiments across multiple models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--models-dir", type=str, required=True,
                        help="Directory containing model subdirectories")
    parser.add_argument("--output-dir", type=str, default="./interpretability_results/batch",
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
    args.experiments = [exp.upper() for exp in args.experiments]
    for exp in args.experiments:
        if exp not in EXPERIMENTS:
            print(f"Unknown experiment: {exp}")
            print(f"Available: {', '.join(EXPERIMENTS.keys())}")
            return
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Discover models
    models_dir = Path(args.models_dir)
    if not models_dir.exists():
        print(f"Error: Models directory does not exist: {models_dir}")
        return
    
    print(f"\nDiscovering models in {models_dir}...")
    models = discover_models(models_dir)
    
    if len(models) == 0:
        print("No valid models found!")
        return
    
    print(f"Found {len(models)} models:")
    for m in models:
        k_status = "INP (uses knowledge)" if m["uses_knowledge"] else "NP (no knowledge)"
        print(f"  - {m['name']}: {k_status}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"batch_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nOutput directory: {output_dir}")
    
    # Store all results
    all_results = {exp_id: [] for exp_id in args.experiments}
    
    # Run experiments sequentially: M1 for all models, then M2 for all models, etc.
    for exp_id in args.experiments:
        name, _ = EXPERIMENTS[exp_id]
        
        print(f"\n{'=' * 70}")
        print(f"{exp_id}: {name}")
        print("=" * 70)
        
        exp_results = []
        
        for model_info in tqdm(models, desc=f"Running {exp_id}"):
            model_name = model_info["name"]
            
            # Load model
            try:
                model, config, dataloader = load_model(model_info, device)
            except Exception as e:
                print(f"  Failed to load {model_name}: {e}")
                exp_results.append({
                    "model": model_name,
                    "experiment": exp_id,
                    "status": "failed",
                    "error": f"Load error: {e}",
                })
                continue
            
            # Run experiment
            result = run_experiment_for_model(
                exp_id, model, config, dataloader, model_info, output_dir, args.quick
            )
            exp_results.append(result)
            
            # Print result
            if result["status"] == "success":
                interp = result.get("interpretation", "N/A")
                print(f"  ✓ {model_name}: {interp[:60]}...")
            elif result["status"] == "skipped":
                print(f"  ○ {model_name}: Skipped - {result.get('reason', 'N/A')}")
            else:
                print(f"  ✗ {model_name}: {result.get('error', 'Unknown error')[:60]}")
            
            # Clear GPU memory
            del model
            torch.cuda.empty_cache()
        
        all_results[exp_id] = exp_results
    
    # Save all results to JSON
    summary_path = output_dir / "all_results.json"
    serializable_results = {
        exp_id: [
            {k: v for k, v in r.items() if k != "results"}
            for r in exp_results
        ]
        for exp_id, exp_results in all_results.items()
    }
    with open(summary_path, "w") as f:
        json.dump(serializable_results, f, indent=2)
    
    # Print final summary
    print("\n" + "=" * 70)
    print("BATCH RUN COMPLETE")
    print("=" * 70)
    
    for exp_id in args.experiments:
        exp_results = all_results[exp_id]
        success = sum(1 for r in exp_results if r["status"] == "success")
        skipped = sum(1 for r in exp_results if r["status"] == "skipped")
        failed = sum(1 for r in exp_results if r["status"] == "failed")
        print(f"{exp_id}: {success} success, {skipped} skipped, {failed} failed")
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
