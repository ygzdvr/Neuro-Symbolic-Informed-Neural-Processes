"""
Base classes and utilities for INP interpretability experiments.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from pathlib import Path
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))))


@dataclass
class ExperimentConfig:
    """Configuration for interpretability experiments."""
    
    # Model and data
    model_path: str = ""
    config_path: str = ""
    dataset: str = "set-trending-sinusoids"
    knowledge_type: str = "abc2"
    uses_knowledge: bool = True  # Whether model uses knowledge integration
    
    # Experiment settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 64
    num_batches: int = 50
    seed: int = 42
    
    # Output
    output_dir: str = "./interpretability_results"
    save_intermediates: bool = True
    
    # Method-specific configs stored as dict
    method_configs: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_path": self.model_path,
            "config_path": self.config_path,
            "dataset": self.dataset,
            "knowledge_type": self.knowledge_type,
            "device": self.device,
            "batch_size": self.batch_size,
            "num_batches": self.num_batches,
            "seed": self.seed,
            "output_dir": self.output_dir,
            "method_configs": self.method_configs,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentConfig":
        return cls(**d)


class InterpretabilityExperiment(ABC):
    """
    Abstract base class for all interpretability experiments.
    
    Each experiment should:
    1. Probe a specific aspect of the INP's mechanism
    2. Return quantitative metrics
    3. Optionally produce visualizations
    """
    
    name: str = "base"
    description: str = "Base interpretability experiment"
    
    def __init__(self, model: nn.Module, config: ExperimentConfig):
        """
        Initialize the experiment.
        
        Args:
            model: The INP model to analyze
            config: Experiment configuration
        """
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.results: Dict[str, Any] = {}
        
        # Ensure model is on correct device and in eval mode
        self.model.to(self.device)
        self.model.eval()
        
        # Create output directory
        self.output_dir = Path(config.output_dir) / self.name
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    @abstractmethod
    def run(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, Any]:
        """
        Run the interpretability experiment.
        
        Args:
            dataloader: DataLoader providing batches of (context, target, knowledge)
            
        Returns:
            Dictionary of results/metrics
        """
        pass
    
    def save_results(self, results: Dict[str, Any], filename: str = "results.json"):
        """Save results to JSON file."""
        save_path = self.output_dir / filename
        
        # Convert numpy arrays and tensors to lists for JSON serialization
        serializable_results = self._make_serializable(results)
        
        with open(save_path, "w") as f:
            json.dump(serializable_results, f, indent=2)
        
        return save_path
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert tensors and arrays to JSON-serializable format."""
        if isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj


class HookManager:
    """
    Utility class for managing forward/backward hooks on model layers.
    
    Useful for extracting activations, gradients, and intermediate representations.
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.activations: Dict[str, torch.Tensor] = {}
        self.gradients: Dict[str, torch.Tensor] = {}
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
    
    def register_forward_hook(self, name: str, module: nn.Module):
        """Register a forward hook to capture activations."""
        def hook(module, input, output):
            if isinstance(output, tuple):
                self.activations[name] = output[0].detach()
            else:
                self.activations[name] = output.detach()
        
        handle = module.register_forward_hook(hook)
        self.hooks.append(handle)
        return handle
    
    def register_backward_hook(self, name: str, module: nn.Module):
        """Register a backward hook to capture gradients."""
        def hook(module, grad_input, grad_output):
            if isinstance(grad_output, tuple):
                self.gradients[name] = grad_output[0].detach()
            else:
                self.gradients[name] = grad_output.detach()
        
        handle = module.register_full_backward_hook(hook)
        self.hooks.append(handle)
        return handle
    
    def clear(self):
        """Clear stored activations and gradients."""
        self.activations.clear()
        self.gradients.clear()
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()


def extract_intermediate_representations(
    model: nn.Module,
    x_context: torch.Tensor,
    y_context: torch.Tensor,
    x_target: torch.Tensor,
    y_target: torch.Tensor,
    knowledge: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    Extract intermediate representations from INP forward pass.
    
    Returns:
        Dictionary containing:
        - x_encoded: Encoded x coordinates
        - R: Global context representation
        - k: Knowledge embedding (if knowledge provided)
        - z_samples: Latent samples
        - q_z_loc: Latent mean
        - q_z_scale: Latent std
    """
    representations = {}
    
    # Encode x coordinates
    x_context_encoded = model.x_encoder(x_context)
    x_target_encoded = model.x_encoder(x_target)
    representations["x_context_encoded"] = x_context_encoded
    representations["x_target_encoded"] = x_target_encoded
    
    # Get global representation R
    R = model.encode_globally(x_context_encoded, y_context, x_target_encoded)
    representations["R"] = R
    
    # Get knowledge embedding if provided
    if knowledge is not None and model.latent_encoder.knowledge_encoder is not None:
        k = model.latent_encoder.knowledge_encoder(knowledge)
        representations["k"] = k
    else:
        representations["k"] = None
    
    # Get latent distribution parameters
    was_training = model.training
    model.eval()
    with torch.no_grad():
        z_samples, q_zCc, q_zCct = model.sample_latent(
            R, x_context_encoded, x_target_encoded, y_target, knowledge
        )
    
    representations["z_samples"] = z_samples
    representations["q_z_loc"] = q_zCc.base_dist.loc
    representations["q_z_scale"] = q_zCc.base_dist.scale
    
    if q_zCct is not None:
        representations["q_zCct_loc"] = q_zCct.base_dist.loc
        representations["q_zCct_scale"] = q_zCct.base_dist.scale
    
    if was_training:
        model.train()
    
    return representations


def compute_gradient_norm(model: nn.Module, loss: torch.Tensor) -> float:
    """Compute the total gradient norm across all parameters."""
    loss.backward(retain_graph=True)
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
