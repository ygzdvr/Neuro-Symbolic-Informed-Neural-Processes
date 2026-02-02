"""
Mechanistic Interpretability Suite for Informed Neural Processes (INPs)

M1: Information Bottleneck Analysis (MINE)
    - Measures I(Z; D) and I(Z; K) to understand how much the latent
      relies on context data vs. knowledge
    - Uses Mutual Information Neural Estimation (MINE)

M2: Loss Landscape Visualization
    - Analyzes flatness of loss landscape using filter-normalized directions
    - Compares basin width with/without knowledge conditioning
    - Tests hypothesis that knowledge convexifies the landscape

M3: Effective Dimensionality Analysis
    - Uses SVD to analyze intrinsic dimensionality of latent space
    - Computes participation ratio (Effective Dimensionality)
    - Tests manifold constraint hypothesis

M4: Gradient Alignment Score
    - Measures alignment between ∇L_NLL and ∇L_KL
    - Tests gradient synergy hypothesis
    - Detects prior-data conflict

M5: Causal Activation Patching
    - Proves causal efficacy of knowledge via intervention
    - Patches knowledge from Task B into Task A
    - Measures transfer ratio and alignment of prediction shift

M6: Knowledge Saliency & Attribution
    - Uses Integrated Gradients to attribute importance to knowledge features
    - For sinusoids: importance of (a, b, c) parameters
    - For text: token-level importance

M7: Linear Probing of Latent Representations
    - Tests if latents linearly encode ground-truth parameters
    - Measures disentanglement via R² scores
    - Compares linear vs MLP probes

M8: Epistemic Uncertainty Decomposition
    - Decomposes uncertainty: Total = Aleatoric + Epistemic
    - Measures "bit-value" of knowledge at N=0 (zero-shot)
    - Plots uncertainty decay curves vs context size

M9: Heavy-Tailed Self-Regularization (WeightWatcher)
    - Analyzes power law exponent (α) of weight spectra
    - α ∈ [2, 4] = Goldilocks zone for generalization
    - Tests implicit regularization from knowledge conditioning

M10: Centered Kernel Alignment (CKA) Similarity
    - Compares representations with vs without knowledge
    - Identifies where knowledge affects processing (info-fusion point)
    - Low CKA = fundamentally different representations
"""

from .base import InterpretabilityExperiment, ExperimentConfig
from .m1_information_bottleneck import (
    InformationBottleneckExperiment,
    MINENetwork,
    MINEEstimator,
    TrainingMITracker,
)
from .m2_loss_landscape import (
    LossLandscapeExperiment,
    get_filter_normalized_direction,
    compare_landscapes,
)
from .m3_effective_dimensionality import (
    EffectiveDimensionalityExperiment,
    compute_ed_for_knowledge_types,
)
from .m4_gradient_alignment import (
    GradientAlignmentExperiment,
    TrainingAlignmentTracker,
    cosine_similarity,
)
from .m5_activation_patching import (
    ActivationPatchingExperiment,
    KnowledgePatcher,
)
from .m6_knowledge_saliency import (
    KnowledgeSaliencyExperiment,
    integrated_gradients,
)
from .m7_linear_probing import (
    LinearProbingExperiment,
    LinearProbe,
    MLPProbe,
)
from .m8_uncertainty_decomposition import (
    UncertaintyDecompositionExperiment,
    gaussian_entropy,
)
from .m9_spectral_analysis import (
    SpectralAnalysisExperiment,
    compute_esd,
    fit_power_law,
)
from .m10_cka_similarity import (
    CKASimilarityExperiment,
    cka,
    hsic,
)
from .enhanced_viz import (
    setup_style,
    save_all_formats,
    viz_m4_gradient_alignment,
    viz_m5_activation_patching,
    viz_m6_knowledge_saliency,
    viz_m7_linear_probing,
    viz_m8_uncertainty,
    viz_m9_spectral,
    viz_m10_cka,
)

__all__ = [
    # Base
    "InterpretabilityExperiment",
    "ExperimentConfig",
    # M1
    "InformationBottleneckExperiment",
    "MINENetwork",
    "MINEEstimator",
    "TrainingMITracker",
    # M2
    "LossLandscapeExperiment",
    "get_filter_normalized_direction",
    "compare_landscapes",
    # M3
    "EffectiveDimensionalityExperiment",
    "compute_ed_for_knowledge_types",
    # M4
    "GradientAlignmentExperiment",
    "TrainingAlignmentTracker",
    "cosine_similarity",
    # M5
    "ActivationPatchingExperiment",
    "KnowledgePatcher",
    # M6
    "KnowledgeSaliencyExperiment",
    "integrated_gradients",
    # M7
    "LinearProbingExperiment",
    "LinearProbe",
    "MLPProbe",
    # M8
    "UncertaintyDecompositionExperiment",
    "gaussian_entropy",
    # M9
    "SpectralAnalysisExperiment",
    "compute_esd",
    "fit_power_law",
    # M10
    "CKASimilarityExperiment",
    "cka",
    "hsic",
    # Visualization utilities
    "setup_style",
    "save_all_formats",
    "viz_m4_gradient_alignment",
    "viz_m5_activation_patching",
    "viz_m6_knowledge_saliency",
    "viz_m7_linear_probing",
    "viz_m8_uncertainty",
    "viz_m9_spectral",
    "viz_m10_cka",
]
