# M1–M10 Interpretability Report

This report summarizes the M1 (Information Bottleneck), M2 (Loss Landscape), M3
(Effective Dimensionality), M4 (Loss Balance), M5 (Activation Patching), M6
(Knowledge Saliency), M7 (Linear Probing), M8 (Uncertainty Decomposition),
M9 (Spectral Analysis), and M10 (CKA Similarity) experiments after stabilizing
the implementation and re-running targeted runs.

## Scope

- Models: INP on sinusoids (example run: `inp_abc__0`)
- Focus: M1, M2, M3, M4, M5, M6, M7, M8, M9, and M10
- Outputs reviewed:
  - M1 debug: `interpretability_results/debug_m1/run_20260201_190009/`
  - M2 fixed: `interpretability_results/m2_fixed/`
  - M3 run: `interpretability_results/m3_effective_dimensionality/`
  - M4 run: `interpretability_results/m4_gradient_alignment/`
  - M5 run: `interpretability_results/m5_activation_patching/`
  - M6 run: `interpretability_results/m6_knowledge_saliency/`
  - M7 run: `interpretability_results/m7_linear_probing/`
  - M8 run: `interpretability_results/m8_uncertainty_decomposition/`
  - M9 run: `interpretability_results/m9_run/`
  - M10 run: `interpretability_results/m10_run/`

## M1: Information Bottleneck (MINE)

### Method summary

M1 estimates mutual information between latent `z` and:

- Data context `D` (I(Z;D))
- Knowledge `K` (I(Z;K))

Stability fixes applied:

- Knowledge dropout disabled during M1 evaluation.
- Stable MI summary computed via peak-plateau trimmed mean.
- Report both stable summary and last-20% summary.

### Key results (debug run)

From `interpretability_results/debug_m1/run_20260201_190009/m1_information_bottleneck/plots/m1_mi_analysis.pdf`:

- I(Z;D) ≈ **0.061**
- I(Z;K) ≈ **0.140**
- Knowledge reliance ≈ **69.7%**

These values are in the expected “balanced to knowledge-leaning” range
(roughly 0.3–0.7 reliance in the experiment spec).

### Diagnostics supporting validity

Additional checks (on `inp_abc__0`) indicate:

- Latent variance is non-trivial (no collapse).
- Knowledge materially shifts latent mean (z changes when K is shuffled).

### Interpretation

The model integrates knowledge and data in the latent. The earlier near‑zero MI
was driven by late‑step decay and stochastic knowledge dropout. With stabilization,
M1 results make sense and are consistent with healthy integration.

### Artifacts

- MI curves and summary:  
  `interpretability_results/debug_m1/run_20260201_190009/m1_information_bottleneck/plots/m1_mi_analysis.pdf`
- MI heatmap:  
  `interpretability_results/debug_m1/run_20260201_190009/m1_information_bottleneck/plots/m1_mi_heatmap.pdf`

## M2: Loss Landscape Visualization

### Method summary

M2 probes local flatness around the trained point by evaluating:

- 1D loss slices along random directions (shared across INP/NP)
- A true 2D loss surface (two directions, grid evaluation)
- Basin width, curvature, and barrier height around **alpha=0**

Stability fixes applied:

- Same perturbation direction for with/without knowledge comparisons.
- Relative basin threshold (`epsilon_abs = max(abs_eps, rel_eps * loss_at_origin)`).
- ELBO evaluation (NLL + beta * KL) instead of NLL only.
- Knowledge dropout disabled during evaluation.
- True 2D surface (heatmap + 3D plot).

### Key results (fixed run)

From `interpretability_results/m2_fixed/m2_loss_landscape/results.json`:

- Basin width mean:
  - INP: **0.048**
  - NP:  **0.032**
  - Ratio: **1.50**
- Curvature mean:
  - INP: **957**
  - NP:  **2465**
  - Ratio: **0.39** (INP is flatter)
- Barrier height mean:
  - INP: **473.6**
  - NP:  **436.3**
  - Reduction: **‑37.3** (small, not decisive)

### Interpretation

The landscape is now a smooth basin around the origin in both 1D and 2D plots.
INP shows a moderately wider basin and lower curvature, consistent with the
expected regularization effect of knowledge.

### Artifacts

- 1D profiles + metrics:  
  `interpretability_results/m2_fixed/m2_loss_landscape/plots/m2_loss_landscape.pdf`
- 2D heatmap:  
  `interpretability_results/m2_fixed/m2_loss_landscape/plots/m2_heatmap.png`
- 3D surface:  
  `interpretability_results/m2_fixed/m2_loss_landscape/plots/m2_3d_surface.png`

## M3: Effective Dimensionality (SVD)

### Method summary

M3 evaluates the intrinsic dimensionality of the latent space using SVD and
participation ratio metrics. The comparison is between:

- INP with knowledge
- NP baseline (separate checkpoint), or ablation if NP checkpoint not provided

Stability fixes applied:

- Knowledge dropout disabled during evaluation.
- Added variance-based participation ratio (PR on σ²) alongside PR on σ.
- Added real PCA/t‑SNE projections of latent codes.

### Key results (NP baseline run)

From `interpretability_results/m3_effective_dimensionality/results.json`:

- ED (PR on σ):
  - INP: **3.95**
  - NP:  **3.97**
  - Ratio: **0.996**
- ED (PR on σ²):
  - INP: **3.45**
  - NP:  **2.80**
- Components to 95% variance:
  - INP: **4**
  - NP:  **3**

### Interpretation

The INP and NP baseline are nearly identical in effective dimensionality.
The PCA/t‑SNE projections and singular value spectra show strong overlap.
This is a valid outcome: the latent space is already low‑dimensional for
sinusoids, and knowledge does not further compress it in this run.

### Artifacts

- Main metrics panel:  
  `interpretability_results/m3_effective_dimensionality/plots/m3_dimensionality.pdf`
- Singular value heatmap:  
  `interpretability_results/m3_effective_dimensionality/plots/m3_sv_heatmap.pdf`
- PCA/t‑SNE projection:  
  `interpretability_results/m3_effective_dimensionality/plots/m3_projection.pdf`
- Manifold schematic:  
  `interpretability_results/m3_effective_dimensionality/plots/m3_manifold.pdf`

## M4: Loss Balance (NLL vs beta*KL)

### Method summary

M4 measures how well the magnitudes of NLL and beta*KL are balanced. For stability,
the experiment uses a loss‑balance proxy instead of true gradient cosine similarity:

```
score = min(|NLL|, beta*|KL|) / max(|NLL|, beta*|KL|, eps)
```

Stability fixes applied:

- Knowledge dropout disabled during evaluation.
- Balance uses absolute values and beta scaling.
- Raw NLL/KL diagnostics logged per batch.

### Key results (inp_abc2_0 run)

From `interpretability_results/m4_gradient_alignment/results.json`:

- Loss balance (mean):
  - With knowledge: **0.408**
  - Random knowledge: **0.092**
  - Improvement: **+0.316**
- Mean |NLL|:
  - With knowledge: **16.98**
  - Random knowledge: **122.62**
- Mean beta*|KL|:
  - With knowledge: **6.91**
  - Random knowledge: **10.23**

### Interpretation

With knowledge, the loss terms are reasonably balanced (moderate score), and the
balance is significantly better than random knowledge. This indicates knowledge
is helping align the two objectives, at least in terms of loss magnitudes.

### Artifacts

- Main panel:  
  `interpretability_results/m4_gradient_alignment/plots/m4_gradient_alignment.pdf`
- Distribution:  
  `interpretability_results/m4_gradient_alignment/plots/m4_alignment_distribution.pdf`

## M5: Causal Activation Patching

### Method summary

M5 performs an interchange intervention: it patches donor knowledge into a
different task’s context and measures how predictions shift toward the donor.

Key implementation fixes applied:

- Donor ground truth now uses the real `y_target_B`, aligned to `x_target_A`.
- Knowledge dropout disabled during evaluation.
- Batch size capped for memory stability.
- Plot scaling fixed (MSE improvement is no longer ×10).

### Key results (inp_abc2_0 run, 4 pairs)

From `interpretability_results/m5_activation_patching/results.json`:

- Transfer ratio: **0.465 ± 0.120**
- Alignment: **0.497 ± 0.052**
- Direct effect: **5.586 ± 1.016**
- MSE improvement: **0.419 ± 0.107**
- Causal efficacy score: **0.585**

### Interpretation

Results indicate **moderate causal efficacy**: patching shifts predictions toward
the donor with partial alignment and consistent MSE improvement on donor ground
truth. This supports a causal role for knowledge while showing that the shift
is not yet fully ideal.

### Artifacts

- Main panel:  
  `interpretability_results/m5_activation_patching/plots/m5_activation_patching.pdf`
- Metrics heatmap:  
  `interpretability_results/m5_activation_patching/plots/m5_metrics_heatmap.pdf`

## M6: Knowledge Saliency (Integrated Gradients)

### Method summary

M6 uses Integrated Gradients (IG) to measure which parts of the knowledge input
drive predictions. For sinusoids, it reports feature importance for (a, b, c).

Key implementation fixes applied:

- Knowledge dropout disabled during IG to ensure determinism.
- Value-only attributions reported (excluding indicator columns).
- Token-level IG supported for text knowledge via input-embedding attribution.

### Key results (inp_abc2_0 run)

From `interpretability_results/m6_knowledge_saliency/results.json` and plots:

- Value-only feature importance:
  - a (amplitude): **15.9%**
  - b (frequency): **59.3%**
  - c (phase): **24.8%**
- IG convergence mean delta: **0.0011** (well below 0.1 threshold)

### Interpretation

The results align with task semantics: frequency dominates, phase is moderate,
and amplitude is lower. The low convergence delta indicates the IG attribution
satisfies completeness and is numerically stable.

### Artifacts

- Feature importance (value-only):  
  `interpretability_results/m6_knowledge_saliency/plots/m6_feature_importance.pdf`
- Convergence check:  
  `interpretability_results/m6_knowledge_saliency/plots/m6_convergence.pdf`

## M7: Linear Probing of Latent Representations

### Method summary

M7 trains linear and MLP probes to predict ground‑truth sinusoid parameters
from latent codes. The comparison tests linear decodability and the benefit
of knowledge conditioning.

Key implementation fixes applied:

- Aligned latents with/without knowledge in the same pass (no sample mismatch).
- Ground‑truth parameters pulled from the dataset when available (not masked).
- Knowledge dropout disabled during probing for determinism.

### Key results (inp_abc2_0 run)

From `interpretability_results/m7_linear_probing/results.json`:

- Overall R² (Linear):
  - With knowledge: **0.785**
  - Without knowledge: **0.582**
  - Benefit: **+0.203**
- Overall R² (MLP):
  - With knowledge: **0.775**
  - Without knowledge: **0.595**
- Per‑parameter R² (Linear):
  - a (amplitude): **0.766** (with K) vs **0.618** (no K)
  - b (frequency): **0.705** (with K) vs **0.325** (no K)
  - c (phase): **0.883** (with K) vs **0.802** (no K)

### Interpretation

Latents are **moderately disentangled** and largely linear (MLP adds little).
Knowledge significantly improves linear decodability, especially for frequency.

### Artifacts

- Main panel:  
  `interpretability_results/m7_linear_probing/plots/m7_linear_probing.pdf`

## M8: Uncertainty Decomposition (Epistemic vs Aleatoric)

### Method summary

M8 decomposes predictive uncertainty into epistemic and aleatoric components
across context sizes, comparing with‑knowledge vs without‑knowledge runs. It
quantifies the “bit‑value” of knowledge as the zero‑shot epistemic reduction.

Key implementation fixes applied:

- Context points are aligned across with/without knowledge comparisons.
- Knowledge dropout disabled during evaluation.
- Text knowledge handled (no silent drop).

### Key results (inp_abc2_0 run)

From `interpretability_results/m8_uncertainty_decomposition/results.json`:

- Zero‑shot epistemic (N=0):
  - With knowledge: **23.250 nats**
  - Without knowledge: **29.832 nats**
  - Reduction: **6.582 nats** (**9.496 bits**)
- Epistemic decreases with context size in both cases, with knowledge consistently lower.
- Aleatoric remains similar (sanity check passed).

### Interpretation

Knowledge provides a substantial zero‑shot reduction in epistemic uncertainty and
maintains a consistent advantage as context grows, indicating meaningful prior
information without inflating aleatoric noise.

### Artifacts

- Main panel:  
  `interpretability_results/m8_uncertainty_decomposition/plots/m8_uncertainty.pdf`

## M9: Spectral Analysis (HTSR)

### Method summary

M9 analyzes the eigenvalue spectra of weight matrices to estimate the power law
exponent (alpha) under Heavy-Tailed Self-Regularization (HTSR). It aggregates
alpha statistics per layer and per module, and visualizes alpha distributions,
stable rank, and spectral norms.

### Key results (m9_run)

From `interpretability_results/m9_run/m9_spectral_analysis/results.json` and plots:

- Mean alpha: **1.843 ± 0.281**
- Range: **1.258-2.607**
- Goldilocks [2,4]: **2 layers (13.3%)**
- Overfit (alpha < 2): **13 layers**
- Underfit (alpha > 6): **0 layers** (per run output)
- Per-module mean alpha:
  - xy_encoder: **2.013 ± 0.426**
  - x_encoder: **2.055 ± 0.000**
  - latent_encoder: **1.764 ± 0.234**
  - decoder: **1.812 ± 0.086**

### Interpretation

The mean alpha is below 2, indicating a heavy-tailed regime and an overfit
tendency by HTSR criteria. The x/xy encoders sit closer to the Goldilocks range,
while latent and decoder layers are more heavy-tailed, which suggests regularization
pressure may be most needed in the knowledge-conditioned path.

### Artifacts

- Main panel:  
  `interpretability_results/m9_run/m9_spectral_analysis/plots/m9_spectral.pdf`
- Layer heatmap:  
  `interpretability_results/m9_run/m9_spectral_analysis/plots/m9_layer_heatmap.pdf`

## M10: CKA Similarity (Representation Alignment)

### Method summary

M10 computes linear-kernel CKA between representations produced with vs without
knowledge on the *same batches*, to localize where knowledge changes the
representation. Key representations include context summary `R`, latent `z_mean`
and `z_std`, and `pred_mean`, along with per-layer hook activations.

Key implementation fixes applied:

- Paired batch collection for with/without knowledge (no sample mismatch).
- Knowledge dropout disabled during evaluation.
- Deterministic decoding with `z_mean` instead of random sampling.

### Key results (m10_run)

From `interpretability_results/m10_run/m10_cka_similarity/results.json` and plots:

- R: **1.000**
- z_mean: **0.820**
- z_std: **0.274**
- pred_mean: **0.916**
- Mean CKA across layers: **0.888 ± 0.207**
- Min CKA layer: **z_std (0.274)**

### Interpretation

Early processing (`R`) is identical with and without knowledge, while the latent
uncertainty (`z_std`) changes the most. The latent mean and predictions remain
highly similar, indicating knowledge acts as a moderate modulation rather than
wholesale representational rewrite in this run.

### Artifacts

- Main panel:  
  `interpretability_results/m10_run/m10_cka_similarity/plots/m10_cka_similarity.pdf`
- Heatmap:  
  `interpretability_results/m10_run/m10_cka_similarity/plots/m10_cka_heatmap.pdf`

## Notes and limitations

- Basin width depends on the chosen epsilon (absolute + relative). If the loss
  scale changes, re-tune `flatness_epsilon_ratio`.
- Barrier height is sensitive to alpha range. Use smaller ranges for strictly
  local geometry.
- M2 comparisons are local; they do not guarantee global generalization.
- M9 alpha fits are sensitive to tail selection; treat them as qualitative
  indicators unless a stricter tail-fit procedure is used.

## Recommended next steps

- Re-run M2 with a tighter alpha range (e.g., `[-0.25, 0.25]`) to isolate local curvature.
- If desired, run M2 on multiple seeds/models and average metrics to reduce variance.

