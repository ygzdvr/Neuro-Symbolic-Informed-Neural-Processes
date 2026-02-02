# Mechanistic Interpretability Suite for Informed Neural Processes

This document provides comprehensive documentation for the 10 mechanistic interpretability experiments designed to understand how Informed Neural Processes (INPs) integrate and utilize external knowledge during meta-learning.

---

## Table of Contents

1. [M1: Information Bottleneck (MINE)](#m1-information-bottleneck-mine)
2. [M2: Loss Landscape Visualization](#m2-loss-landscape-visualization)
3. [M3: Effective Dimensionality (SVD)](#m3-effective-dimensionality-svd)
4. [M4: Gradient Alignment Score](#m4-gradient-alignment-score)
5. [M5: Causal Activation Patching](#m5-causal-activation-patching)
6. [M6: Knowledge Saliency (Integrated Gradients)](#m6-knowledge-saliency-integrated-gradients)
7. [M7: Linear Probing](#m7-linear-probing)
8. [M8: Uncertainty Decomposition](#m8-uncertainty-decomposition)
9. [M9: Spectral Analysis (HTSR)](#m9-spectral-analysis-htsr)
10. [M10: CKA Similarity](#m10-cka-similarity)

---

## Overview: Why Mechanistic Interpretability for INPs?

Informed Neural Processes (INPs) extend Neural Processes by incorporating external knowledge (text descriptions, numeric parameters) during meta-learning. A fundamental question arises: **How does the model actually use this knowledge?**

These 10 experiments provide complementary perspectives:

| Category | Experiments | Question Answered |
|----------|-------------|-------------------|
| Information Flow | M1, M8 | How much information flows from knowledge vs data? |
| Optimization | M2, M4 | Does knowledge improve the optimization landscape? |
| Representation | M3, M7, M10 | How does knowledge shape the latent space? |
| Causality | M5, M6 | Does knowledge causally affect predictions? |
| Generalization | M9 | Is the model well-regularized? |

---

## M1: Information Bottleneck (MINE)

### Conceptual Overview

The Information Bottleneck principle (Tishby et al., 2000) states that an optimal representation Z should:
1. **Compress**: Minimize I(Z; Input) - discard irrelevant details
2. **Predict**: Maximize I(Z; Output) - retain task-relevant information

For INPs, we have two input sources:
- **D**: Context data (x_context, y_context)
- **K**: External knowledge

This experiment measures how much information from each source is retained in the latent z.

### What The Code Does

1. **Collects samples**: For each batch, extracts (z, D, K) triplets where:
   - z = latent mean from q(z|D,K)
   - D = flattened context representation R
   - K = knowledge embedding

2. **Trains MINE networks**: Two separate neural networks T_θ are trained:
   - T_D: Estimates I(Z; D)
   - T_K: Estimates I(Z; K)

3. **MINE Training**: For each estimator, optimizes:
   ```
   max_θ E[T(z,x)] - log E[exp(T(z,x'))]
   ```
   where x' is drawn from the marginal (shuffled pairing)

### Mathematical Foundation

**Mutual Information**:
$$I(X; Y) = H(X) - H(X|Y) = \mathbb{E}_{p(x,y)} \left[ \log \frac{p(x,y)}{p(x)p(y)} \right]$$

**MINE Lower Bound** (Belghazi et al., 2018):
$$I(X; Y) \geq \sup_{\theta \in \Theta} \mathbb{E}_{p(x,y)}[T_\theta(x,y)] - \log \mathbb{E}_{p(x)p(y)}[e^{T_\theta(x,y)}]$$

The neural network T_θ acts as a "critic" that learns to distinguish joint samples (x,y) from product-of-marginals samples (x,y').

**Numerical Stability**: We use:
- Exponential moving average for the marginal term
- Log-sum-exp trick for numerical stability
- Gradient clipping and weight decay
- Output clamping to [-50, 50]
- Non-negativity enforcement (MI ≥ 0)

### Why It's Important

1. **Quantifies Knowledge Utilization**: If I(Z; K) ≈ 0, the model ignores knowledge
2. **Reveals Information Source Preference**: If I(Z; K) >> I(Z; D), knowledge dominates
3. **Validates Knowledge Integration**: Confirms the knowledge pathway is functional

### Code Returns

```python
{
    "mi_data": [list of I(Z;D) estimates per training step],
    "mi_knowledge": [list of I(Z;K) estimates per training step],
    "mi_data_final": float,        # Final I(Z; D) estimate in nats
    "mi_knowledge_final": float,   # Final I(Z; K) estimate in nats
    "knowledge_reliance": float,   # I(Z;K) / (I(Z;D) + I(Z;K))
    "data_contribution": float,    # I(Z;D) / (I(Z;D) + I(Z;K))
    "interpretation": str,         # Human-readable summary
}
```

### Expected Results

| Scenario | I(Z;D) | I(Z;K) | Knowledge Reliance | Interpretation |
|----------|--------|--------|-------------------|----------------|
| Knowledge-heavy | Low | High | >0.6 | Model relies primarily on knowledge |
| Data-heavy | High | Low | <0.3 | Model relies primarily on context data |
| Balanced | Medium | Medium | 0.3-0.6 | Healthy integration of both sources |
| Degenerate | ~0 | ~0 | N/A | Posterior collapse or training issue |

**For a well-functioning INP**: Expect I(Z; K) > 0 and knowledge_reliance between 0.3-0.7.

### Visualizations Generated

1. **MI Training Curves**: Shows I(Z;D) and I(Z;K) over MINE training steps
2. **MI Comparison Bar Chart**: Final values with error bars
3. **Heatmap**: Correlation between data/knowledge MI at different training steps
4. **Pie Chart**: Relative contribution of data vs knowledge

---

## M2: Loss Landscape Visualization

### Conceptual Overview

The geometry of the loss landscape around a trained model reveals important properties:
- **Sharp minima**: High curvature, potentially poor generalization
- **Flat minima**: Low curvature, typically better generalization (Hochreiter & Schmidhuber, 1997)

Knowledge should act as a regularizer, flattening the landscape by constraining the hypothesis space.

### What The Code Does

1. **Caches evaluation batches**: Fixes a set of batches for consistent comparison

2. **Generates random directions**: Creates filter-normalized random perturbation vectors in parameter space:
   ```python
   direction = [torch.randn_like(p) for p in model.parameters()]
   # Normalize each filter/layer to unit norm
   ```

3. **Computes 1D loss slices**: For each direction δ (shared across with/without knowledge):
   ```python
   for alpha in linspace(-1, 1, num_points):
       theta_perturbed = theta_star + alpha * delta
       loss[alpha] = compute_loss(theta_perturbed)
   ```

4. **Measures landscape properties**:
   - **Sharpness/Barrier**: max(loss) - loss at origin (α=0)
   - **Basin width**: Width of region around α=0 where loss ≤ loss(0) + ε
   - **Curvature**: Finite difference second derivative at α=0

5. **Computes true 2D surface**: Evaluates a grid along two random directions
   and visualizes as heatmap + 3D surface.

6. **Compares with vs without knowledge**: Runs the same analysis passing knowledge=None

### Mathematical Foundation

**Filter-Normalized Perturbation** (Li et al., 2018):
For each weight matrix W with shape (n_out, n_in), normalize the random direction D:
$$D_{normalized} = D \cdot \frac{\|W\|_F}{\|D\|_F}$$

This ensures perturbations are scaled appropriately regardless of layer size.

**Sharpness Metric**:
$$\text{Sharpness}(\theta^*, \rho) = \frac{\max_{\|\delta\| \leq \rho} L(\theta^* + \delta) - L(\theta^*)}{1 + L(\theta^*)}$$

**Basin Width**:
$$\text{Width} = \min \{ \alpha > 0 : L(\theta^* + \alpha\delta) > 2 \cdot L(\theta^*) \}$$

### Why It's Important

1. **Generalization Predictor**: Flat minima correlate with generalization (Keskar et al., 2017)
2. **Knowledge as Regularizer**: Knowledge should constrain the model, flattening landscape
3. **Optimization Health**: Sharp spikes indicate potential training issues

### Code Returns

```python
{
    "with_knowledge": {
        "landscape_1d": [(alphas, losses) for each direction],
        "sharpness": float,
        "basin_width": float,
        "loss_at_origin": float,
        "max_loss": float,
        "curvature": float,
    },
    "without_knowledge": {
        # Same structure
    },
    "comparison": {
        "sharpness_ratio": float,    # sharpness_with_k / sharpness_without_k
        "width_ratio": float,        # width_with_k / width_without_k
    },
    "interpretation": str,
}
```

### Expected Results

| Metric | Good INP | Poor INP |
|--------|----------|----------|
| Basin Width Ratio (INP/NP) | >1.5 | <1.0 |
| Sharpness Ratio | <0.7 | >1.0 |

**For a well-functioning INP**: Expect wider basin and lower sharpness compared to NP baseline.

### Visualizations Generated

1. **1D Loss Curves**: Loss vs perturbation magnitude for each direction
2. **2D Loss Heatmap**: Loss surface projected onto 2 random directions
3. **3D Surface Plot**: Interactive-style 3D visualization
4. **Comparison Bar Chart**: Basin width and sharpness with/without knowledge

---

## M3: Effective Dimensionality (SVD)

### Conceptual Overview

The latent space z ∈ ℝ^d has d dimensions, but how many are actually "used"? 
- **High effective dimensionality**: Many dimensions carry information
- **Low effective dimensionality**: Information is concentrated in few dimensions

Knowledge should **constrain** the latent manifold, reducing effective dimensionality.

### What The Code Does

1. **Collects latent samples**: For N tasks, extracts z_mean from q(z|D,K)

2. **Constructs latent matrix**: Z ∈ ℝ^{N×d} where each row is a task's latent

3. **Computes SVD**: Z = UΣV^T

4. **Calculates metrics**:
   - Singular values σ₁ ≥ σ₂ ≥ ... ≥ σ_d
   - Effective dimensionality (PR on σ and on σ²)
   - Variance explained at different k

5. **Compares**: 
   - Default: same checkpoint with knowledge disabled (ablation)
   - Optional: separate NP checkpoint if provided

### Mathematical Foundation

**Singular Value Decomposition**:
$$Z = U \Sigma V^T$$
where Σ = diag(σ₁, σ₂, ..., σ_d) with σ₁ ≥ σ₂ ≥ ... ≥ σ_d ≥ 0

**Effective Dimensionality** (Participation Ratio on σ):
$$\text{ED} = \frac{(\sum_i \sigma_i)^2}{\sum_i \sigma_i^2} = \frac{(\text{trace}(\Sigma))^2}{\|\Sigma\|_F^2}$$

This equals 1 if all variance is in one dimension, and d if variance is uniform across all dimensions.

**Effective Dimensionality** (Participation Ratio on σ² / variance):
$$\text{ED}_{\text{var}} = \frac{(\sum_i \sigma_i^2)^2}{\sum_i \sigma_i^4}$$

**Variance Explained by top-k**:
$$\text{VE}_k = \frac{\sum_{i=1}^k \sigma_i^2}{\sum_{i=1}^d \sigma_i^2}$$

**Condition Number** (ratio of largest to smallest singular value):
$$\kappa = \frac{\sigma_1}{\sigma_d}$$

### Why It's Important

1. **Manifold Hypothesis**: Natural data lies on low-dimensional manifolds
2. **Knowledge as Constraint**: Knowledge should reduce the "degrees of freedom"
3. **Representation Efficiency**: Lower ED means more compressed, efficient representation
4. **Numerical Stability**: Very high condition numbers indicate ill-conditioned representations

### Code Returns

```python
{
    "with_knowledge": {
        "singular_values": list,
        "explained_variance_ratio": list,
        "cumulative_variance": list,
        "effective_dimensionality": float,          # PR on σ
        "effective_dimensionality_variance": float, # PR on σ²
        "effective_dimensionality_entropy": float,
        "components_to_threshold": int,
        "threshold": float,
        "total_variance": float,
        "latent_dim": int,
        "num_samples": int,
    },
    "without_knowledge": { ... },
    "comparison": {
        "ed_ratio": float,                    # ED_with_k / ED_without_k
        "ed_reduction": float,                # ED_without_k - ED_with_k
        "components_reduction": int,
    },
    "interpretation": str,
}
```

### Expected Results

| Metric | Good INP | Interpretation |
|--------|----------|---------------|
| ED Ratio (INP/Baseline) | <0.8 | Knowledge constrains manifold |
| ED Ratio (INP/Baseline) | 0.8-1.0 | Mild/no manifold constraint |
| Variance@90 (INP) | <10 | Highly compressed representation |

**For a well-functioning INP**: Expect ED_with_knowledge < ED_without_knowledge.

### Visualizations Generated

1. **Singular Value Spectrum**: Log-scale plot of σ values, with/without knowledge
2. **Cumulative Variance Explained**: Curve showing VE_k vs k
3. **Singular Value Heatmap**: Matrix visualization of normalized SVs
4. **PCA/t-SNE Projection**: 2D visualization of latent space (real projection)
5. **Manifold Schematic**: ED ratio visualization (schematic, not geometric)

---

## M4: Gradient Alignment Score

### Conceptual Overview

The ELBO objective balances two competing terms:
- **NLL (Reconstruction)**: -E[log p(y|z)] - fit the data
- **KL (Regularization)**: D_KL(q(z|x,k) || p(z)) - stay close to prior

When these gradients are aligned, optimization is smooth. When they conflict, training may be unstable.

### What The Code Does

1. **Computes losses separately**: For each batch, calculates NLL and KL individually

2. **Measures balance**: Computes a "loss balance score":
   ```python
   score = min(|NLL|, beta * |KL|) / max(|NLL|, beta * |KL|, eps)
   ```
   Range: [0, 1], where 1 = perfectly balanced

3. **Compares with shuffled knowledge**: Tests if knowledge improves balance

### Mathematical Foundation

**ELBO Decomposition**:
$$\mathcal{L}(\theta, \phi) = \underbrace{-\mathbb{E}_{q_\phi(z|x,k)}[\log p_\theta(y|z)]}_{\text{NLL (reconstruction)}} + \beta \cdot \underbrace{D_{KL}(q_\phi(z|x,k) \| p(z))}_{\text{KL (regularization)}}$$

**Loss Balance Score**:
$$\text{Score} = \frac{\min(|\text{NLL}|, \beta |\text{KL}|)}{\max(|\text{NLL}|, \beta |\text{KL}|, \epsilon)}$$

This measures how "balanced" the two loss terms are:
- Score → 1: NLL ≈ KL (balanced)
- Score → 0: One term dominates

**Gradient Alignment** (Original formulation - not used due to numerical instability):
$$\text{GAS} = \cos(\nabla_\theta \text{NLL}, \nabla_\theta \text{KL}) = \frac{\nabla \text{NLL} \cdot \nabla \text{KL}}{\|\nabla \text{NLL}\| \|\nabla \text{KL}\|}$$

Note: Direct gradient computation caused CUDA floating point exceptions, so we use the loss balance proxy instead.

### Why It's Important

1. **Optimization Health**: Imbalanced losses indicate training difficulties
2. **β Tuning Guidance**: If KL dominates, β is too high; if NLL dominates, β is too low
3. **Posterior Collapse Detection**: KL → 0 indicates the model ignores the latent

### Code Returns

```python
{
    "with_knowledge": {
        "all_alignment": [score per batch],
        "nll_values": [NLL per batch],
        "kl_values": [KL per batch],
        "kl_scaled_values": [beta * KL per batch],
        "kl_ratio_values": [beta * |KL| / |NLL| per batch],
    },
    "with_random_knowledge": {
        # Same structure with shuffled knowledge
    },
    "aggregated": {
        "mean_alignment_k": float,
        "std_alignment_k": float,
        "mean_alignment_rand": float,
    },
    "interpretation": str,
}
```

### Expected Results

| Score | Interpretation |
|-------|---------------|
| >0.7 | Well-balanced optimization |
| 0.4-0.7 | Moderate imbalance |
| <0.4 | Significant imbalance, potential issues |

**For a well-functioning INP**: Expect score > 0.5 with proper knowledge.

### Visualizations Generated

1. **Balance Comparison Bar Chart**: With knowledge vs random knowledge
2. **Time Series**: Score over batches
3. **Score Distribution Histogram**: Distribution of balance scores
4. **Loss Magnitudes Panel**: |NLL| vs beta*|KL| (proxy diagnostic)

---

## M5: Causal Activation Patching

### Conceptual Overview

Correlation is not causation. Just because knowledge affects z doesn't mean it **causally** influences predictions. This experiment uses **causal intervention** to test genuine causal efficacy.

Inspired by activation patching in mechanistic interpretability (Vig et al., 2020), we:
1. Take Task A with Knowledge A
2. "Patch in" Knowledge B from a different task
3. Measure how much predictions shift toward Task B behavior

### What The Code Does

1. **Creates task pairs**: Samples pairs (Task A, Task B) from dataloader

2. **Runs forward passes**:
   ```python
   pred_A = model(context_A, knowledge_A)           # Original
   pred_patched = model(context_A, knowledge_B)     # Patched
   pred_B = model(context_B, knowledge_B)           # Ideal target
   ```

3. **Computes causal metrics**:
   - **Direct Effect**: ||pred_patched - pred_A||
   - **Transfer Ratio**: direct_effect / ||pred_B - pred_A||
   - **Alignment**: cosine(pred_patched - pred_A, pred_B - pred_A)

4. **Evaluates on same x locations**: Ensures fair comparison by predicting at x_target_A

### Mathematical Foundation

**Causal Effect** (Pearl, 2009):
The causal effect of intervention do(K=k_B) on prediction Y given context D_A:
$$\text{CE} = \mathbb{E}[Y | D_A, \text{do}(K=k_B)] - \mathbb{E}[Y | D_A, \text{do}(K=k_A)]$$

**Direct Effect**:
$$\text{DE} = \|f(D_A, K_B) - f(D_A, K_A)\|_2$$

**Transfer Ratio** (fraction of ideal shift achieved):
$$\text{TR} = \frac{\|f(D_A, K_B) - f(D_A, K_A)\|}{\|f(D_B, K_B) - f(D_A, K_A)\|}$$

If TR = 1, patching in B's knowledge makes A behave exactly like B.
If TR = 0, patching has no effect.

**Alignment** (direction correctness):
$$\text{Align} = \cos\theta = \frac{(\Delta_{\text{actual}}) \cdot (\Delta_{\text{ideal}})}{\|\Delta_{\text{actual}}\| \|\Delta_{\text{ideal}}\|}$$

where Δ_actual = pred_patched - pred_A and Δ_ideal = pred_B - pred_A.

### Why It's Important

1. **Establishes Causality**: Proves knowledge actually causes prediction changes
2. **Distinguishes Correlation from Causation**: High I(Z;K) doesn't imply K causes Y
3. **Tests Knowledge Pathway Functionality**: Low TR suggests knowledge is ignored
4. **Validates Model Design**: Confirms the knowledge integration mechanism works

### Code Returns

```python
{
    "individual_results": [
        {
            "direct_effect": float,           # How much did prediction change?
            "donor_distance": float,          # How far is donor from original?
            "transfer_ratio": float,          # What fraction of shift achieved?
            "alignment": float,               # Did shift go in right direction?
            "mse_original_to_donor_gt": float,
            "mse_patched_to_donor_gt": float,
            "mse_improvement": float,         # MSE reduction from patching
            "mse_degradation_on_original": float,
        },
        ...
    ],
    "aggregated": {
        "transfer_ratio": {"mean": float, "std": float},
        "alignment": {"mean": float, "std": float},
        "mse_improvement": {"mean": float, "std": float},
    },
    "interpretation": str,
}
```

### Expected Results

| Transfer Ratio | Alignment | Interpretation |
|----------------|-----------|---------------|
| >0.5 | >0.5 | Strong causal efficacy |
| 0.2-0.5 | >0 | Moderate causal effect |
| <0.2 | any | Weak/no causal effect |
| any | <0 | Knowledge hurts (wrong direction) |

**For a well-functioning INP**: Expect TR > 0.5 and Alignment > 0.5.

### Visualizations Generated

1. **Transfer Ratio Histogram**: Distribution across pairs
2. **Alignment Scatter Plot**: TR vs Alignment
3. **MSE Comparison**: Before/after patching
4. **Effect Size Bar Chart**: Mean direct effect with confidence intervals

---

## M6: Knowledge Saliency (Integrated Gradients)

### Conceptual Overview

Which parts of the knowledge input matter most for predictions? 

For numeric knowledge (a, b, c parameters in sinusoids), we want to know:
- Does 'a' (amplitude) matter?
- Does 'b' (frequency) matter?
- Does 'c' (phase) matter?

Integrated Gradients provides **axiomatic attribution** - a principled way to assign importance scores to each input feature.

### What The Code Does

1. **Disables knowledge dropout**: Ensures IG is deterministic during attribution

2. **Sets baseline**: Uses zero knowledge as reference (k' = 0)

3. **Interpolates**: Creates path from baseline to actual knowledge:
   ```python
   k_alpha = k' + alpha * (k - k')  # for alpha in [0, 1]
   ```

4. **Accumulates gradients**: Computes gradients at each interpolation point:
   ```python
   for alpha in linspace(0, 1, n_steps):
       k_interp = baseline + alpha * (k - baseline)
       grad = ∂output/∂k_interp
       integrated_grads += grad
   ```

5. **Scales by input-baseline difference**:
   ```python
   attribution = (k - baseline) * (integrated_grads / n_steps)
   ```

6. **Text knowledge**: Computes IG on RoBERTa input embeddings, aggregates to token-level

7. **Sinusoids**: Reports both full (indicator+value) and value-only attributions per (a, b, c)

### Mathematical Foundation

**Integrated Gradients** (Sundararajan et al., 2017):

$$\text{IG}_i(x) = (x_i - x'_i) \times \int_{\alpha=0}^{1} \frac{\partial F(x' + \alpha(x-x'))}{\partial x_i} d\alpha$$

Approximated by Riemann sum:
$$\text{IG}_i(x) \approx (x_i - x'_i) \times \frac{1}{m} \sum_{k=1}^{m} \frac{\partial F(x' + \frac{k}{m}(x-x'))}{\partial x_i}$$

**Key Axioms Satisfied**:
1. **Completeness**: Sum of attributions equals output difference:
   $$\sum_i \text{IG}_i(x) = F(x) - F(x')$$
2. **Sensitivity**: If a feature matters, it gets nonzero attribution
3. **Implementation Invariance**: Same function → same attributions

**Convergence Delta**: Measures how well completeness is satisfied:
$$\delta = |F(x) - F(x') - \sum_i \text{IG}_i(x)|$$

Should be close to 0.

### Why It's Important

1. **Feature Importance**: Identifies which knowledge features drive predictions
2. **Validation**: Verifies model uses semantically meaningful features
3. **Debugging**: Reveals if model focuses on wrong features
4. **Interpretability**: Human-understandable importance scores

### Code Returns

```python
{
    "config": {
        "n_steps": int,           # Integration steps
        "is_text_knowledge": bool,
        "knowledge_dropout_original": float,
        "knowledge_dropout_set_to": float,
    },
    "raw_attributions": [np.array per batch],  # Attributions for raw knowledge
    "convergence_deltas": [float per batch],   # Completeness check
    "aggregated": {
        "mean_importance": np.array,         # Mean importance per feature
        "std_importance": np.array,
        "relative_importance": np.array,     # Percentage importance
    },
    "feature_importance": {                   # Named feature importances (if known)
        "a (amplitude)": float,
        "b (frequency)": float,
        "c (phase)": float,
    },
    "aggregated_value_only": {    # Sinusoids: value-only attribution
        "mean_importance": np.array,
        "std_importance": np.array,
        "relative_importance": np.array,
    },
    "feature_importance_value_only": {        # Named value-only importances
        "a (amplitude)": float,
        "b (frequency)": float,
        "c (phase)": float,
    },
    "indicator_vs_value": {       # Sinusoids: indicator vs value contribution
        "indicator": float,
        "value": float,
        "indicator_fraction": float,
        "value_fraction": float,
    },
    "text_attributions": [np.array per batch],   # Token-level IG (text knowledge)
    "text_tokens": [list per batch],
    "text_samples": [list per batch],
    "token_importance_mean": [(token, score), ...],
    "interpretation": str,
}
```

### Expected Results

For sinusoid data with knowledge (a, b, c):

| Feature | Typical Importance | Reason |
|---------|-------------------|--------|
| b (frequency) | High (40-60%) | Determines oscillation pattern |
| a (amplitude) | Medium (20-35%) | Scales output magnitude |
| c (phase) | Low-Medium (15-30%) | Shifts timing |

**For a well-functioning INP**: Feature importance should align with task semantics.

### Visualizations Generated

1. **Feature Importance Bar Chart**: Normalized importance per feature
2. **Attribution Heatmap**: Per-sample attributions across features
3. **Convergence Plot**: Delta values over samples (should be low)
4. **Waterfall Chart**: Cumulative attribution breakdown

---

## M7: Linear Probing

### Conceptual Overview

A **linear probe** is a simple linear regressor trained on frozen representations to predict task properties. High probe accuracy means the representation **linearly encodes** the target information.

For INPs, we test whether the latent z encodes the true task parameters (a, b, c).

### What The Code Does

1. **Disables knowledge dropout** to keep latents deterministic during probing

2. **Collects aligned latent-parameter pairs**:
   ```python
   for each task:
       z = q(z|D,K).mean
       params = ground_truth (a, b, c)
   ```
   - Latents with/without knowledge are collected in the same pass
   - Ground-truth parameters are pulled from the dataset (full values) when available

3. **Trains probes**: Linear (Ridge) and nonlinear (MLP) regressors:
   ```python
   linear_probe = Ridge(alpha=1.0)
   linear_probe.fit(z_train, params_train)
   r2_linear = linear_probe.score(z_test, params_test)
   
   mlp_probe = MLPRegressor(hidden_layers=(64, 64))
   mlp_probe.fit(z_train, params_train)
   r2_mlp = mlp_probe.score(z_test, params_test)
   ```

4. **Compares**:
   - With vs without knowledge
   - Linear vs nonlinear (measures linearity of encoding)

### Mathematical Foundation

**Linear Probe**:
$$\hat{y} = Wz + b$$

**Training Objective** (Ridge Regression):
$$\min_{W,b} \sum_i \|y_i - (Wz_i + b)\|^2 + \lambda \|W\|_F^2$$

**R² Score** (Coefficient of Determination):
$$R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}$$

- R² = 1: Perfect prediction
- R² = 0: No better than predicting mean
- R² < 0: Worse than mean prediction

**Linear Gap** (MLP R² - Linear R²):
- Small gap: Representation is already linear
- Large gap: Nonlinear structure, linear probe insufficient

### Why It's Important

1. **Disentanglement**: Can we decode each parameter independently?
2. **Representation Quality**: Better representations → higher probe accuracy
3. **Knowledge Benefit**: Does knowledge improve decodability?
4. **Linearity**: Is the encoding linear (simple) or nonlinear (complex)?

### Code Returns

```python
{
    "config": {
        "probe_epochs": int,
        "test_fraction": float,
        "use_sklearn": bool,
        "params_source": "dataset" | "knowledge",
        "knowledge_dropout_original": float,
        "knowledge_dropout_set_to": float,
        "ordered_loader": bool,
        "filtered_masked_params": bool,
    },
    "with_knowledge": {
        "linear": {
            "r2_overall": float,
            "r2_param_0": float,  # R² for parameter a
            "r2_param_1": float,  # R² for parameter b
            "r2_param_2": float,  # R² for parameter c
            "mse": float,
        },
        "mlp": {
            # Same structure
        },
    },
    "without_knowledge": {
        # Same structure
    },
    "comparison": {
        "r2_linear_with_k": float,
        "r2_linear_without_k": float,
        "knowledge_benefit_linear": float,  # Improvement from knowledge
        "linear_gap_with_k": float,         # MLP - Linear (measures nonlinearity)
    },
    "per_parameter": {
        "a (amplitude)": {"r2_linear_with_k": float, ...},
        "b (frequency)": {...},
        "c (phase)": {...},
    },
    "interpretation": str,
}
```

### Expected Results

| Metric | Good INP | Interpretation |
|--------|----------|---------------|
| R² Linear (with K) | >0.7 | Strong disentanglement |
| R² Linear (with K) | 0.4-0.7 | Moderate disentanglement |
| R² Linear (with K) | <0.4 | Weak disentanglement |
| Knowledge Benefit | >0.2 | Knowledge significantly helps |
| Linear Gap | <0.1 | Encoding is already linear |

**For a well-functioning INP**: Expect R² > 0.5 with knowledge benefit > 0.

### Visualizations Generated

1. **R² Comparison Bar Chart**: Linear/MLP with/without knowledge
2. **Per-Parameter R²**: Which parameters are best encoded?
3. **Scatter Plots**: Predicted vs actual for each parameter
4. **Improvement Waterfall**: Knowledge benefit breakdown

---

## M8: Uncertainty Decomposition

### Conceptual Overview

Predictive uncertainty has two sources:
- **Epistemic (Model) Uncertainty**: "I don't know" - reducible with more data
- **Aleatoric (Data) Uncertainty**: "This is inherently noisy" - irreducible

Knowledge should reduce **epistemic** uncertainty by providing task information, but should NOT affect **aleatoric** uncertainty (inherent noise level).

### What The Code Does

1. **Disables knowledge dropout**: Keeps uncertainty estimates deterministic

2. **Aligns context points**: Uses the same context subset for with/without knowledge

3. **Handles text knowledge**: Passes text inputs through the knowledge encoder when present

4. **Samples multiple z values**: For each task, draws S samples from q(z|D,K)

5. **Computes predictions**: For each z_s, gets p(y|x,z_s)

6. **Decomposes variance**:
   ```python
   # Epistemic: variance of means across z samples
   epistemic = Var[E[y|z]]
   
   # Aleatoric: mean of variances
   aleatoric = E[Var[y|z]]
   ```

7. **Compares**: With vs without knowledge, at different context sizes

### Mathematical Foundation

**Law of Total Variance**:
$$\text{Var}[Y] = \underbrace{\mathbb{E}[\text{Var}[Y|Z]]}_{\text{Aleatoric}} + \underbrace{\text{Var}[\mathbb{E}[Y|Z]]}_{\text{Epistemic}}$$

**Monte Carlo Estimation**:
$$\hat{\sigma}^2_{\text{epistemic}} = \frac{1}{S}\sum_{s=1}^S (\mu_s - \bar{\mu})^2$$

$$\hat{\sigma}^2_{\text{aleatoric}} = \frac{1}{S}\sum_{s=1}^S \sigma_s^2$$

where μ_s = E[Y|z_s] and σ²_s = Var[Y|z_s] from the decoder output.

**Zero-Shot Analysis**: Measures uncertainty with 0 context points to isolate knowledge effect.

### Why It's Important

1. **Information Gain**: Knowledge should reduce epistemic uncertainty
2. **Sanity Check**: Aleatoric should stay constant (it's inherent noise)
3. **Uncertainty Calibration**: Well-calibrated models have meaningful uncertainty
4. **Knowledge Informativeness**: Quantifies how informative knowledge is

### Code Returns

```python
{
    "config": {
        "context_sizes": list,
        "num_z_samples": int,
        "num_tasks": int,
        "mc_samples": int,
        "aligned_context": bool,
        "knowledge_dropout_original": float,
        "knowledge_dropout_set_to": float,
    },
    "with_knowledge": {
        "epistemic_mean": float,
        "epistemic_std": float,
        "aleatoric_mean": float,
        "aleatoric_std": float,
        "total_uncertainty": float,
        "uncertainty_ratio": float,  # epistemic / total
        "per_context_size": {
            0: {"epistemic": float, "aleatoric": float},
            5: {...},
            10: {...},
        },
    },
    "without_knowledge": {
        # Same structure
    },
    "comparison": {
        "epistemic_reduction": float,      # How much knowledge reduces epistemic
        "epistemic_reduction_pct": float,  # Percentage reduction
        "aleatoric_difference": float,     # Should be ~0
    },
    "interpretation": str,
}
```

### Expected Results

| Metric | Good INP | Interpretation |
|--------|----------|---------------|
| Epistemic Reduction | >1 nat | Knowledge provides significant information |
| Aleatoric Difference | <0.1 | Sanity check passes |
| Zero-shot Epistemic (with K) | Low | Knowledge alone provides task information |

**For a well-functioning INP**: Expect significant epistemic reduction with minimal aleatoric change.

### Visualizations Generated

1. **Uncertainty Decomposition Bar Chart**: Epistemic vs aleatoric, with/without K
2. **Uncertainty vs Context Size**: How uncertainty decreases with more context
3. **Heatmap**: Task-level uncertainty decomposition
4. **Reduction Waterfall**: Epistemic reduction breakdown

---

## M9: Spectral Analysis (HTSR)

### Conceptual Overview

The Heavy-Tailed Self-Regularization (HTSR) theory (Martin & Mahoney, 2019) suggests that the eigenvalue spectrum of weight matrices reveals model quality:
- **Well-trained models**: Power-law spectrum with exponent α ≈ 2-4
- **Overtrained models**: Heavy tails (α < 2)
- **Undertrained models**: Light tails (α > 4)

This analysis requires no test data - it examines weight matrices directly.

### What The Code Does

1. **Extracts weight matrices**: From all Linear layers

2. **Computes eigenspectra**: For each weight W:
   ```python
   eigenvalues = eig(W.T @ W)
   singular_values = sqrt(eigenvalues)
   ```

3. **Fits power-law**: Estimates α from tail of spectrum:
   ```python
   # Fit p(λ) ∝ λ^(-α) to the tail
   α = fit_power_law(eigenvalues)
   ```

4. **Computes metrics**:
   - **Stable Rank**: ||W||²_F / ||W||²_2 = Σσ²ᵢ / σ²_max
   - **Condition Number**: σ_max / σ_min

### Mathematical Foundation

**Power-Law Distribution**:
$$p(\lambda) \propto \lambda^{-\alpha}$$

**Power-Law Exponent α Estimation**:
Using maximum likelihood on log-log scale:
$$\alpha = 1 + n \left[ \sum_{i=1}^{n} \ln \frac{\lambda_i}{\lambda_{\min}} \right]^{-1}$$

**Stable Rank**:
$$\text{SR}(W) = \frac{\|W\|_F^2}{\|W\|_2^2} = \frac{\sum_i \sigma_i^2}{\sigma_{\max}^2}$$

Measures "effective rank" - how spread out the singular values are.

**HTSR Theory Predictions** (Martin & Mahoney, 2019):
- α ∈ [2, 4]: "Goldilocks zone" - good generalization
- α < 2: Overfitting tendency
- α > 4: Underfitting tendency

### Why It's Important

1. **No Test Data Needed**: Predicts generalization from weights alone
2. **Layer-wise Analysis**: Identifies problematic layers
3. **Training Quality**: Detects over/underfitting
4. **Model Health Check**: Independent of specific task

### Code Returns

```python
{
    "layer_metrics": [
        {
            "name": str,              # Layer name
            "shape": tuple,           # Weight shape
            "alpha": float,           # Power-law exponent
            "stable_rank": float,     # Effective rank
            "condition_number": float,
            "spectral_norm": float,   # Largest singular value
            "frobenius_norm": float,  # ||W||_F
        },
        ...
    ],
    "summary": {
        "mean_alpha": float,
        "std_alpha": float,
        "mean_stable_rank": float,
        "num_layers_analyzed": int,
        "num_heavy_tailed": int,     # Layers with α < 2
        "num_light_tailed": int,     # Layers with α > 4
    },
    "interpretation": str,
}
```

### Expected Results

| Mean α | Interpretation |
|--------|---------------|
| 2.0 - 2.5 | Optimal range, good generalization |
| 2.5 - 4.0 | Good, possibly under-regularized |
| < 2.0 | Heavy-tailed, potential overfit |
| > 4.0 | Under-trained or over-regularized |

**For a well-functioning INP**: Expect mean α in [2, 3] range.

### Visualizations Generated

1. **Eigenvalue Spectra**: Log-log plot for each layer
2. **α Distribution Histogram**: Distribution of α across layers
3. **Layer Heatmap**: α and stable rank per layer
4. **Stable Rank vs α Scatter**: Correlation between metrics

---

## M10: CKA Similarity

### Conceptual Overview

How similar are representations with vs without knowledge at each layer?

**Centered Kernel Alignment (CKA)** measures representational similarity in a way that is:
- Invariant to orthogonal transformations
- Invariant to isotropic scaling
- Sensitive to meaningful representational differences

### What The Code Does

1. **Collects representations**: At each layer, stores activations for the same inputs:
   - With knowledge: h_k = f(x, k)
   - Without knowledge: h_0 = f(x, None)

2. **Computes CKA**: For each layer:
   ```python
   cka = HSIC(h_k, h_0) / sqrt(HSIC(h_k, h_k) * HSIC(h_0, h_0))
   ```

3. **Analyzes patterns**: Which layers are most affected by knowledge?

### Mathematical Foundation

**CKA** (Kornblith et al., 2019):
$$\text{CKA}(X, Y) = \frac{\text{HSIC}(X, Y)}{\sqrt{\text{HSIC}(X, X) \cdot \text{HSIC}(Y, Y)}}$$

**HSIC (Hilbert-Schmidt Independence Criterion)**:
$$\text{HSIC}(X, Y) = \frac{1}{(n-1)^2} \text{tr}(K_X H K_Y H)$$

where:
- K_X = kernel matrix of X (linear: K = XX^T, RBF: K_ij = exp(-||x_i - x_j||²/2σ²))
- H = centering matrix = I - 11^T/n

**Linear Kernel CKA** (what we use):
$$\text{CKA}(X, Y) = \frac{\|Y^T X\|_F^2}{\|X^T X\|_F \|Y^T Y\|_F}$$

**Properties**:
- CKA ∈ [0, 1]
- CKA = 1: Identical representations (up to linear transform)
- CKA = 0: Completely different representations

### Why It's Important

1. **Layer-wise Analysis**: Identifies where knowledge has most impact
2. **Representation Comparison**: Quantifies how much knowledge changes representations
3. **Architecture Insights**: Reveals which components are "knowledge-sensitive"
4. **Validation**: Low CKA in latent layer confirms knowledge affects latent space

### Code Returns

```python
{
    "cka_scores": {
        "R": float,            # Context representation
        "z_mean": float,       # Latent mean
        "z_std": float,        # Latent std
        "pred_mean": float,    # Prediction mean
        "hook_<layer>": float, # Each Linear layer
    },
    "layer_ranking": [(layer_name, cka_score), ...],  # Sorted by CKA
    "summary": {
        "min_cka": float,
        "min_cka_layer": str,
        "max_cka": float,
        "max_cka_layer": str,
        "mean_cka": float,
    },
    "interpretation": str,
}
```

### Expected Results

| Layer | Expected CKA | Reason |
|-------|-------------|--------|
| x_encoder | 0.9-1.0 | Input processing, no knowledge yet |
| xy_encoder (R) | 0.7-0.9 | Context processing, mild effect |
| z_mean | 0.3-0.7 | Latent - knowledge merges here |
| decoder | 0.6-0.8 | Prediction, integrated representation |

**For a well-functioning INP**: Expect z_mean CKA to be lowest (knowledge has most effect on latent).

### Visualizations Generated

1. **CKA Bar Chart**: CKA per layer, sorted
2. **CKA Heatmap**: Matrix showing CKA for all layer pairs
3. **Layer Diagram**: Network architecture colored by CKA
4. **CKA vs Depth**: How CKA changes through network depth

---

## Running the Experiments

### Basic Usage

```bash
# Run all experiments on an INP model
python evaluation/interpretability/run_all.py \
    --model-path saves/INPs_sinusoids/inp_abc2_0/model_best.pt \
    --config-path saves/INPs_sinusoids/inp_abc2_0/config.toml \
    --output-dir ./interpretability_results/inp_abc2

# Run specific experiments
python evaluation/interpretability/run_all.py \
    --model-path saves/INPs_sinusoids/inp_abc2_0/model_best.pt \
    --config-path saves/INPs_sinusoids/inp_abc2_0/config.toml \
    --experiments M1 M3 M7 \
    --output-dir ./interpretability_results/subset

# Quick mode (reduced iterations)
python evaluation/interpretability/run_all.py \
    --model-path saves/INPs_sinusoids/inp_abc2_0/model_best.pt \
    --config-path saves/INPs_sinusoids/inp_abc2_0/config.toml \
    --quick \
    --output-dir ./interpretability_results/quick_run
```

### NP Baseline Models

For Neural Process baselines (without knowledge), some experiments are automatically skipped:
- **Skipped**: M1, M5, M6, M10 (require knowledge integration)
- **Limited mode**: M3, M7, M8 (run without knowledge comparison)
- **Normal**: M2, M4, M9 (analyze model weights/losses only)

### Output Structure

```
interpretability_results/
└── inp_abc2/
    └── run_20260131_170341/
        ├── summary.json              # All results in JSON
        ├── plots/
        │   └── summary_dashboard.png # Overview visualization
        ├── m1_information_bottleneck/
        │   ├── results.json
        │   └── plots/
        │       ├── m1_mi_analysis.png
        │       └── m1_mi_heatmap.png
        ├── m2_loss_landscape/
        │   ├── results.json
        │   └── plots/
        │       ├── m2_landscape.png
        │       └── m2_3d_surface.png
        ... (similar for M3-M10)
```

---

## Summary: Expected Patterns for Well-Functioning INP

| Exp | Key Metric | Good INP | Poor INP | What It Tells You |
|-----|-----------|----------|----------|-------------------|
| M1 | Knowledge Reliance | 0.3-0.7 | <0.1 or >0.9 | Is knowledge used appropriately? |
| M2 | Basin Width Ratio | >1.5 | <1.0 | Does knowledge regularize? |
| M3 | ED Ratio (INP/NP) | <0.8 | >1.0 | Does knowledge constrain manifold? |
| M4 | Loss Balance Score | >0.5 | <0.3 | Is optimization healthy? |
| M5 | Transfer Ratio | >0.5 | <0.2 | Does knowledge causally affect output? |
| M6 | Top Feature Alignment | Semantic | Random | Are the right features used? |
| M7 | Probe R² | >0.5 | <0.3 | Are task parameters encoded? |
| M8 | Epistemic Reduction | >1 nat | <0.1 nat | Does knowledge reduce uncertainty? |
| M9 | Mean α | 2.0-3.0 | <1.5 or >4 | Is model well-regularized? |
| M10 | Latent CKA | <0.7 | >0.95 | Does knowledge change latent space? |

---

## References

1. Tishby, N., Pereira, F. C., & Bialek, W. (2000). "The Information Bottleneck Method"
2. Belghazi, M. I., et al. (2018). "MINE: Mutual Information Neural Estimation"
3. Li, H., et al. (2018). "Visualizing the Loss Landscape of Neural Nets"
4. Hochreiter, S., & Schmidhuber, J. (1997). "Flat Minima"
5. Roy, O., & Bhattacharya, S. (2004). "Effective Dimensionality in PCA"
6. Sundararajan, M., Taly, A., & Yan, Q. (2017). "Axiomatic Attribution for Deep Networks"
7. Alain, G., & Bengio, Y. (2017). "Understanding Intermediate Layers Using Linear Classifier Probes"
8. Depeweg, S., et al. (2018). "Decomposition of Uncertainty in Bayesian Deep Learning"
9. Martin, C. H., & Mahoney, M. W. (2019). "Heavy-Tailed Universality Predicts Generalization"
10. Kornblith, S., et al. (2019). "Similarity of Neural Network Representations Revisited" (CKA)
11. Pearl, J. (2009). "Causality: Models, Reasoning, and Inference"
12. Vig, J., et al. (2020). "Investigating Gender Bias in Language Models Using Causal Mediation Analysis"

python evaluation/interpretability/run_batch.py \
    --models-dir saves/INPs_sinusoids_copy \
    --output-dir ./interpretability_results/sinusoids_batch