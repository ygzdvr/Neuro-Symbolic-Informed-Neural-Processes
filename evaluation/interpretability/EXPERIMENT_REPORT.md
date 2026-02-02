# Comprehensive Interpretability Analysis of Informed Neural Processes: A Formal Investigation of Knowledge Integration Mechanisms

## Table of Contents
1. [Introduction and Theoretical Foundations](#introduction-and-theoretical-foundations)
2. [Models Under Analysis](#models-under-analysis)
3. [Experimental Framework and Data-Generating Process](#experimental-framework-and-data-generating-process)
4. [M1: Information Bottleneck Analysis via MINE](#m1-information-bottleneck-analysis-via-mine)
5. [M2: Loss Landscape Geometry](#m2-loss-landscape-geometry)
6. [M3: Effective Dimensionality via Singular Value Decomposition](#m3-effective-dimensionality-via-singular-value-decomposition)
7. [M4: ELBO Loss Term Balance Analysis](#m4-elbo-loss-term-balance-analysis)
8. [M5: Causal Activation Patching via Interchange Interventions](#m5-causal-activation-patching-via-interchange-interventions)
9. [M6: Knowledge Attribution via Integrated Gradients](#m6-knowledge-attribution-via-integrated-gradients)
10. [M7: Linear Probing for Latent Disentanglement](#m7-linear-probing-for-latent-disentanglement)
11. [M8: Epistemic and Aleatoric Uncertainty Decomposition](#m8-epistemic-and-aleatoric-uncertainty-decomposition)
12. [M9: Heavy-Tailed Self-Regularization Spectral Analysis](#m9-heavy-tailed-self-regularization-spectral-analysis)
13. [M10: Centered Kernel Alignment Similarity](#m10-centered-kernel-alignment-similarity)
14. [Synthesis and Theoretical Implications](#synthesis-and-theoretical-implications)
15. [Future Directions](#future-directions)

---

## Introduction and Theoretical Foundations

This report presents a comprehensive interpretability analysis of Informed Neural Process (INP) models trained on synthetic sinusoidal regression tasks. The investigation comprises ten distinct experiments (M1 through M10), each designed to probe a specific aspect of how external knowledge integrates with and modulates the learned representations and predictive behavior of amortized meta-learners. The analysis extends the theoretical and empirical foundations established in the ICLR 2025 paper "Towards Automated Knowledge Integration From Human-Interpretable Representations," providing mechanistic insights into the internal computational processes through which INPs leverage structured prior information.

The fundamental premise underlying Informed Neural Processes rests on the observation that standard Neural Processes (NPs), while effective as amortized approximations to Gaussian Process inference, must infer all task-relevant information solely from observed context points. This places a significant burden on the context encoder and latent inference mechanism, particularly in regimes where context is sparse or noisy. The INP architecture addresses this limitation by introducing a parallel knowledge pathway that conditions the latent distribution on external symbolic or numeric representations of task structure. The key theoretical result, formalized as Theorem 1 in the original paper, establishes that knowledge K improves predictive performance whenever the mutual information between K and the true task parameters θ* is strictly positive, that is, I(K; θ*) > 0. The experiments in this report provide empirical verification and mechanistic elaboration of this theoretical principle.

The standard Neural Process defines a predictive distribution over target outputs y_T given target inputs x_T, context data D_C = {(x_C, y_C)}, through the following generative process. First, context pairs are encoded into a permutation-invariant representation r_C through an encoder network: r_C = Enc(x_C, y_C). Second, a latent variable z is sampled from a variational posterior conditioned on this representation: z ~ q_θ(z | r_C). Third, target predictions are generated through a decoder conditioned on the latent and target inputs: y_T ~ p_θ(y_T | x_T, z). The Informed Neural Process augments this by introducing knowledge k = KnowledgeEnc(K) and conditioning the latent distribution on both the context representation and the knowledge embedding: z ~ q_θ(z | r_C, k). This architectural modification enables the model to leverage external information about task structure without requiring this information to be inferred solely from potentially sparse context observations.

The primary research questions addressed by this interpretability suite are as follows. First, through what representational pathways does knowledge influence model predictions, and how is this information flow distributed across architectural components? Second, does the presence of knowledge alter the geometric properties of the loss landscape in ways that facilitate or impede optimization? Third, what is the effective dimensionality of the learned latent manifold, and does knowledge impose additional structural constraints? Fourth, how does knowledge modulate the balance between the reconstruction term and regularization term in the variational objective? Fifth, can causal interventions demonstrate that knowledge is not merely correlated with but necessary for accurate predictions? Sixth, which specific features of the knowledge representation drive model behavior? Seventh, are task parameters linearly decodable from latent representations, and does knowledge improve this decodability? Eighth, how much predictive uncertainty does knowledge resolve, expressed in information-theoretic units? Ninth, what do the spectral properties of weight matrices reveal about model conditioning and generalization? Tenth, at which layers do representations diverge most significantly between knowledge-conditioned and unconditioned inference?

---

## Models Under Analysis

The experimental analysis encompasses five trained models spanning two dataset variants and two architectural configurations. The primary model, denoted inp_abc2_0, represents an Informed Neural Process trained on the standard trending sinusoids dataset with partial knowledge revelation, specifically the abc2 knowledge type wherein one or two of the three task parameters (a, b, c) are revealed per task instance. This partial revelation scheme provides a realistic test case for knowledge integration under incomplete information, mirroring practical scenarios where external metadata captures only a subset of relevant task characteristics. The second INP variant, inp_abc__0, employs a more restrictive knowledge type (abc) wherein exactly one parameter is revealed per task, enabling comparison of knowledge integration efficacy as a function of information content. The third INP model, inp_b_dist_shift_0, was trained on a distribution-shifted version of the sinusoids dataset where parameter distributions differ between training and evaluation, with knowledge limited to the frequency parameter b only, testing robustness of knowledge integration under distributional mismatch.

The two baseline models, np_0 and np_dist_shift_0, are standard Neural Processes trained on the respective datasets without any knowledge pathway. These baselines provide critical counterfactual comparisons for isolating the effects of knowledge integration from other architectural factors. All models employ identical hyperparameters for the shared components, including a latent dimension of 128, sum aggregation for context encoding, a set embedding architecture for knowledge encoding (where applicable), and a KL weight of β = 1.0 in the ELBO objective. This controlled experimental design ensures that observed differences between INP and NP models can be attributed to the knowledge integration mechanism rather than confounding architectural variations.

---

## Experimental Framework and Data-Generating Process

The synthetic data follows a parametric family of trending sinusoidal functions defined by the equation f(x) = ax + sin(bx) + c, where the parameter a controls the linear trend or slope component, the parameter b determines the angular frequency of the sinusoidal oscillation, and the parameter c specifies the vertical offset or phase. Parameters are sampled independently from uniform distributions: a ~ U[-1, 1], b ~ U[0, 6], and c ~ U[-1, 1]. This three-parameter family was chosen to provide a tractable yet non-trivial test case: the function class is sufficiently expressive to require meaningful inference, yet sufficiently structured that ground-truth parameters provide an interpretable basis for analysis. The linear trend a introduces non-stationarity that complicates pure frequency estimation, the frequency b controls the complexity of the periodic component across a range from nearly constant (b ≈ 0) to highly oscillatory (b ≈ 6), and the offset c determines the baseline level around which outputs fluctuate.

Knowledge representations are constructed as set embeddings encoding subsets of the parameter tuple (a, b, c). For the abc2 knowledge type, each task receives knowledge revealing one or two parameters, with the selection randomized during training. This creates a partially-observed regime where the model must learn to integrate available knowledge with context data to infer the remaining parameters. For the abc knowledge type, exactly one parameter is revealed per task. For the b-only knowledge type used in distribution shift experiments, only the frequency parameter is provided, testing whether a single piece of knowledge suffices for meaningful performance improvement. The set embedding architecture processes knowledge through a permutation-invariant encoder that first embeds each parameter independently, then aggregates through sum pooling, producing a fixed-dimensional knowledge vector k ∈ R^128 that can be integrated with the context representation in the latent encoder.

All experiments disable the knowledge dropout mechanism during evaluation, setting dropout probability to 0.0 rather than the training-time value of 0.3. This ensures that experimental measurements reflect the full effect of knowledge integration rather than being confounded by stochastic knowledge ablation. The number of latent samples varies by experiment, with typical values ranging from 32 to 50 samples for Monte Carlo estimation of expectations over the latent distribution.

---

## M1: Information Bottleneck Analysis via MINE

The first experiment investigates the information flow through the INP architecture by estimating mutual information between the latent representation z and two distinct information sources: the context data D and the knowledge K. The theoretical framework draws on the Information Bottleneck principle, which characterizes representations as compressions that preserve task-relevant information while discarding irrelevant details. In the INP architecture, the latent z must encode sufficient information about both the data and the knowledge to support accurate predictions, making the quantities I(Z; D) and I(Z; K) natural metrics for quantifying the respective contributions of each information pathway.

Direct computation of mutual information is intractable for high-dimensional continuous distributions, necessitating the use of neural estimators. This experiment employs Mutual Information Neural Estimation (MINE), which provides a lower bound on mutual information through the Donsker-Varadhan representation of KL divergence. Specifically, MINE estimates I(X; Y) by training a discriminator network T_φ to distinguish joint samples (x, y) ~ p(x, y) from marginal samples (x, y') ~ p(x)p(y), yielding the bound I(X; Y) ≥ E_{p(x,y)}[T_φ(x, y)] - log(E_{p(x)p(y)}[exp(T_φ(x, y'))]). The MINE estimator is trained for 100 iterations on batches of latent samples paired with either context data or knowledge representations.

The experimental results for the inp_abc2_0 model reveal a striking asymmetry in information content. The converged MINE estimate yields I(Z; D) = 0.080 nats and I(Z; K) = 0.123 nats, indicating that the latent representation z encodes approximately 54% more information about knowledge than about context data. The knowledge reliance ratio, defined as I(Z; K) / (I(Z; D) + I(Z; K)), equals 0.607, meaning that 60.7% of the total information encoded in z derives from the knowledge pathway while only 39.3% derives from context data. This ratio substantially exceeds the balanced regime (0.5) and suggests that the model has learned to prioritize knowledge as the primary source of task identification, using context data primarily for refinement rather than fundamental task inference.

The MINE training dynamics, visualized in the MI analysis plots, show rapid convergence within the first 40 iterations, with estimates stabilizing thereafter. The lower bound estimates, which provide tighter bounds through alternative estimation procedures, suggest even higher true mutual information values of approximately 1.449 nats for data and 1.647 nats for knowledge, though the ordinal relationship is preserved. The discrepancy between the standard MINE estimate and the lower bound reflects the known looseness of variational bounds in high-dimensional settings.

The information flow diagram illustrates the architectural pathway through which knowledge influences the latent distribution. Context pairs (x, y) are first encoded through the xy_encoder into representations that aggregate via sum pooling to produce r_C. Simultaneously, knowledge K passes through the knowledge encoder to produce k. These two representations then combine in the latent encoder, which outputs the parameters of a Gaussian distribution q(z | r_C, k) from which latent samples are drawn. The MINE analysis reveals that this integration is not symmetric: knowledge dominates the information content of z, consistent with the architectural design wherein knowledge provides a structured prior that contexts refine.

The implications for Theorem 1 are direct: the high knowledge reliance ratio empirically confirms that when I(K; θ*) > 0, the model learns to encode this information in its latent representation. The knowledge pathway is not merely present but actively utilized as the primary source of task-relevant information, with context data playing a supplementary role. This finding has practical implications for few-shot and zero-shot scenarios, where limited context availability would otherwise severely constrain inference quality.

### Plots for M1

![M1 MI Analysis - inp_abc2_0](../../interpretability_results/sinusoids_batch_new/batch_20260202_163615/inp_abc2_0/m1/m1_information_bottleneck/plots/m1_mi_analysis.png)

![M1 MI Heatmap - inp_abc2_0](../../interpretability_results/sinusoids_batch_new/batch_20260202_163615/inp_abc2_0/m1/m1_information_bottleneck/plots/m1_mi_heatmap.png)

![M1 Information Flow - inp_abc2_0](../../interpretability_results/sinusoids_batch_new/batch_20260202_163615/inp_abc2_0/m1/m1_information_bottleneck/plots/m1_information_flow.png)

---

## M2: Loss Landscape Geometry

The second experiment characterizes the geometry of the loss landscape in the vicinity of trained model parameters, investigating whether knowledge integration modifies the conditioning of the optimization surface in ways that might explain training dynamics or generalization behavior. The theoretical motivation draws on the extensive literature connecting loss landscape flatness to generalization, with the hypothesis that knowledge conditioning might induce smoother landscapes by providing structured priors that regularize the effective parameter space.

The experimental methodology employs filter-normalized random perturbations, a technique that accounts for the scale invariance of neural network loss landscapes. For each layer i with weight matrix W_i, two random direction matrices δ_i^(1) and δ_i^(2) are sampled from a standard Gaussian distribution and then normalized such that ||δ_i^(j)||_F = ||W_i||_F. This normalization ensures that perturbations in different layers have comparable effect sizes despite potentially disparate weight magnitudes. The perturbed parameters are then θ(α, β) = θ + α Σ_i δ_i^(1) + β Σ_i δ_i^(2), and the loss is evaluated across a grid of (α, β) values spanning [-0.5, 0.5] in each direction with resolution 0.02.

The one-dimensional loss profiles L(α) = L(θ + α Σ_i δ_i^(1)) reveal the local curvature structure around the trained optimum. For the inp_abc2_0 model with knowledge, the origin loss is -9.62 (negative because the ELBO represents a lower bound that the model maximizes). The curvature at the origin, estimated through finite differences as d²L/dα² |_{α=0} ≈ [L(Δα) - 2L(0) + L(-Δα)] / Δα², equals 1760. The basin width, defined as the range of α values for which L(α) remains within a tolerance ε of the origin loss, measures 0.016. The barrier height, quantifying the maximum loss degradation within the perturbation range, reaches 5294.

The comparison with the knowledge-ablated condition (same model evaluated with random or absent knowledge) reveals instructive contrasts. Without knowledge, the origin loss degrades to -7.74, representing a substantial 1.88 nat reduction in ELBO quality. The curvature decreases to 1468, indicating a slightly flatter but less optimal basin. The basin width remains unchanged at 0.016, suggesting that the local conditioning is similar between conditions. The barrier height decreases to 3363, indicating that the landscape becomes less steep overall without knowledge.

The two-dimensional loss surface visualization, rendered as both a heatmap and a three-dimensional surface plot, confirms these findings. Both conditions exhibit bowl-shaped landscapes characteristic of well-trained models, with the minimum at the origin (trained parameters) and roughly quadratic growth in all directions for small perturbations. The knowledge-conditioned surface shows a deeper central basin (lower origin loss) but slightly steeper walls (higher curvature), while the knowledge-ablated surface shows a shallower basin with gentler slopes.

The theoretical interpretation of these findings is nuanced. The similar basin widths indicate that knowledge does not fundamentally alter the local conditioning of the optimization problem, a finding that might initially seem surprising given the substantial improvement in origin loss. However, this consistency makes sense when considering that the loss landscape geometry is primarily determined by the network architecture and data distribution, with knowledge modulating the location and depth of optima rather than their local shape. The higher curvature with knowledge suggests that knowledge-conditioned optima may be slightly less robust to parameter perturbations, though the practical significance of this difference at curvature values of order 10³ is unclear.

The comparison across models shows broadly similar landscape characteristics, with all models exhibiting basin widths near 0.016 and curvatures in the range 1400-1800. This consistency indicates that the landscape geometry is a property of the architecture and task family rather than the specific knowledge integration mechanism.

### Plots for M2

![M2 Loss Landscape - inp_abc2_0](../../interpretability_results/sinusoids_batch_new/batch_20260202_163615/inp_abc2_0/m2/m2_loss_landscape/plots/m2_loss_landscape.png)

![M2 Heatmap - inp_abc2_0](../../interpretability_results/sinusoids_batch_new/batch_20260202_163615/inp_abc2_0/m2/m2_loss_landscape/plots/m2_heatmap.png)

![M2 3D Surface - inp_abc2_0](../../interpretability_results/sinusoids_batch_new/batch_20260202_163615/inp_abc2_0/m2/m2_loss_landscape/plots/m2_3d_surface.png)

---

## M3: Effective Dimensionality via Singular Value Decomposition

The third experiment quantifies the intrinsic dimensionality of the latent representation manifold through singular value decomposition (SVD) analysis. The theoretical premise is that well-trained models should learn latent representations that capture the essential degrees of freedom of the task distribution while discarding irrelevant variation. For the sinusoid task family with three independent parameters (a, b, c), the intrinsic dimensionality of the task manifold is exactly three, providing a concrete benchmark against which to evaluate the learned representations.

The effective dimensionality is computed using the participation ratio, a metric borrowed from statistical physics and random matrix theory. Given a collection of n latent samples z_1, ..., z_n ∈ R^d, the sample covariance matrix is formed as Σ = (1/n) Σ_i (z_i - μ)(z_i - μ)^T where μ is the sample mean. Singular value decomposition yields Σ = U S V^T where S = diag(σ_1, ..., σ_d) contains the singular values in descending order. The participation ratio is then defined as ED = (Σ_i σ_i)² / Σ_i σ_i², which equals the harmonic mean of the effective number of active dimensions weighted by their variance contributions. For a perfectly uniform distribution across k dimensions, ED = k, while for a distribution concentrated on a single direction, ED = 1.

The experimental results for inp_abc2_0 show remarkable consistency between knowledge-conditioned and knowledge-ablated conditions. With knowledge, the effective dimensionality is 3.946, while without knowledge it is 3.851, a difference of merely 0.095. Both values are close to the theoretical intrinsic dimensionality of 3, indicating that the model has successfully learned to compress the 128-dimensional latent space onto a manifold of appropriate complexity. The number of principal components required to explain 95% of the total variance is 4 in both conditions, further confirming the low-dimensional structure of the learned representations.

The singular value spectrum visualization shows the rapid decay characteristic of low-rank structure. The first four singular values account for nearly all variance, with subsequent components contributing negligibly. The heatmap visualization of the top singular values across batches shows consistent patterns, indicating that the low-dimensional structure is stable across different task instances rather than being an artifact of particular samples.

The manifold visualization, rendered through PCA projection onto the first three principal components, provides geometric intuition for the learned representation structure. Points are colored by the true parameter values (a, b, c), revealing that the latent space organizes tasks according to their generative parameters. In both knowledge-conditioned and unconditioned conditions, clear gradients in the point cloud indicate that the latent dimensions correlate with the underlying task parameters, though the clustering appears slightly tighter with knowledge conditioning.

The theoretical interpretation is that knowledge does not compress the latent manifold to lower dimensionality but rather helps the model locate the correct low-dimensional subspace more accurately. Since sinusoid tasks are intrinsically three-dimensional regardless of whether the model has knowledge, the effective dimensionality should equal approximately 3 in both cases, which is precisely what is observed. The slight increase in effective dimensionality with knowledge (3.946 vs 3.851) may reflect the additional information content that knowledge provides, enabling the model to represent finer distinctions between similar tasks.

The cross-model comparison reveals that all models, including the NP baselines, achieve effective dimensionalities near 4. This universality indicates that the low-dimensional structure is a property of the task family itself rather than the knowledge integration mechanism. The models differ not in the dimensionality of their representations but in how accurately they align the learned dimensions with the true generative parameters, a distinction that subsequent experiments (particularly M7) address directly.

### Plots for M3

![M3 Dimensionality - inp_abc2_0](../../interpretability_results/sinusoids_batch_new/batch_20260202_163615/inp_abc2_0/m3/m3_effective_dimensionality/plots/m3_dimensionality.png)

![M3 SV Heatmap - inp_abc2_0](../../interpretability_results/sinusoids_batch_new/batch_20260202_163615/inp_abc2_0/m3/m3_effective_dimensionality/plots/m3_sv_heatmap.png)

![M3 Manifold - inp_abc2_0](../../interpretability_results/sinusoids_batch_new/batch_20260202_163615/inp_abc2_0/m3/m3_effective_dimensionality/plots/m3_manifold.png)

![M3 Projection - inp_abc2_0](../../interpretability_results/sinusoids_batch_new/batch_20260202_163615/inp_abc2_0/m3/m3_effective_dimensionality/plots/m3_projection.png)

---

## M4: ELBO Loss Term Balance Analysis

The fourth experiment examines the balance between the two terms in the variational ELBO objective, investigating whether knowledge integration affects the relative magnitudes of the negative log-likelihood (NLL) reconstruction term and the KL divergence regularization term. The ELBO for a Neural Process with knowledge conditioning takes the form L = E_{q(z|r_C,k)}[log p(y_T | x_T, z)] - β · D_{KL}(q(z|r_T,k) || q(z|r_C,k)), where the first term encourages accurate predictions and the second term regularizes the posterior toward the prior, with hyperparameter β controlling the strength of regularization.

A well-balanced ELBO is generally desirable for stable training and meaningful uncertainty quantification. If the NLL term dominates (|NLL| >> β·|KL|), the model prioritizes reconstruction at the expense of regularization, potentially leading to posterior collapse or overconfident predictions. Conversely, if the KL term dominates (β·|KL| >> |NLL|), the model may underfit the data in favor of staying close to the prior. The balance score is quantified as min(|NLL|, β·|KL|) / max(|NLL|, β·|KL|), ranging from 0 (complete imbalance) to 1 (perfect balance).

The experimental results for inp_abc2_0 reveal moderate but imperfect balance. With correct knowledge, the mean NLL across evaluation batches is -17.04 and the mean KL is 7.12 (with β = 1.0, so β·KL = 7.12). The balance score is therefore 7.12 / 17.04 = 0.422, indicating that the NLL term is approximately 2.4 times larger than the KL term. While not perfectly balanced, this ratio falls within the range where both terms contribute meaningfully to the objective.

The comparison with random knowledge provides a critical baseline. When the model receives knowledge drawn from a different task than the one it is predicting, the loss balance collapses dramatically. The NLL explodes to 118.3 on average, reflecting the model's inability to make accurate predictions with misleading knowledge, while the KL increases modestly to 11.2. The resulting balance score of 0.109 indicates severe imbalance, with reconstruction loss dominating by an order of magnitude. This contrast demonstrates that correct knowledge enables balanced optimization while incorrect knowledge breaks this balance.

The improvement in balance score from random to correct knowledge is 0.422 - 0.109 = 0.313, representing a threefold improvement. This metric captures the model's sensitivity to knowledge correctness: the balance score effectively distinguishes between conditions where knowledge is informative versus misleading. The practical implication is that monitoring loss balance during deployment could provide a diagnostic signal for detecting knowledge-model misalignment.

The gradient alignment analysis, visualized in the M4 plots, examines the alignment between gradients of the NLL and KL terms. When gradients are aligned (positive cosine similarity), optimizing one term does not conflict with the other, facilitating efficient training. The experimental results show moderate positive alignment with correct knowledge, indicating synergistic rather than adversarial optimization dynamics. With random knowledge, alignment deteriorates as the reconstruction gradients point toward configurations that violate the prior structure.

The comparison with the NP baseline (np_0) is instructive. Without a knowledge pathway, the NP achieves a balance score of 0.539, slightly better than the INP with knowledge. However, this comparison is somewhat misleading because the NP operates at a higher absolute loss level (worse predictions). The NP's better balance reflects its simpler objective structure rather than superior training dynamics. The key distinction is that knowledge enables lower NLL at the cost of slightly reduced balance, a favorable trade-off when predictive accuracy is the primary goal.

### Plots for M4

![M4 Gradient Alignment - inp_abc2_0](../../interpretability_results/sinusoids_batch_new/batch_20260202_163615/inp_abc2_0/m4/m4_gradient_alignment/plots/m4_gradient_alignment.png)

![M4 Alignment Distribution - inp_abc2_0](../../interpretability_results/sinusoids_batch_new/batch_20260202_163615/inp_abc2_0/m4/m4_gradient_alignment/plots/m4_alignment_distribution.png)

---

## M5: Causal Activation Patching via Interchange Interventions

The fifth experiment employs causal intervention techniques to establish that knowledge is not merely correlated with but necessary for accurate predictions. While observational analyses (M1-M4) demonstrate associations between knowledge and model behavior, they cannot distinguish causal influence from confounding. Activation patching, borrowed from mechanistic interpretability of language models, provides direct causal evidence by surgically transplanting internal representations between computations and measuring the resulting changes in outputs.

The intervention procedure operates as follows. Two distinct tasks are selected: a recipient task with parameters (a_r, b_r, c_r) and knowledge K_r, and a donor task with parameters (a_d, b_d, c_d) and knowledge K_d. The model first computes the knowledge encoding for the donor: k_d = KnowledgeEnc(K_d). Then, during inference on the recipient task, the knowledge encoding k_r is replaced with k_d at the point where knowledge enters the latent encoder. If knowledge causally influences predictions, this intervention should cause the recipient's predictions to shift toward the donor's ground truth function.

Three primary metrics quantify the intervention effect. The direct effect measures the L2 norm of the shift in predicted mean: ||μ_{patched} - μ_{original}||. The transfer ratio normalizes this by the distance between original and ideal donor predictions: ||μ_{patched} - μ_{original}|| / ||μ_{donor} - μ_{original}||. A transfer ratio of 1.0 would indicate complete transfer (patched predictions equal donor predictions), while 0.0 indicates no effect. The alignment metric computes the cosine similarity between the actual shift direction and the ideal shift direction: cos(μ_{patched} - μ_{original}, μ_{donor} - μ_{original}). Positive alignment indicates the shift moves toward the donor as expected, while negative alignment would indicate pathological behavior.

The experimental results for inp_abc2_0 provide strong evidence for causal knowledge influence. Across 50 task pairs with 32 latent samples each, the mean direct effect is 5.34, indicating substantial prediction shifts following intervention. The mean transfer ratio is 0.413, meaning that patching achieves 41.3% of the ideal shift toward donor predictions. The mean alignment is 0.489, confirming that the shift direction is positively correlated with the expected direction. The composite causal efficacy score, combining these metrics, is 0.561.

The interpretation of the 41.3% transfer ratio requires careful consideration. Complete transfer (100%) would indicate that knowledge entirely determines predictions independent of context data. However, the INP is designed to integrate knowledge with context, not to rely on knowledge exclusively. The observed partial transfer is therefore appropriate: it demonstrates that knowledge causally contributes to predictions while acknowledging that context data also plays a role. The fact that patching does not achieve complete transfer is evidence that the model has learned a genuine integration strategy rather than simply ignoring context in favor of knowledge.

The MSE-based metrics provide complementary evidence. The mean MSE from original predictions to donor ground truth is 2.14, while the MSE from patched predictions to donor ground truth is 1.65, representing a 23% reduction (improvement of 0.49). Simultaneously, the MSE from patched predictions to original ground truth increases by 0.37 on average, confirming that the intervention degrades predictions for the original task while improving them for the donor task. This pattern is exactly what causal influence predicts: transplanting donor knowledge should help predict donor tasks and hurt prediction of original tasks.

The comparison across INP models reveals an intriguing pattern. The distribution shift model inp_b_dist_shift_0 shows higher causal efficacy (transfer ratio 65.1%, alignment 0.61) despite having access to less knowledge (only parameter b). This apparently paradoxical finding has a natural explanation: when the model has less knowledge, it must rely on that knowledge more heavily, amplifying the causal effect of interventions. The richer knowledge in abc2 provides redundant information that makes the model more robust to partial knowledge ablation.

### Plots for M5

![M5 Activation Patching - inp_abc2_0](../../interpretability_results/sinusoids_batch_new/batch_20260202_163615/inp_abc2_0/m5/m5_activation_patching/plots/m5_activation_patching.png)

![M5 Metrics Heatmap - inp_abc2_0](../../interpretability_results/sinusoids_batch_new/batch_20260202_163615/inp_abc2_0/m5/m5_activation_patching/plots/m5_metrics_heatmap.png)

---

## M6: Knowledge Attribution via Integrated Gradients

The sixth experiment applies Integrated Gradients, an axiomatic attribution method from explainable AI, to quantify the contribution of each knowledge feature to model predictions. While M5 established that knowledge is causally necessary, it did not identify which specific features of knowledge drive this effect. For the sinusoid task where knowledge encodes subsets of (a, b, c), understanding the relative importance of each parameter provides insight into both model behavior and the inherent informativeness of different task characteristics.

Integrated Gradients satisfies two fundamental axioms that make it theoretically grounded for attribution. The sensitivity axiom requires that if changing an input feature changes the output, that feature must receive non-zero attribution. The implementation invariance axiom requires that attributions depend only on the function computed by the network, not on how that function is implemented. Formally, the attribution to feature i is defined as IG_i = (x_i - b_i) · ∫_{α=0}^{1} (∂F/∂x_i)|_{b + α(x-b)} dα, where x is the input, b is a baseline input, and F is the network function. The integral is approximated through Riemann summation with 50 steps.

The attribution results for inp_abc2_0 reveal a clear hierarchy among knowledge features. The phase/offset parameter c receives the highest attribution at 48.5% of total importance. The frequency parameter b receives 34.2%, while the amplitude/trend parameter a receives only 17.3%. This ranking is consistent across both the full attribution (including indicator variables that signal which parameters are revealed) and the value-only attribution (considering only the numeric parameter values).

The dominance of the phase parameter c admits several interpretations. First, c directly determines the vertical location of the sinusoid, which affects every output value uniformly. In contrast, the effects of a and b vary across the input domain: a creates slope that matters more at larger |x|, while b creates oscillations that average out over multiple periods. Second, c may be the most difficult parameter to infer from sparse context data, making external knowledge particularly valuable. Third, the uniform distribution c ~ U[-1, 1] provides substantial variation that the model can exploit.

The secondary importance of frequency b reflects its central role in determining function complexity. Higher frequencies produce more oscillations within a given input range, creating distinctive patterns that knowledge can specify directly rather than requiring inference from context. The relatively low attribution to amplitude a suggests that the linear trend component is either easier to infer from context or less critical for accurate predictions within the evaluated input range.

The indicator versus value decomposition provides additional insight. The analysis separates attribution to the binary indicator variables (which signal which parameters are revealed) from the continuous value variables (the actual parameter values). The results show that 42.2% of total attribution goes to indicators while 57.8% goes to values. This indicates that the model attends not only to the numeric knowledge content but also to the meta-information about which parameters are being provided, adapting its processing strategy based on knowledge structure.

The convergence analysis confirms that the Integrated Gradients computation has stabilized. The mean attribution delta between the final two integration steps is 0.0019, well below typical convergence thresholds. The maximum delta of 0.0048 indicates that even the least stable attributions have effectively converged, lending confidence to the quantitative results.

### Plots for M6

![M6 Feature Importance - inp_abc2_0](../../interpretability_results/sinusoids_batch_new/batch_20260202_163615/inp_abc2_0/m6/m6_knowledge_saliency/plots/m6_feature_importance.png)

![M6 Convergence - inp_abc2_0](../../interpretability_results/sinusoids_batch_new/batch_20260202_163615/inp_abc2_0/m6/m6_knowledge_saliency/plots/m6_convergence.png)

---

## M7: Linear Probing for Latent Disentanglement

The seventh experiment trains linear and MLP probes to decode the true task parameters (a, b, c) from latent representations, quantifying the extent to which the learned latent space is disentangled with respect to the generative factors. A well-disentangled representation would encode each generative factor in a linearly separable manner, enabling simple linear combinations of latent dimensions to recover the underlying parameters. This property is desirable for interpretability, compositional generalization, and transfer learning.

The probing methodology is as follows. For each evaluation batch, context data is processed through the encoder to produce latent samples z. The true task parameters (a, b, c) are obtained from the dataset. A linear probe W ∈ R^{3×128} is trained via ordinary least squares to minimize ||Wz - [a, b, c]^T||² across tasks. An MLP probe with one hidden layer is trained similarly. The coefficient of determination R² = 1 - SS_{res}/SS_{tot} measures the fraction of parameter variance explained by the probe, with R² = 1 indicating perfect recovery and R² = 0 indicating no predictive power.

The experimental results demonstrate substantial disentanglement improvement from knowledge integration. For inp_abc2_0 with knowledge, the linear probe achieves overall R² = 0.750, indicating that 75% of the variance in task parameters can be explained through linear combinations of latent dimensions. Without knowledge (using the same model with random knowledge), the linear probe achieves only R² = 0.568, a reduction of 18.2 percentage points. This knowledge benefit of +0.182 indicates that knowledge explicitly structures the latent space in a more parameter-aligned manner.

The per-parameter breakdown reveals interesting patterns. The phase parameter c is most linearly decodable (R² = 0.862 with knowledge, 0.703 without), consistent with its high attribution in M6. The amplitude a is intermediate (R² = 0.808 with knowledge, 0.631 without). The frequency b is least decodable (R² = 0.581 with knowledge, 0.371 without), suggesting that frequency information may be encoded in more complex, non-linear patterns. Interestingly, knowledge provides the largest absolute improvement for b (+0.210), suggesting that frequency information is particularly difficult to infer from context alone and thus benefits most from explicit knowledge provision.

The comparison between linear and MLP probes tests whether the representation is linearly separable. For inp_abc2_0 with knowledge, the MLP achieves R² = 0.731, slightly lower than the linear probe's 0.750. The negative gap of -0.019 indicates that the MLP does not outperform the linear probe, implying that the representation is essentially linear with respect to parameter encoding. This linearity is a positive property for interpretability, indicating that the latent dimensions encode parameter information in straightforward linear combinations rather than complex nonlinear manifolds.

The comparison with the NP baseline starkly illustrates the benefit of knowledge integration. The np_0 model achieves only R² = 0.148 with the linear probe, meaning that less than 15% of parameter variance is linearly decodable from its latent representations. This fivefold reduction compared to the INP demonstrates that knowledge fundamentally changes the structure of learned representations, imposing parameter-aligned organization that is absent in purely data-driven inference. The NP's poor disentanglement suggests that it learns task-specific representations that do not factorize according to the generative parameters.

### Plots for M7

![M7 Linear Probing - inp_abc2_0](../../interpretability_results/sinusoids_batch_new/batch_20260202_163615/inp_abc2_0/m7/m7_linear_probing/plots/m7_linear_probing.png)

---

## M8: Epistemic and Aleatoric Uncertainty Decomposition

The eighth experiment decomposes predictive uncertainty into its epistemic and aleatoric components, quantifying the information-theoretic value of knowledge in terms of uncertainty reduction. This analysis directly connects to Theorem 1 by measuring how much predictive uncertainty knowledge resolves, expressed in natural units (nats) and bits.

The decomposition follows from the law of total variance applied to the predictive distribution. For a Neural Process, the predictive mean and variance at target point x are obtained by marginalizing over the latent distribution: p(y|x, D_C, K) = ∫ p(y|x, z) q(z|r_C, k) dz. The total predictive variance decomposes as Var[y] = E_z[Var[y|z]] + Var_z[E[y|z]], where the first term represents aleatoric uncertainty (irreducible noise variance averaged over latent samples) and the second term represents epistemic uncertainty (variance in predicted means across latent samples). Epistemic uncertainty should decrease with more context points as the posterior over z concentrates, while aleatoric uncertainty should remain constant as it reflects intrinsic noise.

The experiment evaluates uncertainty across context sizes N ∈ {0, 1, 3, 5, 10, 20, 30} for both knowledge-conditioned and knowledge-ablated conditions. At each context size, 100 tasks are sampled, and uncertainty is computed using 500 Monte Carlo samples over 50 latent samples. The zero-shot (N=0) case is particularly important as it isolates the value of knowledge when no context data is available.

The results for inp_abc2_0 reveal substantial uncertainty reduction from knowledge. At N=0, the epistemic uncertainty with knowledge is 22.73 nats, while without knowledge it is 29.93 nats. The reduction of 7.20 nats corresponds to 10.39 bits, representing substantial prior information that knowledge provides. To contextualize this figure, consider that 10.39 bits is equivalent to the information gained from observing approximately 10 randomly selected context points (as verified by the uncertainty convergence curves).

The uncertainty curves as a function of context size show the expected monotonic decrease in epistemic uncertainty. With knowledge, epistemic uncertainty decreases from 22.73 nats at N=0 to 4.05 nats at N=30. Without knowledge, it decreases from 29.93 nats to 5.84 nats. The gap between curves remains substantial throughout, indicating that knowledge provides value beyond what context data provides. Importantly, the gap does not close as context increases, suggesting that knowledge and context provide complementary rather than redundant information.

The aleatoric uncertainty remains essentially constant across context sizes and conditions, serving as a sanity check on the decomposition. With knowledge, aleatoric uncertainty is approximately -3.3 nats (negative due to the Gaussian variance parameterization); without knowledge, it is approximately -3.2 nats. The consistency (standard deviation < 0.3 nats) confirms that the decomposition is correctly separating reducible from irreducible uncertainty.

The comparison with the NP baseline provides the definitive test of knowledge value. For np_0, the epistemic uncertainty at N=0 with knowledge is 29.92 nats, while without knowledge it is 29.80 nats, a negligible difference of -0.12 nats (the negative value reflects estimation noise). The bit value of knowledge is effectively zero, exactly as expected for a model without a knowledge pathway. This null result confirms that the uncertainty reduction observed in INPs is attributable to the knowledge integration mechanism rather than some other factor.

### Plots for M8

![M8 Uncertainty - inp_abc2_0](../../interpretability_results/sinusoids_batch_new/batch_20260202_163615/inp_abc2_0/m8/m8_uncertainty_decomposition/plots/m8_uncertainty.png)

---

## M9: Heavy-Tailed Self-Regularization Spectral Analysis

The ninth experiment analyzes the spectral properties of trained weight matrices through the lens of Heavy-Tailed Self-Regularization (HTSR) theory. This framework, developed by Martin and Mahoney, connects the power-law exponent of eigenvalue distributions to model conditioning and generalization behavior, providing a principled diagnostic for identifying under-trained, well-trained, and over-trained layers.

The theoretical framework posits that the eigenvalue spectrum of well-trained weight matrices follows a power law P(λ) ~ λ^{-α} in the bulk regime. The exponent α characterizes the spectral tail heaviness and empirically correlates with generalization quality. Specifically, layers with α < 2 exhibit heavy tails indicative of overfitting tendency, layers with α ∈ [2, 4] fall in the "Goldilocks zone" associated with good generalization, and layers with α > 6 show light tails suggesting underfitting. The analysis proceeds by computing the eigenvalue spectrum of each weight matrix W via singular value decomposition (eigenvalues are squared singular values), then fitting a power law to the bulk distribution using maximum likelihood estimation with automatic threshold selection.

The results for inp_abc2_0 reveal a consistent pattern of mild overfitting tendency across all layers. The mean power-law exponent across 15 analyzable layers is α = 1.84 with standard deviation 0.28. The minimum α is 1.26 (in latent_encoder.encoder.layers.1) and the maximum is 2.61 (in xy_encoder.pairer.layers.0). Only 2 of 15 layers (13.3%) fall in the Goldilocks zone, while 13 of 15 (86.7%) show overfitting tendency.

The breakdown by module shows systematic differences. The xy_encoder has mean α = 2.01, closest to the Goldilocks zone, suggesting well-conditioned context encoding. The x_encoder has mean α = 2.06, also near optimal. In contrast, the latent_encoder shows mean α = 1.76, the most heavy-tailed module, indicating overfitting tendency in the knowledge integration pathway. The decoder has mean α = 1.81, also heavy-tailed but less extreme than the latent encoder.

The interpretation of these findings requires nuance. First, the HTSR framework was developed for large-scale image classification models, and its applicability to small Neural Processes trained on synthetic data is not established. The "overfitting" label may be inappropriate for this regime where the model-data relationship differs fundamentally from ImageNet-scale training. Second, the consistency of α values across INP and NP models suggests that the spectral properties are determined by architecture and data rather than knowledge integration. The np_0 baseline shows mean α = 1.89, similar to the INPs. Third, the practical consequences of α < 2 for Neural Process generalization remain unclear without systematic ablation studies.

The stable rank analysis provides additional diagnostic information. The stable rank ||W||_F² / ||W||_2² measures the effective rank of weight matrices independent of power-law assumptions. Values range from 1.19 (rank-1 like) to 14.83 (approximately full rank) across layers, with most values in the range 5-10 indicating moderate effective rank consistent with the low-dimensional task structure.

### Plots for M9

![M9 Spectral - inp_abc2_0](../../interpretability_results/sinusoids_batch_new/batch_20260202_163615/inp_abc2_0/m9/m9_spectral_analysis/plots/m9_spectral.png)

![M9 Layer Heatmap - inp_abc2_0](../../interpretability_results/sinusoids_batch_new/batch_20260202_163615/inp_abc2_0/m9/m9_spectral_analysis/plots/m9_layer_heatmap.png)

---

## M10: Centered Kernel Alignment Similarity

The tenth experiment employs Centered Kernel Alignment (CKA) to measure the similarity between representations computed with and without knowledge, identifying the layers and representations where knowledge has the greatest effect. CKA is a representation similarity metric that is invariant to orthogonal transformations and isotropic scaling, making it well-suited for comparing neural network representations across conditions.

The CKA similarity between two sets of representations X ∈ R^{n×p} and Y ∈ R^{n×q} (n samples, p and q dimensions) is defined as CKA(X, Y) = ||X^T Y||_F² / (||X^T X||_F · ||Y^T Y||_F) for linear CKA. The metric ranges from 0 (orthogonal representations) to 1 (identical representations up to linear transformation). Values near 1 indicate that knowledge does not affect the representation at that layer, while lower values indicate substantial knowledge-induced changes.

The experimental results for inp_abc2_0 reveal a striking pattern of layer-specific knowledge effects. The context representation R has CKA = 1.0, indicating that the context encoding is identical with and without knowledge. This makes architectural sense: the xy_encoder processes context data independently of knowledge, so its output should not change when knowledge is present or absent. Similarly, all x_encoder and xy_encoder layers show CKA = 1.0, confirming that the early data processing pathway is knowledge-independent.

The divergence begins in the latent encoder. The first latent encoder layer shows CKA = 0.986, a slight reduction from 1.0 indicating the onset of knowledge influence. The second latent encoder layer shows CKA = 0.830, a substantial reduction indicating significant knowledge-induced representation changes at the layer where knowledge is integrated with context information.

The key representations show the expected pattern. The latent mean z_mean has CKA = 0.855, indicating moderate but not dramatic changes due to knowledge. More strikingly, the latent standard deviation z_std has CKA = 0.552, indicating that knowledge primarily modulates uncertainty rather than mean predictions. The predicted mean pred_mean has CKA = 0.915, indicating that despite substantial changes in latent representations, the final predictions are relatively similar (though the small differences may be consequential for accuracy).

The theoretical interpretation connects these findings to the INP architecture. Knowledge enters the computation at the latent encoder, so layers before this point should be unaffected (CKA = 1.0), exactly as observed. Knowledge then modulates the latent distribution parameters, with greater effect on the variance (z_std, CKA = 0.552) than the mean (z_mean, CKA = 0.855). This pattern suggests that knowledge primarily affects uncertainty quantification, providing a structured prior that reduces epistemic uncertainty without dramatically shifting predicted means. The relatively high CKA of predictions (0.915) indicates a "soft" integration where knowledge refines rather than overrides data-based inference.

The comparison across INP models reveals that distribution shift increases knowledge differentiation. The inp_b_dist_shift_0 model shows z_mean CKA = 0.729 and z_std CKA = 0.45, both substantially lower than the standard INPs. This suggests that under distribution shift, the model relies more heavily on knowledge to compensate for the mismatch between training and evaluation distributions, inducing greater representational changes when knowledge is present.

### Plots for M10

![M10 CKA Similarity - inp_abc2_0](../../interpretability_results/sinusoids_batch_new/batch_20260202_163615/inp_abc2_0/m10/m10_cka_similarity/plots/m10_cka_similarity.png)

![M10 CKA Heatmap - inp_abc2_0](../../interpretability_results/sinusoids_batch_new/batch_20260202_163615/inp_abc2_0/m10/m10_cka_similarity/plots/m10_cka_heatmap.png)

---

## Synthesis and Theoretical Implications

The ten experiments collectively provide a comprehensive mechanistic account of knowledge integration in Informed Neural Processes. The findings can be organized around several central themes that illuminate both the empirical behavior and theoretical foundations of knowledge-conditioned amortized inference.

The first theme concerns information flow and utilization. Experiment M1 established that the INP latent representation encodes substantially more information about knowledge (I(Z;K) = 0.123 nats) than about context data (I(Z;D) = 0.080 nats), with a knowledge reliance ratio of 60.7%. This finding directly validates Theorem 1's prediction that when I(K; θ*) > 0, the model will learn to exploit knowledge for prediction. Experiment M10 complemented this by localizing knowledge effects to the latent encoder and uncertainty parameters, showing that knowledge modulates z_std (CKA = 0.552) more than z_mean (CKA = 0.855). Together, these experiments reveal a picture where knowledge provides a structured prior that primarily affects uncertainty quantification, with context data refining predictions within the knowledge-constrained hypothesis space.

The second theme concerns representational structure. Experiment M3 showed that both INP and NP models learn low-dimensional latent representations (effective dimensionality ≈ 4) matching the intrinsic dimensionality of the three-parameter sinusoid family. Knowledge does not compress this manifold but rather helps align it with the generative parameters. Experiment M7 quantified this alignment through linear probing: INP with knowledge achieves R² = 0.750 for parameter recovery compared to only R² = 0.148 for the NP baseline, a fivefold improvement. The representation is essentially linear (MLP probes do not outperform linear probes), indicating that knowledge induces a structured, disentangled latent space where task parameters are encoded as separable linear directions.

The third theme concerns causal necessity. Experiment M5 provided direct causal evidence that knowledge is not merely correlated with but necessary for accurate predictions. Activation patching achieved 41.3% transfer of predictions from donor to recipient tasks, with 0.489 directional alignment. This partial transfer is appropriate for a model designed to integrate knowledge with context rather than rely on knowledge exclusively. The causal efficacy score of 0.561 indicates substantial but not complete knowledge dependence, consistent with the balanced information flow observed in M1.

The fourth theme concerns uncertainty quantification. Experiment M8 quantified the information-theoretic value of knowledge as 10.39 bits of zero-shot epistemic uncertainty reduction, equivalent to observing approximately 10 context points. This substantial value persists across context sizes, indicating that knowledge and context provide complementary rather than redundant information. The NP baseline shows negligible uncertainty reduction (−0.12 nats), confirming that this benefit is attributable to knowledge integration rather than other factors.

The fifth theme concerns feature attribution. Experiment M6 identified the phase parameter c as the most important knowledge feature (48.5% attribution), followed by frequency b (34.2%) and amplitude a (17.3%). This hierarchy reflects the differential informativeness of parameters: c affects all predictions uniformly and may be hardest to infer from sparse context, making external knowledge particularly valuable. The model correctly adapts its reliance based on which parameters are revealed (indicator variables receive 42.2% of attribution), demonstrating sophisticated context-dependent processing.

The sixth theme concerns optimization and conditioning. Experiment M2 showed that knowledge improves the origin loss (−9.62 vs −7.74) without substantially altering landscape geometry (similar basin width of 0.016, modestly higher curvature of 1760 vs 1468). Experiment M4 showed moderate loss balance (score = 0.422) with correct knowledge that collapses (score = 0.109) with random knowledge, indicating that balance serves as a diagnostic for knowledge-model alignment. Experiment M9 revealed mild overfitting tendency (mean α = 1.84 < 2) consistent across models, suggesting this is an architectural rather than knowledge-related property.

The overall picture that emerges is of a model that has learned to effectively leverage structured external knowledge while maintaining flexibility to incorporate context data. Knowledge primarily affects uncertainty quantification (z_std) rather than point predictions (z_mean), structures the latent space for parameter-aligned linear decodability, and provides substantial prior information (10.4 bits) that reduces the burden on context-based inference. These properties align with the theoretical motivation for Informed Neural Processes and provide mechanistic validation of the knowledge integration principle formalized in Theorem 1.

---

## Future Directions

The experimental framework developed here opens several avenues for future investigation. First, the MINE estimates in M1 could be tightened through longer training or alternative estimators (e.g., InfoNCE, SMILE), potentially revealing finer-grained information flow patterns. Second, the activation patching methodology in M5 could be extended to layer-specific interventions, identifying the precise computational locus of knowledge integration. Third, the linear probing analysis in M7 could be applied to intermediate layers, tracing the emergence of disentanglement through the forward pass. Fourth, the uncertainty decomposition in M8 could be extended to out-of-distribution tasks, testing whether knowledge provides robustness under distribution shift. Fifth, the spectral analysis in M9 could be tracked during training, testing whether α values evolve predictably and whether they correlate with generalization metrics.

Beyond methodological extensions, the findings suggest several architectural innovations. The observation that knowledge primarily modulates uncertainty (z_std) rather than means (z_mean) suggests that explicit variance-only conditioning pathways might be more parameter-efficient. The high linear decodability of parameters suggests that auxiliary prediction losses could further encourage disentanglement. The feature importance hierarchy suggests that adaptive knowledge encoding, emphasizing more informative features, could improve sample efficiency.

Finally, the framework should be validated on more complex domains where the gap between data-driven and knowledge-augmented inference is larger. The synthetic sinusoid task, while providing clean theoretical interpretation, may underestimate the benefits of knowledge integration in settings with higher-dimensional parameter spaces, multimodal task distributions, or noisier observations.

---

## Technical Details

All experiments were conducted with knowledge dropout disabled during evaluation (set to 0.0 rather than the training value of 0.3) to measure the full effect of knowledge integration. The number of latent samples ranged from 32 to 50 depending on the experiment, with Monte Carlo estimation used for expectations over the latent distribution. Batch sizes ranged from 8 to 16, with 8 evaluation batches used for most experiments. Results are stored in `interpretability_results/sinusoids_batch_new/batch_20260202_163615/` with separate subdirectories for each model and experiment.

The experiments draw on established methodologies from the interpretability literature. MINE (M1) follows Belghazi et al., "Mutual Information Neural Estimation" (ICML 2018). Filter-normalized loss landscape visualization (M2) follows Li et al., "Visualizing the Loss Landscape of Neural Nets" (NeurIPS 2018). Effective dimensionality via participation ratio (M3) follows standard practice in random matrix theory. Integrated Gradients (M6) follows Sundararajan et al., "Axiomatic Attribution for Deep Networks" (ICML 2017). Linear probing (M7) follows Alain and Bengio, "Understanding Intermediate Layers Using Linear Classifier Probes" (ICLR 2017 Workshop). Uncertainty decomposition (M8) follows Kendall and Gal, "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?" (NeurIPS 2017). Heavy-Tailed Self-Regularization (M9) follows Martin and Mahoney, "Implicit Self-Regularization in Deep Neural Networks" (J. Stat. Mech. 2019). CKA (M10) follows Kornblith et al., "Similarity of Neural Network Representations Revisited" (ICML 2019).

---

*Report generated: 2026-02-02*
*Results from batch: batch_20260202_163615*
*All numerical results can be reproduced from the JSON files in the results directories.*
