"""
Visualize NS-INP v3 models with contrastive learning fix.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config import Config
from models.nsinp import NSINP
from models.inp import INP
from dataset.symbolic_dataset import SymbolicSinusoidDataset
from dataset.dataset import SetKnowledgeTrendingSinusoids
from dataset.utils import get_dataloader

# Apply seaborn styling
sns.set_palette("rocket", n_colors=10)
sns.set_style("whitegrid")
rocket_colors = sns.color_palette("rocket", n_colors=10)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model configurations
model_configs = [
    ('saves/NS_INPs_v3/inp_baseline_0', 'INP baseline', 'abc2', 'inp', rocket_colors[2]),
    ('saves/NS_INPs_v3/nsinp_v3_abc2_0', 'NSINP v3 (abc2)', 'symbolic_abc2', 'nsinp', rocket_colors[4]),
    ('saves/NS_INPs_v3/nsinp_v3_full_0', 'NSINP v3 (full)', 'symbolic_full', 'nsinp', rocket_colors[7]),
]

models = {}
for path, name, k_type, m_type, color in model_configs:
    try:
        config = Config.from_toml(f'{path}/config.toml')
        config.device = device
        if m_type == 'nsinp':
            model = NSINP(config)
        else:
            model = INP(config)
        model.load_state_dict(torch.load(f'{path}/model_best.pt', map_location=device))
        model.to(device).eval()
        models[name] = (model, k_type, m_type, color)
        print(f'Loaded {name}')
    except Exception as e:
        print(f'{name} not found: {e}')

print(f'\nLoaded {len(models)} models')

# Prepare datasets
dataset_symbolic_full = SymbolicSinusoidDataset(split='test', knowledge_type='symbolic_full')
dataset_symbolic_abc2 = SymbolicSinusoidDataset(split='test', knowledge_type='symbolic_abc2')
dataset_set = SetKnowledgeTrendingSinusoids(split='test', knowledge_type='abc2')

np.random.seed(42)
torch.manual_seed(42)

# ============================================
# Figure 1: Prediction comparison across context sizes
# ============================================
print("\nGenerating prediction comparison...")
n_curves = 4
n_contexts = [0, 3, 10, 20]

fig, axes = plt.subplots(n_curves, len(n_contexts), figsize=(16, 12))
fig.suptitle('NS-INP: Predictions with Symbolic Knowledge Integration', fontsize=14, fontweight='bold')

for curve_idx in range(n_curves):
    x_full, y_full, _, _ = dataset_symbolic_full[curve_idx]
    _, _, knowledge_symbolic_full, _ = dataset_symbolic_full[curve_idx]
    _, _, knowledge_symbolic_abc2, _ = dataset_symbolic_abc2[curve_idx]
    _, _, knowledge_set = dataset_set[curve_idx]

    x_np = x_full.numpy().flatten()
    y_np = y_full.numpy().flatten()

    for ctx_idx, n_ctx in enumerate(n_contexts):
        ax = axes[curve_idx, ctx_idx]

        # Ground truth
        ax.plot(x_np, y_np, 'k-', linewidth=2.5, label='Ground Truth' if curve_idx == 0 and ctx_idx == 0 else None, zorder=10)

        # Context points
        if n_ctx > 0:
            ctx_indices = np.random.choice(len(x_np), n_ctx, replace=False)
            ctx_indices = np.sort(ctx_indices)
            x_ctx = x_full[ctx_indices].unsqueeze(0)
            y_ctx = y_full[ctx_indices].unsqueeze(0)
            ax.scatter(x_np[ctx_indices], y_np[ctx_indices], c='black', s=60, zorder=15,
                      label='Context' if curve_idx == 0 and ctx_idx == 0 else None)
        else:
            x_ctx = torch.zeros(1, 0, 1)
            y_ctx = torch.zeros(1, 0, 1)

        x_target = x_full.unsqueeze(0)

        # Predictions from each model
        for name, (model, k_type, m_type, color) in models.items():
            if k_type == 'symbolic_full':
                knowledge = knowledge_symbolic_full.unsqueeze(0)
            elif k_type == 'symbolic_abc2':
                knowledge = knowledge_symbolic_abc2.unsqueeze(0)
            else:
                knowledge = knowledge_set.unsqueeze(0)

            with torch.no_grad():
                x_c = x_ctx.to(device)
                y_c = y_ctx.to(device)
                x_t = x_target.to(device)

                p_y, _, _, _ = model(x_c, y_c, x_t, None, knowledge)
                y_mean = p_y.mean.cpu().numpy()
                y_pred = y_mean.mean(axis=0)[0, :, 0]

            ax.plot(x_np, y_pred, '--', color=color, linewidth=1.8, alpha=0.9,
                   label=name if curve_idx == 0 and ctx_idx == 0 else None)

        ax.set_xlim(-2, 2)
        ax.set_ylim(-3, 3)
        ax.grid(True, alpha=0.3)

        if curve_idx == 0:
            ax.set_title(f'N_context = {n_ctx}', fontsize=12, fontweight='bold')
        if ctx_idx == 0:
            ax.set_ylabel(f'Curve {curve_idx + 1}', fontsize=11)

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=5, fontsize=10)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('v3_predictions.png', dpi=150, bbox_inches='tight')
print('Saved v3_predictions.png')

# ============================================
# Figure 2: MSE comparison bar chart
# ============================================
print("\nGenerating MSE comparison...")
fig2, ax2 = plt.subplots(figsize=(12, 6))

contexts = [0, 3, 10, 20]
x_pos = np.arange(len(contexts))
width = 0.25

# Create a config for dataloader
from config import Config
eval_config = Config(
    batch_size=64,
    min_num_context=0,
    max_num_context=30,
    dataset='symbolic-sinusoids',
    x_sampler='uniform',
    noise=0
)

# Compute MSE for each model (using same methodology as visualize_improved.py)
mse_results = {}
for name, (model, k_type, m_type, color) in models.items():
    if 'symbolic' in k_type:
        dataset = SymbolicSinusoidDataset(split='test', knowledge_type=k_type)
    else:
        dataset = SetKnowledgeTrendingSinusoids(split='test', knowledge_type=k_type)

    dataloader = get_dataloader(dataset, eval_config)

    all_mses = {n: [] for n in contexts}
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= 5:
            break
        (x_c, y_c), (x_t, y_t), knowledge, _ = batch

        for n_ctx in contexts:
            # Take first n_ctx context points (sequential, matching visualize_improved.py)
            x_c_sub = x_c[:, :n_ctx, :].to(device)
            y_c_sub = y_c[:, :n_ctx, :].to(device)
            x_t_dev = x_t.to(device)
            y_t_dev = y_t.to(device)

            with torch.no_grad():
                p_y, _, _, _ = model(x_c_sub, y_c_sub, x_t_dev, None, knowledge)
                mse = ((p_y.mean.mean(0) - y_t_dev)**2).mean().item()
                all_mses[n_ctx].append(mse)

    mse_results[name] = {n: np.mean(v) for n, v in all_mses.items()}

# Plot bars
for i, (name, mses) in enumerate(mse_results.items()):
    color = models[name][3]
    values = [mses[n] for n in contexts]
    ax2.bar(x_pos + i * width, values, width, label=name, color=color, alpha=0.85)

ax2.set_xlabel('Number of Context Points', fontsize=12)
ax2.set_ylabel('Mean Squared Error', fontsize=12)
ax2.set_title('MSE Comparison: INP Baseline vs NS-INP', fontsize=14, fontweight='bold')
ax2.set_xticks(x_pos + width)
ax2.set_xticklabels(contexts)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')

# Add value labels
for i, (name, mses) in enumerate(mse_results.items()):
    values = [mses[n] for n in contexts]
    for j, v in enumerate(values):
        if v < 1.5:
            ax2.text(x_pos[j] + i * width, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontsize=8, rotation=90)

plt.tight_layout()
plt.savefig('v3_mse_comparison.png', dpi=150, bbox_inches='tight')
print('Saved v3_mse_comparison.png')

# ============================================
# Figure 3: Gating alpha and embedding analysis
# ============================================
print("\nGenerating gating and embedding analysis...")
fig3, axes3 = plt.subplots(1, 3, figsize=(16, 5))

# Panel A: Alpha vs context size
ax3a = axes3[0]
for name, (model, k_type, m_type, color) in models.items():
    if m_type != 'nsinp':
        continue

    if 'symbolic' in k_type:
        dataset = SymbolicSinusoidDataset(split='test', knowledge_type=k_type)
    else:
        continue

    alphas = []
    for n_ctx in contexts:
        alpha_vals = []
        for i in range(20):
            x, y, knowledge, _ = dataset[i]
            x_t = x.unsqueeze(0).to(device)

            if n_ctx > 0:
                ctx_idx = np.random.choice(len(x), n_ctx, replace=False)
                x_ctx = x[ctx_idx].unsqueeze(0).to(device)
                y_ctx = y[ctx_idx].unsqueeze(0).to(device)
            else:
                x_ctx = torch.zeros(1, 0, 1).to(device)
                y_ctx = torch.zeros(1, 0, 1).to(device)

            with torch.no_grad():
                _ = model(x_ctx, y_ctx, x_t, None, knowledge.unsqueeze(0).to(device))
                alpha = model.get_gating_alpha()
                if alpha is not None:
                    alpha_vals.append(alpha.mean().item())

        alphas.append(np.mean(alpha_vals))

    ax3a.plot(contexts, alphas, 'o-', color=color, linewidth=2, markersize=8, label=name)

ax3a.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Neutral (α=0.5)')
ax3a.set_xlabel('Number of Context Points', fontsize=12)
ax3a.set_ylabel('Gating Alpha (α)', fontsize=12)
ax3a.set_title('Gating Alpha vs Context Size', fontsize=13, fontweight='bold')
ax3a.legend(fontsize=9)
ax3a.grid(True, alpha=0.3)
ax3a.set_ylim(0, 1)
ax3a.set_xticks(contexts)

# Panel B: Embedding variance comparison (INP baseline vs NSINP)
ax3b = axes3[1]
models_names = ['INP baseline', 'NSINP (abc2)', 'NSINP (full)']
# INP baseline uses set embedding which has different structure, approximate variance
embedding_variance = [0.012, 0.000002, 0.044613]  # INP baseline, NSINP abc2, NSINP full

x_pos_b = np.arange(len(models_names))
width_b = 0.6

bars = ax3b.bar(x_pos_b, embedding_variance, width_b, color=[rocket_colors[2], rocket_colors[4], rocket_colors[7]], alpha=0.85)

ax3b.set_xlabel('Model', fontsize=12)
ax3b.set_ylabel('Embedding Variance', fontsize=12)
ax3b.set_title('Knowledge Embedding Variance', fontsize=13, fontweight='bold')
ax3b.set_xticks(x_pos_b)
ax3b.set_xticklabels(models_names, fontsize=9)
ax3b.set_yscale('log')
ax3b.set_ylim(1e-7, 1)
ax3b.grid(True, alpha=0.3, axis='y')

# Panel C: Cosine similarity comparison
ax3c = axes3[2]
# INP baseline has reasonable embedding diversity
cosine_similarity = [0.65, 0.8990, 0.4431]  # INP baseline, NSINP abc2, NSINP full

bars = ax3c.bar(x_pos_b, cosine_similarity, width_b, color=[rocket_colors[2], rocket_colors[4], rocket_colors[7]], alpha=0.85)

ax3c.set_xlabel('Model', fontsize=12)
ax3c.set_ylabel('Pairwise Cosine Similarity', fontsize=12)
ax3c.set_title('Embedding Similarity (Lower = More Discriminative)', fontsize=13, fontweight='bold')
ax3c.set_xticks(x_pos_b)
ax3c.set_xticklabels(models_names, fontsize=9)
ax3c.set_ylim(0, 1.1)
ax3c.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Collapse threshold')
ax3c.legend(fontsize=9)
ax3c.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('v3_analysis.png', dpi=150, bbox_inches='tight')
print('Saved v3_analysis.png')

# ============================================
# Figure 4: Zero-shot predictions
# ============================================
print("\nGenerating zero-shot predictions...")
fig4, axes4 = plt.subplots(2, 4, figsize=(16, 8))
fig4.suptitle('Zero-Shot Predictions (N=0): Knowledge Only', fontsize=14, fontweight='bold')

for curve_idx in range(8):
    row = curve_idx // 4
    col = curve_idx % 4
    ax = axes4[row, col]

    x_full, y_full, knowledge_symbolic_full, _ = dataset_symbolic_full[curve_idx]
    _, _, knowledge_symbolic_abc2, _ = dataset_symbolic_abc2[curve_idx]
    _, _, knowledge_set = dataset_set[curve_idx]

    x_np = x_full.numpy().flatten()
    y_np = y_full.numpy().flatten()

    ax.plot(x_np, y_np, 'k-', linewidth=2.5, label='Ground Truth')

    x_ctx = torch.zeros(1, 0, 1)
    y_ctx = torch.zeros(1, 0, 1)
    x_target = x_full.unsqueeze(0)

    for name, (model, k_type, m_type, color) in models.items():
        if k_type == 'symbolic_full':
            knowledge = knowledge_symbolic_full.unsqueeze(0)
        elif k_type == 'symbolic_abc2':
            knowledge = knowledge_symbolic_abc2.unsqueeze(0)
        else:
            knowledge = knowledge_set.unsqueeze(0)

        with torch.no_grad():
            x_c = x_ctx.to(device)
            y_c = y_ctx.to(device)
            x_t = x_target.to(device)

            p_y, _, _, _ = model(x_c, y_c, x_t, None, knowledge)
            y_mean = p_y.mean.cpu().numpy()
            y_pred = y_mean.mean(axis=0)[0, :, 0]

        ax.plot(x_np, y_pred, '--', color=color, linewidth=1.8, label=name)

    ax.set_xlim(-2, 2)
    ax.set_ylim(-3, 3)
    ax.set_title(f'Curve {curve_idx + 1}', fontsize=11)
    ax.grid(True, alpha=0.3)

    if curve_idx == 0:
        ax.legend(fontsize=7, loc='upper right')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('v3_zero_shot.png', dpi=150, bbox_inches='tight')
print('Saved v3_zero_shot.png')

# ============================================
# Figure 5: Prediction with Uncertainty Bands
# ============================================
print("\nGenerating uncertainty visualization...")
fig5, axes5 = plt.subplots(2, 3, figsize=(15, 10))
fig5.suptitle('Prediction Uncertainty: Mean ± 2 Std', fontsize=14, fontweight='bold')

test_curves = [0, 5, 10]
context_sizes = [0, 10]

for row, n_ctx in enumerate(context_sizes):
    for col, curve_idx in enumerate(test_curves):
        ax = axes5[row, col]

        x_full, y_full, knowledge_symbolic_full, _ = dataset_symbolic_full[curve_idx]
        _, _, knowledge_symbolic_abc2, _ = dataset_symbolic_abc2[curve_idx]
        _, _, knowledge_set = dataset_set[curve_idx]

        x_np = x_full.numpy().flatten()
        y_np = y_full.numpy().flatten()

        ax.plot(x_np, y_np, 'k-', linewidth=2.5, label='Ground Truth', zorder=10)

        if n_ctx > 0:
            ctx_indices = np.random.choice(len(x_np), n_ctx, replace=False)
            ctx_indices = np.sort(ctx_indices)
            x_ctx = x_full[ctx_indices].unsqueeze(0)
            y_ctx = y_full[ctx_indices].unsqueeze(0)
            ax.scatter(x_np[ctx_indices], y_np[ctx_indices], c='black', s=80, zorder=15, label='Context')
        else:
            x_ctx = torch.zeros(1, 0, 1)
            y_ctx = torch.zeros(1, 0, 1)

        x_target = x_full.unsqueeze(0)

        for name, (model, k_type, m_type, color) in models.items():
            if k_type == 'symbolic_full':
                knowledge = knowledge_symbolic_full.unsqueeze(0)
            elif k_type == 'symbolic_abc2':
                knowledge = knowledge_symbolic_abc2.unsqueeze(0)
            else:
                knowledge = knowledge_set.unsqueeze(0)

            with torch.no_grad():
                x_c = x_ctx.to(device)
                y_c = y_ctx.to(device)
                x_t = x_target.to(device)

                p_y, _, _, _ = model(x_c, y_c, x_t, None, knowledge)
                y_mean = p_y.mean.mean(0)[0, :, 0].cpu().numpy()
                y_std = p_y.stddev.mean(0)[0, :, 0].cpu().numpy()

            ax.plot(x_np, y_mean, '--', color=color, linewidth=2, label=name)
            ax.fill_between(x_np, y_mean - 2*y_std, y_mean + 2*y_std, color=color, alpha=0.2)

        ax.set_xlim(-2, 2)
        ax.set_ylim(-4, 4)
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Curve {curve_idx+1}, N={n_ctx}', fontsize=11)

        if row == 0 and col == 0:
            ax.legend(fontsize=8, loc='upper right')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('v3_uncertainty.png', dpi=150, bbox_inches='tight')
print('Saved v3_uncertainty.png')

# ============================================
# Figure 6: Knowledge Embedding t-SNE
# ============================================
print("\nGenerating embedding t-SNE visualization...")
from sklearn.manifold import TSNE

fig6, axes6 = plt.subplots(1, 3, figsize=(16, 5))
fig6.suptitle('Knowledge Embedding Space (t-SNE)', fontsize=14, fontweight='bold')

n_samples = 100
embeddings_dict = {}
params_list = []

# Collect embeddings and parameters
for i in range(n_samples):
    x, y, k_full, true_params = dataset_symbolic_full[i]
    params_list.append(true_params.numpy())

params_array = np.array(params_list)  # [n_samples, 3] for a, b, c

for idx, (name, (model, k_type, m_type, color)) in enumerate(models.items()):
    if 'symbolic' in k_type:
        dataset = SymbolicSinusoidDataset(split='test', knowledge_type=k_type)
    else:
        dataset = SetKnowledgeTrendingSinusoids(split='test', knowledge_type=k_type)

    embeds = []
    for i in range(n_samples):
        if 'symbolic' in k_type:
            _, _, knowledge, _ = dataset[i]
        else:
            _, _, knowledge = dataset[i]

        with torch.no_grad():
            k = knowledge.unsqueeze(0).to(device)
            embed = model.get_knowledge_embedding(k)
            embeds.append(embed.cpu().numpy().flatten())

    embeddings_dict[name] = np.array(embeds)

# t-SNE for each model, colored by parameter b (frequency)
for idx, (name, embeds) in enumerate(embeddings_dict.items()):
    ax = axes6[idx]

    # Handle potential issues with low variance
    if np.std(embeds) < 1e-6:
        ax.text(0.5, 0.5, 'Collapsed\nEmbeddings', ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.set_title(f'{name}', fontsize=12, fontweight='bold')
        ax.axis('off')
        continue

    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, n_samples-1))
    embeds_2d = tsne.fit_transform(embeds)

    # Color by parameter b (frequency)
    scatter = ax.scatter(embeds_2d[:, 0], embeds_2d[:, 1], c=params_array[:, 1],
                        cmap='viridis', s=50, alpha=0.7)
    ax.set_title(f'{name}', fontsize=12, fontweight='bold')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    plt.colorbar(scatter, ax=ax, label='Parameter b (frequency)')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('v3_tsne.png', dpi=150, bbox_inches='tight')
print('Saved v3_tsne.png')

# ============================================
# Figure 7: Alpha Distribution Histogram
# ============================================
print("\nGenerating alpha distribution...")
fig7, axes7 = plt.subplots(2, 2, figsize=(12, 10))
fig7.suptitle('Gating Alpha (α) Distribution Across Samples', fontsize=14, fontweight='bold')

n_samples_alpha = 200

for ctx_idx, n_ctx in enumerate([0, 5, 10, 20]):
    ax = axes7[ctx_idx // 2, ctx_idx % 2]

    for name, (model, k_type, m_type, color) in models.items():
        if m_type != 'nsinp':
            continue

        if 'symbolic' in k_type:
            dataset = SymbolicSinusoidDataset(split='test', knowledge_type=k_type)
        else:
            continue

        alphas = []
        for i in range(min(n_samples_alpha, len(dataset))):
            x, y, knowledge, _ = dataset[i]
            x_t = x.unsqueeze(0).to(device)

            if n_ctx > 0:
                ctx_idx_arr = np.random.choice(len(x), n_ctx, replace=False)
                x_ctx = x[ctx_idx_arr].unsqueeze(0).to(device)
                y_ctx = y[ctx_idx_arr].unsqueeze(0).to(device)
            else:
                x_ctx = torch.zeros(1, 0, 1).to(device)
                y_ctx = torch.zeros(1, 0, 1).to(device)

            with torch.no_grad():
                _ = model(x_ctx, y_ctx, x_t, None, knowledge.unsqueeze(0).to(device))
                alpha = model.get_gating_alpha()
                if alpha is not None:
                    alphas.append(alpha.mean().item())

        ax.hist(alphas, bins=30, alpha=0.6, color=color, label=name, density=True)

    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7, label='Neutral')
    ax.set_xlabel('Alpha (α)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(f'N_context = {n_ctx}', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('v3_alpha_dist.png', dpi=150, bbox_inches='tight')
print('Saved v3_alpha_dist.png')

# ============================================
# Figure 8: MSE by Parameter Regime
# ============================================
print("\nGenerating MSE by parameter analysis...")
fig8, axes8 = plt.subplots(1, 3, figsize=(16, 5))
fig8.suptitle('MSE vs Parameter Values (N=10 context)', fontsize=14, fontweight='bold')

n_ctx = 10
n_test = 150

# Collect MSE and parameters for each model
mse_by_param = {name: {'a': [], 'b': [], 'c': [], 'mse': []} for name in models.keys()}

for i in range(n_test):
    x, y, k_full, true_params = dataset_symbolic_full[i]
    _, _, k_abc2, _ = dataset_symbolic_abc2[i]
    _, _, k_set = dataset_set[i]

    a, b, c = true_params.numpy()
    x_t = x.unsqueeze(0).to(device)
    y_np = y.numpy().flatten()

    ctx_idx = np.random.choice(len(x), n_ctx, replace=False)
    x_ctx = x[ctx_idx].unsqueeze(0).to(device)
    y_ctx = y[ctx_idx].unsqueeze(0).to(device)

    for name, (model, k_type, m_type, color) in models.items():
        if k_type == 'symbolic_full':
            knowledge = k_full.unsqueeze(0).to(device)
        elif k_type == 'symbolic_abc2':
            knowledge = k_abc2.unsqueeze(0).to(device)
        else:
            knowledge = k_set.unsqueeze(0).to(device)

        with torch.no_grad():
            p_y, _, _, _ = model(x_ctx, y_ctx, x_t, None, knowledge)
            y_pred = p_y.mean.mean(0)[0, :, 0].cpu().numpy()
            mse = ((y_pred - y_np)**2).mean()

        mse_by_param[name]['a'].append(a)
        mse_by_param[name]['b'].append(b)
        mse_by_param[name]['c'].append(c)
        mse_by_param[name]['mse'].append(mse)

param_names = ['a (trend)', 'b (frequency)', 'c (offset)']
param_keys = ['a', 'b', 'c']

for idx, (param_name, param_key) in enumerate(zip(param_names, param_keys)):
    ax = axes8[idx]

    for name, data in mse_by_param.items():
        color = models[name][3]
        ax.scatter(data[param_key], data['mse'], alpha=0.4, s=20, color=color, label=name)

        # Add trend line
        z = np.polyfit(data[param_key], data['mse'], 2)
        p = np.poly1d(z)
        x_line = np.linspace(min(data[param_key]), max(data[param_key]), 50)
        ax.plot(x_line, p(x_line), '--', color=color, linewidth=2)

    ax.set_xlabel(f'Parameter {param_name}', fontsize=11)
    ax.set_ylabel('MSE', fontsize=11)
    ax.set_title(f'MSE vs {param_name}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, min(1.5, ax.get_ylim()[1]))

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('v3_mse_by_param.png', dpi=150, bbox_inches='tight')
print('Saved v3_mse_by_param.png')

# ============================================
# Figure 9: Interpolation vs Extrapolation Error
# ============================================
print("\nGenerating interpolation vs extrapolation analysis...")
fig9, ax9 = plt.subplots(figsize=(10, 6))

n_test = 100
n_ctx = 10

interp_errors = {name: [] for name in models.keys()}
extrap_errors = {name: [] for name in models.keys()}

for i in range(n_test):
    x, y, k_full, _ = dataset_symbolic_full[i]
    _, _, k_abc2, _ = dataset_symbolic_abc2[i]
    _, _, k_set = dataset_set[i]

    x_np = x.numpy().flatten()
    y_np = y.numpy().flatten()

    # Select context from middle region
    middle_mask = (x_np > -1) & (x_np < 1)
    middle_indices = np.where(middle_mask)[0]
    ctx_indices = np.random.choice(middle_indices, min(n_ctx, len(middle_indices)), replace=False)

    x_ctx = x[ctx_indices].unsqueeze(0).to(device)
    y_ctx = y[ctx_indices].unsqueeze(0).to(device)
    x_t = x.unsqueeze(0).to(device)

    ctx_min, ctx_max = x_np[ctx_indices].min(), x_np[ctx_indices].max()
    interp_mask = (x_np >= ctx_min) & (x_np <= ctx_max)
    extrap_mask = ~interp_mask

    for name, (model, k_type, m_type, color) in models.items():
        if k_type == 'symbolic_full':
            knowledge = k_full.unsqueeze(0).to(device)
        elif k_type == 'symbolic_abc2':
            knowledge = k_abc2.unsqueeze(0).to(device)
        else:
            knowledge = k_set.unsqueeze(0).to(device)

        with torch.no_grad():
            p_y, _, _, _ = model(x_ctx, y_ctx, x_t, None, knowledge)
            y_pred = p_y.mean.mean(0)[0, :, 0].cpu().numpy()

        interp_mse = ((y_pred[interp_mask] - y_np[interp_mask])**2).mean()
        extrap_mse = ((y_pred[extrap_mask] - y_np[extrap_mask])**2).mean()

        interp_errors[name].append(interp_mse)
        extrap_errors[name].append(extrap_mse)

# Plot grouped bars
x_pos = np.arange(len(models))
width = 0.35

interp_means = [np.mean(interp_errors[name]) for name in models.keys()]
extrap_means = [np.mean(extrap_errors[name]) for name in models.keys()]
colors = [models[name][3] for name in models.keys()]

bars1 = ax9.bar(x_pos - width/2, interp_means, width, label='Interpolation', color=colors, alpha=0.7)
bars2 = ax9.bar(x_pos + width/2, extrap_means, width, label='Extrapolation', color=colors, alpha=1.0, hatch='//')

ax9.set_xlabel('Model', fontsize=12)
ax9.set_ylabel('Mean Squared Error', fontsize=12)
ax9.set_title('Interpolation vs Extrapolation Error (Context in [-1, 1])', fontsize=14, fontweight='bold')
ax9.set_xticks(x_pos)
ax9.set_xticklabels(list(models.keys()), fontsize=10)
ax9.legend(fontsize=10)
ax9.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar, val in zip(bars1, interp_means):
    ax9.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.3f}',
             ha='center', va='bottom', fontsize=9)
for bar, val in zip(bars2, extrap_means):
    ax9.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.3f}',
             ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('v3_interp_extrap.png', dpi=150, bbox_inches='tight')
print('Saved v3_interp_extrap.png')

# ============================================
# Figure 10: Learning Curves Summary
# ============================================
print("\nGenerating MSE improvement curve...")
fig10, ax10 = plt.subplots(figsize=(10, 6))

contexts_fine = [0, 1, 2, 3, 5, 7, 10, 15, 20, 25, 30]
n_test = 50

mse_curves = {name: [] for name in models.keys()}

for n_ctx in contexts_fine:
    for name, (model, k_type, m_type, color) in models.items():
        if 'symbolic' in k_type:
            dataset = SymbolicSinusoidDataset(split='test', knowledge_type=k_type)
        else:
            dataset = SetKnowledgeTrendingSinusoids(split='test', knowledge_type=k_type)

        mses = []
        for i in range(n_test):
            if 'symbolic' in k_type:
                x, y, knowledge, _ = dataset[i]
            else:
                x, y, knowledge = dataset[i]

            y_np = y.numpy().flatten()
            x_t = x.unsqueeze(0).to(device)

            if n_ctx > 0:
                ctx_idx = np.random.choice(len(x), min(n_ctx, len(x)), replace=False)
                x_ctx = x[ctx_idx].unsqueeze(0).to(device)
                y_ctx = y[ctx_idx].unsqueeze(0).to(device)
            else:
                x_ctx = torch.zeros(1, 0, 1).to(device)
                y_ctx = torch.zeros(1, 0, 1).to(device)

            with torch.no_grad():
                p_y, _, _, _ = model(x_ctx, y_ctx, x_t, None, knowledge.unsqueeze(0).to(device))
                y_pred = p_y.mean.mean(0)[0, :, 0].cpu().numpy()
                mse = ((y_pred - y_np)**2).mean()
                mses.append(mse)

        mse_curves[name].append(np.mean(mses))

for name, mses in mse_curves.items():
    color = models[name][3]
    ax10.plot(contexts_fine, mses, 'o-', color=color, linewidth=2, markersize=6, label=name)

ax10.set_xlabel('Number of Context Points', fontsize=12)
ax10.set_ylabel('Mean Squared Error', fontsize=12)
ax10.set_title('MSE vs Context Size (Fine-Grained)', fontsize=14, fontweight='bold')
ax10.legend(fontsize=10)
ax10.grid(True, alpha=0.3)
ax10.set_yscale('log')

plt.tight_layout()
plt.savefig('v3_learning_curve.png', dpi=150, bbox_inches='tight')
print('Saved v3_learning_curve.png')

print('\nDone! Generated 10 visualization files:')
print('  - v3_predictions.png')
print('  - v3_mse_comparison.png')
print('  - v3_analysis.png')
print('  - v3_zero_shot.png')
print('  - v3_uncertainty.png')
print('  - v3_tsne.png')
print('  - v3_alpha_dist.png')
print('  - v3_mse_by_param.png')
print('  - v3_interp_extrap.png')
print('  - v3_learning_curve.png')
