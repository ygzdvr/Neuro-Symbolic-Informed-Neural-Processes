"""
Neuro-Symbolic Informed Neural Process (NS-INP)

Implements:
- SymbolicEquationEncoder: Transformer-based encoder for tokenized equations
- ConflictAwareAggregator: Learned gating between knowledge and data representations
- NSLatentEncoder: Combines equation encoding with conflict-aware aggregation
- NSINP: Main model class with same interface as INP for compatibility
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
from models.modules import XYEncoder, Decoder, XEncoder, MLP
from models.utils import MultivariateNormalDiag


class SymbolicEquationEncoder(nn.Module):
    """
    Transformer-based encoder for tokenized symbolic equations.

    Takes tokenized equations (e.g., "y = 0.5 * x + sin(2.3 * x) + -0.8")
    and produces a fixed-dimension knowledge embedding.

    Uses [CLS] token pooling instead of mean pooling to prevent representation collapse.
    """

    def __init__(self, vocab_size, d_model, max_len=50, num_layers=3, nhead=4, device='cpu'):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        self.device = device

        # Token embedding (vocab_size + 1 for [CLS] token at index vocab_size)
        self.token_embed = nn.Embedding(vocab_size + 1, d_model, padding_idx=0)
        self.cls_token_id = vocab_size  # [CLS] is at the end of vocab

        # Learnable positional encoding (+1 for [CLS] position)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len + 1, d_model) * 0.02)

        # Learnable [CLS] token embedding (in addition to position embedding)
        self.cls_embed = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Layer norm before pooling (helps prevent collapse)
        self.pre_pool_norm = nn.LayerNorm(d_model)

        # Output projection
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model)
        )

        # Auxiliary head for parameter prediction (prevents representation collapse)
        # Predicts 3 parameters: a, b, c
        self.param_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 3)
        )

        # For compatibility with KnowledgeEncoder interface
        self.dim_model = d_model

        # Store last embedding for auxiliary loss computation
        self.last_embedding = None

    def forward(self, equation_tokens):
        """
        Args:
            equation_tokens: [batch, seq_len] token IDs

        Returns:
            z_k: [batch, 1, d_model] knowledge embedding
        """
        # Move to correct device
        equation_tokens = equation_tokens.to(self.token_embed.weight.device)

        batch_size, seq_len = equation_tokens.shape

        # Create padding mask (True for padded positions)
        # [CLS] at position 0 is never masked
        padding_mask = (equation_tokens == 0)
        cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=equation_tokens.device)
        full_padding_mask = torch.cat([cls_mask, padding_mask], dim=1)  # [batch, 1 + seq_len]

        # Token embedding
        x = self.token_embed(equation_tokens)  # [batch, seq_len, d_model]

        # Prepend [CLS] token
        cls_tokens = self.cls_embed.expand(batch_size, -1, -1)  # [batch, 1, d_model]
        x = torch.cat([cls_tokens, x], dim=1)  # [batch, 1 + seq_len, d_model]

        # Add positional embedding
        x = x + self.pos_embed[:, :x.shape[1], :]

        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=full_padding_mask)

        # Use [CLS] token output (position 0) instead of mean pooling
        cls_output = x[:, 0, :]  # [batch, d_model]

        # Layer norm before projection
        cls_output = self.pre_pool_norm(cls_output)

        # Project to output
        z_k = self.head(cls_output)  # [batch, d_model]

        # Add dimension for compatibility: [batch, 1, d_model]
        return z_k.unsqueeze(1)

    def predict_params(self, z_k):
        """Predict parameters from embedding (for auxiliary loss)."""
        return self.param_predictor(z_k)


class ConflictAwareAggregator(nn.Module):
    """
    Learned gating mechanism that dynamically balances knowledge and data.

    Computes: z_post = alpha * z_k + (1 - alpha) * z_d
    where alpha = sigmoid(GateNet(z_k, z_d))

    The alpha value indicates how much to trust knowledge vs data:
    - alpha ~ 1: Trust knowledge (sparse/noisy data)
    - alpha ~ 0: Trust data (dense data contradicts knowledge)
    - alpha ~ 0.5: Balanced integration
    """

    def __init__(self, d_model, hidden_dim=64, init_bias=0.0):
        super().__init__()
        self.d_model = d_model

        # Gate network: takes concatenated [z_d, z_k] and outputs scalar alpha
        self.gate_net = nn.Sequential(
            nn.Linear(d_model * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        # Initialize final layer bias to favor knowledge initially
        # init_bias > 0 means alpha starts closer to 1 (trust knowledge)
        nn.init.zeros_(self.gate_net[-1].weight)
        nn.init.constant_(self.gate_net[-1].bias, init_bias)

        # Store last alpha for interpretability access (M4/M8 experiments)
        self.last_alpha = None

    def forward(self, z_d, z_k):
        """
        Args:
            z_d: [batch, 1, d_model] data representation
            z_k: [batch, 1, d_model] knowledge representation

        Returns:
            z_post: [batch, 1, d_model] gated combination
        """
        # Concatenate along feature dimension
        combined = torch.cat([z_d, z_k], dim=-1)  # [batch, 1, d_model * 2]

        # Compute gating value (sigmoid applied here, not in Sequential)
        alpha = torch.sigmoid(self.gate_net(combined))  # [batch, 1, 1]

        # Store for interpretability (detached to avoid affecting gradients)
        self.last_alpha = alpha.detach()

        # Gated combination
        z_post = alpha * z_k + (1 - alpha) * z_d

        return z_post

    def get_alpha(self):
        """Return the last computed alpha value for interpretability."""
        return self.last_alpha


class NSLatentEncoder(nn.Module):
    """
    Neuro-Symbolic Latent Encoder combining equation encoding with conflict-aware gating.

    Maintains compatibility with LatentEncoder interface:
    - Has .knowledge_encoder attribute
    - Supports knowledge_dropout
    - Same forward signature
    """

    def __init__(self, config):
        super().__init__()

        self.knowledge_dim = config.knowledge_dim
        self.knowledge_dropout = config.knowledge_dropout
        self.use_gating = getattr(config, 'use_gating', True)

        # Symbolic equation encoder (acts as knowledge_encoder for compatibility)
        vocab_size = getattr(config, 'equation_vocab_size', 64)
        max_len = getattr(config, 'equation_max_len', 50)

        if config.use_knowledge:
            self.knowledge_encoder = SymbolicEquationEncoder(
                vocab_size=vocab_size,
                d_model=config.knowledge_dim,
                max_len=max_len,
                num_layers=3,
                nhead=4
            )
        else:
            self.knowledge_encoder = None

        # Conflict-aware aggregator
        if self.use_gating and config.use_knowledge:
            gating_hidden_dim = getattr(config, 'gating_hidden_dim', 64)
            gating_init_bias = getattr(config, 'gating_init_bias', 0.0)
            self.aggregator = ConflictAwareAggregator(
                d_model=config.hidden_dim,
                hidden_dim=gating_hidden_dim,
                init_bias=gating_init_bias
            )
        else:
            self.aggregator = None

        # Knowledge projection (if knowledge_dim != hidden_dim)
        if config.use_knowledge and config.knowledge_dim != config.hidden_dim:
            self.knowledge_proj = nn.Linear(config.knowledge_dim, config.hidden_dim)
        else:
            self.knowledge_proj = None

        # Final encoder to produce latent distribution parameters
        input_dim = config.hidden_dim
        if config.latent_encoder_num_hidden > 0:
            self.encoder = MLP(
                input_size=input_dim,
                hidden_size=config.hidden_dim,
                num_hidden=config.latent_encoder_num_hidden,
                output_size=2 * config.hidden_dim,
            )
        else:
            self.encoder = nn.Linear(input_dim, 2 * config.hidden_dim)

        self.config = config

    def forward(self, R, knowledge, n):
        """
        Infer the latent distribution given the global representation and knowledge.

        Args:
            R: [batch, 1, hidden_dim] data representation
            knowledge: [batch, seq_len] tokenized equation or None
            n: number of context points (unused, for interface compatibility)

        Returns:
            q_z_stats: [batch, 1, 2*hidden_dim] latent distribution parameters
        """
        drop_knowledge = torch.rand(1) < self.knowledge_dropout

        # Encode knowledge if available
        if drop_knowledge or knowledge is None or self.knowledge_encoder is None:
            # No knowledge: just use data representation
            encoder_input = F.relu(R)
        else:
            # Encode symbolic equation
            k = self.knowledge_encoder(knowledge)  # [batch, 1, knowledge_dim]

            # Project if needed
            if self.knowledge_proj is not None:
                k = self.knowledge_proj(k)  # [batch, 1, hidden_dim]

            # Apply conflict-aware gating or simple sum
            if self.aggregator is not None:
                encoder_input = self.aggregator(R, k)  # [batch, 1, hidden_dim]
            else:
                encoder_input = F.relu(R + k)

        # Produce latent distribution parameters
        q_z_stats = self.encoder(encoder_input)

        return q_z_stats

    def get_knowledge_embedding(self, knowledge):
        """Get knowledge embedding for interpretability experiments."""
        if self.knowledge_encoder is None:
            batch_size = knowledge.shape[0] if isinstance(knowledge, torch.Tensor) else 1
            device = knowledge.device if isinstance(knowledge, torch.Tensor) else 'cpu'
            return torch.zeros((batch_size, 1, self.knowledge_dim)).to(device)

        k = self.knowledge_encoder(knowledge)
        if self.knowledge_proj is not None:
            k = self.knowledge_proj(k)
        return k

    def get_gating_alpha(self):
        """Get the last gating alpha for interpretability."""
        if self.aggregator is not None:
            return self.aggregator.get_alpha()
        return None


class NSINP(nn.Module):
    """
    Neuro-Symbolic Informed Neural Process.

    Same interface as INP class for compatibility with existing experiments.
    Uses symbolic equation encoding and conflict-aware gating instead of
    text/set embedding with additive aggregation.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.xy_encoder = XYEncoder(config)
        self.latent_encoder = NSLatentEncoder(config)
        self.decoder = Decoder(config)
        self.x_encoder = XEncoder(config)
        self.train_num_z_samples = config.train_num_z_samples
        self.test_num_z_samples = config.test_num_z_samples

    def forward(self, x_context, y_context, x_target, y_target, knowledge=None, true_params=None):
        """
        Forward pass with same signature as INP.

        Args:
            x_context: [batch, num_context, input_dim]
            y_context: [batch, num_context, output_dim]
            x_target: [batch, num_target, input_dim]
            y_target: [batch, num_target, output_dim] or None
            knowledge: [batch, seq_len] tokenized equation or None
            true_params: [batch, 3] true (a,b,c) params for auxiliary loss (optional)

        Returns:
            p_yCc: predicted distribution
            z_samples: sampled latent variables
            q_z_Cc: prior distribution
            q_zCct: posterior distribution (training only)
        """
        # Compute auxiliary losses FIRST with a dedicated forward pass
        # This must happen before any other operations to avoid graph conflicts
        aux_loss = None
        contrastive_loss = None
        if (true_params is not None and
            knowledge is not None and
            self.latent_encoder.knowledge_encoder is not None and
            self.training):
            # Encode knowledge and predict params in one go
            encoder = self.latent_encoder.knowledge_encoder
            k_embed = encoder(knowledge)  # [batch, 1, d_model]
            k_embed_flat = k_embed.squeeze(1)  # [batch, d_model]

            # 1. Parameter prediction loss (prevents collapse by forcing encoder to preserve info)
            pred_params = encoder.predict_params(k_embed_flat)  # [batch, 3]
            aux_loss = F.mse_loss(pred_params, true_params)

            # 2. Contrastive loss (pushes different equations apart)
            # Use InfoNCE-style contrastive loss
            batch_size = k_embed_flat.shape[0]
            if batch_size > 1:
                # Normalize embeddings for cosine similarity
                k_norm = F.normalize(k_embed_flat, dim=1)
                # Similarity matrix
                sim_matrix = torch.mm(k_norm, k_norm.t())  # [batch, batch]
                # Temperature scaling
                temperature = 0.1
                sim_matrix = sim_matrix / temperature
                # Labels: each sample is its own class (diagonal should be highest)
                labels = torch.arange(batch_size, device=sim_matrix.device)
                # Cross entropy loss (pushes diagonal up, off-diagonal down)
                contrastive_loss = F.cross_entropy(sim_matrix, labels)

        x_context = self.x_encoder(x_context)
        x_target = self.x_encoder(x_target)

        R = self.encode_globally(x_context, y_context, x_target)

        z_samples, q_z_Cc, q_zCct = self.sample_latent(
            R, x_context, x_target, y_target, knowledge
        )

        R_target = self.target_dependent_representation(R, x_target, z_samples)

        p_yCc = self.decode_target(x_target, R_target)

        # Store auxiliary losses for retrieval
        self._aux_loss = aux_loss
        self._contrastive_loss = contrastive_loss

        return p_yCc, z_samples, q_z_Cc, q_zCct

    def encode_globally(self, x_context, y_context, x_target):
        """Encode context set all together."""
        R = self.xy_encoder(x_context, y_context, x_target)

        if x_context.shape[1] == 0:
            R = torch.zeros((R.shape[0], 1, R.shape[-1])).to(R.device)

        return R

    def get_knowledge_embedding(self, knowledge):
        """Get knowledge embedding for interpretability experiments."""
        return self.latent_encoder.get_knowledge_embedding(knowledge)

    def get_gating_alpha(self):
        """Get the gating alpha for interpretability experiments."""
        return self.latent_encoder.get_gating_alpha()

    def sample_latent(self, R, x_context, x_target, y_target, knowledge):
        """Sample latent variable z given the global representation."""
        q_zCc = self.infer_latent_dist(R, knowledge, x_context.shape[1])

        if y_target is not None and self.training:
            R_from_target = self.encode_globally(x_target, y_target, x_target)
            q_zCct = self.infer_latent_dist(R_from_target, knowledge, x_target.shape[1])
            sampling_dist = q_zCct
        else:
            q_zCct = None
            sampling_dist = q_zCc

        if self.training:
            z_samples = sampling_dist.rsample([self.train_num_z_samples])
        else:
            z_samples = sampling_dist.rsample([self.test_num_z_samples])

        return z_samples, q_zCc, q_zCct

    def infer_latent_dist(self, R, knowledge, n):
        """Infer the latent distribution given the global representation."""
        q_z_stats = self.latent_encoder(R, knowledge, n)
        q_z_loc, q_z_scale = q_z_stats.split(self.config.hidden_dim, dim=-1)
        q_z_scale = 0.01 + 0.99 * F.softplus(q_z_scale)
        q_zCc = MultivariateNormalDiag(q_z_loc, q_z_scale)
        return q_zCc

    def target_dependent_representation(self, R, x_target, z_samples):
        """Compute the target dependent representation of the context set."""
        R_target = z_samples
        R_target = R_target.expand(-1, -1, x_target.shape[1], -1)
        return R_target

    def decode_target(self, x_target, R_target):
        """Decode the target set given the target dependent representation."""
        p_y_stats = self.decoder(x_target, R_target)
        p_y_loc, p_y_scale = p_y_stats.split(self.config.output_dim, dim=-1)
        p_y_scale = 0.1 + 0.9 * F.softplus(p_y_scale)
        p_yCc = MultivariateNormalDiag(p_y_loc, p_y_scale)
        return p_yCc

    def get_auxiliary_loss(self):
        """
        Get the auxiliary loss computed during the forward pass.

        Returns:
            aux_loss: scalar tensor or None
        """
        return getattr(self, '_aux_loss', None)

    def get_contrastive_loss(self):
        """
        Get the contrastive loss computed during the forward pass.

        Returns:
            contrastive_loss: scalar tensor or None
        """
        return getattr(self, '_contrastive_loss', None)

    def get_param_predictions(self, knowledge):
        """Get parameter predictions for given knowledge."""
        if self.latent_encoder.knowledge_encoder is None:
            return None
        encoder = self.latent_encoder.knowledge_encoder
        k_embed = encoder(knowledge)
        return encoder.predict_params(k_embed.squeeze(1))


if __name__ == "__main__":
    from argparse import Namespace

    # Test configuration
    config = Namespace(
        # model
        input_dim=1,
        output_dim=1,
        xy_encoder_num_hidden=2,
        xy_encoder_hidden_dim=384,
        data_agg_func="mean",
        latent_encoder_num_hidden=2,
        decoder_hidden_dim=128,
        decoder_num_hidden=2,
        decoder_activation="gelu",
        hidden_dim=128,
        x_transf_dim=128,
        x_encoder_num_hidden=1,
        test_num_z_samples=16,
        train_num_z_samples=1,
        knowledge_dropout=0.3,
        knowledge_dim=128,
        use_knowledge=True,
        # NS-INP specific
        model_type="nsinp",
        equation_vocab_size=64,
        equation_max_len=50,
        use_gating=True,
        gating_hidden_dim=64,
    )
    config.device = "cpu"

    # Create model
    model = NSINP(config)
    print(f"Model created successfully")
    print(f"Has latent_encoder: {hasattr(model, 'latent_encoder')}")
    print(f"Has knowledge_encoder: {hasattr(model.latent_encoder, 'knowledge_encoder')}")
    print(f"Has get_knowledge_embedding: {callable(getattr(model, 'get_knowledge_embedding', None))}")

    # Test forward pass
    batch_size = 4
    num_context = 10
    num_target = 50
    seq_len = 30

    x_context = torch.randn(batch_size, num_context, 1)
    y_context = torch.randn(batch_size, num_context, 1)
    x_target = torch.randn(batch_size, num_target, 1)
    y_target = torch.randn(batch_size, num_target, 1)
    knowledge = torch.randint(1, 64, (batch_size, seq_len))  # Token IDs

    model.train()
    p_yCc, z_samples, q_z_Cc, q_zCct = model(x_context, y_context, x_target, y_target, knowledge)

    print(f"\nForward pass successful:")
    print(f"  p_yCc mean shape: {p_yCc.mean.shape}")
    print(f"  z_samples shape: {z_samples.shape}")
    print(f"  Gating alpha: {model.get_gating_alpha().mean().item():.3f}")

    # Test knowledge embedding
    k_embed = model.get_knowledge_embedding(knowledge)
    print(f"  Knowledge embedding shape: {k_embed.shape}")
