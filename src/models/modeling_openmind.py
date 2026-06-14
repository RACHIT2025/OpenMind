"""
OpenMind Transformer Model Architecture.

A decoder-only transformer (GPT-style) with:
- Rotary Positional Embeddings (RoPE)
- RMSNorm (pre-norm architecture)
- SwiGLU activation in feed-forward network
- Grouped Query Attention (GQA) support
- KV-Cache for efficient autoregressive inference
- Hugging Face save/load compatibility
"""

import math
import json
import os
from dataclasses import asdict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .config_openmind import OpenMindConfig
except ImportError:
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from config_openmind import OpenMindConfig


# ─── RMSNorm ──────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


# ─── Rotary Positional Embeddings ─────────────────────────────────────────────

class RotaryEmbedding(nn.Module):
    """Rotary Positional Embeddings (Su et al., 2021)."""

    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        # Precompute frequency matrix
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute cos/sin cache
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int, offset: int = 0):
        if seq_len + offset > self.max_seq_len:
            self._build_cache(seq_len + offset)
        cos = self.cos_cached[offset: offset + seq_len]
        sin = self.sin_cached[offset: offset + seq_len]
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary positional embeddings to query and key tensors."""
    # cos/sin shape: (seq_len, head_dim) -> broadcast to (1, 1, seq_len, head_dim)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# ─── Grouped Query Attention ──────────────────────────────────────────────────

class GroupedQueryAttention(nn.Module):
    """
    Multi-Head Attention with optional Grouped Query Attention (GQA).

    When n_kv_heads < n_heads, key/value heads are shared across groups,
    reducing memory and compute for KV cache during inference.
    """

    def __init__(self, config: OpenMindConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.n_rep = self.n_heads // self.n_kv_heads  # Number of query heads per KV head

        self.q_proj = nn.Linear(config.dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, config.dim, bias=False)

        self.dropout = nn.Dropout(config.dropout)
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_seq_len=config.max_seq_len,
            theta=config.rope_theta,
        )

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """Repeat KV heads to match number of query heads (for GQA)."""
        if self.n_rep == 1:
            return x
        bs, n_kv, seq_len, head_dim = x.shape
        x = x[:, :, None, :, :].expand(bs, n_kv, self.n_rep, seq_len, head_dim)
        return x.reshape(bs, self.n_heads, seq_len, head_dim)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        bsz, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Determine offset for RoPE (for KV cache continuation)
        kv_seq_len = seq_len
        offset = 0
        if past_key_value is not None:
            offset = past_key_value[0].shape[-2]
            kv_seq_len += offset

        # Apply rotary embeddings
        cos, sin = self.rotary_emb(q, seq_len, offset=offset)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Handle KV cache
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)

        new_cache = (k, v) if use_cache else None

        # Expand KV heads for GQA
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)

        # Apply causal mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, new_cache


# ─── SwiGLU Feed-Forward Network ──────────────────────────────────────────────

class SwiGLU(nn.Module):
    """SwiGLU activation function (Shazeer, 2020) with gated linear unit."""

    def __init__(self, config: OpenMindConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.dim, config.intermediate_dim, bias=False)
        self.up_proj = nn.Linear(config.dim, config.intermediate_dim, bias=False)
        self.down_proj = nn.Linear(config.intermediate_dim, config.dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


# ─── Transformer Block ────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """Single transformer decoder block with pre-norm architecture."""

    def __init__(self, config: OpenMindConfig):
        super().__init__()
        self.attention_norm = RMSNorm(config.dim)
        self.attention = GroupedQueryAttention(config)
        self.ffn_norm = RMSNorm(config.dim)
        self.feed_forward = SwiGLU(config)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        # Pre-norm attention with residual
        residual = x
        x = self.attention_norm(x)
        x, cache = self.attention(x, attention_mask, position_ids, past_key_value, use_cache)
        x = residual + x

        # Pre-norm feed-forward with residual
        residual = x
        x = self.ffn_norm(x)
        x = self.feed_forward(x)
        x = residual + x

        return x, cache


# ─── Full Model ───────────────────────────────────────────────────────────────

class OpenMindModel(nn.Module):
    """
    OpenMind Decoder-Only Transformer Language Model.

    Architecture: GPT-style with RoPE, RMSNorm, SwiGLU, and optional GQA.
    """

    def __init__(self, config: OpenMindConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.dim)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        # Final normalization
        self.norm = RMSNorm(config.dim)

        # Language model head
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

        # Tie embeddings
        if config.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        # Initialize weights
        self.apply(self._init_weights)

        # Report parameter count
        n_params = sum(p.numel() for p in self.parameters())
        n_params_non_embed = n_params - self.embed_tokens.weight.numel()
        print(f"OpenMind Model initialized:")
        print(f"  Total parameters: {n_params:,}")
        print(f"  Non-embedding parameters: {n_params_non_embed:,}")

    def _init_weights(self, module: nn.Module):
        """Initialize weights using GPT-2 style initialization."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _make_causal_mask(
        self,
        seq_len: int,
        dtype: torch.dtype,
        device: torch.device,
        past_len: int = 0,
    ) -> torch.Tensor:
        """Create causal attention mask."""
        mask = torch.full(
            (seq_len, seq_len + past_len), float("-inf"), dtype=dtype, device=device
        )
        mask = torch.triu(mask, diagonal=past_len + 1)
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, total_len)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Forward pass.

        Args:
            input_ids: Token IDs, shape (batch, seq_len)
            attention_mask: Optional mask
            position_ids: Optional position indices
            past_key_values: Optional KV cache from previous steps
            use_cache: Whether to return updated KV cache
            labels: Optional labels for computing loss (shifted internally)

        Returns:
            Dictionary with 'logits', 'loss' (if labels provided), 'past_key_values'
        """
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        # Determine past length for KV cache
        past_len = 0
        if past_key_values is not None and past_key_values[0] is not None:
            past_len = past_key_values[0][0].shape[-2]

        # Embed tokens
        h = self.embed_tokens(input_ids)

        # Create causal mask
        causal_mask = self._make_causal_mask(seq_len, h.dtype, device, past_len)

        # Pass through transformer layers
        new_caches = []
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None
            h, cache = layer(h, causal_mask, position_ids, past_kv, use_cache)
            new_caches.append(cache)

        # Final norm and LM head
        h = self.norm(h)
        logits = self.lm_head(h)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift logits and labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return {
            "logits": logits,
            "loss": loss,
            "past_key_values": new_caches if use_cache else None,
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        eos_token_id: int = 0,
        do_sample: bool = True,
        repetition_penalty: float = 1.0,
    ) -> torch.Tensor:
        """
        Autoregressive text generation with KV-cache.

        Args:
            input_ids: Starting token IDs, shape (batch, prefix_len)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (1.0 = neutral)
            top_k: Top-k filtering (0 = disabled)
            top_p: Nucleus sampling threshold (1.0 = disabled)
            eos_token_id: Token ID that signals end of generation
            do_sample: If False, use greedy decoding
            repetition_penalty: Repetition penalty (1.0 = disabled)

        Returns:
            Generated token IDs including the input prefix
        """
        self.eval()
        past_key_values = [None] * self.config.n_layers
        generated = input_ids

        for _ in range(max_new_tokens):
            # Only feed the last token if we have KV cache
            if past_key_values[0] is not None:
                curr_input = generated[:, -1:]
            else:
                curr_input = generated

            outputs = self.forward(
                curr_input,
                past_key_values=past_key_values,
                use_cache=True,
            )

            logits = outputs["logits"][:, -1, :]
            past_key_values = outputs["past_key_values"]

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for i in range(logits.shape[0]):
                    for token_id in set(generated[i].tolist()):
                        logit = logits[i, token_id].item()
                        if logit < 0:
                            logits[i, token_id] = logit * repetition_penalty
                        else:
                            logits[i, token_id] = logit / repetition_penalty

            if do_sample:
                # Apply temperature
                logits = logits / max(temperature, 1e-8)

                # Top-k filtering
                if top_k > 0:
                    top_k_vals = torch.topk(logits, min(top_k, logits.size(-1)))
                    indices_to_remove = logits < top_k_vals.values[..., -1, None]
                    logits[indices_to_remove] = float("-inf")

                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(
                        F.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = float("-inf")

                # Sample
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=-1)

            # Stop if EOS token generated (for all sequences in batch)
            if (next_token == eos_token_id).all():
                break

        return generated

    def count_parameters(self) -> dict:
        """Count model parameters by component."""
        counts = {}
        counts["embedding"] = self.embed_tokens.weight.numel()
        counts["attention"] = sum(
            p.numel() for layer in self.layers
            for p in layer.attention.parameters()
        )
        counts["ffn"] = sum(
            p.numel() for layer in self.layers
            for p in layer.feed_forward.parameters()
        )
        counts["norm"] = sum(
            p.numel() for layer in self.layers
            for p in layer.attention_norm.parameters()
        ) + sum(
            p.numel() for layer in self.layers
            for p in layer.ffn_norm.parameters()
        ) + sum(p.numel() for p in self.norm.parameters())
        counts["lm_head"] = 0 if self.config.tie_embeddings else self.lm_head.weight.numel()
        counts["total"] = sum(p.numel() for p in self.parameters())
        return counts

    def save_pretrained(self, output_dir: str) -> None:
        """Save model weights and config to directory."""
        os.makedirs(output_dir, exist_ok=True)

        # Save config
        self.config.save_pretrained(output_dir)

        # Save model weights
        model_path = os.path.join(output_dir, "model.safetensors")
        try:
            from safetensors.torch import save_file
            save_file(self.state_dict(), model_path)
        except ImportError:
            model_path = os.path.join(output_dir, "pytorch_model.bin")
            torch.save(self.state_dict(), model_path)

        print(f"Model saved to {output_dir}/")

    @classmethod
    def from_pretrained(cls, model_dir: str, device: str = "cpu") -> "OpenMindModel":
        """Load model weights and config from directory."""
        config = OpenMindConfig.from_pretrained(model_dir)
        model = cls(config)

        # Try safetensors first, then PyTorch format
        safetensors_path = os.path.join(model_dir, "model.safetensors")
        pytorch_path = os.path.join(model_dir, "pytorch_model.bin")
        pt_path = os.path.join(model_dir, "model.pt")

        if os.path.exists(safetensors_path):
            from safetensors.torch import load_file
            state_dict = load_file(safetensors_path)
        elif os.path.exists(pytorch_path):
            state_dict = torch.load(pytorch_path, map_location=device)
        elif os.path.exists(pt_path):
            state_dict = torch.load(pt_path, map_location=device)
        else:
            raise FileNotFoundError(f"No model weights found in {model_dir}")

        model.load_state_dict(state_dict)
        model = model.to(device)

        print(f"Model loaded from {model_dir}/")
        return model
