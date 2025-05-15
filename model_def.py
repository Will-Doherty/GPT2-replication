import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from configs import ModelConfig
import tiktoken
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

class Embed(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        nn.init.normal_(self.embedding.weight, std=cfg.init_std_dev)

    def forward(self, tokens):
        return self.embedding(tokens)

class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.d_model // cfg.num_heads
        assert self.head_dim * cfg.num_heads == cfg.d_model, "d_model must be divisible by num_heads"

        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=True)

        nn.init.normal_(self.q_proj.weight, std=cfg.init_std_dev)
        nn.init.normal_(self.k_proj.weight, std=cfg.init_std_dev)
        nn.init.normal_(self.v_proj.weight, std=cfg.init_std_dev)
        nn.init.normal_(self.out_proj.weight, std=cfg.init_std_dev)

    def forward(self, x, block_mask):
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        sin, cos = self._get_rotary_embeddings(seq_len, self.head_dim, x.device, dtype=x.dtype)
        q, k = self._apply_rope(q, k, sin, cos)

        # attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_output = flex_attention(q, k, v, block_mask=block_mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.out_proj(attn_output)

    def _get_rotary_embeddings(self, seq_len, head_dim, device, *, dtype, base=1024):
        """
        Create sin and cos matrices for RoPE.
        """
        # half-truncated RoPE is used
        # base has been tuned
        inv_freq = (1.0 / base) ** torch.linspace(0.0, 1.0, steps=head_dim // 4, device=device, dtype=dtype)
        inv_freq = torch.cat([inv_freq, inv_freq.new_zeros(head_dim // 4)])
        positions = torch.arange(seq_len, device=inv_freq.device, dtype=inv_freq.dtype)
        sinusoid_input = torch.einsum("i,j->ij", positions, inv_freq)

        # applying sin and cos to 0 (in the second half of the dimensions) results in the identity transformation
        sin = nn.Buffer(sinusoid_input.sin(), persistent=False)
        cos = nn.Buffer(sinusoid_input.cos(), persistent=False)
        return sin, cos

    def _apply_rope(self, q, k, sin, cos):
        sin, cos = sin[None, None, :, :].to(q.device), cos[None, None, :, :].to(q.device)
        q1, q2 = q.chunk(2, dim=-1)
        k1, k2 = k.chunk(2, dim=-1)
        q_rotated = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
        k_rotated = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)
        return q_rotated, k_rotated

class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.mlp_layer1 = nn.Linear(cfg.d_model, cfg.d_mlp)
        self.mlp_layer2 = nn.Linear(cfg.d_mlp, cfg.d_model)

        nn.init.normal_(self.mlp_layer1.weight, std=cfg.init_std_dev)
        nn.init.normal_(self.mlp_layer2.weight, std=cfg.init_std_dev)

    def forward(self, resid):
        resid_mlp_hidden = self.mlp_layer1(resid)
        mlp_hidden_activations = F.gelu(resid_mlp_hidden)
        resid_post_mlp = self.mlp_layer2(mlp_hidden_activations)
        return resid_post_mlp

class Unembed(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

    def forward(self, resid_post_mlp):
        return F.linear(resid_post_mlp, self.weight)  # tied weights to the embedding weights

class TransformerLayer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn = Attention(cfg)
        self.mlp = MLP(cfg)

    def forward(self, resid, block_mask):
        resid_after_norm = F.rms_norm(resid, (resid.size(-1),))
        resid_after_attn = resid_after_norm + self.attn(resid_after_norm, block_mask)
        resid_after_mlp = resid_after_attn + self.mlp(resid_after_attn)
        return resid_after_mlp

class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = Embed(cfg)
        self.layers = nn.ModuleList([TransformerLayer(cfg) for _ in range(cfg.num_layers)])
        self.unembed = Unembed(cfg)

    def forward(self, tokens):
        B, S = tokens.shape
        device = tokens.device
        doc_id = (tokens == self.cfg.sep_id).cumsum(dim=1).to(device)

        def mask_mod(b, h, q_idx, k_idx):
            return (q_idx >= k_idx) & (doc_id[b, q_idx] == doc_id[b, k_idx])

        block_mask = create_block_mask(
            mask_mod,
            B=B,
            H=None,
            Q_LEN=S,
            KV_LEN=S,
            device=device
        )

        x = self.embed(tokens)
        for layer in self.layers:
            x = layer(x, block_mask)
        return F.linear(x, self.embed.embedding.weight)