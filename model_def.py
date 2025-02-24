import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from configs import ModelConfig

class Embed(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        nn.init.normal_(self.embedding.weight, std=cfg.init_std_dev)

    def forward(self, tokens):
        return self.embedding(tokens)

class PositionalEncoding(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.pos_embedding = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        nn.init.normal_(self.pos_embedding.weight, std=cfg.init_std_dev)

    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        pos_embed = self.pos_embedding(positions)
        return pos_embed

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

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True) # use flash attention
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        return self.out_proj(attn_output)

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
        return F.linear(resid_post_mlp, self.weight) # no weights because they are tied to the embedding weights

class TransformerLayer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn = Attention(cfg)
        self.mlp = MLP(cfg)

    def forward_hook(self, module, inp, out):
        self.captured_activations = out.detach()

    def set_activation_hook(self, enable=False):
        if enable:
            self.hook_handle = self.register_forward_hook(self.forward_hook)

    def forward(self, resid):
        resid_after_attn = resid + self.attn(resid)
        resid_after_mlp = resid_after_attn + self.mlp(resid_after_attn)
        return resid_after_mlp

class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = Embed(cfg)
        self.pos_encoding = PositionalEncoding(cfg)
        self.layers = nn.ModuleList([TransformerLayer(cfg) for _ in range(cfg.num_layers)])
        self.unembed = Unembed(cfg)
        self.unembed.weight = self.embed.embedding.weight

    def forward(self, tokens):
        x = self.embed(tokens)
        x = x + self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x)
        logits = self.unembed(x)
        return logits