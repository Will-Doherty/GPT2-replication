import tiktoken
from dataclasses import dataclass
import torch

@dataclass
class TrainingConfig:
    max_seq_len = 1024
    minibatch_size = 32
    model_save_path = 'model.pth'
    loss_save_path = 'losses.json'
    weight_decay = 0.1
    warmup_steps = 2000
    cooldown_frac = 0.6
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_dataloader_workers = 0
    checkpoint_interval = 5000  # number of steps before saving
    muon_momentum_warmup_steps = 1e5
    print_loss_stride = 10
    momentum_coef1 = 0.85
    momentum_coef2 = 0.95
    
@dataclass    
class ModelConfig:
    vocab_size = 50304
    d_model = 768
    num_heads = 12
    d_mlp = 3072
    num_layers = 12
    init_std_dev = 0.02
    max_seq_len = 1024
    tokenizer = tiktoken.get_encoding("gpt2")
    sep_id = 50256