import tiktoken
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    max_seq_len = 1024
    minibatch_size = 16
    total_batch_size = 2 ** 19  # from gpt-3 paper
    grad_accum_steps = total_batch_size // (minibatch_size * max_seq_len)
    device = "cuda"
    max_learning_rate = 6e-4
    min_learning_rate = max_learning_rate * 0.1
    warmup_tokens = 1e9  # 100m warmup tokens i.e. 1% of total training dataset
    warmup_steps = warmup_tokens // total_batch_size
    total_steps = 1e11 // total_batch_size
    num_epochs = 1
    model_save_path = 'model.pth'
    loss_save_path = 'losses.json'
    weight_decay = 0.1
    
@dataclass
class ModelConfig:
    vocab_size = 50304
    max_seq_len = 1024
    d_model = 768
    num_heads = 8
    d_mlp = 3072
    num_layers = 12
    init_std_dev = 0.02
    tokenizer = tiktoken.get_encoding("gpt2")
    device = "cuda"
