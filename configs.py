import tiktoken
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    training_batch_size = 16
    learning_rate = 1e-4
    num_epochs = 1
    model_save_path = 'model_weights_and_results/model.pth'
    loss_save_path = 'model_weights_and_results/losses.json'
    weight_decay = 0
    
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