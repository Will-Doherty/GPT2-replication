import torch
import torch.nn.functional as F
import json
from model_def import ModelConfig, Transformer
from load_data import get_data
from configs import TrainingConfig
import math
from statistics import mean
from torch.optim.lr_scheduler import LambdaLR

def save_state_dict(model, optimizer):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, TrainingConfig.model_save_path)
    print(f"Model saved to {TrainingConfig.model_save_path}")

def save_losses(training_losses, validation_losses):
    with open(TrainingConfig.loss_save_path, 'w') as f:
        json.dump({
            'training_losses': training_losses,
            'validation_losses': validation_losses
        }, f)
    print(f"Losses saved to {TrainingConfig.loss_save_path}")

def train_model(model_cfg, training_cfg):
    torch_dataset = get_data()
    batched_dataset = torch_dataset.batch(batch_size=training_cfg.training_batch_size)

    device = model_cfg.device
    torch.set_float32_matmul_precision('medium')

    torch.manual_seed(123)
    torch.cuda.manual_seed(123)

    model = Transformer(model_cfg)
    model.to(device)
    model = torch.compile(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_cfg.max_learning_rate,
        betas=(0.9, 0.95),
        weight_decay=training_cfg.weight_decay
    )

    def lr_lambda(current_step):
        if current_step <= training_cfg.warmup_steps:
            warmup_ratio = current_step / float(training_cfg.warmup_steps)
            lr = training_cfg.min_learning_rate + (training_cfg.max_learning_rate - training_cfg.min_learning_rate) * warmup_ratio
        else:
            progress = (current_step - training_cfg.warmup_steps) / float(training_cfg.total_steps - training_cfg.warmup_steps)
            lr = (training_cfg.min_learning_rate
                  + 0.5 * (training_cfg.max_learning_rate - training_cfg.min_learning_rate)
                  * (1 + math.cos(math.pi * progress)))
        return lr / training_cfg.max_learning_rate

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    model.train()
    training_losses = []
    validation_losses = []
    accumulation_counter = 0
    current_step = 0

    for i, dset_dict in enumerate(batched_dataset):
        max_input_len = max(tensor.size(0) for tensor in dset_dict['input_ids'])
        inputs = [torch.cat([tensor, torch.full([max_input_len - tensor.size(0)], 50256)])[:model_cfg.max_seq_len] for tensor in dset_dict['input_ids']]
        max_target_len = max(tensor.size(0) for tensor in dset_dict['labels'])
        targets = [torch.cat([tensor, torch.full([max_target_len - tensor.size(0)], 50256)])[:model_cfg.max_seq_len] for tensor in dset_dict['labels']]
        x = torch.stack(inputs)
        y = torch.stack(targets)
        x, y = x.to(device), y.to(device)

        logits = model(x)
        raw_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        training_losses.append(raw_loss.item())

        loss = raw_loss / training_cfg.grad_accum_steps
        loss.backward()
        accumulation_counter += 1

        if accumulation_counter % training_cfg.grad_accum_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            current_step += 1

        if i > 0 and i % 10 == 0:
            recent_mean_loss = mean(training_losses[-10:])
            print(f"Minibatch {i}: training loss (avg of last 10) = {recent_mean_loss:.4f}")

    save_state_dict(model, optimizer, training_cfg)
    save_losses(training_losses, validation_losses, training_cfg)

    return model, training_losses, validation_losses, optimizer


if __name__ == '__main__':
    train_model(ModelConfig(), TrainingConfig())
