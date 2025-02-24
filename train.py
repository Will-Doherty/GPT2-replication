import torch
import torch.nn.functional as F
import json
from model_def import ModelConfig, Transformer
from load_data import get_data
from configs import TrainingConfig
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast

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
        lr=training_cfg.learning_rate, 
        weight_decay=training_cfg.weight_decay
    )

    validation_losses = []
    training_losses = []

    model.train()

    for step, dset_dict in enumerate(batched_dataset):
        max_input_len = max(tensor.size(0) for tensor in dset_dict['input_ids'])
        inputs = [torch.cat([tensor, torch.full([max_input_len - tensor.size(0)], 50256)])[:model_cfg.max_seq_len] for tensor in dset_dict['input_ids']]
        max_target_len = max(tensor.size(0) for tensor in dset_dict['labels'])
        targets = [torch.cat([tensor, torch.full([max_target_len - tensor.size(0)], 50256)])[:model_cfg.max_seq_len] for tensor in dset_dict['labels']]
        x = torch.stack(inputs)
        y = torch.stack(targets)
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        if step % 10 != 0 or step == 0:
            if step % 10 == 9:
                training_losses.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        else:
            validation_losses.append(loss.item())
            print(f"Step {step}: training loss = {training_losses[-1]}, validation loss = {loss.item()}")

        if step > 0 and step % 30 == 0:
            break

    save_state_dict(model, optimizer)
    save_losses(training_losses, validation_losses)

    return model, training_losses, validation_losses, optimizer


if __name__ == '__main__':
    train_model(ModelConfig(), TrainingConfig())
