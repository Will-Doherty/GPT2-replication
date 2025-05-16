import torch
import torch.nn.functional as F
import json
from load_data import get_data
from configs import TrainingConfig, ModelConfig
import math
from statistics import mean
import time
from optimizers import Muon
import os
import torch.distributed as dist
from torch.utils.data import DataLoader
from model_def import Transformer
from torch.amp import autocast, GradScaler
import argparse
import torch._dynamo

def get_rank_and_world_size():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1

def save_losses(training_losses, training_cfg):
    with open(training_cfg.loss_save_path, 'w') as f:
        json.dump({
            'training_losses': training_losses,
        }, f)
    print(f"Losses saved to {training_cfg.loss_save_path}")

def train_model(model_cfg, training_cfg):
    local_rank = int(os.environ["LOCAL_RANK"])
    local_device = torch.device("cuda", local_rank)
    print(f"Using {local_device}")

    model = Transformer(model_cfg)

    model.to(local_device).bfloat16()
    torch._dynamo.config.suppress_errors = True
    model = torch.compile(model, dynamic=False)

    multi = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if multi and not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    
    if dist.is_initialized():
        dist.barrier()
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)

    core = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    rank, world_size = get_rank_and_world_size()
    hidden_matrix_params = [p for p in core.layers.parameters() if p.ndim == 2]
    scalar_params = [p for p in core.layers.parameters() if p.ndim == 1]
    embed_params = [p for n, p in core.named_parameters() if "embed" in n]
    adam_params = [dict(params=embed_params, lr=0.005 * world_size),
                dict(params=scalar_params, lr=0.005 * world_size)
                ]
    adam_optimizer = torch.optim.Adam(adam_params, betas=(0.8, 0.95), eps=1e-10, fused=True)
    muon_optimizer = Muon(hidden_matrix_params, lr=0.005 * world_size, momentum=0.95, rank=rank, world_size=world_size, device=local_device)
    optimizers = [adam_optimizer, muon_optimizer]
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]

    torch.set_float32_matmul_precision("medium")

    torch.cuda.set_device(local_rank)
    
    def lr_mult_adam(step: int, total_steps: int) -> float:
        if step < training_cfg.warmup_steps:
            return step / training_cfg.warmup_steps
        progress = step / total_steps
        if progress < 1 - training_cfg.cooldown_frac:
            return 1.0
        w = (1 - progress) / training_cfg.cooldown_frac
        return w + (1 - w) * 0.1

    def lr_mult_muon(step: int, total_steps: int) -> float:
        progress = step / total_steps
        if progress < 1 - training_cfg.cooldown_frac:
            return 1.0
        w = (1 - progress) / training_cfg.cooldown_frac
        return w + (1 - w) * 0.1

    dset = get_data(rank, world_size)
    
    batched_dataset = DataLoader(
        dset,
        batch_size=training_cfg.minibatch_size,
        num_workers=training_cfg.num_dataloader_workers,
        pin_memory=True
    )

    scaler = GradScaler()
    model.train()
    training_losses = []
    start_time = time.time()
    for ministep, batch in enumerate(batched_dataset):
        x = batch["input_ids"].to(local_device, non_blocking=True)
        y = batch["labels"].to(local_device, non_blocking=True)
        with autocast(device_type=local_device.type, dtype=torch.bfloat16):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        training_losses.append(loss.item())
        scaler.scale(loss).backward()

        for group in adam_optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lr_mult_adam(ministep, training_cfg.total_steps)
        for group in muon_optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lr_mult_muon(ministep, training_cfg.total_steps)
        mu_frac = min(1.0, ministep / training_cfg.muon_momentum_warmup_steps)
        for group in muon_optimizer.param_groups:
            group["momentum"] = (1 - mu_frac) * training_cfg.momentum_coef1 + mu_frac * training_cfg.momentum_coef2

        for opt in optimizers:
            scaler.step(opt)
        scaler.update()
        model.zero_grad(set_to_none=True)

        if ministep > 0 and ministep % training_cfg.print_loss_stride == 0:
            minibatch_time = (time.time() - start_time) / training_cfg.print_loss_stride
            recent_mean_loss = mean(training_losses[-training_cfg.print_loss_stride:])
            print(f"Minibatch {ministep}: training loss (avg of last {training_cfg.print_loss_stride}) = {recent_mean_loss:.4f}, time = {minibatch_time}")
            start_time = time.time()

        if recent_mean_loss < training_cfg.target_loss:
            break

    if rank == 0:
        save_losses(training_losses, training_cfg)


if __name__ == '__main__':
    train_model(ModelConfig(), TrainingConfig())
