import matplotlib.pyplot as plt
import json
import torch
from train import TrainingConfig

with open(TrainingConfig.loss_save_path, 'r') as f:
    data = json.load(f)

training_losses = data['training_losses']

plt.figure(figsize=(10,6))
x_values = torch.arange(0, len(training_losses))
plt.plot(x_values, training_losses, label='Training Loss', marker='o', linestyle='-', alpha=0.7)
plt.xlabel("Batch Number")
plt.ylabel("Loss")
plt.xticks(ticks=x_values[::1000])
plt.legend()
plt.grid(False)
plt.show()