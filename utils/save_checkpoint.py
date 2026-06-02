import os
import torch


def save_checkpoint(state, save_dir, filename):
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, filename)
    print(f"start saving model to {checkpoint_path}")
    torch.save(state, checkpoint_path)
    return checkpoint_path
