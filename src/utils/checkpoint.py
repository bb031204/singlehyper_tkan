import os
import torch
from typing import Optional


def save_checkpoint(state: dict, checkpoint_dir: str, filename: str = 'last.pt', is_best: bool = False):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, filename)
    torch.save(state, path)
    torch.save(state, os.path.join(checkpoint_dir, 'last.pt'))
    if is_best:
        torch.save(state, os.path.join(checkpoint_dir, 'best_model.pt'))


def load_checkpoint(path: str, model, optimizer=None, scheduler=None, device='cpu'):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer is not None and ckpt.get('optimizer_state_dict') is not None:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scheduler is not None and ckpt.get('scheduler_state_dict') is not None:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    return ckpt


def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    if not os.path.exists(checkpoint_dir):
        return None
    best = os.path.join(checkpoint_dir, 'best_model.pt')
    last = os.path.join(checkpoint_dir, 'last.pt')
    if os.path.exists(best):
        return best
    if os.path.exists(last):
        return last
    pts = [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
    return max(pts, key=os.path.getmtime) if pts else None
