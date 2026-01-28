# utils.py
import os
import json
import time
import random
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def get_device(device: Optional[str] = None) -> torch.device:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def now_str() -> str:
    return time.strftime("%Y%m%d-%H%M%S", time.localtime())


class AverageMeter:
 
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.sum = 0.0
        self.cnt = 0
        self.avg = 0.0

    def update(self, val: float, n: int = 1):
        self.val = float(val)
        self.sum += float(val) * n
        self.cnt += n
        self.avg = self.sum / max(1, self.cnt)


@torch.no_grad()
def accuracy_topk(logits: torch.Tensor, target: torch.Tensor, topk=(1,)) -> Dict[str, float]:
    """
    logits: [B, C], target: [B]
    returns dict like {"top1":0.83, "top5":0.95}
    """
    maxk = max(topk)
    B = target.size(0)

    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)  # [B, maxk]
    pred = pred.t()  # [maxk, B]
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    out = {}
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0).item()
        out[f"top{k}"] = correct_k / B
    return out


def save_checkpoint(
    save_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    epoch: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    ensure_dir(os.path.dirname(save_path) or ".")
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "extra": extra or {},
    }
    torch.save(payload, save_path)


def load_checkpoint(
    ckpt_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    map_location: str = "cpu",
) -> Tuple[int, Dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location=map_location)
    model.load_state_dict(ckpt["model"], strict=True)

    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])

    epoch = int(ckpt.get("epoch") or 0)
    extra = ckpt.get("extra") or {}
    return epoch, extra


class JSONLLogger:
  
    def __init__(self, log_path: str):
        ensure_dir(os.path.dirname(log_path) or ".")
        self.log_path = log_path

    def log(self, record: Dict[str, Any]) -> None:
        record = dict(record)
        record["_time"] = time.time()
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


class EMA:
    """
  
    """
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().clone()

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            assert name in self.shadow
            new_avg = (1.0 - self.decay) * p.detach() + self.decay * self.shadow[name]
            self.shadow[name] = new_avg.clone()

    @torch.no_grad()
    def apply_to(self, model: torch.nn.Module):
        self.backup = {}
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            self.backup[name] = p.detach().clone()
            p.data.copy_(self.shadow[name].data)

    @torch.no_grad()
    def restore(self, model: torch.nn.Module):
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            p.data.copy_(self.backup[name].data)
        self.backup = {}
