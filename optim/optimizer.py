# optim/optimizer.py
from __future__ import annotations

from typing import Dict, Any, Iterable, Optional
import torch


def _get_param_groups(
    model: torch.nn.Module,
    weight_decay: float,
    no_decay_keywords: tuple[str, ...] = ("bias", "bn", "norm", "ln", "layernorm"),
):
    """
    论文复现常用：对 norm/bias 不做 weight decay。
    - no_decay: bias / BatchNorm / LayerNorm 等
    - decay: 其他参数
    """
    decay_params = []
    no_decay_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        lname = name.lower()
        if (
            p.ndim == 1  # 
            or any(k in lname for k in no_decay_keywords)
        ):
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    param_groups = []
    if decay_params:
        param_groups.append({"params": decay_params, "weight_decay": weight_decay})
    if no_decay_params:
        param_groups.append({"params": no_decay_params, "weight_decay": 0.0})
    return param_groups


def build_optimizer(cfg: Dict[str, Any], model: torch.nn.Module) -> torch.optim.Optimizer:
    """
    cfg 对齐 configs/default.yaml:
      optimizer:
        name: adamw
        lr: 0.0003
        weight_decay: 0.05
        betas: [0.9, 0.999]
    """
    opt_cfg = cfg.get("optimizer", {})
    name = str(opt_cfg.get("name", "adamw")).lower()
    lr = float(opt_cfg.get("lr", 3e-4))
    weight_decay = float(opt_cfg.get("weight_decay", 0.0))


    param_groups = _get_param_groups(model, weight_decay=weight_decay)

    if name in ("adamw", "adam_w"):
        betas = opt_cfg.get("betas", [0.9, 0.999])
        eps = float(opt_cfg.get("eps", 1e-8))
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=lr,
            betas=(float(betas[0]), float(betas[1])),
            eps=eps,
        )
        return optimizer

    if name in ("adam",):
        betas = opt_cfg.get("betas", [0.9, 0.999])
        eps = float(opt_cfg.get("eps", 1e-8))
        optimizer = torch.optim.Adam(
            param_groups,
            lr=lr,
            betas=(float(betas[0]), float(betas[1])),
            eps=eps,
        )
        return optimizer

    if name in ("sgd",):
        momentum = float(opt_cfg.get("momentum", 0.9))
        nesterov = bool(opt_cfg.get("nesterov", True))
        optimizer = torch.optim.SGD(
            param_groups,
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
        )
        return optimizer

    raise ValueError(f"Unsupported optimizer name: {name}")
