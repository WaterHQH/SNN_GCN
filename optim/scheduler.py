# optim/scheduler.py
from __future__ import annotations

import math
from typing import Dict, Any, Optional, Tuple

import torch


class WarmupThenScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
   
    """
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        after_scheduler: torch.optim.lr_scheduler._LRScheduler,
        last_epoch: int = -1,
    ):
        self.warmup_steps = int(max(0, warmup_steps))
        self.after_scheduler = after_scheduler
        self.finished_warmup = (self.warmup_steps == 0)
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        if self.finished_warmup:
            return self.after_scheduler.get_last_lr()

       
        step = max(0, self.last_epoch) 
        scale = float(step + 1) / float(max(1, self.warmup_steps))
        return [base_lr * scale for base_lr in self.base_lrs]

    def step(self, epoch: Optional[int] = None):
        if self.finished_warmup:
           
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch)
            self._last_lr = self.after_scheduler.get_last_lr()
            return

        
        super().step(epoch)

        if self.last_epoch + 1 >= self.warmup_steps:
            self.finished_warmup = True
       
            self.after_scheduler.base_lrs = self.base_lrs
            self.after_scheduler.step(0)
            self._last_lr = self.after_scheduler.get_last_lr()


def build_scheduler(
    cfg: Dict[str, Any],
    optimizer: torch.optim.Optimizer,
    steps_per_epoch: int,
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    cfg 对齐 configs/default.yaml:
      scheduler:
        name: cosine
        warmup_epochs: 5
        min_lr: 0.000001

    约定：
    - scheduler.step() 以“每 step”调用（推荐），因此这里用 steps_per_epoch 推导总步数/暖启动步数
    """
    sch_cfg = cfg.get("scheduler", {})
    name = str(sch_cfg.get("name", "cosine")).lower()

 
    if name in ("none", "null", ""):
        return None

    warmup_epochs = float(sch_cfg.get("warmup_epochs", 0))
    warmup_steps = int(max(0, round(warmup_epochs * steps_per_epoch)))

 
    epochs = int(cfg.get("train", {}).get("epochs", 1))
    total_steps = max(1, epochs * steps_per_epoch)

    if name in ("cosine", "cosineannealing", "cosine_annealing"):
        min_lr = float(sch_cfg.get("min_lr", 1e-6))

  
        t_max = max(1, total_steps - warmup_steps)
        after = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=t_max,
            eta_min=min_lr,
        )
        if warmup_steps > 0:
            return WarmupThenScheduler(optimizer, warmup_steps=warmup_steps, after_scheduler=after)
        return after

    if name in ("step", "steplr"):
        step_size_epochs = int(sch_cfg.get("step_size", 30))
        gamma = float(sch_cfg.get("gamma", 0.1))
    
        step_size = max(1, step_size_epochs * steps_per_epoch)
        after = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        if warmup_steps > 0:
            return WarmupThenScheduler(optimizer, warmup_steps=warmup_steps, after_scheduler=after)
        return after

    if name in ("multistep", "multisteplr"):
        milestones_epochs = sch_cfg.get("milestones", [60, 80])
        gamma = float(sch_cfg.get("gamma", 0.1))
        milestones = sorted([max(1, int(m) * steps_per_epoch) for m in milestones_epochs])
        after = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        if warmup_steps > 0:
            return WarmupThenScheduler(optimizer, warmup_steps=warmup_steps, after_scheduler=after)
        return after

    if name in ("plateau", "reducelronplateau"):
      
        factor = float(sch_cfg.get("factor", 0.1))
        patience = int(sch_cfg.get("patience", 10))
        min_lr = float(sch_cfg.get("min_lr", 1e-6))
     
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",        
            factor=factor,
            patience=patience,
            min_lr=min_lr,
        )

    raise ValueError(f"Unsupported scheduler name: {name}")
