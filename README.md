# Paper Model PyTorch Implementation

This repository provides a **PyTorch implementation of the model proposed in our paper**  
> *[Label Distribution Learning via Implicit Distribution Representation]*

# model_arch
<img width="693" height="531" alt="image" src="https://github.com/user-attachments/assets/31be7238-80a7-4b0f-848c-c4617f76155f" />

---

## 📌 Overview

- **Framework**: PyTorch
- **Goal**: Faithful reproduction of the model and experiments described in the paper
- **Features**:
  - Modular model design
  - YAML-based experiment configuration
  - Support for warmup + cosine LR schedule
  - Optional EMA, AMP training
  - Clear separation of model / data / training / optimization logic

---

## 📁 Project Structure

```text
model_pytorch/
├── configs/                 # Experiment configurations
├── models/                  # Model definitions
├── datasets/                # Dataset wrappers
├── losses/                  # Loss functions
├── optim/                   # Optimizers & LR schedulers              # Training / validation loops
├── utils/                   # Utilities (seed, logging, checkpoint, EMA)         
├── outputs/                 # Logs & checkpoints (ignored by git)
├── train.py                 # Main training entry
├── evaluate.py              # Evaluation script
├── requirements.txt
└── README.md


