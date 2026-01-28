# train.py
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from arch.models import model
from dataset import DummyDataset
from losses import LabelSmoothingCE


def accuracy(logits, y):
    pred = logits.argmax(dim=1)
    return (pred == y).float().mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    ds = DummyDataset(n=800, num_classes=args.num_classes, image_size=args.image_size)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)

    model = Model(num_classes=args.num_classes, embed_dim=256, depth=4, num_heads=8, use_fusion=False)
    model.to(args.device)

    criterion = LabelSmoothingCE(smoothing=0.1)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)

    model.train()
    for ep in range(1, args.epochs + 1):
        total_loss = 0.0
        total_acc = 0.0
        for x, y in dl:
            x = x.to(args.device)
            y = torch.tensor(y, device=args.device)

            optim.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optim.step()

            total_loss += loss.item()
            total_acc += accuracy(logits.detach(), y)

        print(f"Epoch {ep}: loss={total_loss/len(dl):.4f}, acc={total_acc/len(dl):.4f}")


if __name__ == "__main__":
    main()
