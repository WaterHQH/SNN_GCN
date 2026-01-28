# evaluate.py
import argparse
import json
import torch
from torch.utils.data import DataLoader

from arch.models import Model
from dataset import DummyDataset
from utils import get_device, accuracy_topk, load_checkpoint


@torch.no_grad()
def run_eval(model, dl, device):
    model.eval()
    top1_sum, top5_sum, n = 0.0, 0.0, 0

    for x, y in dl:
        x = x.to(device)
        y = torch.tensor(y, device=device)

        logits = model(x)
        acc = accuracy_topk(logits, y, topk=(1, 5))
        bsz = x.size(0)

        top1_sum += acc["top1"] * bsz
        top5_sum += acc["top5"] * bsz
        n += bsz

    return {"top1": top1_sum / max(1, n), "top5": top5_sum / max(1, n), "num_samples": n}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="", help="checkpoint path (optional)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--out_json", type=str, default="", help="save metrics to json file")
    args = parser.parse_args()

    device = get_device(args.device)

 
    ds = DummyDataset(n=500, num_classes=args.num_classes, image_size=args.image_size)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = Model(num_classes=args.num_classes, embed_dim=256, depth=4, num_heads=8, use_fusion=False).to(device)

    if args.ckpt:
        load_checkpoint(args.ckpt, model, map_location=str(device))

    metrics = run_eval(model, dl, device)
    print("Eval:", metrics)

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"Saved to: {args.out_json}")


if __name__ == "__main__":
    main()
