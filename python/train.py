"""Train the cell classifier (empty / X / O) on cells extracted from synthetic photos, export it to ONNX.

Usage: python python/train.py [--photos 4000]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from rectify import INPUT, read
from synth import sample

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["empty", "X", "O"]


class CellNet(nn.Module):
    def __init__(self):
        super().__init__()
        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
        self.features = nn.Sequential(block(1, 16), block(16, 32), block(32, 64))
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(64 * 4 * 4, 64), nn.ReLU(),
                                  nn.Linear(64, len(CLASSES)))

    def forward(self, x):
        return self.head(self.features(x))


def build(seeds):
    xs, ys = [], []
    for s in seeds:
        img, labels, _ = sample(int(s), hard=int(s) % 10 < 3)
        r = read(img)
        if r is None:
            continue
        xs.append(r[0])
        ys.append(labels)
    return np.concatenate(xs).astype(np.float32), np.concatenate(ys).astype(np.int64)


def augment(x: torch.Tensor) -> torch.Tensor:
    shift = torch.randint(-3, 4, (2,))
    x = torch.roll(x, shifts=(int(shift[0]), int(shift[1])), dims=(2, 3))
    if torch.rand(1) < 0.5:
        x = torch.flip(x, dims=(3,))
    return x * torch.empty(x.shape[0], 1, 1, 1).uniform_(0.7, 1.3) + 0.05 * torch.randn_like(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photos", type=int, default=4000)
    ap.add_argument("--epochs", type=int, default=12)
    args = ap.parse_args()
    torch.manual_seed(0)
    t = time.time()
    Xtr, ytr = build(range(args.photos))
    Xva, yva = build(range(100_000, 100_000 + args.photos // 8))
    print(f"cells: train {len(ytr)}, val {len(yva)} ({time.time() - t:.0f} s)", flush=True)

    model = CellNet()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 4e-3, total_steps=args.epochs * (len(ytr) // 256 + 1))
    Xtr_t, ytr_t = torch.from_numpy(Xtr[:, None]), torch.from_numpy(ytr)
    Xva_t = torch.from_numpy(Xva[:, None])
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(ytr_t))
        for i in range(0, len(perm), 256):
            b = perm[i:i + 256]
            loss = nn.functional.cross_entropy(model(augment(Xtr_t[b])), ytr_t[b], label_smoothing=0.05)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            acc = (model(Xva_t).argmax(1).numpy() == yva).mean()
        print(f"epoch {epoch + 1}: loss {loss.item():.3f}, val accuracy {acc:.4f}", flush=True)

    out = ROOT / "models"
    out.mkdir(exist_ok=True)
    torch.save(model.state_dict(), out / "cellnet.pt")
    torch.onnx.export(model, torch.zeros(1, 1, INPUT, INPUT), out / "cellnet.onnx", input_names=["cells"],
                      output_names=["logits"], dynamic_axes={"cells": {0: "batch"}, "logits": {0: "batch"}},
                      opset_version=11, dynamo=False)
    (ROOT / "results" / "training.json").write_text(json.dumps(
        {"train_cells": int(len(ytr)), "val_cells": int(len(yva)), "val_accuracy": float(acc),
         "parameters": sum(p.numel() for p in model.parameters())}, indent=2))
    print("exported models/cellnet.onnx")


if __name__ == "__main__":
    main()
