"""Train a CPU model to denoise grayscale document images."""

import argparse
import json
import math
from pathlib import Path
import random

import torch
from torch import nn
from torch.utils.data import DataLoader

from data_loaders import DocumentDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    random.seed(manifest["seed"])
    torch.manual_seed(manifest["seed"])
    torch.set_num_threads(2)

    crop_size = 128
    batch_size = 8
    epochs = 2
    learning_rate = 0.001
    training = DocumentDataset(args.data, manifest, "train", crop_size=crop_size)
    validation = DocumentDataset(args.data, manifest, "validation")
    train_batches = DataLoader(training, batch_size=batch_size, shuffle=True)
    # Full pages have different dimensions, so evaluate one page at a time.
    validation_batches = DataLoader(validation, batch_size=1)
    model = nn.Sequential(
        nn.Conv2d(1, 8, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 1, 3, padding=1),
        nn.Sigmoid(),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    total_loss = 0.0
    for _ in range(epochs):
        for noisy, clean in train_batches:
            optimizer.zero_grad()
            loss = criterion(model(noisy), clean)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * noisy.size(0)

    model.eval()
    squared_error = 0.0
    pixel_count = 0
    with torch.inference_mode():
        for noisy, clean in validation_batches:
            squared_error += (model(noisy) - clean).square().sum().item()
            pixel_count += clean.numel()

    torch.save(model.state_dict(), "model.pt")
    print(json.dumps({
        "model": "three-layer denoising CNN",
        "crop_size": crop_size,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "train_loss": total_loss / (epochs * len(training)),
        "validation_examples": len(validation),
        "validation_rmse": math.sqrt(squared_error / pixel_count),
    }))


if __name__ == "__main__":
    main()
