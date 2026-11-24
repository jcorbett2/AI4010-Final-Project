"""Reference CPU training script and offline fixture for the leaf agent."""

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from data_loaders import LeafDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    torch.manual_seed(manifest["seed"])
    torch.set_num_threads(2)

    training = LeafDataset(args.data, manifest, "train", image_size=64)
    validation = LeafDataset(args.data, manifest, "validation", image_size=64)
    train_batches = DataLoader(training, batch_size=32, shuffle=True)
    validation_batches = DataLoader(validation, batch_size=32)
    model = nn.Sequential(
        nn.Conv2d(3, 16, 5, stride=2, padding=2),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((4, 4)),
        nn.Flatten(),
        nn.Linear(32 * 4 * 4, len(training.classes)),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    for images, labels in train_batches:
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)

    model.eval()
    correct = 0
    with torch.inference_mode():
        for images, labels in validation_batches:
            correct += (model(images).argmax(dim=1) == labels).sum().item()

    torch.save(model.state_dict(), "model.pt")
    print(json.dumps({
        "epochs": 1,
        "batch_size": 32,
        "learning_rate": 0.001,
        "image_size": 64,
        "model": "two-layer CNN",
        "train_loss": total_loss / len(training),
        "validation_accuracy": correct / len(validation),
        "validation_examples": len(validation),
    }))


if __name__ == "__main__":
    main()
