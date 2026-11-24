"""Score the recovered Gemini revisions on their held-out test splits."""

import argparse
import json
import math
from pathlib import Path
import runpy

import torch
from torch import nn
from torch.utils.data import DataLoader

from data_loaders import DocumentDataset, LeafDataset


def leaf_model(classes):
    # Match the architecture in the saved leaf revision.
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((4, 4)),
        nn.Flatten(),
        nn.Linear(128 * 4 * 4, classes),
    )


def load_checkpoint(model, revision):
    model.load_state_dict(torch.load(revision / "model.pt", map_location="cpu", weights_only=True))
    return model.eval()


@torch.inference_mode()
def evaluate_leaves(data, manifest, revision):
    test = LeafDataset(data, manifest, "test", image_size=64)
    model = load_checkpoint(leaf_model(len(test.classes)), revision)
    correct = 0
    for images, labels in DataLoader(test, batch_size=32):
        correct += (model(images).argmax(dim=1) == labels).sum().item()
    return {"examples": len(test), "correct": correct, "accuracy": correct / len(test)}


@torch.inference_mode()
def evaluate_documents(data, manifest, revision):
    test = DocumentDataset(data, manifest, "test")
    model_class = runpy.run_path(str(revision / "train.py"))["DenoiseNet"]
    model = load_checkpoint(model_class(), revision)
    squared_error = 0.0
    pixels = 0
    for noisy, clean in DataLoader(test, batch_size=1):
        squared_error += (model(noisy).clamp(0, 1) - clean).square().sum().item()
        pixels += clean.numel()
    return {"examples": len(test), "pixels": pixels, "rmse": math.sqrt(squared_error / pixels)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leaf-data", required=True, type=Path)
    parser.add_argument("--document-data", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=Path("splits.json"))
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    args = parser.parse_args()

    torch.set_num_threads(2)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    results = {
        "split": "test",
        "leaves": evaluate_leaves(
            args.leaf_data, manifest, args.runs / "leaf-gemini-recovered" / "revision",
        ),
        "documents": evaluate_documents(
            args.document_data, manifest, args.runs / "documents-gemini-recovered" / "revision",
        ),
    }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
