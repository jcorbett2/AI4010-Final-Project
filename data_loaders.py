"""Load PyTorch batches from the saved leaf and document splits."""

import argparse
import csv
import json
import random
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset

from inspect_data import dataset_entry


def read_image(folder, filename, mode):
    with (folder / filename).open("rb") as stream, Image.open(stream) as image:
        return image.convert(mode)


def image_tensor(image):
    pixels = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    return pixels.reshape(image.height, image.width, len(image.getbands())).permute(
        2, 0, 1
    ).float().div_(255)


class LeafDataset(Dataset):
    """Return a padded RGB image and its integer species label."""

    def __init__(self, root, manifest, split, image_size=224):
        self.root = Path(root)
        self.image_ids = manifest["leaves"][split]
        self.image_size = image_size
        with dataset_entry(self.root, "train.csv") as labels, labels.open(
            "r", encoding="utf-8-sig"
        ) as stream:
            species_by_id = {
                row["id"]: row["species"] for row in csv.DictReader(stream)
            }
        self.classes = sorted(set(species_by_id.values()))
        self.class_to_index = {species: index for index, species in enumerate(self.classes)}
        self.labels = [self.class_to_index[species_by_id[item]] for item in self.image_ids]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        with dataset_entry(self.root, "images") as folder:
            image = read_image(folder, f"{self.image_ids[index]}.jpg", "RGB")
        image = ImageOps.pad(
            image, (self.image_size, self.image_size),
            method=Image.Resampling.BILINEAR, color="white",
        )
        return image_tensor(image), self.labels[index]


class DocumentDataset(Dataset):
    """Return aligned grayscale crops for training and full pages for evaluation."""

    def __init__(self, root, manifest, split, crop_size=128):
        self.root = Path(root)
        self.filenames = manifest["documents"][split]
        self.training = split == "train"
        self.crop_size = crop_size

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]
        with dataset_entry(self.root, "train") as folder:
            noisy = read_image(folder, filename, "L")
        with dataset_entry(self.root, "train_cleaned") as folder:
            clean = read_image(folder, filename, "L")
        if noisy.size != clean.size:
            raise ValueError(f"Document pair has different dimensions: {filename}")
        if self.training:
            noisy, clean = self.crop_pair(noisy, clean)
        return image_tensor(noisy), image_tensor(clean)

    def crop_pair(self, noisy, clean):
        size = self.crop_size
        width, height = noisy.size
        padding = (0, 0, max(0, size - width), max(0, size - height))
        if padding[2] or padding[3]:
            noisy = ImageOps.expand(noisy, border=padding, fill=255)
            clean = ImageOps.expand(clean, border=padding, fill=255)
        left = random.randint(0, noisy.width - size)
        top = random.randint(0, noisy.height - size)
        box = (left, top, left + size, top + size)
        return noisy.crop(box), clean.crop(box)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("leaf_path", type=Path)
    parser.add_argument("documents_path", type=Path)
    parser.add_argument("--manifest", type=Path, default=Path("splits.json"))
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    leaves = LeafDataset(args.leaf_path, manifest, "train")
    documents = DocumentDataset(args.documents_path, manifest, "train")
    images, labels = next(iter(DataLoader(leaves, batch_size=4)))
    noisy, clean = next(iter(DataLoader(documents, batch_size=4)))
    print(f"Leaf images {tuple(images.shape)}, labels {tuple(labels.shape)}")
    print(f"Noisy documents {tuple(noisy.shape)}, clean documents {tuple(clean.shape)}")


if __name__ == "__main__":
    main()
