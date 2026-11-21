"""Check the labeled leaf images and paired document images without extracting ZIPs."""

import argparse
import csv
import json
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from zipfile import Path as ZipPath, ZipFile

from PIL import Image


@contextmanager
def dataset_entry(root, name):
    """Open an extracted entry or its counterpart inside the Kaggle ZIP."""
    entry = root / name
    if entry.exists():
        yield entry
        return

    with ZipFile(root / f"{name}.zip") as archive:
        yield ZipPath(archive, name + ("/" if "." not in name else ""))


def image_size(path):
    with path.open("rb") as stream, Image.open(stream) as image:
        return image.size


def inspect_leaves(root):
    with dataset_entry(root, "train.csv") as labels, labels.open(
        "r", encoding="utf-8-sig"
    ) as stream:
        rows = [(row["id"], row["species"]) for row in csv.DictReader(stream)]

    species_counts = Counter(species for _, species in rows)
    with dataset_entry(root, "images") as images:
        missing = [
            image_id for image_id, _ in rows
            if not (images / f"{image_id}.jpg").is_file()
        ]

    return {
        "labeled_rows": len(rows),
        "species_count": len(species_counts),
        "examples_per_species": dict(species_counts),
        "missing_image_ids": missing,
    }


def png_names(folder):
    return {
        entry.name for entry in folder.iterdir()
        if entry.is_file() and entry.name.lower().endswith(".png")
    }


def inspect_documents(root):
    mismatches = []
    with dataset_entry(root, "train") as noisy, dataset_entry(
        root, "train_cleaned"
    ) as clean:
        noisy_names = png_names(noisy)
        clean_names = png_names(clean)
        paired_names = noisy_names & clean_names
        for name in sorted(paired_names):
            noisy_size = image_size(noisy / name)
            clean_size = image_size(clean / name)
            if noisy_size != clean_size:
                mismatches.append({
                    "filename": name,
                    "noisy_size": noisy_size,
                    "clean_size": clean_size,
                })

    return {
        "matched_pairs": len(paired_names),
        "missing_clean_images": sorted(noisy_names - clean_names),
        "missing_noisy_images": sorted(clean_names - noisy_names),
        "dimension_mismatches": mismatches,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("leaf_path", type=Path, help="Leaf Classification directory")
    parser.add_argument("documents_path", type=Path, help="Dirty Documents directory")
    args = parser.parse_args()
    leaves = inspect_leaves(args.leaf_path)
    documents = inspect_documents(args.documents_path)
    print(json.dumps({"leaves": leaves, "documents": documents}, indent=2, sort_keys=True))
    return int(
        leaves["labeled_rows"] == 0
        or documents["matched_pairs"] == 0
        or bool(leaves["missing_image_ids"])
        or bool(documents["missing_clean_images"])
        or bool(documents["missing_noisy_images"])
        or bool(documents["dimension_mismatches"])
    )


if __name__ == "__main__":
    raise SystemExit(main())
