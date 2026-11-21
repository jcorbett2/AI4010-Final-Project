"""Create fixed train, validation, and test splits from the labeled Kaggle data."""

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from inspect_data import dataset_entry, png_names


def split_leaves(root, rng):
    ids_by_species = defaultdict(list)
    seen_ids = set()
    with dataset_entry(root, "train.csv") as labels, labels.open(
        "r", encoding="utf-8-sig"
    ) as stream:
        for row in csv.DictReader(stream):
            image_id, species = row["id"], row["species"]
            if not image_id or not species or image_id in seen_ids:
                raise ValueError(f"Missing label or invalid or duplicate leaf ID: {image_id!r}")
            seen_ids.add(image_id)
            ids_by_species[species].append(image_id)

    if len(ids_by_species) != 99:
        raise ValueError(f"Expected 99 leaf species, found {len(ids_by_species)}")

    splits = {"train": [], "validation": [], "test": []}
    for species, image_ids in sorted(ids_by_species.items()):
        if len(image_ids) != 10:
            raise ValueError(f"Expected 10 images for {species}, found {len(image_ids)}")
        image_ids.sort()
        rng.shuffle(image_ids)
        splits["train"].extend(image_ids[:8])
        splits["validation"].append(image_ids[8])
        splits["test"].append(image_ids[9])
    return splits


def split_documents(root, rng):
    with dataset_entry(root, "train") as noisy, dataset_entry(
        root, "train_cleaned"
    ) as clean:
        noisy_names = png_names(noisy)
        clean_names = png_names(clean)

    if noisy_names != clean_names:
        raise ValueError("Noisy and clean document filenames differ. Run inspect_data.py.")
    if len(noisy_names) != 144:
        raise ValueError(f"Expected 144 matched documents, found {len(noisy_names)}")

    filenames = sorted(noisy_names)
    rng.shuffle(filenames)
    return {
        "train": filenames[:116],
        "validation": filenames[116:130],
        "test": filenames[130:],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("leaf_path", type=Path, help="Leaf Classification directory")
    parser.add_argument("documents_path", type=Path, help="Dirty Documents directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("splits.json"))
    args = parser.parse_args()

    try:
        manifest = {
            "seed": args.seed,
            "leaves": split_leaves(args.leaf_path, random.Random(args.seed)),
            "documents": split_documents(args.documents_path, random.Random(args.seed)),
        }
        serialized = json.dumps(manifest, indent=2) + "\n"
        if args.output.exists():
            if args.output.read_text(encoding="utf-8") != serialized:
                parser.error(f"{args.output} differs from the generated manifest. Choose another --output.")
        else:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(serialized)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    print(f"Split manifest: {args.output}")
    for dataset in ("leaves", "documents"):
        counts = ", ".join(f"{name}={len(items)}" for name, items in manifest[dataset].items())
        print(f"{dataset}: {counts}")


if __name__ == "__main__":
    main()
