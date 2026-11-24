"""Generate and execute a Leaf Classification baseline experiment."""

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from inspect_data import inspect_leaves


PROJECT = Path(__file__).resolve().parent


def generate_script(model, key, prompt):
    request = Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urlopen(request, timeout=120) as response:
        source = json.load(response)["choices"][0]["message"]["content"]
    if not source or not source.strip():
        raise ValueError("The model returned no Python source.")
    source = source.strip()
    lines = source.splitlines()
    if lines[0] in ("```python", "```") and lines[-1] == "```":
        source = "\n".join(lines[1:-1])
    return source + "\n"


def run_experiment(output, data, manifest, source):
    script = output / "train.py"
    script.write_text(source, encoding="utf-8")
    compile(source, str(script), "exec")
    command = [
        sys.executable, "-B", str(script), "--data", str(data), "--manifest", str(manifest),
    ]
    env = {"PATH": os.defpath, "PYTHONPATH": str(PROJECT)}
    with (output / "training.log").open("w", encoding="utf-8") as log:
        subprocess.run(
            command, cwd=output, env=env, stdout=log, stderr=subprocess.STDOUT,
            check=True, timeout=900,
        )
    log_path = output / "training.log"
    lines = log_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError("Training produced no output.")
    metrics = json.loads(lines[-1])
    accuracy = metrics.get("validation_accuracy") if isinstance(metrics, dict) else None
    if type(accuracy) not in (float, int) or not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
        raise ValueError("Training did not report a valid validation_accuracy.")
    if not (output / "model.pt").is_file():
        raise ValueError("Training did not save model.pt.")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return accuracy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=PROJECT / "splits.json")
    parser.add_argument("--output", required=True, type=Path, help="New directory for run artifacts")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--model", help="OpenAI API model whose generated Python will be run locally")
    mode.add_argument("--offline", action="store_true", help="Run the reference script without an LLM")
    args = parser.parse_args()
    if args.model and not (key := os.getenv("OPENAI_API_KEY")):
        parser.error("Set OPENAI_API_KEY to run with --model.")

    data, manifest, output = args.data.resolve(), args.manifest.resolve(), args.output.resolve()
    try:
        summary = inspect_leaves(data)
        if not summary["labeled_rows"] or summary["missing_image_ids"]:
            raise ValueError("Leaf data inspection failed. Run inspect_data.py for details.")
        splits = json.loads(manifest.read_text(encoding="utf-8"))["leaves"]
        reference = (PROJECT / "leaf_baseline.py").read_text(encoding="utf-8")
        prompt = f"""Build a short CPU PyTorch baseline for Leaf Classification.
Predict species from raw JPGs; CSV fields may supply only IDs and labels.
Use the existing LeafDataset interface and train and validation splits as in the
reference below. Keep the test split held out. Train on the entire train split
for one or two epochs. Use torch and Pillow only, with no downloads.
Keep the reference CLI and save model.pt in the working directory.
The last line on stdout must be JSON with validation_accuracy in [0, 1],
validation_examples, train_loss, model description, and training settings.
Return only the complete Python script.

Dataset: {summary["labeled_rows"]} labeled images, {summary["species_count"]} species.
Train examples: {len(splits["train"])}. Validation examples: {len(splits["validation"])}.

Reference:
{reference}"""
        output.mkdir(parents=True)
        (output / "run.json").write_text(json.dumps({
            "mode": "offline" if args.offline else "llm",
            "model": args.model,
            "data": str(data),
            "manifest": str(manifest),
            "inspection": summary,
        }, indent=2) + "\n", encoding="utf-8")
        if args.offline:
            source = reference
        else:
            (output / "prompt.txt").write_text(prompt, encoding="utf-8")
            source = generate_script(args.model, key, prompt)
        accuracy = run_experiment(output, data, manifest, source)
    except HTTPError as error:
        parser.exit(1, f"Model API request failed (HTTP {error.code}).\n")
    except (OSError, URLError, ValueError, KeyError, SyntaxError, subprocess.SubprocessError) as error:
        parser.exit(1, f"Experiment failed: {error}\nRun artifacts: {output}\n")
    print(f"Validation accuracy: {accuracy:.2%}")
    print(f"Run artifacts: {output}")


if __name__ == "__main__":
    main()
