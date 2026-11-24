"""Run a Leaf Classification baseline and an LLM-driven revision."""

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from inspect_data import inspect_leaves


PROJECT = Path(__file__).resolve().parent


def generate_script(model, key, prompt, *, provider="openai"):
    endpoint = {
        "openai": "https://api.openai.com/v1/chat/completions",
        "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
    }[provider]
    request = Request(
        endpoint,
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
    python = [sys.executable, "-X", "faulthandler", "-B"]
    arguments = ["--data", str(data), "--manifest", str(manifest)]
    command = [*python, str(script), *arguments]
    env = {"PATH": os.defpath, "PYTHONPATH": str(PROJECT)}
    log_path = output / "training.log"
    for attempt in range(2):
        try:
            with log_path.open("w", encoding="utf-8") as log:
                subprocess.run(
                    command, cwd=output, env=env, stdout=log, stderr=subprocess.STDOUT,
                    check=True, timeout=900,
                )
            break
        except subprocess.CalledProcessError as error:
            if attempt or error.returncode != -signal.SIGSEGV:
                raise
            log_path.rename(output / "training-attempt-1.log")
            (output / "model.pt").unlink(missing_ok=True)
            # Try the standard CPU kernels if the optimized backend crashed.
            command = [
                *python, "-c",
                "import runpy, sys, torch; "
                "torch.backends.mkldnn.enabled = False; "
                "sys.argv.pop(0); runpy.run_path(sys.argv[0], run_name='__main__')",
                str(script), *arguments,
            ]
    lines = log_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError("Training produced no output.")
    metrics = json.loads(lines[-1])
    accuracy = validation_accuracy(metrics)
    if not (output / "model.pt").is_file():
        raise ValueError("Training did not save model.pt.")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return accuracy


def validation_accuracy(metrics):
    accuracy = metrics.get("validation_accuracy") if isinstance(metrics, dict) else None
    if type(accuracy) not in (float, int) or not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
        raise ValueError("Training did not report a valid validation_accuracy.")
    return accuracy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=PROJECT / "splits.json")
    parser.add_argument("--output", required=True, type=Path, help="New directory for run artifacts")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--model", help="API model whose generated Python will be run locally")
    mode.add_argument("--offline", action="store_true", help="Run the reference script without an LLM")
    parser.add_argument("--provider", choices=("openai", "gemini"), default="openai")
    initial = parser.add_mutually_exclusive_group()
    initial.add_argument("--initial-script", type=Path, help="Replay a saved initial training script before requesting a revision")
    initial.add_argument("--initial-run", type=Path, help="Reuse a completed initial run for the same dataset and split manifest")
    args = parser.parse_args()
    if (args.initial_script or args.initial_run) and not args.model:
        parser.error("--initial-script and --initial-run require --model.")
    key_name = {"openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY"}[args.provider]
    if args.model and not (key := os.getenv(key_name)):
        parser.error(f"Set {key_name} to run with --model.")

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
            "provider": args.provider if args.model else None,
            "data": str(data),
            "manifest": str(manifest),
            "inspection": summary,
            "initial_script": str(args.initial_script.resolve()) if args.initial_script else None,
            "initial_run": str(args.initial_run.resolve()) if args.initial_run else None,
        }, indent=2) + "\n", encoding="utf-8")
        if args.initial_run:
            source = (args.initial_run / "train.py").read_text(encoding="utf-8")
            metrics = json.loads((args.initial_run / "metrics.json").read_text(encoding="utf-8"))
            accuracy = validation_accuracy(metrics)
            if metrics.get("validation_examples") != len(splits["validation"]):
                raise ValueError("Saved run has a different number of validation examples.")
            for name in ("train.py", "metrics.json", "training.log", "model.pt"):
                shutil.copyfile(args.initial_run / name, output / name)
        elif args.offline:
            source = reference
        else:
            (output / "prompt.txt").write_text(prompt, encoding="utf-8")
            if args.initial_script:
                source = args.initial_script.read_text(encoding="utf-8")
            else:
                source = generate_script(args.model, key, prompt, provider=args.provider)
        if not args.initial_run:
            accuracy = run_experiment(output, data, manifest, source)
        print(f"Initial validation accuracy: {accuracy:.2%}")
        if args.model:
            revision = output / "revision"
            revision.mkdir()
            feedback = (output / "metrics.json").read_text(encoding="utf-8")
            revision_prompt = f"""{prompt}

The initial experiment completed. Its measured validation feedback was:
{feedback}

Here is the actual initial training code:
{source}

Use the observed feedback to choose one concrete change intended to improve
validation accuracy. Keep the same dataset splits and compute budget.
Return the complete revised Python script. Start it with a Python comment
explaining what the metrics showed, what you changed, and why.
"""
            (revision / "prompt.txt").write_text(revision_prompt, encoding="utf-8")
            revised_source = generate_script(
                args.model, key, revision_prompt, provider=args.provider,
            )
            revised_accuracy = run_experiment(revision, data, manifest, revised_source)
            print(f"Revised validation accuracy: {revised_accuracy:.2%}")
    except HTTPError as error:
        parser.exit(1, f"Model API request failed (HTTP {error.code}).\n")
    except (OSError, URLError, ValueError, KeyError, SyntaxError, subprocess.SubprocessError) as error:
        parser.exit(1, f"Experiment failed: {error}\nRun artifacts: {output}\n")
    print(f"Run artifacts: {output}")


if __name__ == "__main__":
    main()
