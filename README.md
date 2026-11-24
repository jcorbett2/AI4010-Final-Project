# Autonomous Deep Learning Agent for Computer Vision

**Author** — Jared  
**Course** — AI 4010, Ohio University, Fall 2026  
**Project track** — Track B 

The agent inspects a dataset, runs a PyTorch training experiment, and uses validation results to revise its approach and run a second experiment. The same agent trains a separate model for each dataset.

## Datasets

- **Leaf Classification** — Predict the species from a raw leaf image.
- **Denoising Dirty Documents** — Produce a clean document image from a noisy one.

## Run the experiments

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch pillow kagglehub
python download_data.py
```

Authenticate with Kaggle and accept each competition's rules before downloading.
Set `LEAF_DATA` and `DOCUMENT_DATA` to the dataset directories printed by
`download_data.py`. In the same shell, export your Gemini API key:

```bash
read -rsp 'Gemini API key: ' GEMINI_API_KEY
printf '\n'
export GEMINI_API_KEY

python agent.py --dataset leaves --provider gemini --model gemini-3.1-flash-lite \
  --data "$LEAF_DATA" --output runs/leaves

python agent.py --dataset documents --provider gemini --model gemini-3.1-flash-lite \
  --data "$DOCUMENT_DATA" --output runs/documents
```

For a local training check, replace `--provider gemini --model gemini-3.1-flash-lite`
with `--offline` and choose a new output directory. API mode executes the
generated Python in a local subprocess, so run it in a dedicated environment.
Each run saves the training script, log, checkpoint, and validation metrics.
API mode also saves the prompts and stores the feedback-driven rerun under
`revision/`. Leaf runs report classification accuracy. Document runs report
pixel RMSE over full validation pages with values scaled to [0, 1].

## Evaluate the submitted checkpoints

From the root of the extracted submission, set the same dataset paths used
above and run:

```bash
python evaluate_checkpoints.py --leaf-data "$LEAF_DATA" --document-data "$DOCUMENT_DATA"
```

The evaluator reads the revised checkpoints under `runs/leaf-gemini-recovered/`
and `runs/documents-gemini-recovered/`. It reports held-out metrics without
calling Gemini or retraining. See `RESULTS.md` for the analysis and
`test_metrics.json` for the recorded scores.
