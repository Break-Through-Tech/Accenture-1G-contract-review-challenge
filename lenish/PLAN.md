# Plan: reverse-engineer the CUAD data into the paper

Goal: take the raw dataset in `lenish/data/` and reproduce the paper's claims
(Hendrycks et al., *CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review*,
NeurIPS 2021, arXiv 2103.06268) from the data up. For every number, table and figure in
the paper, say which script produces it, and record where our result differs.

Everything lives in `lenish/`. The repo's own files (`train.py`, `utils.py`,
`evaluate.py`, `run.sh`, ...) stay untouched.

## Where we are

| Have | What it gives us |
|---|---|
| `data/CUADv1.json` | all 510 contracts, 41 categories, SQuAD format |
| `data/train_separate_questions.json` | 408 train contracts |
| `data/test.json` | 102 test contracts (80/20 split) |
| `visualize_data.py`, `report_pandas.py` | dataset dashboards (HTML) |
| `explore.py` | CLI: categories, spans, one contract, regex search |
| `analysis/category_summary.csv` | per-category rate, span count, span length, position |

Verified so far: 510 contracts, 41 categories, 13,823 annotated spans, 102 / 408 split.

## Steps

### 1. Pin down the targets (paper → checklist)
- Read the paper and list every quantitative claim in `targets.md`: the dataset size,
  contract types, label count, category frequencies, contract lengths, the train/test
  split, and the model results table and figures.
- For each one, note the section/table/figure, the paper's value, and an empty "ours" column.
- Download the PDF into `lenish/paper/` yourself; I'd ask before downloading it.

### 2. Dataset statistics (no GPU, do now)
Extend `report_pandas.py`, or add `paper_stats.py`, to output each dataset number:
- **Counts**: contracts, categories, labels. The abstract says "over 13,000"; we have 13,823.
  Work out what they counted (spans? answered questions = 6,702?) and write it down.
- **Contract types (25 in the paper)**: the type is the tail of each title, but it comes
  out as 144 raw strings (`SERVICE` vs `SERVICES`, `LICENSE AND HOSTING`, ...). Build a
  hand-made mapping `raw → paper type` in `analysis/contract_types.csv` until the
  counts match the paper's table.
- **Category frequency**: we already have this (`category_summary.csv`); compare it with
  the paper's figure.
- **"Needle in a haystack"**: the share of each contract's characters that is
  annotated, and the length distribution. Compare with the paper's claim.
- **Split check**: confirm test ∪ train = CUADv1 with no contract in both, and that
  category rates are similar across the split.

Output: `analysis/paper_stats.csv` plus a "Dataset" section in `targets.md`, filled in.

### 3. Metrics without training (no GPU)
`evaluate.py` computes the paper's metrics: AUPR, Precision@80% recall, and
Precision@90% recall, with a Jaccard IoU ≥ 0.5 match.
- Read it closely and write down how a prediction counts as correct. This is what the
  results table depends on.
- Sanity-run it on the test set with fake predictions (ground truth → should score ~1.0;
  empty → ~0) so we trust the scoring before spending GPU time.

### 4. Get the model results (needs GPU → WSL Ubuntu)
Two options, cheapest first:
- **a. Paper checkpoints**: download the released RoBERTa-base / RoBERTa-large /
  DeBERTa-xlarge checkpoints (Zenodo link in `readme.md`), predict on `test.json`, and
  run `evaluate.py`. This reproduces the results table without training.
- **b. Retrain**: RoBERTa-base with `run.sh` (4 epochs, lr 1e-4, seq 512, stride 256).

Set up WSL yourself (I stopped at this step earlier):
1. On Windows, run `nvidia-smi` inside Ubuntu (`wsl`). If it shows the GPU, the
   Windows driver is enough; don't install a Linux NVIDIA driver.
2. In Ubuntu: `sudo apt install python3-venv`, then `python3 -m venv ~/cuad-env`.
3. `pip install torch` (a CUDA build from pytorch.org) plus
   `transformers==4.4.2 numpy pandas scikit-learn tqdm`. `train.py` was written for
   Transformers 4.3/4.4 and imports `transformers.data.processors.squad`,
   which newer versions change.
4. Work from `/mnt/c/Users/lpand003/Desktop/Research/BBT/cuad`. For speed, copy
   `lenish/data` into the Linux home directory.
5. **Data paths**: `run.sh` expects `./data/...`, but the data is now in `lenish/data`.
   Copy `run.sh` to `lenish/run_wsl.sh` and point it at `lenish/data/...`, with
   `CUDA_VISIBLE_DEVICES=0` if you have one GPU. Then the root files stay unchanged.
6. `per_gpu_train_batch_size=40` at seq 512 won't fit on a consumer GPU. Lower it
   (e.g. 8) and add `--gradient_accumulation_steps` to keep an effective batch of about 40.

### 5. Figures the paper draws from training
- **Dataset size vs performance**: retrain on 3–4 fractions of the training contracts
  (subset by contract, not by question) and plot AUPR against size.
- **Per-category performance**: `evaluate.py` with `category=` for each of the 41
  categories, then join that onto `category_summary.csv`. Does performance track
  how often a category appears?

### 6. Write it up
`lenish/REPRODUCTION.md`: for each paper claim, give the paper value, our value, the
script that produced it, and the reason for any gap (label counting, contract-type
mapping, hardware or batch size, library version).

## Order of work
1 → 2 → 3 need no GPU and can be done now on Windows. 4a before 4b. 5 only if 4 works.
