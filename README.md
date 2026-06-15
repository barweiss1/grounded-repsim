# Grounding Representation Similarity with Statistical Testing

This repository contains code for reproducing and extending the experiments in
[Grounding Representation Similarity with Statistical Testing](https://arxiv.org/abs/2108.01661).
The benchmark evaluates representation-similarity measures by comparing
dissimilarities between neural network representations with downstream accuracy
differences.

The expensive embeddings and accuracy scores are provided as precomputed
resources. This repository contains the experiment and metric code that consumes
those resources.

## Repository Layout

- `dists/` implements representation dissimilarity metrics and pairwise scoring.
- `experiments/` contains the experiment-specific pipelines:
  - `layer_exp`: different random seeds and layer depths.
  - `pca_deletion`: different numbers of principal components deleted.
  - `feather`: different finetuning seeds.
  - `pretrain_finetune`: different pretraining and finetuning seeds.
- `experiments/common.py` contains shared runner path, config alias, metric
  filtering, result writing, and plot helpers.
- `run_experiments.py` is the YAML-driven runner for staged experiment runs.
- `configs/` contains example and fast-run YAML configs.
- `paths.py` points to the local `sim_metric_resources` directory.

## Setup

Use Python `>=3.10,<3.12` and install the dependencies:

```bash
pip install -r requirements.txt
```

Download `sim_metric_resources.tar` from
<https://zenodo.org/record/5117844> and extract it. By default the code expects
a `sim_metric_resources` directory next to this repository. On servers or other
machines, set an override:

```bash
export SIM_METRIC_RESOURCES=/path/to/sim_metric_resources
```

The runner expects precomputed `rep.npy` files under
`$SIM_METRIC_RESOURCES/embeddings/...`. It does not compute BERT embeddings on
demand; if an embedding is missing, fix the resource path or download the full
resources archive.

The resources directory is expected to contain:

- `embeddings/`: representation arrays used for pairwise scoring.
- `dists/`: precomputed dissimilarities.
- `scores/`: accuracy/probing scores.
- `full_dfs/`: dataframes joining metric dissimilarities with accuracy
  differences.
- `results/`: runner output directories, if using `run_experiments.py`.

## Running Experiments

Prefer the YAML runner for new work:

```bash
python run_experiments.py --config configs/layer_exp_fast.yaml --device auto
```

The singular wrapper also works for server scripts that call
`run_experiment.py`:

```bash
python run_experiment.py --config configs/layer_exp_fast.yaml --device auto
```

Each enabled experiment can run up to three stages:

1. `compute_dists`: writes `dists_self_computed.csv`.
2. `compute_full_df`: writes `full_df_self_computed.csv`.
3. `script`: writes `results.txt` and `rank_corrs_<task>.png`.

By default, the runner executes every configured stage in that order. To reuse
existing distances and run only later stages, add `run_stages` to the experiment
config:

```yaml
pretrain_finetune:
  enabled: true
  run_stages: ["compute_full_df", "script"]
  compute_dists:
    ...
  compute_full_df:
    ...
  script:
    ...
```

Requested stages must still have config blocks, and stage execution always
follows the canonical order: `compute_dists`, `compute_full_df`, then `script`.

`compute_dists` stages support CPU-parallel, resumable distance computation.
`num_workers` can be set in YAML or passed at runtime with `--num-workers`; if
neither is provided, a stage runs with a single worker. Other useful keys under
each `compute_dists` block:

```yaml
write_every: 50
overwrite_dists: false
embedding_cache_size: 2
torch_num_threads_per_worker: 1
```

For SLURM CPU jobs, request matching cores and prevent each worker from spawning
extra BLAS threads:

```bash
#SBATCH --cpus-per-task=8

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python run_experiment.py \
  --config configs/pretrain_finetune.yaml \
  --device cpu \
  --num-workers "$SLURM_CPUS_PER_TASK"
```

Runner outputs are written under:

```text
<resources_path>/<results_base>/<experiment>/<run_id>/
```

For example, with `results_base: "results"` and
`run_id: "fast_last_layers_run"`, the layer experiment writes to:

```text
sim_metric_resources/results/layer_exp/fast_last_layers_run/
```

`run_experiments.py` is the only supported experiment execution path. Experiment
modules expose runner-callable functions and should not be run directly. The
notebooks remain exploratory artifacts and may read precomputed files from
`sim_metric_resources/full_dfs/`.

## Config Conventions

Configs must use canonical lowercase keys:

- `layers`
- `ref_seeds`
- `task`

Older aliases such as `LAYERS`, `REF_SEEDS`, and `probe_task` are no longer
supported by the runner.

Useful configs:

- `configs/feather.yaml`: Feather experiment with a limited `rep1` seed sweep.
- `configs/layer_exp.yaml`: full layer experiment.
- `configs/layer_exp_fast.yaml`: smaller layer experiment for iteration.
- `configs/pca_deletion_fast.yaml`: smaller PCA-deletion experiment.
- `configs/pca_deletion.yaml`: full PCA-deletion runner config.
- `configs/pretrain_finetune_fast.yaml`: smaller pretrain/finetune experiment.
- `configs/pretrain_finetune.yaml`: full pretrain/finetune runner config.
- `configs/experiment_runner.yaml`: example runner config.

Sweep analysis notebooks live at
`experiments/<experiment>/similarity_signals.ipynb`. They use
`experiments/sweep_analysis.py` to load AUC sweep JSON files from a run's
`sweeps/` directory and save similarity-signal plots under
`<results_dir>/figures/sweep_signals/`.

The primary research metrics in this repo are the RWKA family:
`softmax_rwka`, `rbf_rwka`, and especially their AUC variants
`softmax_rwka_auc` and `rbf_rwka_auc`. Fast-run configs should include these
metrics by default so iteration results stay focused on the main research
questions.

## Reproducing Paper Results

To use the precomputed paper data, read the relevant `full_df.csv` files from
`sim_metric_resources/full_dfs/<experiment>/` in the notebooks or scripts. The
notebooks remain useful for exploratory analysis and visualization.

To recompute dissimilarities, use the runner with a config that enables
`compute_dists`, then `compute_full_df`, then `script`. Recomputing can be slow
and can produce many CSV/JSON/PNG outputs, especially for AUC sweep metrics.

## Adding a New Metric

Classic metrics are implemented in `dists/scoring.py` and dispatched in
`dists/score_pair.py`.

Newer alignment metrics live in `dists/metrics.py`:

1. Add the metric name to `AlignmentMetrics.SUPPORTED_METRICS`.
2. Add sweep metadata to `AlignmentMetrics.SWEEP_PARAMS`; use `"param": None`
   for metrics without a swept hyperparameter.
3. Add dispatch logic in `AlignmentMetrics.measure` if the metric is an alias
   or needs special keyword arguments.
4. Update `dists/metrics_utils.py` if the metric introduces a new swept
   hyperparameter.
5. Add the metric name, or `<metric>_auc` for AUC-over-sweep distance, to the
   relevant config `metrics` lists.

Most newer metrics return similarities; saved distances are generally
`1 - score`.

## Development Checks

Run a syntax check:

```bash
python -m py_compile run_experiments.py dists/*.py experiments/*/*.py
```

Run import checks:

```bash
python -c "import experiments.common; import experiments.feather.script; import experiments.layer_exp.script; import experiments.pca_deletion.script; import experiments.pretrain_finetune.script"
```

If bytecode or Matplotlib caches are not writable in your environment, redirect
them to a temporary directory:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/grounded-repsim-pycache \
MPLCONFIGDIR=/private/tmp/grounded-repsim-mpl \
XDG_CACHE_HOME=/private/tmp/grounded-repsim-xdg \
python -m py_compile run_experiments.py dists/*.py experiments/*/*.py
```
