#!/usr/local/linux/anaconda3.8/bin/python

import pathlib
import sys

BASE_DIR = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR / "dists"))

from experiments.common import get_run_paths, make_pair_job, run_pair_jobs

# --- Refactored for wrapper ---
def run_compute_dists(cfg, results_dir, resources_path, device=None):
    metrics = cfg.get('metrics')
    dataset = cfg.get('dataset')
    architecture = cfg.get('architecture')
    step = cfg.get('step')
    num_layers = cfg.get('num_layers')
    num_seeds = cfg.get('num_seeds')
    seed1_seeds = cfg.get('seed1_seeds', list(range(num_seeds)))

    result_filename = get_run_paths(results_dir)["dists"]
    result_filename.parent.mkdir(parents=True, exist_ok=True)

    jobs = []
    for seed1 in seed1_seeds:
        for seed2 in range(seed1, num_seeds):
            for layer in range(num_layers):
                rep1_dict = {"dataset": dataset, "architecture": architecture, "seed": seed1, "step": step, "layer": layer}
                rep2_dict = {"dataset": dataset, "architecture": architecture, "seed": seed2, "step": step, "layer": layer}
                jobs.append(make_pair_job(rep1_dict, rep2_dict))

    run_pair_jobs(jobs, metrics, result_filename, cfg, desc="feather distance pairs")
