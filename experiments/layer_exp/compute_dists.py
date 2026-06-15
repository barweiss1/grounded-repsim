#!/usr/local/linux/anaconda3.8/bin/python

import pathlib
import sys

BASE_DIR = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))
from experiments.common import get_run_paths, make_pair_job, run_pair_jobs

def run_compute_dists(cfg, results_dir, resources_path, device=None):
    metrics = cfg.get('metrics', ["CKA", "Procrustes", 'cka_rbf', 'cka_rbf_quantile', 'cka_rbf_auc', 'mutual_knn', 'mutual_knn_auc', 'cknna'])
    dataset = cfg.get('dataset', 'mnli_matched_100')
    architecture = cfg.get('architecture', 'base')
    step = cfg.get('step', 2000000)
    num_layers = cfg.get('num_layers', 12)
    num_seeds = cfg.get('num_seeds', 10)
    seeds_list = cfg.get('seeds_list', list(range(1, num_seeds + 1)))
    layer_indices = cfg.get('layers')

    if layer_indices is None:
        layer_indices = list(range(num_layers))
    else:
        layer_indices = [int(layer) for layer in layer_indices]

    result_filename = get_run_paths(results_dir)["dists"]
    result_filename.parent.mkdir(parents=True, exist_ok=True)

    jobs = []
    for idx, seed1 in enumerate(seeds_list):
        for seed2 in seeds_list[idx:]:
            for layer1 in layer_indices:
                for layer2 in layer_indices:
                    if seed1 == seed2 and layer1 > layer2:
                        continue
                    rep1_dict = {"dataset": dataset, "architecture": architecture, "seed": seed1, "step": step, "layer": layer1}
                    rep2_dict = {"dataset": dataset, "architecture": architecture, "seed": seed2, "step": step, "layer": layer2}
                    jobs.append(make_pair_job(rep1_dict, rep2_dict))

    run_pair_jobs(jobs, metrics, result_filename, cfg, desc="layer_exp distance pairs")
