#!/usr/local/linux/anaconda3.8/bin/python

import pathlib
from icecream import ic
import os
import sys

BASE_DIR = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))
from paths import resources_path
from experiments.common import get_run_paths

sys.path.append(str(BASE_DIR / "dists"))
from score_pair import score_pair_to_csv

def run_compute_dists(cfg, results_dir, resources_path, device=None):
    from tqdm import tqdm

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
    open(result_filename, 'w').close()

    pbar = tqdm(total=len(seeds_list) * len(seeds_list) * len(layer_indices) * len(layer_indices))
    for idx, seed1 in enumerate(seeds_list):
        for seed2 in seeds_list[idx:]:
            for layer1 in layer_indices:
                for layer2 in layer_indices:
                    pbar.update(1)
                    if seed1 == seed2 and layer1 > layer2:
                        pass
                    else:
                        ic(seed1, seed2, layer1, layer2)
                        rep1_dict = {"dataset": dataset, "architecture": architecture, "seed": seed1, "step": step, "layer": layer1}
                        rep2_dict = {"dataset": dataset, "architecture": architecture, "seed": seed2, "step": step, "layer": layer2}
                        score_pair_to_csv(rep1_dict, rep2_dict, result_filename, metrics)


if __name__ == '__main__':
    # legacy behavior
    cfg = {}
    results_dir = resources_path / pathlib.Path('dists/layer_exp/')
    run_compute_dists(cfg, results_dir, resources_path)
