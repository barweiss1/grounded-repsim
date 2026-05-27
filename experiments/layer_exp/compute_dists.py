#!/usr/local/linux/anaconda3.8/bin/python

import pathlib
from icecream import ic
import os
import sys

sys.path.append(os.path.abspath("../../"))
from paths import resources_path

sys.path.append(os.path.abspath("../../dists/"))
from score_pair import score_pair_to_csv

def run_compute_dists(cfg, results_dir, resources_path):
    import pathlib
    from tqdm import tqdm

    metrics = cfg.get('metrics', ["CKA", "Procrustes", 'cka_rbf', 'cka_rbf_quantile', 'cka_rbf_auc', 'mutual_knn', 'mutual_knn_auc', 'cknna'])
    dataset = cfg.get('dataset', 'mnli_matched_100')
    architecture = cfg.get('architecture', 'base')
    step = cfg.get('step', 2000000)
    num_layers = cfg.get('num_layers', 12)
    seeds_list = cfg.get('seeds_list', list(range(1, 11)))

    result_filename = pathlib.Path(results_dir) / 'dists_self_computed.csv'
    open(result_filename, 'w').close()

    pbar = tqdm(total=len(seeds_list) * len(seeds_list) * num_layers * num_layers)
    for idx, seed1 in enumerate(seeds_list):
        for seed2 in seeds_list[idx:]:
            for layer1 in range(num_layers):
                for layer2 in range(num_layers):
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
