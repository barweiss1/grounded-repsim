#!/usr/local/linux/anaconda3.8/bin/python

import pathlib
from icecream import ic
import os
import sys
import itertools

sys.path.append(os.path.abspath("../.."))
from paths import resources_path

sys.path.append(os.path.abspath("../../dists/"))
from score_pair import score_pair_to_csv

def run_compute_dists(cfg, results_dir, resources_path):
    metrics = cfg.get('metrics', ["PWCCA", "mean_cca_corr", "mean_sq_cca_corr", "CSA", "CKA", "Procrustes"])
    dataset = cfg.get('dataset', 'mnli_matched_100')
    architecture = cfg.get('architecture', 'medium_finetuned')
    num_layers = cfg.get('num_layers', 8)
    num_seeds = cfg.get('num_seeds', 10)
    num_fseeds = cfg.get('num_fseeds', 10)

    all_model_seeds = list(itertools.product(range(1, 1 + num_seeds), range(1, 1 + num_fseeds)))

    result_filename = pathlib.Path(results_dir) / 'dists_self_computed.csv'
    if not result_filename.parent.exists():
        result_filename.parent.mkdir(parents=True, exist_ok=True)
    open(result_filename, 'w').close()

    for idx, (pseed1, fseed1) in enumerate(all_model_seeds):
        for (pseed2, fseed2) in all_model_seeds[idx:]:
            for layer in range(num_layers):
                ic(pseed1, fseed1, pseed2, fseed2, layer)
                rep1_dict = {"dataset": dataset, "architecture": architecture, "seed": pseed1, "step": fseed1, "layer": layer}
                rep2_dict = {"dataset": dataset, "architecture": architecture, "seed": pseed2, "step": fseed2, "layer": layer}
                score_pair_to_csv(rep1_dict, rep2_dict, result_filename, metrics)


if __name__ == '__main__':
    cfg = {}
    results_dir = resources_path / pathlib.Path('dists/pretrain_finetune/')
    run_compute_dists(cfg, results_dir, resources_path)
