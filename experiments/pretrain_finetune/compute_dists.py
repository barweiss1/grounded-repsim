#!/usr/local/linux/anaconda3.8/bin/python

import pathlib
import sys
import itertools

BASE_DIR = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))
from experiments.common import get_run_paths, make_pair_job, run_pair_jobs

def run_compute_dists(cfg, results_dir, resources_path, device=None):
    metrics = cfg.get('metrics', ["PWCCA", "mean_cca_corr", "mean_sq_cca_corr", "CSA", "CKA", "Procrustes"])
    dataset = cfg.get('dataset', 'mnli_matched_100')
    architecture = cfg.get('architecture', 'medium_finetuned')
    num_layers = cfg.get('num_layers', 8)
    layer_indices = cfg.get('layers')
    num_seeds = cfg.get('num_seeds', 10)
    num_fseeds = cfg.get('num_fseeds', 10)
    pre_seeds = cfg.get('pre_seeds', list(range(1, 1 + num_seeds)))
    fine_seeds = cfg.get('fine_seeds', list(range(1, 1 + num_fseeds)))

    if layer_indices is None:
        layer_indices = list(range(num_layers))
    else:
        layer_indices = [int(layer) for layer in layer_indices]

    all_model_seeds = list(itertools.product(pre_seeds, fine_seeds))

    result_filename = get_run_paths(results_dir)["dists"]
    result_filename.parent.mkdir(parents=True, exist_ok=True)

    jobs = []
    for idx, (pseed1, fseed1) in enumerate(all_model_seeds):
        for (pseed2, fseed2) in all_model_seeds[idx:]:
            for layer in layer_indices:
                rep1_dict = {"dataset": dataset, "architecture": architecture, "seed": pseed1, "step": fseed1, "layer": layer}
                rep2_dict = {"dataset": dataset, "architecture": architecture, "seed": pseed2, "step": fseed2, "layer": layer}
                jobs.append(make_pair_job(rep1_dict, rep2_dict))

    run_pair_jobs(
        jobs,
        metrics,
        result_filename,
        cfg,
        desc="pretrain_finetune distance pairs",
    )
