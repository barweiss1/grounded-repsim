#!/usr/local/linux/anaconda3.8/bin/python

import numpy as np
import os
import pathlib
import sys

BASE_DIR = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))
from paths import resources_path

sys.path.append(str(BASE_DIR / "dists"))
from score_pair import score_local_pair, load_embedding

def run_compute_dists(cfg, results_dir, resources_path):
    metrics = cfg.get('metrics', ["PWCCA", "mean_cca_corr", "mean_sq_cca_corr", "CSA", "CKA", "Procrustes"])
    dataset = cfg.get('dataset', 'mnli_matched_100')
    architecture = cfg.get('architecture', 'base')
    step = cfg.get('step', 2000000)
    num_layers = cfg.get('num_layers', 12)
    layer_indices = cfg.get('layers')
    num_seeds = cfg.get('num_seeds', 10)
    dims_deleted = np.array(cfg.get('dims_deleted', [0,100,200,300,400,500,600,650,700,725,750,758,763,767]))
    TOTAL_DIM = cfg.get('total_dim', 768)

    result_filename = pathlib.Path(results_dir) / 'dists_self_computed.csv'
    if not result_filename.parent.exists():
        result_filename.parent.mkdir(parents=True, exist_ok=True)
    open(result_filename, 'w').close()

    if layer_indices is None:
        layer_indices = list(range(num_layers))
    else:
        layer_indices = [int(layer) for layer in layer_indices]
        
    # add progress bar
    from tqdm import tqdm
    total_iterations = num_seeds * num_seeds * len(layer_indices)
    pbar = tqdm(total=total_iterations)

    for seed1 in range(1, num_seeds + 1):
        for seed2 in range(1, num_seeds + 1):
            for layer in layer_indices:
                pbar.update(1)
                rep1 = load_embedding(dataset, architecture, seed1, step, layer)
                rep1 = rep1 - rep1.mean(axis=1, keepdims=True)

                rep2 = load_embedding(dataset, architecture, seed2, step, layer)
                rep2 = rep2 - rep2.mean(axis=1, keepdims=True)
                u, s, vh = np.linalg.svd(rep2, full_matrices=False)

                for dim in dims_deleted:
                    dims_kept = TOTAL_DIM - dim
                    # Keep representation in the original feature space after deleting
                    # high-rank principal components so metrics see matching shapes.
                    basis = u[:, :dims_kept]
                    deleted_rep = basis @ (basis.T @ rep2)
                    deleted_rep = deleted_rep * TOTAL_DIM / dims_kept

                    results = {
                        "dataset1": dataset,
                        "architecture1": architecture,
                        "seed1": seed1,
                        "step1": step,
                        "layer1": layer,
                        "dataset2": dataset,
                        "architecture2": architecture,
                        "seed2": seed2,
                        "step2": step,
                        "layer2": layer,
                        "dims_deleted": int(dim),
                        "dims_kept": int(dims_kept),
                    }
                    # build sim params from cfg to allow configuring adaptive sweeps
                    sim_params = {
                        'metric_param_sweep_len': cfg.get('metric_param_sweep_len', 30),
                        'auc_integration_method': cfg.get('auc_integration_method', 'average'),
                        'auc_logscale': cfg.get('auc_logscale', True),
                        'auc_adaptive_rbf_sigma': cfg.get('auc_adaptive_rbf_sigma', False),
                        'auc_adaptive_quantiles': tuple(cfg.get('auc_adaptive_quantiles', (0.01, 0.8))),
                        'auc_adaptive_temperature': cfg.get('auc_adaptive_temperature', False),
                        'auc_adaptive_temperature_quantiles': tuple(cfg.get('auc_adaptive_temperature_quantiles', (0.01, 0.8))),
                    }

                    score_local_pair(
                        rep1=rep1,
                        rep2=deleted_rep,
                        metadata=results,
                        metrics=metrics,
                        filename=result_filename,
                        sim_params=sim_params,
                    )


if __name__ == '__main__':
    cfg = {}
    results_dir = resources_path / pathlib.Path('dists/pca_deletion/')
    run_compute_dists(cfg, results_dir, resources_path)
