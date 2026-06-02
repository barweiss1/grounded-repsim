#!/usr/local/linux/anaconda3.8/bin/python

import pathlib
import sys

BASE_DIR = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR / "dists"))

from experiments.common import get_run_paths
from score_pair import score_pair_to_csv

# --- Refactored for wrapper ---
def run_compute_dists(cfg, results_dir, resources_path, device=None):
    metrics = cfg.get('metrics')
    dataset = cfg.get('dataset')
    architecture = cfg.get('architecture')
    step = cfg.get('step')
    num_layers = cfg.get('num_layers')
    num_seeds = cfg.get('num_seeds')

    result_filename = get_run_paths(results_dir)["dists"]
    result_filename.parent.mkdir(parents=True, exist_ok=True)
    open(result_filename, 'w').close()

    for seed1 in range(num_seeds):
        for seed2 in range(seed1, num_seeds):
            for layer in range(num_layers):
                rep1_dict = {"dataset": dataset, "architecture": architecture, "seed": seed1, "step": step, "layer": layer}
                rep2_dict = {"dataset": dataset, "architecture": architecture, "seed": seed2, "step": step, "layer": layer}
                score_pair_to_csv(rep1_dict, rep2_dict, result_filename, metrics)

# --- Legacy CLI fallback ---
if __name__ == "__main__":
    from paths import resources_path
    cfg = {
        'metrics': ["PWCCA", "mean_cca_corr", "mean_sq_cca_corr", "CSA", "CKA", "Procrustes", 'cka_rbf', 'cka_rbf_quantile', 'cka_rbf_auc', 'mutual_knn', 'mutual_knn_auc', 'cknna'],
        'dataset': "mnli_matched_100",
        'architecture': "feather",
        'step': 0,
        'num_layers': 12,
        'num_seeds': 100
    }
    results_dir = resources_path / pathlib.Path("dists/feather/")
    run_compute_dists(cfg, results_dir, resources_path)
