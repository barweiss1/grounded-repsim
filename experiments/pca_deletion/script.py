import pathlib
import pandas as pd
import pickle as pkl
import numpy as np
import sys

BASE_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))
REPO_DIR = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_DIR))
try:
    from .compute_full_df import get_acc
except ImportError:
    from compute_full_df import get_acc
from experiments.common import (
    filter_available_metrics,
    get_run_paths,
    resolve_resource,
    save_rank_corr_plot,
    write_rank_corr_results,
)
try:
    from ..utils import aggregate_rank_corrs
except ImportError:
    from utils import aggregate_rank_corrs

def run_experiment_script(cfg, results_dir, resources_path, device=None):
    paths = get_run_paths(results_dir)
    scores_path = resolve_resource(resources_path, cfg.get('scores_path', 'scores/pca_deletion/scores.pkl'))
    full_df_path = paths["full_df"]
    results_path = paths["results"]

    full_df = pd.read_csv(full_df_path)
    data_dict = pkl.load(open(scores_path, "rb"))

    ref_seeds = cfg.get('ref_seeds', [1,2,3,4,5,6,7])
    task = cfg.get('task', 'SST-2')

    def pca_sub_df(df, task_name, ref_depth):
        accs = [
            get_acc(data_dict, task_name, seed, layer=ref_depth, dims=0, run="average")
            for seed in ref_seeds
        ]
        acc_dict = dict(zip(ref_seeds, accs))
        best_seed = max(acc_dict, key=acc_dict.get)
        return df[
            (df.layer1 == ref_depth)
            & (df.layer2 == ref_depth)
            & ((df.seed1 == best_seed) | (df.seed2 == best_seed))
        ]

    metrics = cfg.get('metrics', ["Procrustes", "CKA", "PWCCA"])
    num_layers = cfg.get('num_layers', 12)
    layers = cfg.get('layers', [7,8,9,10,11])
    metrics_filtered = filter_available_metrics(metrics, full_df)

    rho, rho_p, tau, tau_p, bad_fracs = aggregate_rank_corrs(
        full_df, task, num_layers, metrics_filtered, pca_sub_df, list_layers=layers
    )

    write_rank_corr_results(results_path, metrics_filtered, rho, rho_p, tau, tau_p, bad_fracs)
    save_rank_corr_plot(results_dir, rho, rho_p, tau, tau_p, metrics_filtered, task)
