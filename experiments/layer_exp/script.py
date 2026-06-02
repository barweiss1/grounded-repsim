import numpy as np
import pandas as pd
import pathlib
import pickle as pkl
import sys
import os

BASE_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))
REPO_DIR = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_DIR))
from experiments.common import (
    filter_available_metrics,
    get_config_value,
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
    scores_path = resolve_resource(resources_path, cfg.get('scores_path', 'scores/layer_exp/scores.pkl'))
    full_df_path = resolve_resource(resources_path, cfg['full_df_path']) if 'full_df_path' in cfg else paths["full_df"]
    results_path = paths["results"]

    full_df = pd.read_csv(full_df_path)
    data_dict = pkl.load(open(scores_path, "rb"))

    def best_probing_seed(task, ref_depth, list_ref_seeds):
        list_to_max = [
            np.mean(data_dict[task][seed][ref_depth + 1][0][0]) for seed in list_ref_seeds
        ]
        idx, _ = max(enumerate(list_to_max), key=lambda x: x[1])
        return list_ref_seeds[idx]

    def layer_sub_df(df, task_name, ref_depth):
        ref_seed = best_probing_seed(task_name, ref_depth, ref_seeds)
        sub_df = df.loc[
            ((df["seed1"] == ref_seed) & (df["layer1"] == ref_depth))
            | ((df["seed2"] == ref_seed) & (df["layer2"] == ref_depth))
        ].reset_index()
        return sub_df

    task = cfg.get('task', 'SST-2')
    layers = get_config_value(cfg, 'layers', cfg.get('layer_depths', [11]), 'LAYERS')
    ref_seeds = get_config_value(cfg, 'ref_seeds', cfg.get('list_ref_seeds', list(range(1, 11))), 'REF_SEEDS')
    num_layers = cfg.get('num_layers', 12)
    metrics = cfg.get('metrics', ["PWCCA", "mean_cca_corr", "mean_sq_cca_corr", "CSA", "CKA", "Procrustes", 'cka_rbf', 'cka_rbf_quantile', 'cka_rbf_auc', 'mutual_knn', 'mutual_knn_auc', 'cknna'])

    metrics_filtered = filter_available_metrics(metrics, full_df)

    rho, rho_p, tau, tau_p, bad_fracs = aggregate_rank_corrs(
        full_df, task, num_layers, metrics_filtered, layer_sub_df, list_layers=layers
    )

    save_rank_corr_plot(results_dir, rho, rho_p, tau, tau_p, metrics_filtered, task)
    write_rank_corr_results(results_path, metrics_filtered, rho, rho_p, tau, tau_p, bad_fracs)


if __name__ == '__main__':
    sys.path.append(os.path.abspath('../../'))
    from paths import resources_path
    cfg = {
        'scores_path': 'scores/layer_exp/scores.pkl',
        'task': 'SST-2',
        'layers': [11],
        'ref_seeds': list(range(1, 11)),
        'metrics': ["PWCCA", "mean_cca_corr", "mean_sq_cca_corr", "CSA", "CKA", "Procrustes", 'cka_rbf', 'cka_rbf_quantile', 'cka_rbf_auc', 'mutual_knn', 'mutual_knn_auc', 'cknna'],
    }
    results_dir = resources_path / pathlib.Path("full_dfs/layer_exp/")
    run_experiment_script(cfg, results_dir, resources_path)
