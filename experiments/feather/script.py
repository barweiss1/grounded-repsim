import pathlib
import sys

import pandas as pd
from tqdm import tqdm

BASE_DIR = pathlib.Path(__file__).resolve().parents[1]
REPO_DIR = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))
sys.path.append(str(REPO_DIR))

try:
    from .compute_full_df import rename_scores
except ImportError:
    from compute_full_df import rename_scores
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
    scores_path = resolve_resource(resources_path, cfg.get('scores_path', 'scores/feather/scores.csv'))
    full_df_path = paths["full_df"]
    results_path = paths["results"]

    full_df = pd.read_csv(full_df_path)
    for metric in ["mean_cca_corr", "mean_sq_cca_corr"]:
        if metric in full_df.columns:
            full_df[metric] = 1 - full_df[metric]

    scores_df = rename_scores(pd.read_csv(scores_path))

    def feather_sub_df(df, task, ref_depth):
        seeds = list(df.seed1.unique())
        accs = [scores_df.iloc[seed][task] for seed in seeds]
        acc_dict = dict(zip(seeds, accs))
        best_seed = max(acc_dict, key=acc_dict.get)
        return df[
            (df.layer1 == ref_depth)
            & (df.layer2 == ref_depth)
            & ((df.seed1 == best_seed) | (df.seed2 == best_seed))
        ]

    metrics = cfg.get('metrics', ["PWCCA", "mean_cca_corr", "mean_sq_cca_corr", "CSA", "CKA", "Procrustes", 'cka_rbf', 'cka_rbf_quantile', 'cka_rbf_auc', 'mutual_knn', 'mutual_knn_auc', 'cknna'])
    task = cfg.get('task', "lex_nonent")
    num_layers = cfg.get('num_layers', 12)
    metrics_filtered = filter_available_metrics(metrics, full_df)

    with tqdm(total=1, desc="feather analysis tasks", unit="task") as pbar:
        rho, rho_p, tau, tau_p, bad_fracs = aggregate_rank_corrs(
            full_df, task, num_layers, metrics_filtered, feather_sub_df
        )
        pbar.update(1)

    write_rank_corr_results(results_path, metrics_filtered, rho, rho_p, tau, tau_p, bad_fracs)
    save_rank_corr_plot(results_dir, rho, rho_p, tau, tau_p, metrics_filtered, task)
