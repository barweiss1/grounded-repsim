import pathlib
import pandas as pd
import sys
import os

BASE_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))
REPO_DIR = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_DIR))
try:
    from .compute_full_df import collect_scores
except ImportError:
    from compute_full_df import collect_scores
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
    scores_path = resolve_resource(resources_path, cfg.get('scores_path', 'scores/pretrain_finetune/scores.pkl'))
    full_df_path = resolve_resource(resources_path, cfg['full_df_path']) if 'full_df_path' in cfg else paths["full_df"]
    results_path = paths["results"]

    full_df = pd.read_csv(full_df_path)
    _, acc_dict = collect_scores(scores_path)

    ref_seed_pair = cfg.get('ref_seed_pair')

    def best_seed_pair(task, df):
        if ref_seed_pair is not None:
            return int(ref_seed_pair[0]), int(ref_seed_pair[1])

        candidates = set(
            zip(df["pre_seed1"].astype(int), df["fine_seed1"].astype(int))
        ) | set(zip(df["pre_seed2"].astype(int), df["fine_seed2"].astype(int)))
        return max(
            candidates,
            key=lambda seeds: acc_dict[task][seeds[0] - 1][seeds[1] - 1],
        )

    def ftvft_sub_df(df, task, ref_depth):
        best_pre_seed, best_fine_seed = best_seed_pair(task, df)
        sub_df = df[
            (df.layer1 == ref_depth)
            & (df.layer2 == ref_depth)
            & (
                ((df.pre_seed1 == best_pre_seed) & (df.fine_seed1 == best_fine_seed))
                | ((df.pre_seed2 == best_pre_seed) & (df.fine_seed2 == best_fine_seed))
            )
        ]
        return sub_df

    metrics = cfg.get('metrics', ["Procrustes", "CKA", "PWCCA"])
    num_layers = cfg.get('num_layers', 8)
    layers = get_config_value(cfg, 'layers', None, 'LAYERS')
    tasks = cfg.get('tasks', ["STRESS_ANTONYMY", "STRESS_NUMERICAL"])
    metrics_filtered = filter_available_metrics(metrics, full_df)

    for idx, task in enumerate(tasks):
        rho, rho_p, tau, tau_p, bad_fracs = aggregate_rank_corrs(
            full_df, task, num_layers, metrics_filtered, ftvft_sub_df, list_layers=layers
        )
        write_rank_corr_results(
            results_path,
            metrics_filtered,
            rho,
            rho_p,
            tau,
            tau_p,
            bad_fracs,
            header=f"task: {task}",
            mode="w" if idx == 0 else "a",
        )
        save_rank_corr_plot(results_dir, rho, rho_p, tau, tau_p, metrics_filtered, task)


if __name__ == '__main__':
    from paths import resources_path
    cfg = {
        'scores_path': 'scores/pretrain_finetune/scores.pkl',
        'full_df_path': 'full_dfs/pretrain_finetune/full_df.csv',
        'tasks': ["STRESS_ANTONYMY"],
        'num_layers': 8,
        'metrics': ["Procrustes", "CKA", "PWCCA"],
    }
    results_dir = resources_path / pathlib.Path("full_dfs/pretrain_finetune/")
    run_experiment_script(cfg, results_dir, resources_path)
