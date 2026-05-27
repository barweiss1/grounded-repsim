avg_rho = {metric: round(np.mean(rho[metric]), 3) for metric in metrics_filtered}
import numpy as np
import pandas as pd
import pathlib
import pickle as pkl
import sys
import os
from icecream import ic


sys.path.append(os.path.abspath("../"))
from utils import plot_rank_corrs, get_rank_corrs

def run_experiment_script(cfg, results_dir, resources_path):
    scores_path = resources_path / pathlib.Path(cfg.get('scores_path', 'scores/layer_exp/scores.pkl'))
    full_df_path = pathlib.Path(results_dir) / 'full_df_self_computed.csv'
    results_path = pathlib.Path(results_dir) / 'results.txt'

    full_df = pd.read_csv(full_df_path)

    def best_probing_seed(task, ref_depth, list_ref_seeds):
        data_dict = pkl.load(open(scores_path, "rb"))
        list_to_max = [
            np.mean(data_dict[task][seed][ref_depth + 1][0][0]) for seed in list_ref_seeds
        ]
        idx, _ = max(enumerate(list_to_max), key=lambda x: x[1])
        return list_ref_seeds[idx]

    def layer_sub_df(df, ref_depth, ref_seed):
        sub_df = df.loc[
            ((df["seed1"] == ref_seed) & (df["layer1"] == ref_depth))
            | ((df["seed2"] == ref_seed) & (df["layer2"] == ref_depth))
        ].reset_index()
        return sub_df

    task = cfg.get('task', 'SST-2')
    layer_depths = cfg.get('layer_depths', [11])
    list_ref_seeds = cfg.get('list_ref_seeds', list(range(1, 11)))
    METRICS = cfg.get('metrics', ["PWCCA", "mean_cca_corr", "mean_sq_cca_corr", "CSA", "CKA", "Procrustes", 'cka_rbf', 'cka_rbf_quantile', 'cka_rbf_auc', 'mutual_knn', 'mutual_knn_auc', 'cknna'])

    metrics_filtered = [metric for metric in METRICS if metric in full_df.columns]

    rho, rho_p, tau, tau_p, bad_fracs = aggregate_rank_corrs(
        full_df, task, layer_depths, list_ref_seeds, metrics_filtered, layer_sub_df
    )

    plot_rank_corrs(rho, rho_p, tau, tau_p, metrics_filtered, title=task, save_path=pathlib.Path(results_dir) / f"rank_corrs_{task}.png")

    avg_rho = {metric: round(np.mean(rho[metric]), 3) for metric in metrics_filtered}
    avg_rho_p = {metric: round(np.mean(rho_p[metric]), 3) for metric in metrics_filtered}
    avg_tau = {metric: round(np.mean(tau[metric]), 3) for metric in metrics_filtered}
    avg_tau_p = {metric: round(np.mean(tau_p[metric]), 3) for metric in metrics_filtered}

    with open(results_path, "w") as f:
        for metric in metrics_filtered:
            f.write(f"metric: {metric}\n")
            f.write(f"avg_rho: {avg_rho[metric]}\n")
            f.write(f"avg_rho_p: {avg_rho_p[metric]}\n")
            f.write(f"avg_tau: {avg_tau[metric]}\n")
            f.write(f"avg_tau_p: {avg_tau_p[metric]}\n")
            f.write("\n")


if __name__ == '__main__':
    sys.path.append(os.path.abspath('../../'))
    from paths import resources_path
    scores_path = resources_path / pathlib.Path("scores/layer_exp/scores.pkl")
    full_df_path = resources_path / pathlib.Path("full_dfs/layer_exp/full_df_self_computed.csv")
    # original behavior
    full_df = pd.read_csv(full_df_path)
    # keep existing plotting behavior
    task = "SST-2"
    layer_depths = [11]
    list_ref_seeds = list(range(1,11))
    METRICS = ["PWCCA", "mean_cca_corr", "mean_sq_cca_corr", "CSA", "CKA", "Procrustes", 'cka_rbf', 'cka_rbf_quantile', 'cka_rbf_auc', 'mutual_knn', 'mutual_knn_auc', 'cknna']
    metrics_filtered = [metric for metric in METRICS if metric in full_df.columns]
    rho, rho_p, tau, tau_p, bad_fracs = aggregate_rank_corrs(full_df, task, layer_depths, list_ref_seeds, metrics_filtered, lambda df, d, s: df)
    plot_rank_corrs(rho, rho_p, tau, tau_p, metrics_filtered, title=task)
