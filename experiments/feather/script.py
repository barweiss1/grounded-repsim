scores_df = rename_scores(scores_df)
def feather_sub_df(df, task, ref_depth):
    # find best seed for the task
    seeds = list(df.seed1.unique())
    accs = [scores_df.iloc[seed][task] for seed in seeds]
    acc_dict = dict(zip(seeds, accs))
    best_seed = max(acc_dict, key=acc_dict.get)

    # select rows of full_df corresponding to the reference layer (layer depth and seed)
    sub_df = df[
        (df.layer1 == ref_depth)
        & (df.layer2 == ref_depth)
        & ((df.seed1 == best_seed) | (df.seed2 == best_seed))
    ]

    return sub_df
rho, rho_p, tau, tau_p, bad_fracs = aggregate_rank_corrs(
    full_df, task, num_layers, METRICS, feather_sub_df
)

# --- Refactored for wrapper ---
import numpy as np
import pandas as pd
import pathlib
from compute_full_df import rename_scores
from utils import aggregate_rank_corrs, plot_rank_corrs

def run_experiment_script(cfg, results_dir, resources_path):
    # Paths
    scores_path = resources_path / pathlib.Path("scores/feather/scores.csv")
    full_df_path = pathlib.Path(results_dir) / 'full_df_self_computed.csv'
    results_path = pathlib.Path(results_dir) / 'results.txt'

    # Load full_df
    full_df = pd.read_csv(full_df_path)
    full_df["mean_cca_corr"] = 1 - full_df["mean_cca_corr"]
    full_df["mean_sq_cca_corr"] = 1 - full_df["mean_sq_cca_corr"]

    # Load scores
    scores_df = pd.read_csv(scores_path)
    scores_df = rename_scores(scores_df)

    def feather_sub_df(df, task, ref_depth):
        seeds = list(df.seed1.unique())
        accs = [scores_df.iloc[seed][task] for seed in seeds]
        acc_dict = dict(zip(seeds, accs))
        best_seed = max(acc_dict, key=acc_dict.get)
        sub_df = df[
            (df.layer1 == ref_depth)
            & (df.layer2 == ref_depth)
            & ((df.seed1 == best_seed) | (df.seed2 == best_seed))
        ]
        return sub_df

    METRICS = cfg.get('metrics', ["PWCCA", "mean_cca_corr", "mean_sq_cca_corr", "CSA", "CKA", "Procrustes", 'cka_rbf', 'cka_rbf_quantile', 'cka_rbf_auc', 'mutual_knn', 'mutual_knn_auc', 'cknna'])
    task = cfg.get('task', "lex_nonent")
    num_layers = cfg.get('num_layers', 12)

    rho, rho_p, tau, tau_p, bad_fracs = aggregate_rank_corrs(
        full_df, task, num_layers, METRICS, feather_sub_df
    )
    metrics_filtered = [metric for metric in METRICS if metric in full_df.columns]
    with open(results_path, "w") as f:
        for metric in metrics_filtered:
            avg_rho = round(np.mean(rho[metric]), 3)
            avg_rho_p = format(np.mean(rho_p[metric]), ".1e")
            avg_tau = round(np.mean(tau[metric]), 3)
            avg_tau_p = format(np.mean(tau_p[metric]), ".1e")
            avg_bad_frac = round(np.mean(bad_fracs[metric]), 3)
            f.write(f"metric: {metric}\n")
            f.write(f"avg_rho: {avg_rho}\n")
            f.write(f"avg_rho_p: {avg_rho_p}\n")
            f.write(f"avg_tau: {avg_tau}\n")
            f.write(f"avg_tau_p: {avg_tau_p}\n")
            f.write(f"avg_bad_frac: {avg_bad_frac}\n\n")
    # Save plot to results_dir
    plot_path = pathlib.Path(results_dir) / f"rank_corrs_{task}.png"
    plot_rank_corrs(rho, rho_p, tau, tau_p, metrics_filtered, title=task)
