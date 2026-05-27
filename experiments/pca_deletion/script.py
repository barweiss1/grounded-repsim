import pathlib
import pandas as pd
import pickle as pkl
import numpy as np
from icecream import ic
import sys
import os

from compute_full_df import get_acc

BASE_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))
from utils import get_rank_corrs, plot_rank_corrs, aggregate_rank_corrs

def run_experiment_script(cfg, results_dir, resources_path):
    scores_path = resources_path / pathlib.Path(cfg.get('scores_path', 'scores/pca_deletion/scores.pkl'))
    full_df_path = pathlib.Path(results_dir) / 'full_df_self_computed.csv'
    results_path = pathlib.Path(results_dir) / 'results.txt'

    full_df = pd.read_csv(full_df_path)

    REF_SEEDS = cfg.get('REF_SEEDS', [1,2,3,4,5,6,7])

    def pca_sub_df(df, task, ref_depth):
        data_dict = pkl.load(open(scores_path, "rb"))
        accs = [
            get_acc(data_dict, probe_task, seed, layer=ref_depth, dims=0, run="average")
            for seed in REF_SEEDS
        ]
        acc_dict = dict(zip(REF_SEEDS, accs))
        best_seed = max(acc_dict, key=acc_dict.get)
        sub_df = df[
            (df.layer1 == ref_depth)
            & (df.layer2 == ref_depth)
            & ((df.seed1 == best_seed) | (df.seed2 == best_seed))
        ]
        return sub_df

    probe_task = cfg.get('probe_task', 'SST-2')
    METRICS = cfg.get('metrics', ["Procrustes", "CKA", "PWCCA"])
    num_layers = cfg.get('num_layers', 12)
    LAYERS = cfg.get('LAYERS', [7,8,9,10,11])

    rho, rho_p, tau, tau_p, bad_fracs = aggregate_rank_corrs(
        full_df, probe_task, num_layers, METRICS, pca_sub_df, list_layers=LAYERS
    )

    with open(results_path, "w") as f:
        for metric in METRICS:
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
            f.write("\n")

    plot_rank_corrs(rho, rho_p, tau, tau_p, METRICS, title=probe_task, save_path=pathlib.Path(results_dir) / f"rank_corrs_{probe_task}.png")


if __name__ == '__main__':
    # legacy behavior
    full_df_path = resources_path / pathlib.Path("full_dfs/pca_deletion/full_df.csv")
    full_df = pd.read_csv(full_df_path)
    probe_task = "SST-2"
    METRICS = ["Procrustes", "CKA", "PWCCA"]
    num_layers = 12
    LAYERS = [7,8,9,10,11]
    rho, rho_p, tau, tau_p, bad_fracs = aggregate_rank_corrs(full_df, probe_task, num_layers, METRICS, lambda df, t, d: df, list_layers=LAYERS)
    plot_rank_corrs(rho, rho_p, tau, tau_p, METRICS, title=probe_task)
