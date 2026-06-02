#!/usr/local/linux/anaconda3.8/bin/python

import numpy as np
import pandas as pd
import pathlib
import pickle as pkl
import sys

BASE_DIR = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))
from experiments.common import get_run_paths, resolve_resource

def get_acc(data_dict, task, seed, layer, dims, run="average"):
    if run == "average":
        return np.mean(data_dict[task][seed][layer + 1][dims])
    elif run == "std":
        return np.std(data_dict[task][seed][layer + 1][dims])
    else:
        return data_dict[task][seed][layer + 1][dims][run]

def get_acc_diff(data_dict, row, task):
    acc1 = get_acc(data_dict, task=task, seed=row["seed1"], layer=row["layer1"], dims=0, run="average")
    acc2 = get_acc(data_dict, task=task, seed=row["seed2"], layer=row["layer2"], dims=row["dims_deleted"], run="average")
    return np.abs(acc1 - acc2)

def get_full_df(scores_path, dists_path, full_df_path, ref_seeds, layers, task):
    dists_df = pd.read_csv(dists_path)
    print("got dists_df")
    full_df = pd.DataFrame(
        dists_df[
            (dists_df["seed1"].isin(ref_seeds))
            & (dists_df["seed2"].isin(ref_seeds))
            & (dists_df["layer1"].isin(layers))
            & (dists_df["layer2"].isin(layers))
        ]
    )
    print("filtered full_df layers and seeds")
    print("adding probing scores to get full_df")
    data_dict = pkl.load(open(scores_path, "rb"))
    f = lambda row: get_acc_diff(data_dict, row, task)
    full_df[f"{task}_diff"] = full_df.apply(f, axis=1)
    print("got full_df, saving:")
    full_df.to_csv(full_df_path)
    print("saved")
    return full_df

def run_compute_full_df(cfg, results_dir, resources_path, device=None):
    paths = get_run_paths(results_dir)
    scores_path = resolve_resource(resources_path, cfg.get('scores_path', 'scores/pca_deletion/scores.pkl'))
    dists_path = paths["dists"]
    full_df_path = paths["full_df"]
    ref_seeds = cfg.get('ref_seeds', [1, 2, 3, 4, 5, 6, 7])
    layers = cfg.get('layers', [7, 8, 9, 10, 11])
    task = cfg.get('task', 'SST-2')
    get_full_df(scores_path, dists_path, full_df_path, ref_seeds=ref_seeds, layers=layers, task=task)
