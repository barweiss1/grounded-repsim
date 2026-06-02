#!/usr/local/linux/anaconda3.8/bin/python

import pickle as pkl
import numpy as np
import pathlib
import pandas as pd
import sys

BASE_DIR = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))
from experiments.common import get_run_paths, resolve_resource

# list of probing tasks
task_list = ["QNLI", "SST-2"]


def get_probing_accuracy(data_dict, task, seed, depth):
    """
    average accuracy of model finetuned with finetuning seed seed on mnli
    when probing layer layer on task
    """
    # delete 0 neurons with deletion seed 0
    return np.mean(data_dict[task][seed][depth + 1][0][0])


# TODO: this is different from but better than what we had originally
def get_full_df(scores_path, dists_path, full_df_path):
    # get pairwise distances df
    dists_df = pd.read_csv(dists_path)
    print("got dists_df")

    # add probing accuracy differences to dists_df to get full_df
    print("adding probing scores to get full_df")
    full_df = dists_df
    data_dict = pkl.load(open(scores_path, "rb"))
    for task in task_list:
        task_diff_list = []
        for _, row in dists_df.iterrows():
            acc1 = get_probing_accuracy(data_dict, task, row["seed1"], row["layer1"])
            acc2 = get_probing_accuracy(data_dict, task, row["seed2"], row["layer2"])
            task_diff_list.append(np.abs(acc1 - acc2))

        full_df[f"{task}_diff"] = np.array(task_diff_list)

    print("got full_df, saving:")
    full_df.to_csv(full_df_path)
    print("saved")
    return full_df

def run_compute_full_df(cfg, results_dir, resources_path, device=None):
    paths = get_run_paths(results_dir)
    scores_path = resolve_resource(resources_path, cfg.get('scores_path', 'scores/layer_exp/scores.pkl'))
    dists_path = paths["dists"]
    full_df_path = paths["full_df"]
    get_full_df(scores_path, dists_path, full_df_path)
