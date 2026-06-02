#!/usr/local/linux/anaconda3.8/bin/python


import pandas as pd
import pathlib
import sys
from tqdm import tqdm

BASE_DIR = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))
from experiments.common import get_run_paths, resolve_resource

def get_acc_diff(row, scores_df, task_list):
    score_row1 = scores_df.iloc[row["seed1"]]
    score_row2 = scores_df.iloc[row["seed2"]]
    for task in task_list:
        acc1 = score_row1[task]
        acc2 = score_row2[task]
        row[f"{task}_diff"] = abs(acc1 - acc2)
    return row

def rename_scores(scores_df):
    scores_df = scores_df.rename(
        columns={
            "MNLI dev acc.": "mnli_dev_acc",
            "Lexical (entailed)": "lex_ent",
            "Subseq (entailed)": "sub_ent",
            "Constituent (entailed)": "const_ent",
            "Lexical (nonent)": "lex_nonent",
            "Subseq (nonent)": "sub_nonent",
            "Constituent (nonent)": "const_nonent",
            "Overall accuracy": "overall_accuracy",
        }
    )
    return scores_df

def get_full_df(scores_path, dists_path, full_df_path):
    scores_df = pd.read_csv(scores_path)[0:100]
    scores_df = rename_scores(scores_df)
    task_list = list(scores_df.columns[1:9])
    print("got scores_df")
    dists_df = pd.read_csv(dists_path)
    print("got dists_df")
    print("getting full_df, will take a while")
    tqdm.pandas(desc="feather score diffs")
    full_df = dists_df.progress_apply(
        lambda row: get_acc_diff(row, scores_df, task_list), axis=1
    )
    print("got full_df, saving:")
    full_df.to_csv(full_df_path)
    print("saved")
    return full_df

# --- Refactored for wrapper ---
def run_compute_full_df(cfg, results_dir, resources_path, device=None):
    # Default paths
    paths = get_run_paths(results_dir)
    scores_path = resolve_resource(resources_path, cfg.get('scores_path', 'scores/feather/scores.csv'))
    dists_path = paths["dists"]
    full_df_path = paths["full_df"]
    get_full_df(scores_path, dists_path, full_df_path)
