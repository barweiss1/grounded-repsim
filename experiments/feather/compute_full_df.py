#!/usr/local/linux/anaconda3.8/bin/python


import pandas as pd
import pathlib
import os
import sys

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
    from tqdm import tqdm
    tqdm.pandas()
    full_df = dists_df.progress_apply(
        lambda row: get_acc_diff(row, scores_df, task_list), axis=1
    )
    print("got full_df, saving:")
    full_df.to_csv(full_df_path)
    print("saved")
    return full_df

# --- Refactored for wrapper ---
def run_compute_full_df(cfg, results_dir, resources_path):
    import pathlib
    # Default paths
    scores_path = resources_path / pathlib.Path("scores/feather/scores.csv")
    dists_path = pathlib.Path(results_dir) / 'dists_self_computed.csv'
    full_df_path = pathlib.Path(results_dir) / 'full_df_self_computed.csv'
    get_full_df(scores_path, dists_path, full_df_path)

# --- Legacy CLI fallback ---
if __name__ == "__main__":
    from paths import resources_path
    scores_path = resources_path / pathlib.Path("scores/feather/scores.csv")
    dists_path = resources_path / pathlib.Path("dists/feather/dists_self_computed.csv")
    full_df_path = resources_path / pathlib.Path("full_dfs/feather/full_df_self_computed.csv")
    get_full_df(scores_path, dists_path, full_df_path)