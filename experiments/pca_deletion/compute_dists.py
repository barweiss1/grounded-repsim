#!/usr/local/linux/anaconda3.8/bin/python

import numpy as np
import pathlib
import sys

BASE_DIR = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))
from experiments.common import (
    PAIR_KEY_COLUMNS,
    get_run_paths,
    get_sim_params,
    make_pair_job,
    pair_job_key,
    run_result_jobs,
)

sys.path.append(str(BASE_DIR / "dists"))
from score_pair import score_local_pair, load_embedding

PCA_KEY_COLUMNS = PAIR_KEY_COLUMNS + ["dims_deleted"]


def _pca_job_key(job, key_columns=PCA_KEY_COLUMNS):
    return pair_job_key(job, key_columns)


def _score_pca_deletion_job(payload):
    index, job, cfg = payload
    metrics = cfg.get('metrics', ["PWCCA", "mean_cca_corr", "mean_sq_cca_corr", "CSA", "CKA", "Procrustes"])
    total_dim = cfg.get('total_dim', 768)
    rep1_dict = job["rep1"]
    rep2_dict = job["rep2"]
    metadata = dict(job.get("metadata", {}))
    dim = int(metadata["dims_deleted"])
    dims_kept = int(total_dim - dim)

    rep1 = load_embedding(
        rep1_dict["dataset"],
        rep1_dict["architecture"],
        rep1_dict["seed"],
        rep1_dict["step"],
        rep1_dict["layer"],
    )
    rep1 = rep1 - rep1.mean(axis=1, keepdims=True)

    rep2 = load_embedding(
        rep2_dict["dataset"],
        rep2_dict["architecture"],
        rep2_dict["seed"],
        rep2_dict["step"],
        rep2_dict["layer"],
    )
    rep2 = rep2 - rep2.mean(axis=1, keepdims=True)
    u, s, vh = np.linalg.svd(rep2, full_matrices=False)

    basis = u[:, :dims_kept]
    deleted_rep = basis @ (basis.T @ rep2)
    deleted_rep = deleted_rep * total_dim / dims_kept

    row, sweep_payloads = score_local_pair(
        rep1=rep1,
        rep2=deleted_rep,
        metadata=metadata,
        metrics=metrics,
        sim_params=get_sim_params(cfg),
    )
    return index, row, sweep_payloads


def run_compute_dists(cfg, results_dir, resources_path, device=None):
    metrics = cfg.get('metrics', ["PWCCA", "mean_cca_corr", "mean_sq_cca_corr", "CSA", "CKA", "Procrustes"])
    dataset = cfg.get('dataset', 'mnli_matched_100')
    architecture = cfg.get('architecture', 'base')
    step = cfg.get('step', 2000000)
    num_layers = cfg.get('num_layers', 12)
    layer_indices = cfg.get('layers')
    num_seeds = cfg.get('num_seeds', 10)
    dims_deleted = np.array(cfg.get('dims_deleted', [0,100,200,300,400,500,600,650,700,725,750,758,763,767]))
    TOTAL_DIM = cfg.get('total_dim', 768)

    result_filename = get_run_paths(results_dir)["dists"]
    result_filename.parent.mkdir(parents=True, exist_ok=True)

    if layer_indices is None:
        layer_indices = list(range(num_layers))
    else:
        layer_indices = [int(layer) for layer in layer_indices]
        
    jobs = []
    for seed1 in range(1, num_seeds + 1):
        for seed2 in range(1, num_seeds + 1):
            for layer in layer_indices:
                rep1_dict = {"dataset": dataset, "architecture": architecture, "seed": seed1, "step": step, "layer": layer}
                rep2_dict = {"dataset": dataset, "architecture": architecture, "seed": seed2, "step": step, "layer": layer}
                for dim in dims_deleted:
                    dims_kept = int(TOTAL_DIM - dim)
                    metadata = {
                        "dataset1": dataset,
                        "architecture1": architecture,
                        "seed1": seed1,
                        "step1": step,
                        "layer1": layer,
                        "dataset2": dataset,
                        "architecture2": architecture,
                        "seed2": seed2,
                        "step2": step,
                        "layer2": layer,
                        "dims_deleted": int(dim),
                        "dims_kept": dims_kept,
                    }
                    jobs.append(make_pair_job(rep1_dict, rep2_dict, metadata=metadata))

    run_result_jobs(
        jobs,
        _score_pca_deletion_job,
        result_filename,
        cfg,
        desc="pca_deletion distance pairs",
        key_columns=PCA_KEY_COLUMNS,
        job_key_fn=_pca_job_key,
    )
