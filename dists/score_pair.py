import numpy as np
import sys
import os
import csv
import pathlib
import logging

from scoring import *
from compute_embeddings import compute_embeddings
from metrics import AlignmentMetrics
import json
import pathlib

sys.path.append(os.path.abspath(".."))
from paths import resources_path


def get_embedding_folder(dataset, architecture, seed, step, layer):
    suffix = pathlib.Path(f"embeddings/{dataset}/{architecture}/{seed}/{step}/{layer}")
    return resources_path / suffix

DEFAULT_SIM_PARAMS = {
    'metric_param_sweep_len': 30,
    'auc_integration_method': 'average',
    'auc_logscale': True,
    'auc_adaptive_rbf_sigma': False,
    'auc_adaptive_quantiles': (0.01, 0.8),
}

# new adaptive temperature config for softmax_rwka
DEFAULT_SIM_PARAMS.update({
    'auc_adaptive_temperature': False,
    'auc_adaptive_temperature_quantiles': (0.01, 0.8),
})


def _to_jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def load_embedding(
    dataset: str,
    architecture: str,  # in practice this can be feather, base, or medium_finetuned
    seed: int,
    step: int,
    layer: int,
) -> np.ndarray:

    # path to look for embedding
    folder_path = get_embedding_folder(dataset, architecture, seed, step, layer)

    # check if representations have already been computed; otherwise compute them:
    if not os.path.exists(folder_path):
        print("Computing representations for model")
        os.makedirs(folder_path)
        rep = compute_embeddings(dataset, architecture, seed, step, layer)

    else:
        print("Representation already exists...loading...")
        rep = np.load(folder_path / pathlib.Path("rep.npy"))
    return rep


def score_pair_to_csv(
    rep1_dict: dict,
    rep2_dict: dict,
    filename: str,
    metrics: list,
) -> None:
    """
    Compute metric distance between two representations and save it to a csv file

    Args:
        rep1_dict (dict): dictionary specifying configuration of representation 1, to load its representation from disk
        rep2_dict (dict): dictionary specifying configuration of representation 2, to load its representation from disk
        filename (str): output filename to save results to
        metrics (list, optional): list of metrics to apply, eg CCA and/or CKA and/or GLD (by default all)
    """

    rep1 = load_embedding(
        rep1_dict["dataset"],
        rep1_dict["architecture"],
        rep1_dict["seed"],
        rep1_dict["step"],
        rep1_dict["layer"],
    )
    rep2 = load_embedding(
        rep2_dict["dataset"],
        rep2_dict["architecture"],
        rep2_dict["seed"],
        rep2_dict["step"],
        rep2_dict["layer"],
    )

    logging.info(f"representation 1 shape: {rep1.shape}")
    logging.info(f"representation 2 shape: {rep2.shape}")

    results = {
        "dataset1": rep1_dict["dataset"],
        "architecture1": rep1_dict["architecture"],
        "seed1": rep1_dict["seed"],
        "step1": rep1_dict["step"],
        "layer1": rep1_dict["layer"],
        "dataset2": rep2_dict["dataset"],
        "architecture2": rep2_dict["architecture"],
        "seed2": rep2_dict["seed"],
        "step2": rep2_dict["step"],
        "layer2": rep2_dict["layer"],
    }

    score_local_pair(
        rep1=rep1, rep2=rep2, metrics=metrics, filename=filename, metadata=results
    )


def score_local_pair(
    rep1: np.ndarray,
    rep2: np.ndarray,
    filename: str,
    metrics: list,
    metadata: dict = {},
    sim_params: dict = DEFAULT_SIM_PARAMS,
) -> None:
    """
    Compute metric distances between two representations (in numpy array format) and
    save results to a csv file

    Args:
        rep1 (np.ndarray): representation 1 to compare
        rep2 (np.ndarray): representation 2 to compare
        filename (str): file name for output csv
        metrics (list, optional): list of metrics to apply (by default all)
        metadata (dict, optional): metadata for the representations to print to the csv (by default empty)
    """

    # center each row
    rep1 = rep1 - rep1.mean(axis=1, keepdims=True)
    rep2 = rep2 - rep2.mean(axis=1, keepdims=True)

    # normalize each representation
    rep1 = rep1 / np.linalg.norm(rep1)
    rep2 = rep2 / np.linalg.norm(rep2)

    results = metadata

    ## CCA like: first decompose, then compute metrics themselves
    if (
        "PWCCA" in metrics
        or "mean_sq_cca_corr" in metrics
        or "mean_cca_corr" in metrics
    ):
        logging.info("Computing CCA decomposition...")
        cca_u, cca_rho, cca_vh, transformed_rep1, transformed_rep2 = cca_decomp(
            rep1, rep2
        )

        if "PWCCA" in metrics:
            logging.info("Computing PWCCA distance...")
            results["PWCCA"] = pwcca_dist(rep1, cca_rho, transformed_rep1)
        if "mean_sq_cca_corr" in metrics:
            logging.info("Computing mean square CCA corelation...")
            results["mean_sq_cca_corr"] = mean_sq_cca_corr(cca_rho)
        if "mean_cca_corr" in metrics:
            logging.info("Computing mean CCA corelation...")
            results["mean_cca_corr"] = mean_cca_corr(cca_rho)

    ## CKA like
    if "CKA" in metrics:
        logging.info("Computing Linear CKA dist...")
        lin_cka_sim = lin_cka_dist(rep1, rep2)
        results["CKA"] = lin_cka_sim

    if "CKA'" in metrics:
        logging.info("Computing Linear CKA' dist...")
        lin_cka_sim = lin_cka_prime_dist(rep1, rep2)
        results["CKA'"] = lin_cka_sim

    ## Procrustes
    if "Procrustes" in metrics:
        logging.info("Computing GLD dist...")
        results["Procrustes"] = procrustes(rep1, rep2)

    # My metrics: compute metric distance and also AUC distance if specified
    for metric in AlignmentMetrics.SUPPORTED_METRICS:
        if metric in metrics:
            logging.info(f"Computing {metric} dist...")
            try:
                results[metric] = calc_metric_dist(
                    feats_A=rep1, feats_B=rep2, metric_name=metric
                )
            except Exception as e:
                logging.warning(f"Skipping metric {metric} due to error: {e}")
        if metric + "_auc" in metrics:
            logging.info(f"Computing {metric}_auc dist...")
            try:
                auc_result = calc_metric_auc_dist(
                    feats_A=rep1,
                    feats_B=rep2,
                    metric_name=metric,
                    sweep_len=sim_params['metric_param_sweep_len'],
                    integration_method=sim_params['auc_integration_method'],
                    logscale=sim_params['auc_logscale'],
                    adaptive_rbf_sigma=sim_params.get('auc_adaptive_rbf_sigma', False),
                    adaptive_quantiles=sim_params.get('auc_adaptive_quantiles', (0.01, 0.8)),
                    adaptive_temperature=sim_params.get('auc_adaptive_temperature', False),
                    adaptive_temperature_quantiles=sim_params.get('auc_adaptive_temperature_quantiles', (0.01, 0.8)),
                    return_sweep=True,
                )
                # auc_result can be (auc_dist, param_vec, scores)
                if isinstance(auc_result, tuple) and len(auc_result) == 3:
                    auc_dist, param_vec, scores = auc_result
                    results[metric + "_auc"] = auc_dist

                    # save sweep data to a json file alongside the CSV
                    try:
                        csv_path = pathlib.Path(filename)
                        sweeps_dir = csv_path.parent / pathlib.Path("sweeps")
                        sweeps_dir.mkdir(parents=True, exist_ok=True)
                        # construct a short, unique filename using metadata
                        meta_parts = [
                            str(results.get('architecture1', 'a1')),
                            f"s{results.get('seed1', 's1')}",
                            f"l{results.get('layer1', 'l1')}",
                            "vs",
                            str(results.get('architecture2', 'a2')),
                            f"s{results.get('seed2', 's2')}",
                            f"l{results.get('layer2', 'l2')}",
                        ]
                        # Include dims_deleted if present to avoid overwrites for different deletion levels
                        dims_deleted = results.get('dims_deleted', None)
                        if dims_deleted is not None:
                            meta_parts.append(f"d{dims_deleted}")
                        safe_name = "__".join(meta_parts)
                        sweep_file = sweeps_dir / pathlib.Path(f"{metric}_auc__{safe_name}.json")
                        # include sweep param name and values for downstream post-processing
                        sweep_param = AlignmentMetrics.SWEEP_PARAMS.get(metric, {}).get('param')
                        sweep_payload = {
                            'metadata': results,
                            'metric': metric,
                            'param_name': sweep_param,
                            'param_values': list(param_vec),
                            'scores': list(scores),
                        }
                        with open(sweep_file, 'w') as sf:
                            json.dump(_to_jsonable(sweep_payload), sf)
                    except Exception as e:
                        logging.warning(f"Failed saving sweep file for {metric}_auc: {e}")
                else:
                    # fallback if older return format
                    results[metric + "_auc"] = auc_result
            except Exception as e:
                logging.warning(f"Skipping metric {metric}_auc due to error: {e}")
    


    ## your metric here
    # function: my_metric_fn(rep1, rep2)
    # name: "my_new_metric"
    # if "my_new_metric" in metrics:
    #     logging.info("Computing my_new_metric dist...")
    #     results["my_new_metric"] = my_metric_fn(rep1, rep2)

    # Save results to file
    with open(filename, mode="a") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=results.keys())
        if csv_file.tell() == 0:
            writer.writeheader()
        writer.writerow(results)