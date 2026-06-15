import numpy as np
import sys
import os
import csv
import pathlib
import logging
from collections import OrderedDict

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

_EMBEDDING_CACHE = OrderedDict()
_EMBEDDING_CACHE_SIZE = 0


def set_embedding_cache_size(cache_size: int) -> None:
    global _EMBEDDING_CACHE_SIZE
    _EMBEDDING_CACHE_SIZE = max(0, int(cache_size))
    if _EMBEDDING_CACHE_SIZE == 0:
        _EMBEDDING_CACHE.clear()


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
    cache_key = (dataset, architecture, seed, step, layer)
    if _EMBEDDING_CACHE_SIZE > 0 and cache_key in _EMBEDDING_CACHE:
        rep = _EMBEDDING_CACHE.pop(cache_key)
        _EMBEDDING_CACHE[cache_key] = rep
        return rep

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
    if _EMBEDDING_CACHE_SIZE > 0:
        _EMBEDDING_CACHE[cache_key] = rep
        while len(_EMBEDDING_CACHE) > _EMBEDDING_CACHE_SIZE:
            _EMBEDDING_CACHE.popitem(last=False)
    return rep


def score_pair(
    rep1_dict: dict,
    rep2_dict: dict,
    metrics: list,
    sim_params: dict = DEFAULT_SIM_PARAMS,
) -> tuple[dict, list[dict]]:
    """
    Compute metric distance between two representations and return the row plus
    any AUC sweep payloads.

    Args:
        rep1_dict (dict): dictionary specifying configuration of representation 1, to load its representation from disk
        rep2_dict (dict): dictionary specifying configuration of representation 2, to load its representation from disk
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

    return score_local_pair(
        rep1=rep1, rep2=rep2, metrics=metrics, metadata=results, sim_params=sim_params
    )


def score_local_pair(
    rep1: np.ndarray,
    rep2: np.ndarray,
    filename: str = None,
    metrics: list = None,
    metadata: dict = None,
    sim_params: dict = DEFAULT_SIM_PARAMS,
) -> tuple[dict, list[dict]]:
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

    metrics = metrics or []

    # center each row
    rep1 = rep1 - rep1.mean(axis=1, keepdims=True)
    rep2 = rep2 - rep2.mean(axis=1, keepdims=True)

    # normalize each representation
    rep1 = rep1 / np.linalg.norm(rep1)
    rep2 = rep2 / np.linalg.norm(rep2)

    results = dict(metadata or {})
    sweep_payloads = []

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

                    sweep_param = AlignmentMetrics.SWEEP_PARAMS.get(metric, {}).get('param')
                    sweep_payloads.append(
                        {
                            'metadata': dict(results),
                            'metric': metric,
                            'param_name': sweep_param,
                            'param_values': list(param_vec),
                            'scores': list(scores),
                        }
                    )
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

    if filename is not None:
        append_score_row_to_csv(results, filename)
        save_sweep_payloads(sweep_payloads, filename)

    return results, sweep_payloads


def _sweep_safe_name(metadata: dict) -> str:
    meta_parts = [
        str(metadata.get('architecture1', 'a1')),
        f"s{metadata.get('seed1', 's1')}",
        f"st{metadata.get('step1', 'st1')}",
        f"l{metadata.get('layer1', 'l1')}",
        "vs",
        str(metadata.get('architecture2', 'a2')),
        f"s{metadata.get('seed2', 's2')}",
        f"st{metadata.get('step2', 'st2')}",
        f"l{metadata.get('layer2', 'l2')}",
    ]
    dims_deleted = metadata.get('dims_deleted', None)
    if dims_deleted is not None:
        meta_parts.append(f"d{dims_deleted}")
    return "__".join(meta_parts)


def save_sweep_payloads(sweep_payloads: list[dict], csv_filename: str) -> None:
    if not sweep_payloads:
        return
    csv_path = pathlib.Path(csv_filename)
    sweeps_dir = csv_path.parent / pathlib.Path("sweeps")
    sweeps_dir.mkdir(parents=True, exist_ok=True)
    for sweep_payload in sweep_payloads:
        try:
            metadata = sweep_payload.get('metadata', {})
            metric = sweep_payload['metric']
            safe_name = _sweep_safe_name(metadata)
            sweep_file = sweeps_dir / pathlib.Path(f"{metric}_auc__{safe_name}.json")
            tmp_file = sweep_file.with_suffix(sweep_file.suffix + ".tmp")
            with open(tmp_file, 'w') as sf:
                json.dump(_to_jsonable(sweep_payload), sf)
            os.replace(tmp_file, sweep_file)
        except Exception as e:
            logging.warning(f"Failed saving sweep file for {sweep_payload.get('metric', 'unknown')}_auc: {e}")


def append_score_row_to_csv(results: dict, filename: str) -> None:
    with open(filename, mode="a") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=results.keys())
        if csv_file.tell() == 0:
            writer.writeheader()
        writer.writerow(results)


def score_pair_to_csv(
    rep1_dict: dict,
    rep2_dict: dict,
    filename: str,
    metrics: list,
    sim_params: dict = DEFAULT_SIM_PARAMS,
) -> None:
    results, sweep_payloads = score_pair(rep1_dict, rep2_dict, metrics, sim_params=sim_params)
    append_score_row_to_csv(results, filename)
    save_sweep_payloads(sweep_payloads, filename)
