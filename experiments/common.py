import pathlib
import re
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from .utils import plot_rank_corrs
except ImportError:
    from utils import plot_rank_corrs

BASE_DIR = pathlib.Path(__file__).resolve().parents[1]
DISTS_DIR = BASE_DIR / "dists"
if str(DISTS_DIR) not in sys.path:
    sys.path.append(str(DISTS_DIR))


DISTS_FILENAME = "dists_self_computed.csv"
FULL_DF_FILENAME = "full_df_self_computed.csv"
RESULTS_FILENAME = "results.txt"
PLOT_PREFIX = "rank_corrs"
PAIR_KEY_COLUMNS = [
    "dataset1",
    "architecture1",
    "seed1",
    "step1",
    "layer1",
    "dataset2",
    "architecture2",
    "seed2",
    "step2",
    "layer2",
]


def get_run_paths(results_dir):
    results_dir = pathlib.Path(results_dir)
    return {
        "results_dir": results_dir,
        "dists": results_dir / DISTS_FILENAME,
        "full_df": results_dir / FULL_DF_FILENAME,
        "results": results_dir / RESULTS_FILENAME,
    }


def resolve_resource(resources_path, relative_path):
    path = pathlib.Path(relative_path)
    if path.is_absolute():
        return path
    return pathlib.Path(resources_path) / path


def filter_available_metrics(requested_metrics, df, warn=True):
    metrics = [metric for metric in requested_metrics if metric in df.columns]
    missing = [metric for metric in requested_metrics if metric not in df.columns]
    if warn and missing:
        warnings.warn(
            "Skipping metrics not found in dataframe columns: "
            + ", ".join(missing),
            RuntimeWarning,
            stacklevel=2,
        )
    return metrics


def _mean_or_nan(values):
    if len(values) == 0:
        return np.nan
    return np.mean(values)


def write_rank_corr_results(
    results_path,
    metrics,
    rho,
    rho_p,
    tau,
    tau_p,
    bad_fracs=None,
    header=None,
    mode="w",
):
    with open(results_path, mode) as f:
        if header:
            f.write(f"{header}\n")
        for metric in metrics:
            f.write(f"metric: {metric}\n")
            f.write(f"avg_rho: {round(_mean_or_nan(rho[metric]), 3)}\n")
            f.write(f"avg_rho_p: {format(_mean_or_nan(rho_p[metric]), '.1e')}\n")
            f.write(f"avg_tau: {round(_mean_or_nan(tau[metric]), 3)}\n")
            f.write(f"avg_tau_p: {format(_mean_or_nan(tau_p[metric]), '.1e')}\n")
            if bad_fracs is not None:
                f.write(f"avg_bad_frac: {round(_mean_or_nan(bad_fracs[metric]), 3)}\n")
            f.write("\n")


def _safe_plot_name(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_") or "plot"


def save_rank_corr_plot(results_dir, rho, rho_p, tau, tau_p, metrics, title):
    save_path = pathlib.Path(results_dir) / f"{PLOT_PREFIX}_{_safe_plot_name(title)}.png"
    plot_rank_corrs(rho, rho_p, tau, tau_p, metrics, title=title, save_path=save_path)
    return save_path


def get_sim_params(cfg):
    from score_pair import DEFAULT_SIM_PARAMS

    sim_params = dict(DEFAULT_SIM_PARAMS)
    for key in sim_params:
        if key in cfg:
            sim_params[key] = cfg[key]
    return sim_params


def make_pair_job(rep1_dict, rep2_dict, metadata=None):
    return {
        "rep1": dict(rep1_dict),
        "rep2": dict(rep2_dict),
        "metadata": dict(metadata or {}),
    }


def pair_job_key(job, key_columns=PAIR_KEY_COLUMNS):
    rep1 = job["rep1"]
    rep2 = job["rep2"]
    metadata = {
        "dataset1": rep1["dataset"],
        "architecture1": rep1["architecture"],
        "seed1": rep1["seed"],
        "step1": rep1["step"],
        "layer1": rep1["layer"],
        "dataset2": rep2["dataset"],
        "architecture2": rep2["architecture"],
        "seed2": rep2["seed"],
        "step2": rep2["step"],
        "layer2": rep2["layer"],
        **job.get("metadata", {}),
    }
    return tuple(_normalize_key_value(metadata.get(column)) for column in key_columns)


def _normalize_key_value(value):
    if pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _completed_keys_from_df(df, key_columns):
    if df.empty or any(column not in df.columns for column in key_columns):
        return set()
    return {
        tuple(_normalize_key_value(row[column]) for column in key_columns)
        for _, row in df.iterrows()
    }


def _read_existing_dists(dists_path, overwrite):
    dists_path = pathlib.Path(dists_path)
    if overwrite or not dists_path.exists() or dists_path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(dists_path)


def _write_rows_atomic(dists_path, existing_df, new_rows, key_columns):
    dists_path = pathlib.Path(dists_path)
    dists_path.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame(new_rows)
    combined = pd.concat([existing_df, new_df], ignore_index=True, sort=False)
    if not combined.empty and all(column in combined.columns for column in key_columns):
        combined = combined.drop_duplicates(subset=key_columns, keep="last")
    tmp_path = dists_path.with_suffix(dists_path.suffix + ".tmp")
    combined.to_csv(tmp_path, index=False)
    os.replace(tmp_path, dists_path)
    return combined


def _write_sweeps(sweep_payloads, dists_path):
    from score_pair import save_sweep_payloads

    for sweep_payload in sweep_payloads:
        save_sweep_payloads([sweep_payload], dists_path)


def _init_pair_worker(torch_threads, embedding_cache_size):
    os.environ["OMP_NUM_THREADS"] = str(torch_threads)
    os.environ["MKL_NUM_THREADS"] = str(torch_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(torch_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(torch_threads)
    try:
        import torch

        torch.set_num_threads(int(torch_threads))
    except Exception:
        pass
    from score_pair import set_embedding_cache_size

    set_embedding_cache_size(embedding_cache_size)


def _score_pair_job_worker(payload):
    index, job, metrics, sim_params = payload
    from score_pair import score_pair

    row, sweep_payloads = score_pair(
        job["rep1"], job["rep2"], metrics, sim_params=sim_params
    )
    extra_metadata = job.get("metadata", {})
    if extra_metadata:
        row.update(extra_metadata)
        for sweep_payload in sweep_payloads:
            sweep_payload.setdefault("metadata", {}).update(extra_metadata)
    return index, row, sweep_payloads


def run_pair_jobs(
    jobs,
    metrics,
    dists_path,
    cfg,
    desc="distance pairs",
    key_columns=PAIR_KEY_COLUMNS,
):
    dists_path = pathlib.Path(dists_path)
    num_workers = int(cfg.get("num_workers", 1))
    write_every = max(1, int(cfg.get("write_every", 50)))
    overwrite = bool(cfg.get("overwrite_dists", False))
    embedding_cache_size = int(cfg.get("embedding_cache_size", 2))
    torch_threads = int(cfg.get("torch_num_threads_per_worker", 1))
    sim_params = get_sim_params(cfg)

    existing_df = _read_existing_dists(dists_path, overwrite)
    completed_keys = set() if overwrite else _completed_keys_from_df(existing_df, key_columns)
    pending_jobs = [
        (index, job)
        for index, job in enumerate(jobs)
        if pair_job_key(job, key_columns) not in completed_keys
    ]

    skipped = len(jobs) - len(pending_jobs)
    if skipped:
        tqdm.write(f"[INFO] Skipping {skipped} completed distance pairs.")
    if not pending_jobs:
        if not dists_path.exists():
            _write_rows_atomic(dists_path, existing_df, [], key_columns)
        return

    next_index = 0
    pending_results = {}
    buffered_rows = []
    buffered_sweeps = []
    payloads = [
        (pending_index, job, metrics, sim_params)
        for pending_index, (_, job) in enumerate(pending_jobs)
    ]

    def handle_result(result):
        nonlocal existing_df, next_index, buffered_rows, buffered_sweeps
        index, row, sweep_payloads = result
        pending_results[index] = (row, sweep_payloads)
        while next_index in pending_results:
            ready_row, ready_sweeps = pending_results.pop(next_index)
            buffered_rows.append(ready_row)
            buffered_sweeps.extend(ready_sweeps)
            next_index += 1
        if len(buffered_rows) >= write_every:
            existing_df = _write_rows_atomic(dists_path, existing_df, buffered_rows, key_columns)
            _write_sweeps(buffered_sweeps, dists_path)
            buffered_rows = []
            buffered_sweeps = []

    with tqdm(total=len(payloads), desc=desc, unit="pair") as pbar:
        if num_workers <= 1:
            _init_pair_worker(torch_threads, embedding_cache_size)
            for payload in payloads:
                handle_result(_score_pair_job_worker(payload))
                pbar.update(1)
        else:
            try:
                with ProcessPoolExecutor(
                    max_workers=num_workers,
                    initializer=_init_pair_worker,
                    initargs=(torch_threads, embedding_cache_size),
                ) as executor:
                    future_to_payload = {
                        executor.submit(_score_pair_job_worker, payload): payload
                        for payload in payloads
                    }
                    for future in as_completed(future_to_payload):
                        handle_result(future.result())
                        pbar.update(1)
            except PermissionError as exc:
                tqdm.write(
                    "[WARN] Process pool unavailable in this environment; "
                    f"falling back to serial compute. Original error: {exc}"
                )
                _init_pair_worker(torch_threads, embedding_cache_size)
                for payload in payloads:
                    handle_result(_score_pair_job_worker(payload))
                    pbar.update(1)

    if buffered_rows:
        existing_df = _write_rows_atomic(dists_path, existing_df, buffered_rows, key_columns)
        _write_sweeps(buffered_sweeps, dists_path)


def run_result_jobs(
    jobs,
    worker_fn,
    dists_path,
    cfg,
    desc="distance pairs",
    key_columns=PAIR_KEY_COLUMNS,
    job_key_fn=pair_job_key,
):
    dists_path = pathlib.Path(dists_path)
    num_workers = int(cfg.get("num_workers", 1))
    write_every = max(1, int(cfg.get("write_every", 50)))
    overwrite = bool(cfg.get("overwrite_dists", False))
    embedding_cache_size = int(cfg.get("embedding_cache_size", 2))
    torch_threads = int(cfg.get("torch_num_threads_per_worker", 1))

    existing_df = _read_existing_dists(dists_path, overwrite)
    completed_keys = set() if overwrite else _completed_keys_from_df(existing_df, key_columns)
    pending_jobs = [
        (index, job)
        for index, job in enumerate(jobs)
        if job_key_fn(job, key_columns) not in completed_keys
    ]

    skipped = len(jobs) - len(pending_jobs)
    if skipped:
        tqdm.write(f"[INFO] Skipping {skipped} completed distance pairs.")
    if not pending_jobs:
        if not dists_path.exists():
            _write_rows_atomic(dists_path, existing_df, [], key_columns)
        return

    next_index = 0
    pending_results = {}
    buffered_rows = []
    buffered_sweeps = []
    payloads = [
        (pending_index, job, cfg)
        for pending_index, (_, job) in enumerate(pending_jobs)
    ]

    def handle_result(result):
        nonlocal existing_df, next_index, buffered_rows, buffered_sweeps
        index, row, sweep_payloads = result
        pending_results[index] = (row, sweep_payloads)
        while next_index in pending_results:
            ready_row, ready_sweeps = pending_results.pop(next_index)
            buffered_rows.append(ready_row)
            buffered_sweeps.extend(ready_sweeps)
            next_index += 1
        if len(buffered_rows) >= write_every:
            existing_df = _write_rows_atomic(dists_path, existing_df, buffered_rows, key_columns)
            _write_sweeps(buffered_sweeps, dists_path)
            buffered_rows = []
            buffered_sweeps = []

    with tqdm(total=len(payloads), desc=desc, unit="pair") as pbar:
        if num_workers <= 1:
            _init_pair_worker(torch_threads, embedding_cache_size)
            for payload in payloads:
                handle_result(worker_fn(payload))
                pbar.update(1)
        else:
            try:
                with ProcessPoolExecutor(
                    max_workers=num_workers,
                    initializer=_init_pair_worker,
                    initargs=(torch_threads, embedding_cache_size),
                ) as executor:
                    future_to_payload = {
                        executor.submit(worker_fn, payload): payload
                        for payload in payloads
                    }
                    for future in as_completed(future_to_payload):
                        handle_result(future.result())
                        pbar.update(1)
            except PermissionError as exc:
                tqdm.write(
                    "[WARN] Process pool unavailable in this environment; "
                    f"falling back to serial compute. Original error: {exc}"
                )
                _init_pair_worker(torch_threads, embedding_cache_size)
                for payload in payloads:
                    handle_result(worker_fn(payload))
                    pbar.update(1)

    if buffered_rows:
        existing_df = _write_rows_atomic(dists_path, existing_df, buffered_rows, key_columns)
        _write_sweeps(buffered_sweeps, dists_path)
