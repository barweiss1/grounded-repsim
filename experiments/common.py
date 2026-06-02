import pathlib
import re
import warnings

import numpy as np

try:
    from .utils import plot_rank_corrs
except ImportError:
    from utils import plot_rank_corrs


DISTS_FILENAME = "dists_self_computed.csv"
FULL_DF_FILENAME = "full_df_self_computed.csv"
RESULTS_FILENAME = "results.txt"
PLOT_PREFIX = "rank_corrs"


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
