import json
import pathlib
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import math


def _has_dim_suffix(path):
    return re.search(r"__d\d+$", pathlib.Path(path).stem) is not None


def _is_legacy_dimless_pca_sweep(sweep_file, payload):
    metadata = payload.get("metadata", {})
    return "dims_deleted" in metadata and not _has_dim_suffix(sweep_file)


def _read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def list_sweep_metrics(sweeps_dir, ignore_legacy_dimless_pca=True):
    sweeps_dir = pathlib.Path(sweeps_dir)
    metrics = set()
    for path in sweeps_dir.glob("*_auc__*.json"):
        if ignore_legacy_dimless_pca and not _has_dim_suffix(path):
            try:
                if _is_legacy_dimless_pca_sweep(path, _read_json(path)):
                    continue
            except json.JSONDecodeError:
                continue
        name = path.name
        if "_auc__" in name:
            metrics.add(name.split("_auc__", 1)[0])
    return sorted(metrics)


def load_sweep_entries(sweeps_dir, metric_name=None, ignore_legacy_dimless_pca=True):
    sweeps_dir = pathlib.Path(sweeps_dir)
    if metric_name is None:
        patterns = ["*_auc__*.json"]
    else:
        base_metric = str(metric_name).removesuffix("_auc")
        patterns = [f"{base_metric}_auc__*.json"]

    entries = []
    for pattern in patterns:
        for sweep_file in sorted(sweeps_dir.glob(pattern)):
            payload = _read_json(sweep_file)
            if ignore_legacy_dimless_pca and _is_legacy_dimless_pca_sweep(sweep_file, payload):
                continue
            metric = payload.get("metric")
            if metric is None:
                metric = sweep_file.name.split("_auc__", 1)[0]
            entries.append(
                {
                    "path": sweep_file,
                    "metadata": payload.get("metadata", {}),
                    "metric": metric,
                    "metric_auc": f"{metric}_auc",
                    "param_name": payload.get("param_name"),
                    "param_values": np.asarray(payload.get("param_values", []), dtype=float),
                    "scores": np.asarray(payload.get("scores", []), dtype=float),
                }
            )
    return entries


def entries_to_dataframe(entries):
    rows = []
    for entry in entries:
        row = {
            "path": pathlib.Path(entry["path"]),
            "metric": entry["metric"],
            "metric_auc": entry.get("metric_auc", f"{entry['metric']}_auc"),
            "param_name": entry.get("param_name"),
            "param_values": np.asarray(entry.get("param_values", []), dtype=float),
            "scores": np.asarray(entry.get("scores", []), dtype=float),
        }
        row.update(entry.get("metadata", {}))
        rows.append(row)
    return pd.DataFrame(rows)


def _as_array(values):
    return np.asarray(values, dtype=float)


def _safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "plot"


def is_logscale_sweep(param_values, ratio_tol=0.05):
    param_values = _as_array(param_values)
    if len(param_values) < 3 or np.any(param_values <= 0):
        return False
    ratios = param_values[1:] / param_values[:-1]
    ratio_mean = np.mean(ratios)
    ratio_std = np.std(ratios)
    return ratio_mean > 1.01 and (ratio_std / ratio_mean) < ratio_tol


def align_sweep_curves(curves, target_param_values=None):
    valid_curves = []
    for param_values, scores in curves:
        param_values = _as_array(param_values)
        scores = _as_array(scores)
        if len(param_values) == 0 or len(scores) == 0 or len(param_values) != len(scores):
            continue
        order = np.argsort(param_values)
        valid_curves.append((param_values[order], scores[order]))

    if not valid_curves:
        return None, None

    if target_param_values is None:
        target_param_values = max(valid_curves, key=lambda curve: len(curve[0]))[0]
    else:
        target_param_values = _as_array(target_param_values)

    aligned_scores = [
        np.interp(target_param_values, param_values, scores)
        for param_values, scores in valid_curves
    ]
    return target_param_values, np.vstack(aligned_scores)


def _apply_filters(df, filters):
    if filters is None:
        return df
    if callable(filters):
        return df[filters(df)].copy()

    mask = pd.Series(True, index=df.index)
    for column, allowed in filters.items():
        if not isinstance(allowed, (list, tuple, set, np.ndarray)):
            allowed = [allowed]
        mask &= df[column].isin(allowed)
    return df[mask].copy()


def aggregate_sweep_curves(df, group_cols, filters=None, target_param_values=None):
    filtered_df = _apply_filters(df, filters)
    if filtered_df.empty:
        return pd.DataFrame()

    group_cols = list(group_cols)
    rows = []
    for group_key, group_df in filtered_df.groupby(group_cols, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        curves = list(zip(group_df["param_values"], group_df["scores"]))
        param_values, aligned_scores = align_sweep_curves(curves, target_param_values)
        if aligned_scores is None:
            continue
        row = dict(zip(group_cols, group_key))
        row.update(
            {
                "param_name": group_df["param_name"].dropna().iloc[0]
                if group_df["param_name"].notna().any()
                else None,
                "param_values": param_values,
                "mean_scores": aligned_scores.mean(axis=0),
                "std_scores": aligned_scores.std(axis=0),
                "n": aligned_scores.shape[0],
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def plot_sweep_curves(
    aggregated_df,
    line_label_cols,
    title,
    save_path=None,
    facet_col=None,
    ylabel="Similarity score",
):
    if aggregated_df is None or aggregated_df.empty:
        print(f"No aggregated data for {title}")
        return None

    line_label_cols = list(line_label_cols)
    if facet_col is None:
        facets = [(None, aggregated_df)]
    else:
        facets = list(aggregated_df.groupby(facet_col, dropna=False))

    fig, axes = plt.subplots(
        len(facets),
        1,
        figsize=(12, max(4.5, 4.2 * len(facets))),
        squeeze=False,
    )
    axes = axes[:, 0]

    for ax, (facet_value, facet_df) in zip(axes, facets):
        cmap = plt.colormaps.get_cmap("viridis").resampled(max(len(facet_df), 1))
        use_logscale = False
        for idx, (_, row) in enumerate(facet_df.sort_values(line_label_cols).iterrows()):
            param_values = _as_array(row["param_values"])
            mean_scores = _as_array(row["mean_scores"])
            std_scores = _as_array(row["std_scores"])
            label = ", ".join(f"{col}={row[col]}" for col in line_label_cols)
            color = cmap(idx)
            ax.plot(param_values, mean_scores, label=label, color=color, linewidth=2.2)
            ax.fill_between(
                param_values,
                mean_scores - std_scores,
                mean_scores + std_scores,
                color=color,
                alpha=0.18,
            )
            use_logscale = use_logscale or is_logscale_sweep(param_values)

        if use_logscale:
            ax.set_xscale("log")

        param_name = facet_df["param_name"].dropna().iloc[0] if facet_df["param_name"].notna().any() else "sweep parameter"
        ax.set_xlabel(param_name)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9, framealpha=0.95)
        if facet_col is not None:
            ax.set_title(f"{facet_col}={facet_value}")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()

    if save_path is not None:
        save_path = pathlib.Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
        print(f"Saved figure to {save_path}")

    return fig


def sample_similarity_at_param(df, target_param, metric_name=None, output_col="similarity_at_param"):
    sampled_df = df.copy()
    if metric_name is not None:
        base_metric = str(metric_name).removesuffix("_auc")
        sampled_df = sampled_df[sampled_df["metric"] == base_metric].copy()

    values = []
    for _, row in sampled_df.iterrows():
        param_values = _as_array(row["param_values"])
        scores = _as_array(row["scores"])
        if len(param_values) == 0 or len(scores) == 0:
            values.append(np.nan)
            continue
        order = np.argsort(param_values)
        values.append(float(np.interp(target_param, param_values[order], scores[order])))
    sampled_df[output_col] = values
    sampled_df["sampled_param"] = target_param
    return sampled_df


def compute_variance_weighted_curve_auc(df, metric_name, output_col=None):
    base_metric = str(metric_name).removesuffix("_auc")
    output_col = output_col or f"{base_metric}_var_weighted_auc"
    metric_df = df[df["metric"] == base_metric].copy()
    if metric_df.empty:
        return metric_df.assign(**{output_col: []})

    valid_rows = []
    curves = []
    for idx, row in metric_df.iterrows():
        param_values = _as_array(row["param_values"])
        scores = _as_array(row["scores"])
        if len(param_values) == 0 or len(scores) == 0 or len(param_values) != len(scores):
            continue
        order = np.argsort(param_values)
        valid_rows.append(idx)
        curves.append((param_values[order], scores[order]))

    metric_df[output_col] = np.nan
    if not curves:
        return metric_df

    target_param_values = max(curves, key=lambda curve: len(curve[0]))[0]
    aligned_scores = np.vstack(
        [
            np.interp(target_param_values, param_values, scores)
            for param_values, scores in curves
        ]
    )
    weights = np.nanvar(aligned_scores, axis=0)
    if not np.all(np.isfinite(weights)) or float(np.nansum(weights)) <= 0:
        weights = np.ones(aligned_scores.shape[1], dtype=float)
    weights = weights / weights.sum()

    # Match the full dataframe convention: metric columns are distances.
    metric_df.loc[valid_rows, output_col] = 1 - aligned_scores @ weights
    metric_df[f"{output_col}_param_values"] = [target_param_values] * len(metric_df)
    metric_df[f"{output_col}_weights"] = [weights] * len(metric_df)
    return metric_df


PRETRAIN_SWEEP_COLUMN_RENAMES = {
    "seed1": "pre_seed1",
    "step1": "fine_seed1",
    "seed2": "pre_seed2",
    "step2": "fine_seed2",
}

PRETRAIN_JOIN_COLUMNS = [
    "dataset1",
    "architecture1",
    "pre_seed1",
    "fine_seed1",
    "layer1",
    "dataset2",
    "architecture2",
    "pre_seed2",
    "fine_seed2",
    "layer2",
]


def normalize_pretrain_sweep_metadata(df):
    normalized_df = df.copy()
    renames = {
        source: target
        for source, target in PRETRAIN_SWEEP_COLUMN_RENAMES.items()
        if source in normalized_df.columns and target not in normalized_df.columns
    }
    normalized_df = normalized_df.rename(columns=renames)
    for column in PRETRAIN_JOIN_COLUMNS:
        if column in normalized_df.columns and column not in {"dataset1", "dataset2", "architecture1", "architecture2"}:
            normalized_df[column] = pd.to_numeric(normalized_df[column], errors="coerce").astype("Int64")
    return normalized_df


def merge_pretrain_weighted_sweep_features(full_df, weighted_feature_dfs, feature_cols=None):
    merged_df = normalize_pretrain_sweep_metadata(full_df)
    if isinstance(weighted_feature_dfs, pd.DataFrame):
        weighted_feature_dfs = [weighted_feature_dfs]

    for feature_df in weighted_feature_dfs:
        feature_df = normalize_pretrain_sweep_metadata(feature_df)
        missing_join_cols = [column for column in PRETRAIN_JOIN_COLUMNS if column not in feature_df.columns]
        if missing_join_cols:
            raise ValueError(f"Missing pretrain join columns in sweep features: {missing_join_cols}")

        if feature_cols is None:
            these_feature_cols = [
                column
                for column in feature_df.columns
                if column.endswith("_var_weighted_auc") or column.endswith("_at_param")
            ]
        else:
            these_feature_cols = [column for column in feature_cols if column in feature_df.columns]
        if not these_feature_cols:
            continue

        feature_df = feature_df[PRETRAIN_JOIN_COLUMNS + these_feature_cols].drop_duplicates(PRETRAIN_JOIN_COLUMNS)
        merged_df = merged_df.merge(feature_df, on=PRETRAIN_JOIN_COLUMNS, how="left")

    return merged_df


def _correlate(x, y, method):
    return x.corr(y, method=method)


def _bootstrap_corr_ci(x, y, method, n_boot=1000, ci=0.95, random_state=0):
    if n_boot is None or n_boot <= 0:
        return np.nan, np.nan
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3:
        return np.nan, np.nan

    rng = np.random.default_rng(random_state)
    boot_corrs = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, len(x), len(x))
        bx = pd.Series(x[idx])
        by = pd.Series(y[idx])
        if bx.nunique(dropna=True) < 2 or by.nunique(dropna=True) < 2:
            continue
        corr = _correlate(bx, by, method)
        if np.isfinite(corr):
            boot_corrs.append(corr)

    if not boot_corrs:
        return np.nan, np.nan
    alpha = (1 - ci) / 2
    return tuple(np.quantile(boot_corrs, [alpha, 1 - alpha]))

def _get_spearman_ci(x, y, alpha=0.05):
    """
    Computes Spearman correlation and CI using SciPy.
    alpha: Significance level (e.g., 0.05 for 95% confidence, 0.01 for 99% confidence)
    """
    # 1. Calculate Spearman rho
    rho, _ = stats.spearmanr(x, y)
    n = len(x)
    
    # 2. Fisher z-transform
    z = math.atanh(rho)
    stderr = 1.0 / math.sqrt(n - 3)
    
    # 3. Use SciPy to get exact margin of error for the changing alpha
    # (e.g., if alpha=0.05, confidence_level=0.95)
    confidence_level = 1 - alpha
    z_lower, z_upper = stats.norm.interval(confidence_level, loc=z, scale=stderr)
    
    # 4. Transform back to correlation scale
    lower_ci = math.tanh(z_lower)
    upper_ci = math.tanh(z_upper)
    
    return rho, (lower_ci, upper_ci)


def compute_task_metric_correlations(
    df,
    metrics,
    tasks=None,
    method="spearman",
    n_boot=1000,
    ci=0.95,
    random_state=0,
):
    if tasks is None:
        tasks = sorted(column.removesuffix("_diff") for column in df.columns if column.endswith("_diff"))

    rows = []
    for task in tasks:
        task_col = task if str(task).endswith("_diff") else f"{task}_diff"
        if task_col not in df.columns:
            continue
        y = pd.to_numeric(df[task_col], errors="coerce")
        for metric in metrics:
            if metric not in df.columns:
                continue
            x = pd.to_numeric(df[metric], errors="coerce")
            valid = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
            valid_x = x[valid]
            valid_y = y[valid]
            if len(valid_x) < 2 or valid_x.nunique(dropna=True) < 2 or valid_y.nunique(dropna=True) < 2:
                corr = np.nan
                ci_low, ci_high = np.nan, np.nan
            else:
                if method == "spearman":
                    corr, (ci_low, ci_high) = _get_spearman_ci(valid_x, valid_y, alpha=1-ci)
                else:
                    corr = _correlate(valid_x, valid_y, method)
                    ci_low, ci_high = _bootstrap_corr_ci(
                        valid_x,
                        valid_y,
                        method,
                        n_boot=n_boot,
                        ci=ci,
                        random_state=random_state,
                    )
            rows.append(
                {
                    "task": str(task).removesuffix("_diff"),
                    "task_col": task_col,
                    "metric": metric,
                    "method": method,
                    "correlation": corr,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "n": int(len(valid_x)),
                }
            )
    return pd.DataFrame(rows)


def plot_task_metric_bars(corr_df, task, save_path=None, title=None, metric_order=None):
    task_name = str(task).removesuffix("_diff")
    task_df = corr_df[corr_df["task"] == task_name].copy()
    if task_df.empty:
        print(f"No correlations found for task {task_name}")
        return None

    if metric_order is not None:
        order = {metric: idx for idx, metric in enumerate(metric_order)}
        task_df["metric_order"] = task_df["metric"].map(order).fillna(len(order))
        task_df = task_df.sort_values(["metric_order", "metric"], ascending=[False, False])
    else:
        task_df = task_df.sort_values("correlation", ascending=True, na_position="first")
    colors = np.where(task_df["correlation"].fillna(0) >= 0, "tab:blue", "tab:orange")
    fig, ax = plt.subplots(figsize=(9, max(3.8, 0.45 * len(task_df))))
    xerr = None
    if {"ci_low", "ci_high"}.issubset(task_df.columns):
        lower = task_df["correlation"] - task_df["ci_low"]
        upper = task_df["ci_high"] - task_df["correlation"]
        lower = lower.where(lower.notna() & (lower >= 0), 0)
        upper = upper.where(upper.notna() & (upper >= 0), 0)
        xerr = np.vstack([lower.to_numpy(dtype=float), upper.to_numpy(dtype=float)])
    ax.barh(
        task_df["metric"],
        task_df["correlation"],
        xerr=xerr,
        color=colors,
        alpha=0.82,
        error_kw={"elinewidth": 1.2, "capsize": 3, "alpha": 0.8},
    )
    ax.axvline(0, color="black", linewidth=1)
    # ax.set_xlim(-1, 1)
    ax.set_xlabel(f"{task_df['method'].iloc[0].title()} correlation with task difference")
    ax.set_ylabel("Metric")
    ax.set_title(title or f"{task_name}: metric signal vs task difference")
    ax.grid(True, axis="x", alpha=0.3, linestyle="--")
    fig.tight_layout()

    if save_path is not None:
        save_path = pathlib.Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
        print(f"Saved figure to {save_path}")

    return fig


def summarize_task_metric_correlations(corr_df, quantiles=(0.25, 0.75)):
    valid_df = corr_df.dropna(subset=["correlation"]).copy()
    if valid_df.empty:
        return pd.DataFrame(
            columns=["metric", "median", "mean", "q_low", "q_high", "min", "max", "n_tasks"]
        )

    q_low, q_high = quantiles
    summary = (
        valid_df.groupby("metric")["correlation"]
        .agg(
            median="median",
            mean="mean",
            q_low=lambda values: values.quantile(q_low),
            q_high=lambda values: values.quantile(q_high),
            min="min",
            max="max",
            n_tasks="count",
        )
        .reset_index()
    )
    return summary


def plot_task_correlation_boxplot(
    corr_df,
    save_path=None,
    title=None,
    metric_order=None,
    quantiles=(0.25, 0.75),
    show_quantile_labels=True,
):
    plot_df = corr_df.dropna(subset=["correlation"]).copy()
    if plot_df.empty:
        print("No task correlations to plot")
        return None

    summary = summarize_task_metric_correlations(plot_df, quantiles=quantiles)
    if metric_order is None:
        metric_order = summary.sort_values("median", ascending=True)["metric"].tolist()
    else:
        metric_order = [metric for metric in metric_order if metric in set(plot_df["metric"])]

    data = [
        plot_df.loc[plot_df["metric"] == metric, "correlation"].to_numpy(dtype=float)
        for metric in metric_order
    ]
    fig_height = max(4.0, 0.45 * len(metric_order))
    fig, ax = plt.subplots(figsize=(10, fig_height))
    ax.boxplot(
        data,
        vert=False,
        labels=metric_order,
        patch_artist=True,
        showmeans=False,
        boxprops={"facecolor": "tab:blue", "alpha": 0.35, "edgecolor": "tab:blue"},
        medianprops={"color": "black", "linewidth": 1.8},
        whiskerprops={"color": "tab:blue"},
        capprops={"color": "tab:blue"},
        flierprops={"marker": "o", "markersize": 3, "alpha": 0.45},
    )
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel(f"{plot_df['method'].iloc[0].title()} correlation across tasks")
    ax.set_ylabel("Metric")
    ax.set_title(title or "Task correlation distribution by metric")
    ax.grid(True, axis="x", alpha=0.3, linestyle="--")

    if show_quantile_labels:
        summary_by_metric = summary.set_index("metric")
        x_min, x_max = ax.get_xlim()
        label_x = x_max + 0.02 * (x_max - x_min)
        for y_pos, metric in enumerate(metric_order, start=1):
            row = summary_by_metric.loc[metric]
            label = f"{row['median']:.2f} [{row['q_low']:.2f}, {row['q_high']:.2f}]"
            ax.text(label_x, y_pos, label, va="center", fontsize=8)
        ax.set_xlim(x_min, label_x + 0.2 * (x_max - x_min))

    fig.tight_layout()

    if save_path is not None:
        save_path = pathlib.Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
        print(f"Saved figure to {save_path}")

    return fig


def compute_pretrain_param_task_correlations(
    full_df,
    sweep_df,
    metric_name,
    tasks=None,
    method="spearman",
    target_param_values=None,
    n_boot=0,
    ci=0.95,
    random_state=0,
):
    base_metric = str(metric_name).removesuffix("_auc")
    metric_df = sweep_df[sweep_df["metric"] == base_metric].copy()
    if metric_df.empty:
        return pd.DataFrame()

    curves = list(zip(metric_df["param_values"], metric_df["scores"]))
    aligned_params, aligned_scores = align_sweep_curves(curves, target_param_values=target_param_values)
    if aligned_scores is None:
        return pd.DataFrame()

    feature_df = normalize_pretrain_sweep_metadata(metric_df).reset_index(drop=True)
    rows = []
    for param_idx, param_value in enumerate(aligned_params):
        feature_col = f"{base_metric}_similarity_at_{param_idx}"
        # the sweeps are stored as similarities so we need to convert to 
        # distances to match the full_df convention
        feature_df[feature_col] = 1 - aligned_scores[:, param_idx]
        analysis_df = merge_pretrain_weighted_sweep_features(
            full_df,
            feature_df,
            feature_cols=[feature_col],
        )
        corr_df = compute_task_metric_correlations(
            analysis_df,
            metrics=[feature_col],
            tasks=tasks,
            method=method,
            n_boot=n_boot,
            ci=ci,
            random_state=random_state,
        )
        if corr_df.empty:
            continue
        corr_df["metric"] = base_metric
        corr_df["param_idx"] = param_idx
        corr_df["param_value"] = float(param_value)
        rows.append(corr_df)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def plot_param_task_heatmap(param_corr_df, metric_name=None, save_path=None, title=None):
    plot_df = param_corr_df.copy()
    if metric_name is not None:
        plot_df = plot_df[plot_df["metric"] == str(metric_name).removesuffix("_auc")]
    if plot_df.empty:
        print("No parameter correlations to plot")
        return None

    plot_df["param_label"] = plot_df["param_value"].map(lambda value: f"{value:.3g}")
    heatmap_df = plot_df.pivot_table(
        index="task",
        columns="param_label",
        values="correlation",
        aggfunc="mean",
        sort=False,
    )
    fig, ax = plt.subplots(figsize=(max(8, 0.45 * heatmap_df.shape[1]), max(4, 0.35 * heatmap_df.shape[0])))
    im = ax.imshow(heatmap_df.to_numpy(dtype=float), aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(heatmap_df.shape[1]))
    ax.set_xticklabels(heatmap_df.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(heatmap_df.shape[0]))
    ax.set_yticklabels(heatmap_df.index)
    ax.set_xlabel("Sweep parameter value")
    ax.set_ylabel("Task")
    ax.set_title(title or f"{metric_name or plot_df['metric'].iloc[0]}: task correlation by parameter")
    fig.colorbar(im, ax=ax, label="Correlation")
    fig.tight_layout()

    if save_path is not None:
        save_path = pathlib.Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
        print(f"Saved figure to {save_path}")

    return fig


def add_experiment_preset_columns(df, experiment, ref_layer=None):
    df = df.copy()

    if {"seed1", "seed2"}.issubset(df.columns):
        df["same_seed"] = df["seed1"] == df["seed2"]
        df["seed_mode"] = np.where(df["same_seed"], "same_seed", "different_seed")

    if experiment == "layer_exp":
        if ref_layer is not None:
            layer1_is_ref = df["layer1"] == ref_layer
            layer2_is_ref = df["layer2"] == ref_layer
            df["other_layer"] = np.where(layer1_is_ref, df["layer2"], np.where(layer2_is_ref, df["layer1"], np.nan))
            df["uses_ref_layer"] = layer1_is_ref | layer2_is_ref
        else:
            df["other_layer"] = np.nan
            df["uses_ref_layer"] = True
        df["layer_gap"] = (df["layer1"] - df["layer2"]).abs()

    if experiment == "pretrain_finetune":
        df["preseed1"] = df["seed1"]
        df["preseed2"] = df["seed2"]
        df["fineseed1"] = df["step1"]
        df["fineseed2"] = df["step2"]
        df["same_preseed"] = df["preseed1"] == df["preseed2"]
        df["same_fineseed"] = df["fineseed1"] == df["fineseed2"]
        df["same_model"] = df["same_preseed"] & df["same_fineseed"]
        df["pretrain_finetune_mode"] = np.select(
            [
                df["same_model"],
                df["same_preseed"],
                df["same_fineseed"],
            ],
            [
                "same_model",
                "same_preseed",
                "same_fineseed",
            ],
            default="different_model",
        )

    return df


def preset_specs(experiment, ref_layer=None):
    if experiment == "feather":
        return [
            {
                "name": "seed_mode_by_layer",
                "filters": None,
                "group_cols": ["metric", "seed_mode", "layer1"],
                "line_label_cols": ["layer1"],
                "facet_col": "seed_mode",
            }
        ]

    if experiment == "layer_exp":
        return [
            {
                "name": "ref_layer_same_seed",
                "filters": {"uses_ref_layer": True, "seed_mode": "same_seed"},
                "group_cols": ["metric", "seed_mode", "other_layer"],
                "line_label_cols": ["other_layer"],
                "facet_col": None,
            },
            {
                "name": "ref_layer_different_seed",
                "filters": {"uses_ref_layer": True, "seed_mode": "different_seed"},
                "group_cols": ["metric", "seed_mode", "other_layer"],
                "line_label_cols": ["other_layer"],
                "facet_col": None,
            },
            {
                "name": "ref_layer_all",
                "filters": {"uses_ref_layer": True},
                "group_cols": ["metric", "other_layer"],
                "line_label_cols": ["other_layer"],
                "facet_col": None,
            },
        ]

    if experiment == "pca_deletion":
        return [
            {
                "name": "same_seed_by_layer_dims",
                "filters": {"seed_mode": "same_seed"},
                "group_cols": ["metric", "seed_mode", "layer1", "dims_deleted"],
                "line_label_cols": ["dims_deleted"],
                "facet_col": "layer1",
            },
            {
                "name": "different_seed_by_layer_dims",
                "filters": {"seed_mode": "different_seed"},
                "group_cols": ["metric", "seed_mode", "layer1", "dims_deleted"],
                "line_label_cols": ["dims_deleted"],
                "facet_col": "layer1",
            },
        ]

    if experiment == "pretrain_finetune":
        return [
            {
                "name": "pretrain_finetune_modes_by_layer",
                "filters": None,
                "group_cols": ["metric", "pretrain_finetune_mode", "layer1"],
                "line_label_cols": ["layer1"],
                "facet_col": "pretrain_finetune_mode",
            }
        ]

    raise ValueError(f"Unknown experiment preset: {experiment}")


def save_preset_plots(df, experiment, metrics, figures_dir, ref_layer=None):
    figures_dir = pathlib.Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    df = add_experiment_preset_columns(df, experiment, ref_layer=ref_layer)
    specs = preset_specs(experiment, ref_layer=ref_layer)
    saved = []

    for metric in metrics:
        metric_df = df[df["metric"] == str(metric).removesuffix("_auc")]
        if metric_df.empty:
            print(f"No sweep entries for metric {metric}")
            continue
        for spec in specs:
            aggregated = aggregate_sweep_curves(
                metric_df,
                group_cols=spec["group_cols"],
                filters=spec["filters"],
            )
            if aggregated.empty:
                print(f"No aggregated data for {metric} / {spec['name']}")
                continue
            save_path = figures_dir / f"{experiment}_{_safe_name(metric)}_{spec['name']}.png"
            fig = plot_sweep_curves(
                aggregated,
                line_label_cols=spec["line_label_cols"],
                facet_col=spec["facet_col"],
                title=f"{experiment}: {metric} ({spec['name']})",
                save_path=save_path,
            )
            if fig is not None:
                saved.append(save_path)
    return saved
