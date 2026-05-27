from metrics import AlignmentMetrics
import numpy as np
import torch

def _to_torch_tensor(feats):
    if torch.is_tensor(feats):
        return feats
    return torch.from_numpy(np.asarray(feats))


def _pairwise_distance_ratios(feats, q_low=0.01, q_high=0.8, eps=1e-8):
    feats_t = _to_torch_tensor(feats).double()
    if feats_t.ndim != 2:
        raise ValueError("feats must be a 2D array or tensor.")

    if feats_t.shape[0] < 2:
        return 1.0, 1.0

    dists = torch.cdist(feats_t, feats_t)
    tril = torch.tril_indices(dists.shape[0], dists.shape[1], offset=-1)
    vals = dists[tril[0], tril[1]]
    if vals.numel() == 0:
        return 1.0, 1.0

    median = torch.quantile(vals, 0.5).item()
    median = max(median, eps)
    low_val = torch.quantile(vals, float(q_low)).item()
    high_val = torch.quantile(vals, float(q_high)).item()

    return low_val / median, high_val / median


def _inner_product_ratios(feats, q_low=0.01, q_high=0.8, eps=1e-8):
    feats_t = _to_torch_tensor(feats).double()
    if feats_t.ndim != 2:
        raise ValueError("feats must be a 2D array or tensor.")

    if feats_t.shape[0] < 2:
        return 1.0, 1.0

    ip = torch.abs(feats_t @ feats_t.T)
    tril = torch.tril_indices(ip.shape[0], ip.shape[1], offset=-1)
    vals = ip[tril[0], tril[1]]
    if vals.numel() == 0:
        return 1.0, 1.0

    median = torch.quantile(vals, 0.5).item()
    median = max(median, eps)
    low_val = torch.quantile(vals, float(q_low)).item()
    high_val = torch.quantile(vals, float(q_high)).item()

    return low_val / median, high_val / median


def get_adaptive_rbf_sigma_sweep(
    feats_A,
    feats_B,
    sweep_len,
    logscale=False,
    quantiles=(0.01, 0.8),
    eps=1e-8,
):
    q_low, q_high = quantiles
    a_low, a_high = _pairwise_distance_ratios(feats_A, q_low=q_low, q_high=q_high, eps=eps)
    b_low, b_high = _pairwise_distance_ratios(feats_B, q_low=q_low, q_high=q_high, eps=eps)

    min_ratio = max(min(a_low, b_low), eps)
    max_ratio = max(a_high, b_high)
    if max_ratio <= min_ratio:
        max_ratio = min_ratio * 1.05

    if logscale:
        return np.geomspace(min_ratio, max_ratio, num=sweep_len)
    return np.linspace(min_ratio, max_ratio, num=sweep_len)

def get_adaptive_temperature_sweep(
    feats_A,
    feats_B,
    sweep_len,
    logscale=False,
    quantiles=(0.01, 0.8),
    eps=1e-8,
):
    """
    Builds an adaptive temperature sweep based on quantiles of the absolute inner-products
    between feature vectors. Temperature must be positive, so we use absolute values of
    inner products and take the specified quantile range as bounds.
    """
    q_low, q_high = quantiles
    a_low, a_high = _inner_product_ratios(feats_A, q_low=q_low, q_high=q_high, eps=eps)
    b_low, b_high = _inner_product_ratios(feats_B, q_low=q_low, q_high=q_high, eps=eps)

    min_ratio = max(min(a_low, b_low), eps)
    max_ratio = max(a_high, b_high)
    if max_ratio <= min_ratio:
        max_ratio = min_ratio * 1.05

    if logscale:
        return np.geomspace(min_ratio, max_ratio, num=sweep_len)
    return np.linspace(min_ratio, max_ratio, num=sweep_len)



def integrate_metric_over_param(param_vec, scores, integration_method='trapezoidal'):
    """
    Integrates a metric curve over a hyperparameter sweep using the specified integration method.
    """
    if len(param_vec) < 2:
        return scores[0]  # If we only have one point, return that score
    if integration_method == 'trapezoidal':
        return np.trapezoid(scores, param_vec) / (param_vec[-1] - param_vec[0])
    elif integration_method == 'average':
        return sum(scores) / len(scores)
    else:
        raise ValueError(f"Unsupported integration method: {integration_method}")
    

def map_param_name_to_kwargs(param_name, param_value=None, local=False):
    """
    Maps a sweep parameter name and value to the corresponding keyword arguments for AlignmentMetrics.measure.
    """
    if param_name is None:
        kwargs = {}
    elif param_name == 'rbf_sigma':
        val = param_value if param_value is not None else 1.0
        if local:
            val /= 5.0  # for local mode, use a smaller sigma to focus on local neighborhoods 
        kwargs = {'rbf_sigma': val}
    elif param_name == 'quantile':
        val = param_value if param_value is not None else 0.1
        kwargs = {'quantile': val}
    elif param_name == 'topk':
        val = param_value if param_value is not None else 20
        kwargs = {'topk': int(val)}
    elif param_name == 'temperature':
        val = param_value if param_value is not None else 0.5
        kwargs = {'temperature': val}
    else:
        raise ValueError(f"Unsupported sweep parameter: {param_name}")
    return kwargs


def get_param_sweep_for_metric(
    metric_name,
    sweep_len,
    sweep_values=None,
    logscale=False,
    adaptive_rbf_sigma=False,
    adaptive_temperature=False,
    feats_A=None,
    feats_B=None,
    adaptive_quantiles=(0.01, 0.8),
):
    """
    Builds a vector of hyperparameter values to sweep over for a given metric, based on the metric's sweep configuration.
    If sweep_values is provided, it will be used directly. Otherwise, the values will be
    generated according to the metric's sweep configuration and the specified sweep_len and logscale parameters.

    """
    if sweep_values is not None:
        return np.asarray(sweep_values, dtype=float)

    sweep_config = AlignmentMetrics.SWEEP_PARAMS[metric_name]
    param_name = sweep_config["param"]
    if param_name is None:
        raise ValueError(f"Metric {metric_name} does not define a sweepable hyperparameter")

    if param_name == "rbf_sigma" and adaptive_rbf_sigma:
        if feats_A is None or feats_B is None:
            raise ValueError("Adaptive rbf_sigma sweep requires feats_A and feats_B.")
        return get_adaptive_rbf_sigma_sweep(
            feats_A=feats_A,
            feats_B=feats_B,
            sweep_len=sweep_len,
            logscale=logscale,
            quantiles=adaptive_quantiles,
        )

    # adaptive temperature sweep for softmax_rwka
    if param_name == "temperature" and adaptive_temperature:
        if feats_A is None:
            raise ValueError("Adaptive temperature sweep requires feats_A.")
        return get_adaptive_temperature_sweep(
            feats_A=feats_A,
            feats_B=feats_B,
            sweep_len=sweep_len,
            logscale=logscale,
            quantiles=adaptive_quantiles,
        )

    if logscale:
        sweep_vec = np.geomspace(sweep_config["min"], sweep_config["max"], num=sweep_len)
    else:
        sweep_vec = np.linspace(sweep_config["min"], sweep_config["max"], num=sweep_len)
    
    if param_name == "topk":
        return np.round(sweep_vec).astype(int)
    if param_name == "temperature":
        return sweep_vec.astype(float)
    if param_name == "rbf_sigma":
        return sweep_vec.astype(float)
    if param_name == "quantile":
        return sweep_vec.astype(float)


def sweep_metric_over_param(
    feats_A,
    feats_B,
    metric_name,
    sweep_config,
    sweep_len,
    logscale=False,
    adaptive_rbf_sigma=False,
    adaptive_temperature=False,
    adaptive_quantiles=(0.01, 0.8),
):
    """
    Sweeps a specified alignment metric across a range of hyperparemeter values defined in sweep_config,
    and returns the vector of metric scores corresponding to each hyperparameter value.
    """
    param_name = sweep_config['param']
    param_vec = get_param_sweep_for_metric(
        metric_name=metric_name,
        sweep_len=sweep_len,
        sweep_values=None,
        logscale=logscale,
        adaptive_rbf_sigma=adaptive_rbf_sigma,
        adaptive_temperature=adaptive_temperature,
        feats_A=feats_A,
        feats_B=feats_B,
        adaptive_quantiles=adaptive_quantiles,
    )
    scores = []
    for param_value in param_vec:
        kwargs = map_param_name_to_kwargs(param_name, param_value=param_value)
        score = AlignmentMetrics.measure(feats_A=feats_A, 
                                         feats_B=feats_B, 
                                         metric=metric_name, **kwargs)
        scores.append(score)
    return param_vec, scores