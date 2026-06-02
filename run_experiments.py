import os
import sys
import yaml
import json
import pathlib
from datetime import datetime
from importlib import import_module

from tqdm import tqdm

try:
    import torch
except Exception:
    torch = None

from paths import resources_path

REPO_DIR = pathlib.Path(__file__).resolve().parent
DISTS_DIR = REPO_DIR / "dists"
if str(DISTS_DIR) not in sys.path:
    sys.path.append(str(DISTS_DIR))

def load_config(config_path):
    config_path = pathlib.Path(config_path)
    if not config_path.is_absolute():
        cwd_path = pathlib.Path.cwd() / config_path
        repo_path = REPO_DIR / config_path
        config_path = cwd_path if cwd_path.exists() else repo_path
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)

def run_experiment(exp_name, exp_cfg, results_base, run_id, device=None):
    exp_dir = os.path.join(results_base, exp_name, run_id)
    ensure_dir(exp_dir)
    save_json(exp_cfg, os.path.join(exp_dir, 'run_config.json'))

    exp_path = REPO_DIR / 'experiments' / exp_name
    sys.path.insert(0, str(exp_path))
    stages = [
        ("compute_dists", f"experiments.{exp_name}.compute_dists", "run_compute_dists"),
        ("compute_full_df", f"experiments.{exp_name}.compute_full_df", "run_compute_full_df"),
        ("script", f"experiments.{exp_name}.script", "run_experiment_script"),
    ]

    enabled_stages = [stage for stage in stages if exp_cfg.get(stage[0], None)]
    for stage_name, module_name, function_name in tqdm(
        enabled_stages,
        desc=f"{exp_name} stages",
        unit="stage",
    ):
        mod = import_module(module_name)
        if hasattr(mod, function_name):
            func = getattr(mod, function_name)
            func(exp_cfg[stage_name], exp_dir, resources_path, device=device)
        else:
            tqdm.write(f"[WARN] {module_name} has no {function_name} function.")

    sys.path.pop(0)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run grounded-repsim experiments from YAML config')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config')
    parser.add_argument('--device', type=str, default='auto', help='Device to run on: auto|cpu|cuda[:idx]')
    args = parser.parse_args()

    cfg = load_config(args.config)
    # Resolve device
    device_arg = args.device
    resolved_device = None
    if device_arg == 'auto':
        if torch is not None and torch.cuda.is_available():
            resolved_device = torch.device('cuda')
        else:
            resolved_device = torch.device('cpu')
    else:
        if torch is not None:
            try:
                resolved_device = torch.device(device_arg)
            except Exception:
                print(f"[WARN] could not parse device '{device_arg}', falling back to cpu")
                resolved_device = torch.device('cpu')
        else:
            print("[WARN] PyTorch not available; defaulting device to 'cpu'")
            resolved_device = 'cpu'
    run_id = cfg.get('run_id') or datetime.now().strftime('%Y%m%d_%H%M%S')
    results_base = os.path.join(resources_path, cfg.get('results_base', 'results'))
    print(f"[INFO] Using resources_path: {resources_path}")
    print(f"[INFO] Writing results under: {results_base}")

    enabled_experiments = [
        (exp_name, exp_cfg)
        for exp_name, exp_cfg in cfg['experiments'].items()
        if exp_cfg.get('enabled', False)
    ]
    for exp_name, exp_cfg in tqdm(enabled_experiments, desc="experiments", unit="experiment"):
        if exp_cfg.get('enabled', False):
            tqdm.write(f"[INFO] Running experiment: {exp_name}")
            run_experiment(exp_name, exp_cfg, results_base, run_id, device=resolved_device)

if __name__ == '__main__':
    main()
