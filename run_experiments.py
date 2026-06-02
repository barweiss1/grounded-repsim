import os
import sys
import yaml
import json
import pathlib
from datetime import datetime
from importlib import import_module

try:
    import torch
except Exception:
    torch = None

from paths import resources_path
sys.path.append(os.path.abspath("dists/"))

def load_config(config_path):
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

    exp_path = pathlib.Path(__file__).parent / 'experiments' / exp_name
    sys.path.insert(0, str(exp_path))

    # 1. compute_dists
    if exp_cfg.get('compute_dists', None):
        mod = import_module(f'experiments.{exp_name}.compute_dists')
        if hasattr(mod, 'run_compute_dists'):
            func = getattr(mod, 'run_compute_dists')
            func(exp_cfg['compute_dists'], exp_dir, resources_path, device=device)
        else:
            print(f"[WARN] {exp_name}/compute_dists.py has no run_compute_dists function.")

    # 2. compute_full_df
    if exp_cfg.get('compute_full_df', None):
        mod = import_module(f'experiments.{exp_name}.compute_full_df')
        if hasattr(mod, 'run_compute_full_df'):
            func = getattr(mod, 'run_compute_full_df')
            func(exp_cfg['compute_full_df'], exp_dir, resources_path, device=device)
        else:
            print(f"[WARN] {exp_name}/compute_full_df.py has no run_compute_full_df function.")

    # 3. script
    if exp_cfg.get('script', None):
        mod = import_module(f'experiments.{exp_name}.script')
        if hasattr(mod, 'run_experiment_script'):
            func = getattr(mod, 'run_experiment_script')
            func(exp_cfg['script'], exp_dir, resources_path, device=device)
        else:
            print(f"[WARN] {exp_name}/script.py has no run_experiment_script function.")

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

    for exp_name, exp_cfg in cfg['experiments'].items():
        if exp_cfg.get('enabled', False):
            print(f"[INFO] Running experiment: {exp_name}")
            run_experiment(exp_name, exp_cfg, results_base, run_id, device=resolved_device)

if __name__ == '__main__':
    main()
