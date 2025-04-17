import os
import time
import yaml
import glob
import pandas as pd

from src.mask import MaskingProcessor
from src.construct import ReconstructionProcessor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

MODEL_SCRIPTS = {
    'model1': 'data/{d}/model1.py',
    'model2': 'data/{d}/model2.py',
    'model3': 'data/{d}/model3.py',
    'model4': 'data/{d}/model4.py'
}

ALGORITHMS = ['ipf_with_1d', 'ipf_without_1d', 'sampling']
METRICS = ['Delta_MI', 'Delta_Chi2']
TARGET_VARIABLES = {1: 'Air Quality', 2: 'is_booking', 3: 'income'}
TOTAL_CONFIGS = 50

def generate_configs_for_dataset(dataset):
    print(f"\n[INFO] Generating configurations for Dataset {dataset}...")
    from generate_configs import main as gen_main
    args = [
        'generate_configs.py',
        '--original', f'data/{dataset}/original_data.csv',
        '--target_variable', TARGET_VARIABLES[dataset],
        '--model_script', MODEL_SCRIPTS['model1'].format(d=dataset),
        '--output_dir', f'configs/{dataset}',
        '--dataset_tag', str(dataset),
        '--num_configs', str(TOTAL_CONFIGS),
        '--algorithm', ALGORITHMS[0]
    ]
    import sys
    old_argv = sys.argv
    sys.argv = args
    try:
        gen_main()
    finally:
        sys.argv = old_argv
    print(f"[SUCCESS] Configurations generated for Dataset {dataset}.")

def update_configs_yaml(dataset, field_path, new_value):
    print(f"[INFO] Updating configurations for Dataset {dataset}...")
    cfg_dir = f'configs/{dataset}'
    for path in glob.glob(f'{cfg_dir}/*.yaml'):
        with open(path) as f:
            cfg = yaml.safe_load(f)
        node = cfg
        for key in field_path[:-1]:
            node = node[key]
        node[field_path[-1]] = new_value
        with open(path, 'w') as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"[SUCCESS] Configurations updated for Dataset {dataset}.")

def load_config(path):
    print(f"[INFO] Loading configuration from {path}...")
    with open(path) as f:
        return yaml.safe_load(f)

def run_reconstruction(dataset, metric):
    print(f"[INFO] Running reconstruction for Dataset {dataset} using Metric '{metric}'...")
    cfg_dir = f'configs/{dataset}'
    best_vals = []
    elapsed = 0
    for cfg in glob.glob(f'{cfg_dir}/*.yaml'):
        start = time.time()
        mp = MaskingProcessor(cfg)
        mp.process()
        rp = ReconstructionProcessor(cfg)
        vals = [rp.process_attribute(attr)[metric] for attr in rp.attributes]
        best_vals.append(sum(vals) / len(vals))
        elapsed += time.time() - start
    print(f"[SUCCESS] Reconstruction completed for Dataset {dataset} (Metric: {metric}). Time: {elapsed:.2f}s")
    return min(best_vals), elapsed

def run_model(dataset, model_name):
    print(f"[INFO] Running Model '{model_name}' for Dataset {dataset}...")
    cfg_dir = f'configs/{dataset}'
    best_cfg = f'{cfg_dir}/config00.yaml'
    cfg = load_config(best_cfg)
    mod_path = MODEL_SCRIPTS[model_name].format(d=dataset).replace('/', '.').replace('.py', '')
    start = time.time()
    module = __import__(mod_path, fromlist=['CustomModelProcessor'])
    Processor = getattr(module, 'CustomModelProcessor')
    proc = Processor(best_cfg)
    proc.process()
    elapsed = time.time() - start
    print(f"[SUCCESS] Model '{model_name}' completed for Dataset {dataset}. Time: {elapsed:.2f}s")
    return elapsed, best_cfg

def main():
    print("\n[INFO] Starting the main process...")
    results_algo = []
    results_model = []

    for dataset in [1, 2, 3]:
        print(f"\n[INFO] Processing Dataset {dataset}...")
        generate_configs_for_dataset(dataset)

        for algo in ALGORITHMS:
            print(f"\n[INFO] Evaluating Algorithm '{algo}' for Dataset {dataset}...")
            update_configs_yaml(dataset, ['reconstruction', 'algorithm'], algo)
            for metric in METRICS:
                best_val, t = run_reconstruction(dataset, metric)
                results_algo.append({
                    'dataset': dataset,
                    'algorithm': algo,
                    'metric': metric,
                    'time': t,
                    'best_metric': best_val
                })
                print(f"[RESULT] Algorithm: {algo}, Metric: {metric}, Time: {t:.2f}s, Best Value: {best_val}")

        for model in MODEL_SCRIPTS:
            print(f"\n[INFO] Evaluating Model '{model}' for Dataset {dataset}...")
            update_configs_yaml(dataset, ['dataset', 'model_script'], MODEL_SCRIPTS[model].format(d=dataset))
            t_model, best_cfg = run_model(dataset, model)
            t_total_model = t_model * TOTAL_CONFIGS
            results_model.append({
                'dataset': dataset,
                'model': model,
                'time': t_total_model,
                'best_config': os.path.basename(best_cfg)
            })
            print(f"[RESULT] Model: {model}, Total Time: {t_total_model:.2f}s, Best Config: {os.path.basename(best_cfg)}")

    pd.DataFrame(results_algo).to_csv('time_algo.csv', index=False)
    pd.DataFrame(results_model).to_csv('time_model.csv', index=False)
    print("\n[INFO] Process completed. Results saved to 'time_algo.csv' and 'time_model.csv'.")

if __name__ == '__main__':
    main()
