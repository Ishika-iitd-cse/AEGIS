import argparse
import os
import random
import yaml
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description='Randomly generate YAML config files for data masking.')
    parser.add_argument('--original', type=str, required=True, help='Path to the original CSV data.')
    parser.add_argument('--target_variable', type=str, required=True, help='Name of the target variable (excluded from masking).')
    parser.add_argument('--model_script', type=str, default='data/1/model.py', help='Path to the custom model script for this dataset.')
    parser.add_argument('--output_dir', type=str, default='configs/1', help='Directory to save the generated config files.')
    parser.add_argument('--dataset_tag', type=str, default='1', help='Used to build output paths for masked CSVs and reconstruction results (e.g., "1").')
    parser.add_argument('--num_configs', type=int, default=3, help='Number of config files to generate.')
    parser.add_argument('--algorithm', type=str, choices=['sampling', 'ipf_with_1d', 'ipf_without_1d'], default='ipf_without_1d', help='Reconstruction algorithm: "ipf_with_1d" or "ipf_without_1d".')
    return parser.parse_args()

def random_masking_function(is_numeric=True):
    if is_numeric:
        choices = [
            ('generalize', {'M': random.randint(2, 5)}),
            ('erase_digits', {'num_digits': random.randint(1, 3)}),
            ('suppress', {})
        ]
    else:
        choices = [
            ('suppress', {}),
            (None, None)
        ]
    func, params = random.choice(choices)
    return func, params

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_csv(args.original, nrows=200)
    all_columns = df.columns.tolist()
    if args.target_variable not in all_columns:
        print(f"WARNING: Target variable '{args.target_variable}' not found in the CSV columns.")
    columns_to_mask = [col for col in all_columns if col != args.target_variable]
    numeric_cols = df.select_dtypes(include=['int', 'float']).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c in columns_to_mask]
    categorical_cols = [c for c in columns_to_mask if c not in numeric_cols]
    num_digits = len(str(args.num_configs))
    for i in range(1, args.num_configs + 1):
        masking_dict = {}
        for col in numeric_cols:
            func, params = random_masking_function(is_numeric=True)
            if func is None:
                continue
            masking_dict.setdefault(col, [])
            masking_dict[col].append({
                'function': func,
                'params': params
            })
        for col in categorical_cols:
            func, params = random_masking_function(is_numeric=False)
            if func is None:
                continue
            masking_dict.setdefault(col, [])
            masking_dict[col].append({
                'function': func,
                'params': params
            })
        config_name = f"config{i:0{num_digits}d}"
        masked_csv = f"data/{args.dataset_tag}/{config_name}_masked.csv"
        reconstruction_out = f"results/{args.dataset_tag}/{config_name}_reconstruction.tsv"
        config_data = {
            'dataset': {
                'original_path': args.original,
                'masked_path': masked_csv,
                'target_variable': args.target_variable,
                'model_script': args.model_script
            },
            'masking': {
                'attributes': masking_dict
            },
            'reconstruction': {
                'ipf_params': {
                    'max_iter': 1000,
                    'tol': 1e-6
                },
                'algorithm': args.algorithm,
                'output_path': reconstruction_out
            }
        }
        output_file = os.path.join(args.output_dir, f"{config_name}.yaml")
        with open(output_file, 'w') as f:
            yaml.safe_dump(config_data, f, sort_keys=False)
        print(f"Generated {output_file}")

if __name__ == "__main__":
    main()
