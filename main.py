import os
import shutil
import subprocess
import pandas as pd
import glob
import argparse
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import re
import time

class ConfigProcessor:
    def __init__(self, config_path, result_dir):
        self.config_path = config_path
        self.result_dir = result_dir
        self.config = self.load_config()
    
    def load_config(self):
        with open(self.config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def run_script(self, script_path, extra_args=None, capture=False):
        if extra_args is None:
            extra_args = []
        cmd = ['python3', script_path, '--config', self.config_path] + extra_args
        if capture:
            return subprocess.run(cmd, capture_output=True, text=True, check=True)
        subprocess.run(cmd, check=True)
    
    def process_mask(self):
        mask_script = 'src/mask.py'
        start = time.time()
        self.run_script(mask_script)
        t = time.time() - start

        masked_src = self.config['dataset']['masked_path']
        masked_dst = os.path.join(self.result_dir, 'masked_data.csv')
        if os.path.exists(masked_src):
            shutil.copy(masked_src, masked_dst)
            return masked_dst, t
        return None, t
    
    def process_construct(self):
        construct_script = 'src/construct.py'
        start = time.time()
        self.run_script(construct_script)
        t = time.time() - start

        output_path = self.config['reconstruction']['output_path']
        if not os.path.exists(output_path):
            return None, t
        dst_path = os.path.join(self.result_dir, 'reconstruction_results.tsv')
        shutil.copy(output_path, dst_path)
        df = pd.read_csv(output_path, sep="\t")
        avg_metrics = {col: df[col].mean() for col in df.columns if col != "Attribute"}
        return avg_metrics, t
    
    def process_model(self):
        model_script = self.config['dataset'].get('model_script')
        if not model_script or not os.path.exists(model_script):
            print(f"No valid model_script specified in config, or path not found: {model_script}")
            return None, 0.0
        try:
            start = time.time()
            result = self.run_script(model_script, capture=True)
            t = time.time() - start

            stdout = result.stdout
            def extract_value(pattern):
                m = re.search(pattern, stdout)
                return float(m.group(1)) if m else None

            orig_acc = extract_value(r'Original Accuracy:\s*([\d.]+)')
            masked_acc = extract_value(r'Masked Accuracy:\s*([\d.]+)')
            acc_diff = extract_value(r'Accuracy Difference:\s*([\d.]+)')

            return {
                'Original_Accuracy': orig_acc,
                'Masked_Accuracy': masked_acc,
                'Accuracy_Difference': acc_diff
            }, t

        except subprocess.CalledProcessError as e:
            print("An error occurred in process_model:")
            print(e)
            return None, 0.0
        except Exception as e:
            print(f"Unexpected error in process_model: {e}")
            return None, 0.0
    
    def process(self):
        masked_dst, t_mask = self.process_mask()
        if not masked_dst:
            return None

        construct_stats, t_construct = self.process_construct()
        if construct_stats is None:
            return None

        model_stats, t_model = self.process_model()
        if model_stats is None:
            return None

        final_stats = {
            **construct_stats,
            **model_stats,
            'Time_Mask': t_mask,
            'Time_Construct': t_construct,
            'Time_Model': t_model
        }
        return final_stats

def save_summary(results, path):
    df = pd.DataFrame(results).sort_values(by='Config')
    df.to_csv(path, index=False)
    return df

def create_plot(df, x_key, y_key, filename):
    plt.style.use('seaborn-v0_8-muted')
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(data=df, x=x_key, y=y_key, hue='Config', palette='tab20', s=150, ax=ax)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.title(f'{y_key} vs {x_key}', fontsize=16, weight='bold')
    plt.xlabel(x_key, fontsize=14)
    plt.ylabel(y_key, fontsize=14)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs_dir', type=str, default='configs/', help='Directory containing config.yaml files')
    parser.add_argument('--results_dir', type=str, default='results/', help='Directory to store results')
    args = parser.parse_args()
    
    os.makedirs(args.results_dir, exist_ok=True)
    config_files = sorted(glob.glob(os.path.join(args.configs_dir, '*.yaml')))
    if not config_files:
        print(f"No config files found in {args.configs_dir}")
        return
    
    summary = []
    for config_file in config_files:
        print(f'\nProcessing {config_file}')
        config_name = os.path.splitext(os.path.basename(config_file))[0]
        config_result_dir = os.path.join(args.results_dir, config_name)
        os.makedirs(config_result_dir, exist_ok=True)
        
        processor = ConfigProcessor(config_file, config_result_dir)
        stats = processor.process()
        if stats:
            row = {
                'Config': config_name,
                'G1_masked': stats.get('G1_masked'),
                'G1_recon': stats.get('G1_recon'),
                'Delta_G1': stats.get('Delta_G1'),
                'G2_masked': stats.get('G2_masked'),
                'G2_recon': stats.get('G2_recon'),
                'Delta_G2': stats.get('Delta_G2'),
                'G3_masked': stats.get('G3_masked'),
                'G3_recon': stats.get('G3_recon'),
                'Delta_G3': stats.get('Delta_G3'),
                'KL_masked': stats.get('KL_masked'),
                'KL_recon': stats.get('KL_recon'),
                'Delta_KL': stats.get('Delta_KL'),
                'MI_masked': stats.get('MI_masked'),
                'MI_recon': stats.get('MI_recon'),
                'Delta_MI': stats.get('Delta_MI'),
                'Chi2_masked': stats.get('Chi2_masked'),
                'Chi2_recon': stats.get('Chi2_recon'),
                'Delta_Chi2': stats.get('Delta_Chi2'),
                'TVD_masked': stats.get('TVD_masked'),
                'TVD_recon': stats.get('TVD_recon'),
                'Delta_TVD': stats.get('Delta_TVD'),
                'Original_Accuracy': stats.get('Original_Accuracy'),
                'Masked_Accuracy': stats.get('Masked_Accuracy'),
                'Accuracy_Difference': stats.get('Accuracy_Difference'),
                'Time_Mask': stats.get('Time_Mask'),
                'Time_Construct': stats.get('Time_Construct'),
                'Time_Model': stats.get('Time_Model')
            }
            summary.append(row)
    
    if summary:
        summary_path = os.path.join(args.results_dir, 'summary_results.csv')
        df_summary = save_summary(summary, summary_path)
        for metric in ['Delta_G1', 'Delta_G2', 'Delta_G3', 'Delta_KL', 'Delta_MI', 'Delta_Chi2', 'Delta_TVD']:
            create_plot(df_summary, metric, 'Accuracy_Difference', os.path.join(args.results_dir, f'{metric.lower()}_vs_acc.png'))
        print(f"Summary saved to {summary_path}")

if __name__ == "__main__":
    main()