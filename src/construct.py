import os
import re
import argparse
import yaml
import pandas as pd
import numpy as np
from metrics import aggregator

class ReconstructionProcessor:
    def __init__(self, config_path):
        self.config = self.load_config(config_path)
        self.orig_df = pd.read_csv(self.config['dataset']['original_path'])
        self.masked_df = pd.read_csv(self.config['dataset']['masked_path'])
        self.attributes = list(self.config['masking']['attributes'].keys())
        self.target = self.config['dataset']['target_variable']
        self.output_path = self.config['reconstruction']['output_path']
        ipf_params = self.config['reconstruction'].get('ipf_params', {})
        self.max_iter = ipf_params.get('max_iter', 1000)
        self.tol = ipf_params.get('tol', 1e-6)
        self.label_categories = sorted(self.orig_df[self.target].unique())
        self.algorithm = self.config['reconstruction'].get('algorithm', "ipf_without_1d")

    def load_config(self, path):
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def parse_range_bucket(self, bucket_str):
        try:
            start, end = bucket_str.split(':')
            return float(start), float(end)
        except Exception:
            return None

    def ipf_with_1d(self, row_sums, col_sums):
        matrix = np.ones((len(row_sums), len(col_sums)), dtype=float)
        for _ in range(self.max_iter):
            rs = matrix.sum(axis=1)
            rs[rs == 0] = 1e-12
            for i in range(len(row_sums)):
                matrix[i, :] *= row_sums[i] / rs[i]
            cs = matrix.sum(axis=0)
            cs[cs == 0] = 1e-12
            for j in range(len(col_sums)):
                matrix[:, j] *= col_sums[j] / cs[j]
            new_rs = matrix.sum(axis=1)
            new_cs = matrix.sum(axis=0)
            if np.allclose(new_rs, row_sums, atol=self.tol) and np.allclose(new_cs, col_sums, atol=self.tol):
                break
        return matrix

    def ipf_without_1d(self, num_candidates, col_sums, alpha=0.1):
        matrix = np.full((num_candidates, len(col_sums)), 0.0)
        for j in range(len(col_sums)):
            matrix[:, j] = col_sums[j] / num_candidates
        prev_cost = np.inf
        for _ in range(self.max_iter):
            col_means = matrix.mean(axis=0)
            cost = np.sum(np.abs(matrix - col_means))
            if abs(prev_cost - cost) < self.tol:
                break
            prev_cost = cost
            matrix = matrix + alpha * (col_means - matrix)
            for j in range(len(col_sums)):
                col_total = matrix[:, j].sum()
                if col_total != 0:
                    matrix[:, j] *= col_sums[j] / col_total
        return matrix

    def reconstruct_distribution(self, attribute):
        orig_counts = self.orig_df.groupby(attribute).size().to_dict()
        orig_values = sorted(orig_counts.keys())
        masked_cont = self.masked_df.groupby([attribute, self.target]).size().unstack(fill_value=0).sort_index()
        reconstructed = {}
        if self.algorithm == "ipf_with_1d":
            for bucket in masked_cont.index:
                rng = self.parse_range_bucket(str(bucket))
                if rng is None:
                    continue
                mb_start, mb_end = rng
                candidates = [v for v in orig_values if mb_start <= float(v) <= mb_end]
                if not candidates:
                    continue
                row_sums = [orig_counts[v] for v in candidates]
                col_sums = [masked_cont.at[bucket, label] if bucket in masked_cont.index and label in masked_cont.columns else 0 for label in self.label_categories]
                matrix = self.ipf_with_1d(row_sums, col_sums)
                for i, candidate in enumerate(candidates):
                    for j, label in enumerate(self.label_categories):
                        key = (candidate, label)
                        reconstructed[key] = reconstructed.get(key, 0) + matrix[i, j]
        elif self.algorithm == "ipf_without_1d":
            for bucket in masked_cont.index:
                rng = self.parse_range_bucket(str(bucket))
                if rng is None:
                    continue
                mb_start, mb_end = rng
                candidates = [v for v in orig_values if mb_start <= float(v) <= mb_end]
                if not candidates:
                    continue
                num_candidates = len(candidates)
                row_sums = [orig_counts[v] for v in candidates]
                col_sums = [masked_cont.at[bucket, label] if bucket in masked_cont.index and label in masked_cont.columns else 0 for label in self.label_categories]
                matrix = self.ipf_without_1d(num_candidates, col_sums)
                for i, candidate in enumerate(candidates):
                    for j, label in enumerate(self.label_categories):
                        key = (candidate, label)
                        reconstructed[key] = reconstructed.get(key, 0) + matrix[i, j]
        elif self.algorithm == "sampling":
            for bucket in masked_cont.index:
                rng = self.parse_range_bucket(str(bucket))
                if rng is None:
                    continue
                mb_start, mb_end = rng
                total_count = int(masked_cont.loc[bucket].sum())
                bucket_rows = self.masked_df[self.masked_df[attribute] == bucket]
                if bucket_rows.empty:
                    continue
                sampled = bucket_rows.sample(n=total_count, replace=True, random_state=42)
                for _, row in sampled.iterrows():
                    key = (row[attribute], row[self.target])
                    reconstructed[key] = reconstructed.get(key, 0) + 1
        else:
            raise ValueError("Algorithm not recognized. Use 'ipf_with_1d', 'ipf_without_1d', or 'sampling'.")
        return reconstructed

    def process_attribute(self, attribute):
        reconstructed_dist = self.reconstruct_distribution(attribute)
        metric_vals = aggregator.compute_metrics(reconstructed_dist, self.orig_df, self.masked_df, attribute, self.target)
        metric_vals['Attribute'] = attribute
        return metric_vals

    def process(self):
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        results = []
        for attr in self.attributes:
            res = self.process_attribute(attr)
            results.append(res)
        with open(self.output_path, 'w') as f:
            header = "\t".join(["Attribute", 
                                "G3_masked", "G3_recon", "Delta_G3",
                                "G1_masked", "G1_recon", "Delta_G1",
                                "G2_masked", "G2_recon", "Delta_G2",
                                "KL_masked", "KL_recon", "Delta_KL",
                                "MI_masked", "MI_recon", "Delta_MI",
                                "Chi2_masked", "Chi2_recon", "Delta_Chi2", 
                                "TVD_masked", "TVD_recon", "Delta_TVD"]) + "\n"
            f.write(header)
            for r in results:
                line = "\t".join([str(r.get(k, "")) for k in ["Attribute", 
                                                                "G3_masked", "G3_recon", "Delta_G3",
                                                                "G1_masked", "G1_recon", "Delta_G1",
                                                                "G2_masked", "G2_recon", "Delta_G2",
                                                                "KL_masked", "KL_recon", "Delta_KL",
                                                                "MI_masked", "MI_recon", "Delta_MI",
                                                                "Chi2_masked", "Chi2_recon", "Delta_Chi2",
                                                                "TVD_masked", "TVD_recon", "Delta_TVD"]])
                f.write(line + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the config.yaml file')
    args = parser.parse_args()
    processor = ReconstructionProcessor(args.config)
    processor.process()

if __name__ == "__main__":
    main()
