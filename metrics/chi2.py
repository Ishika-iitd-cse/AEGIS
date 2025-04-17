import numpy as np
import pandas as pd

def compute_contingency_from_distribution(distribution):
    table = {}
    for (x, y), count in distribution.items():
        table.setdefault(x, {})
        table[x][y] = table[x].get(y, 0) + count
    df = pd.DataFrame(table).fillna(0).T
    return df

def compute_chi_square(contingency):
    total = contingency.values.sum()
    row_totals = contingency.sum(axis=1)
    col_totals = contingency.sum(axis=0)
    chi2 = 0.0
    for row in contingency.index:
        for col in contingency.columns:
            expected = (row_totals[row] * col_totals[col]) / total if total > 0 else 0
            observed = contingency.at[row, col]
            if expected > 0:
                chi2 += ((observed - expected) ** 2) / expected
    return chi2

def compute_metric(reconstructed_dist, masked_df, attribute, target):
    contingency_recon = compute_contingency_from_distribution(reconstructed_dist)
    chi2_recon = compute_chi_square(contingency_recon)
    
    contingency_masked = masked_df.groupby([attribute, target]).size().unstack(fill_value=0).sort_index()
    chi2_masked = compute_chi_square(contingency_masked)
    
    total = masked_df.shape[0]
    norm_recon = chi2_recon / total if total > 0 else 0
    norm_masked = chi2_masked / total if total > 0 else 0
    delta = abs(norm_recon - norm_masked)
    
    return {
        'Chi2_masked': norm_masked,
        'Chi2_recon': norm_recon,
        'Delta_Chi2': delta
    }
