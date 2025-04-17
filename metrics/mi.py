import numpy as np
import pandas as pd

def compute_mutual_information(p_x, p_y, p_xy, epsilon=1e-12):
    mi = 0.0
    p_x = p_x / p_x.sum() if p_x.sum() != 0 else p_x
    p_y = p_y / p_y.sum() if p_y.sum() != 0 else p_y
    total = p_xy.values.sum()
    p_xy = p_xy / total if total != 0 else p_xy
    for x in p_xy.index:
        for y in p_xy.columns:
            p_xy_val = p_xy.at[x, y]
            if p_xy_val > 0:
                mi += p_xy_val * np.log2(p_xy_val / (p_x.get(x, 0) * p_y.get(y, 0) + epsilon))
    return mi

def compute_metric(reconstructed_dist, masked_df, attribute, target, epsilon=1e-12):
    recon_df = pd.DataFrame([(k[0], k[1], v) for k, v in reconstructed_dist.items()], columns=[attribute, target, 'count'])
    if not recon_df.empty:
        recon_contingency = recon_df.pivot_table(index=attribute, columns=target, values='count', aggfunc='sum', fill_value=0).sort_index()
    else:
        recon_contingency = pd.DataFrame()
    masked_contingency = masked_df.groupby([attribute, target]).size().unstack(fill_value=0).sort_index()
    total_recon = recon_contingency.values.sum() if not recon_contingency.empty else 0
    total_masked = masked_contingency.values.sum() if not masked_contingency.empty else 0
    mi_recon = compute_mutual_information(recon_contingency.sum(axis=1), recon_contingency.sum(axis=0), recon_contingency, epsilon) if total_recon > 0 else 0
    mi_masked = compute_mutual_information(masked_contingency.sum(axis=1), masked_contingency.sum(axis=0), masked_contingency, epsilon) if total_masked > 0 else 0
    delta_mi = abs(mi_recon - mi_masked)
    return {
        'MI_masked': mi_masked,
        'MI_recon': mi_recon,
        'Delta_MI': delta_mi
    }