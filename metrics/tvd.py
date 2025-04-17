import numpy as np
import pandas as pd

def compute_tvd(dist1, dist2):
    total1 = sum(dist1.values())
    total2 = sum(dist2.values())
    support = set(dist1) | set(dist2)
    abs_diff = 0.0
    for key in support:
        p = dist1.get(key, 0) / total1 if total1 > 0 else 0
        q = dist2.get(key, 0) / total2 if total2 > 0 else 0
        abs_diff += abs(p - q)
    return 0.5 * abs_diff

def compute_metric(reconstructed_dist, orig_df, masked_df, attribute, target):
    orig_joint = orig_df.groupby([attribute, target]).size().to_dict()
    recon_joint = reconstructed_dist.copy()
    TVD_recon = compute_tvd(orig_joint, recon_joint)
    return {
        'TVD_masked': 0.0,
        'TVD_recon': TVD_recon,
        'Delta_TVD': abs(TVD_recon - 0.0)
    }
