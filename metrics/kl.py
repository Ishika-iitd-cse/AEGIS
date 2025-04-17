import numpy as np
import pandas as pd

def compute_histogram_from_recon(distribution):
    hist = {}
    for (x, _), cnt in distribution.items():
        hist[x] = hist.get(x, 0) + cnt
    return hist

def compute_kl_divergence(P, Q, epsilon=1e-12):
    kl = 0.0
    for k, p in P.items():
        q = Q.get(k, epsilon)
        if p > 0:
            kl += p * np.log(p / (q + epsilon))
    return kl

def compute_metric(reconstructed_dist, orig_df, masked_df, attribute, target=None, epsilon=1e-12):
    orig_counts = orig_df.groupby(attribute).size().to_dict()
    total_orig = sum(orig_counts.values())
    P = {k: v / total_orig for k, v in orig_counts.items()} if total_orig > 0 else {}
    recon_counts = compute_histogram_from_recon(reconstructed_dist)
    total_recon = sum(recon_counts.values())
    Q = {k: v / total_recon for k, v in recon_counts.items()} if total_recon > 0 else {}
    KL_recon = compute_kl_divergence(P, Q, epsilon)
    return {
        'KL_masked': 0.0,
        'KL_recon': KL_recon,
        'Delta_KL': abs(KL_recon - 0.0)
    }