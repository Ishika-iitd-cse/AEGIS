from metrics import g1, g2, g3, kl, mi, chi2, tvd

def compute_metrics(reconstructed_dist, orig_df, masked_df, attribute, target):
    metrics = {}
    metrics.update(g3.compute_metric(reconstructed_dist, masked_df, attribute, target))
    metrics.update(g1.compute_metric(reconstructed_dist, masked_df, attribute, target))
    metrics.update(g2.compute_metric(reconstructed_dist, masked_df, attribute, target))
    metrics.update(kl.compute_metric(reconstructed_dist, orig_df, masked_df, attribute, target))
    metrics.update(mi.compute_metric(reconstructed_dist, masked_df, attribute, target))
    metrics.update(chi2.compute_metric(reconstructed_dist, masked_df, attribute, target))
    metrics.update(tvd.compute_metric(reconstructed_dist, orig_df, masked_df, attribute, target))
    return metrics