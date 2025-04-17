def compute_g3(distribution):
    x_groups = {}
    for (x, y), cnt in distribution.items():
        x_groups.setdefault(x, {})
        x_groups[x][y] = x_groups[x].get(y, 0) + cnt
    total = sum(distribution.values())
    sum_majority = sum(max(group.values()) for group in x_groups.values())
    G3 = total - sum_majority
    norm = G3 / total if total > 0 else 0
    return G3, norm

def masked_g3(df, attribute, target):
    grouped = df.groupby([attribute, target]).size().reset_index(name='count')
    total = len(df)
    sum_majority = 0
    for _, subdf in grouped.groupby(attribute):
        sum_majority += subdf['count'].max()
    G3 = total - sum_majority
    norm = G3 / total if total > 0 else 0
    return G3, norm

def compute_metric(reconstructed_dist, masked_df, attribute, target):
    G3_recon, norm_recon = compute_g3(reconstructed_dist)
    G3_masked, norm_masked = masked_g3(masked_df, attribute, target)
    delta = abs(norm_recon - norm_masked)
    return {
        'G3_masked': norm_masked,
        'G3_recon': norm_recon,
        'Delta_G3': delta
    }
