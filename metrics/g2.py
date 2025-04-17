def compute_g2(distribution):
    x_groups = {}
    for (x, y), cnt in distribution.items():
        x_groups.setdefault(x, {})
        x_groups[x][y] = x_groups[x].get(y, 0) + cnt
    G2 = 0
    for group in x_groups.values():
        if len(group) > 1:
            G2 += sum(group.values())
    total = sum(distribution.values())
    norm = G2 / total if total > 0 else 0
    return G2, norm

def masked_g2(df, attribute, target):
    grouped = df.groupby([attribute, target]).size().reset_index(name='count')
    x_groups = {}
    for _, row in grouped.iterrows():
        x = row[attribute]
        y = row[target]
        cnt = row['count']
        x_groups.setdefault(x, {})
        x_groups[x][y] = x_groups[x].get(y, 0) + cnt
    G2 = 0
    for group in x_groups.values():
        if len(group) > 1:
            G2 += sum(group.values())
    total = len(df)
    norm = G2 / total if total > 0 else 0
    return G2, norm

def compute_metric(reconstructed_dist, masked_df, attribute, target):
    G2_recon, norm_recon = compute_g2(reconstructed_dist)
    G2_masked, norm_masked = masked_g2(masked_df, attribute, target)
    delta = abs(norm_recon - norm_masked)
    return {
        'G2_masked': norm_masked,
        'G2_recon': norm_recon,
        'Delta_G2': delta
    }
