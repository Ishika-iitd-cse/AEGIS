def compute_g1(distribution):
    x_groups = {}
    for (x, y), cnt in distribution.items():
        x_groups.setdefault(x, {})
        x_groups[x][y] = x_groups[x].get(y, 0) + cnt
    G1 = 0
    for group in x_groups.values():
        T = sum(group.values())
        if T <= 1:
            continue
        same_pairs = sum(freq * (freq - 1) / 2 for freq in group.values())
        all_pairs = T * (T - 1) / 2
        G1 += (all_pairs - same_pairs)
    total = sum(distribution.values())
    norm = G1 / (total**2) if total > 0 else 0
    return G1, norm

def masked_g1(df, attribute, target):
    grouped = df.groupby([attribute, target]).size().reset_index(name='count')
    x_groups = {}
    for _, row in grouped.iterrows():
        x = row[attribute]
        y = row[target]
        cnt = row['count']
        x_groups.setdefault(x, {})
        x_groups[x][y] = x_groups[x].get(y, 0) + cnt
    G1 = 0
    for group in x_groups.values():
        T = sum(group.values())
        if T <= 1:
            continue
        same_pairs = sum(freq * (freq - 1) / 2 for freq in group.values())
        all_pairs = T * (T - 1) / 2
        G1 += (all_pairs - same_pairs)
    total = len(df)
    norm = G1 / (total**2) if total > 0 else 0
    return G1, norm

def compute_metric(reconstructed_dist, masked_df, attribute, target):
    G1_recon, norm_recon = compute_g1(reconstructed_dist)
    G1_masked, norm_masked = masked_g1(masked_df, attribute, target)
    delta = abs(norm_recon - norm_masked)
    return {
        'G1_masked': norm_masked,
        'G1_recon': norm_recon,
        'Delta_G1': delta
    }
