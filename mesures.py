from scipy.spatial.distance import cdist
import numpy as np

# based on how well we classify the biggest cluster (wich should be the one with clean laps)    
def biggest_cluster_quality_score(X_scaled, labels):
    # Points in the biggest cluseter
    unique_labels, counts = np.unique(labels, return_counts=True)
    biggest_cluster_label = unique_labels[np.argmax(counts)]
    cluster0_points = X_scaled[labels == biggest_cluster_label]
    # Points NOT in cluster 0 (including noise)
    other_points = X_scaled[labels != biggest_cluster_label]

    if len(cluster0_points) < 2:
        # Not enough points in cluster 0 to evaluate compactness
        return -1

    # Intra-cluster distances within cluster 0 (pairwise distances)
    intra_dists = cdist(cluster0_points, cluster0_points, metric='chebyshev')
    # Remove diagonal zeros by taking upper triangle distances only
    intra_dists = intra_dists[np.triu_indices_from(intra_dists, k=1)]
    mean_intra = np.mean(intra_dists)

    if len(other_points) == 0:
        # No other points, so no separation possible
        # Smaller intra-cluster distance is better
        return mean_intra

    # Inter-cluster distances between cluster 0 and other points
    inter_dists = cdist(cluster0_points, other_points, metric='chebyshev')
    mean_inter = np.mean(inter_dists)

    # Score = separation / compactness (higher is better)
    score = mean_inter / (mean_intra + 1e-10)
    return score
