from scipy.cluster.hierarchy import linkage
import numpy as np
from tqdm import tqdm


def shc_linkage(y, method='single', metric='euclidean', optimal_ordering=False):
    """
    Perform Statistical Significance for Hierarchical Clustering (SHC) [1].
    This function wraps scipy.cluster.hierarchy.linkage and uses the same
    arguments.

    References
    ----------
    [1] Kimes PK, Liu Y, Neil Hayes D, Marron JS. "Statistical significance
        for hierarchical clustering," Biometrics, vol. 73, no. 3, pp. 811-821,
        Sep 2017. :doi:`10.1111/biom.12647`.
    """

    # Initialize a mapping from cluster indices to observation indices
    # At first, each observation is its own cluster
    n = y.shape[0]
    cluster_map = {i: [i] for i in range(n)}

    # Step 1: Run HAC on the original data
    z = linkage(y, method, metric, optimal_ordering)
    p_values = np.empty(n - 1)

    # Helper function to simulate null datasets and compute HAC
    def simulate_hac_and_get_distances(null_mean, null_cov, n_points,
                                       num_simulations=2):
        null_distances = []
        for _ in tqdm(range(num_simulations)):
            # Step 4: Simulate n points from the Gaussian distribution
            simulated_data = np.random.multivariate_normal(null_mean, null_cov,
                                                           size=n_points)
            # Compute HAC on the simulated dataset
            z_simulated = linkage(simulated_data, method, metric)
            # Collect the root split distance (last distance in linkage matrix)
            null_distances.append(z_simulated[-1, 2])  # Distance at the root
        return np.array(null_distances)

    # Step 2: Traverse the hierarchy from the leaves and test each split for
    # significance
    for i in range(len(z)):
        # Extract cluster indices and distance from the linkage matrix
        cluster_1, cluster_2, observed_distance, num_points = z[i]
        cluster_1, cluster_2 = int(cluster_1), int(cluster_2)

        # Find the observations that belong to each cluster
        points_in_both_clusters = np.vstack([y[cluster_map[cluster_1]],
                                             y[cluster_map[cluster_2]]])

        # Step 3: Fit a Gaussian to the combined group of points at this level
        n_points = points_in_both_clusters.shape[0]
        null_mean = np.mean(points_in_both_clusters, axis=0)
        null_cov = np.cov(points_in_both_clusters, rowvar=False)

        # Steps 4 & 5: Simulate null datasets and calculate p-value
        null_distances = simulate_hac_and_get_distances(null_mean,
                                                        null_cov,
                                                        n_points)

        # Calculate the p-value as the proportion of null distances greater
        # than the observed distance
        p_values[i] = np.mean(null_distances >= observed_distance)

        # Update the cluster map to reflect the new merged cluster
        cluster_map[n + i] = cluster_map[cluster_1] + cluster_map[cluster_2]

    return z, p_values