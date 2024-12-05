from scipy.cluster.hierarchy import linkage
import numpy as np
from tqdm import tqdm
import multiprocess as mp


def precompute_cluster_memberships(Z, n):
    """Precompute observation indices for all clusters."""
    cluster_map = {i: [i] for i in range(n)}  # Initially, each observation is its own cluster

    for i, row in enumerate(Z):
        cluster_1, cluster_2 = int(row[0]), int(row[1])
        new_cluster = cluster_map[cluster_1] + cluster_map[cluster_2]
        cluster_map[n + i] = new_cluster  # Update cluster map

    return cluster_map


def simulate_hac_for_split(args):
    null_mean, null_cov, n_points, num_simulations = args
    null_distances = []
    for _ in range(num_simulations):
        # Simulate n points from the Gaussian distribution
        simulated_data = np.random.multivariate_normal(null_mean, null_cov, size=n_points)
        # Compute HAC on the simulated dataset
        Z_simulated = linkage(simulated_data, method='ward')
        # Collect the root split distance (last distance in linkage matrix)
        null_distances.append(Z_simulated[-1, 2])  # Distance at the root
    return np.array(null_distances)


def process_linkage_row(args):
    row_index, Z, target_embeddings, cluster_map, num_simulations = args

    cluster_1, cluster_2, distance, _ = Z[row_index]
    cluster_1, cluster_2 = int(cluster_1), int(cluster_2)

    # Use precomputed observation indices for the clusters
    points_in_both_clusters = np.vstack([
        target_embeddings[cluster_map[cluster_1]],
        target_embeddings[cluster_map[cluster_2]]
    ])

    # Fit a Gaussian to the combined group of n points at this level
    n_points = points_in_both_clusters.shape[0]
    null_mean = np.mean(points_in_both_clusters, axis=0)
    null_cov = np.cov(points_in_both_clusters, rowvar=False)

    # Simulate HAC and calculate null distances
    null_distances = simulate_hac_for_split((null_mean, null_cov, n_points, num_simulations))

    # Calculate the p-value as the proportion of null distances greater than the observed distance
    observed_distance = distance  # Original HAC split distance for the two clusters
    p_value = np.mean(null_distances >= observed_distance)

    return row_index, p_value


def shc_linkage(target_embeddings, num_simulations=10, num_processes=8):
    # TODO: check that num_processes < len(target_embeddings)

    # Step 1: Run HAC to get the linkage matrix
    Z = linkage(target_embeddings, method='ward')

    # Step 2: Precompute memberships for all clusters
    n = target_embeddings.shape[0]
    cluster_map = precompute_cluster_memberships(Z, n)

    # Step 3: Prepare arguments for multiprocessing
    tasks = [
        (i, Z, target_embeddings, cluster_map, num_simulations)
        for i in range(len(Z))
    ]

    with mp.Pool(processes=num_processes) as pool:
        # Wrap the tasks in tqdm to show progress
        unordered_results = list(tqdm(pool.imap(process_linkage_row, tasks), total=len(tasks), desc="Processing linkage rows"))

    # Sort the results by row index to preserve order
    results = sorted(unordered_results, key=lambda x: x[0])  # Sort by row_index

    # Step 5: Process results
    for row_index, p_value in results:
        if p_value < 0.05:
            print(f"Split at level {row_index} is statistically significant with p-value {p_value:.4f}")
        else:
            print(f"Split at level {row_index} is NOT statistically significant (p-value {p_value:.4f})")

    return Z


