import numpy as np

def find_contiguous_subsets(positions, n=1):
    positions = np.sort(np.unique(positions))  # Ensure sorted unique positions
    subsets = []
    start = positions[0]
    prev = start
        
    for pos in positions[1:]:
        if pos != prev + n - 1:
            subsets.append((start, prev))
            start = pos
        prev = pos
    subsets.append((start, prev))  # Append last subset
        
    return subsets

def compute_measurements(positions):
    subsets = find_contiguous_subsets(positions)
    measurements = []
    std_devs = []
        
    for start, stop in subsets:
        subset = np.arange(start, stop + 1)
        mean_val = np.mean(subset)
        std_val = np.std(subset)
        measurements.append(mean_val)
        std_devs.append(std_val)
        
    # Covariance matrix R: Diagonal with variances
    # R = np.diag(np.array(std_devs) ** 2)
      
    return measurements, std_devs

def cluster_and_compute_stats(measurements, spacing=1):
    """
        Given a sorted array of pixel positions, clusters contiguous pixels and computes
        the centroid and standard deviation of each cluster.

        Parameters:
            measurements (np.ndarray): 1D array of pixel positions (sorted or unsorted).
            spacing (int): Maximum allowed gap between consecutive pixels to form a cluster.

        Returns:
            centroids (np.ndarray): Array of centroids for each cluster.
            stds (np.ndarray): Array of standard deviations for each cluster.
    """
        
    if len(measurements) == 0:
        return np.array([]), np.array([])

        # Sort the measurements (if not already sorted)
    measurements = np.sort(measurements)

        # Find cluster boundaries
    cluster_splits = np.where(np.diff(measurements) > spacing)[0] + 1

        # Split into clusters
    clusters = np.split(measurements, cluster_splits)

        # Compute centroid and std for each cluster
    centroids = np.array([np.mean(cluster) for cluster in clusters])
    stds = np.array([np.std(cluster) if len(cluster) > 1 else 1.0 for cluster in clusters])  # std=1.0 if singleton

    return centroids, stds

def test_equivalence(measurements_list):
    for i, positions in enumerate(measurements_list, 1):
        print(f"\nTest case {i}: positions = {positions}")
        # cm_means, cm_stds = compute_measurements(positions)
        cl_means, cl_stds = cluster_and_compute_stats(positions, spacing=1)

        # print(f"Method 1 (compute_measurements):")
        # print(f"  Means: {cm_means}")
        # print(f"  Stds : {cm_stds}")
        print(f"Method 2 (cluster_and_compute_stats):")
        print(f"  Means: {cl_means}")
        print(f"  Stds : {cl_stds}")

# === Tricky test cases ===

test_cases = [
    [1, 2, 3, 10, 11, 12],            # two contiguous clusters
    [1, 2, 3, 7, 8, 20],              # non-uniform cluster gaps
    [1, 2, 2, 3, 3, 3],               # duplicates (should be removed)
    [5],                              # singleton
    [],                               # empty input
    [1, 5, 6, 10, 11, 12, 13],        # mix of singleton and a long cluster
    list(range(1, 11)) + [20, 21],    # long cluster + short one
    [1, 4, 7, 10],                    # all singletons (no adjacent points)
    [1, 2, 3, 4, 100, 101, 102, 103], # two well-separated blocks
]

test_equivalence(test_cases)