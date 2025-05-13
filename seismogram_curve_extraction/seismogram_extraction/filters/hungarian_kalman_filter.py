import numpy as np
from scipy.optimize import linear_sum_assignment

class HungarianKalmanFilter:
    def __init__(self, F, H, Q, R):
        self.F = F  # State transition matrix
        self.H = H  # Observation matrix
        self.Q = Q  # Process noise covariance
        self.R = R  # Measurement noise covariance
    
    def predict(self):
        X, P, F, Q = self.X, self.P, self.F, self.Q
        # check if X has 1 or 2 dimension(s)
        if len(X.shape) == 1:
            self.X = F @ X
            self.P = F @ P @ F.T + Q
        else:
            # multiple states in parallel
            for i in range(X.shape[0]):
                self.X[i] = F @ X[i]
                self.P[i] = F @ P[i] @ F.T + Q
        return self.X, self.P
    
    def update(self, X, P, Z):
        H, R = self.H, self.R
        if len(X.shape) == 1:
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            X = X + K @ (Z - H @ X)
            P = P - K @ H @ P
        else:
            for i in range(X.shape[0]):
                S = H @ P[i] @ H.T + R
                K = P[i] @ H.T @ np.linalg.inv(S)
                X[i] = X[i] + K @ (Z[i] - H @ X[i])
                P[i] = P[i] - K @ H @ P[i]
        return X, P

    def process_sequence(self, sequence, X_0, P_0, step=1):
        """
        Processes a sequence of inputs as if using an RNN. This function mimics an RNN behavior,
        where each time step input updates the state.

        Args:
            sequence (np.ndarray): Shape (batch_size, 1, height, width) - Sequence of batch_size-dimensional inputs.
            X_0 (np.ndarray): Shape (batch_size, N_traces, N_states) - Initial state estimates for N_states traces.
            P_0 (np.ndarray): Shape (batch_size, N_traces, N_states, N_states) - Initial covariance matrices for N_states traces.

        Returns:
            X_results (np.ndarray): Shape (batch_size, width, N_traces, N_states) - Updated state estimates for each time step (width).
            P_results (np.ndarray): Shape (batch_size, width, N_traces, N_states, N_states) - Updated covariance matrices for each time step.
        """

        X_results = np.full((sequence.shape[0], sequence.shape[-1], X_0.shape[1], X_0.shape[2]), np.nan)
        P_results = np.full((sequence.shape[0], sequence.shape[-1], P_0.shape[1], P_0.shape[2], P_0.shape[3]), np.nan)

        # Initial state
        X_results[:, 0, :, :] = X_0
        P_results[:, 0, :, :, :] = P_0
        avg_N_components = X_results.shape[2]
        for i, batch in enumerate(sequence):
            self.X = X_0[i]
            self.P = P_0[i]

            image = batch[0] 
            self.check_binary_image_background(image) # ! ensure the image background is 0 and foreground is 1
            tresh = 0.5
            image_stepped = image[:, ::step]

            for k in range(1, image_stepped.shape[1]):
                col = image_stepped[:, k] # find the non 0 value pixels, with a given treshold

                measurements = (np.where(col > tresh)[0]).astype(np.float64)
                centroids, stds = self.cluster_and_compute_stats(measurements, spacing=1)
                M = len(centroids)
                if M == 0:
                    # Predict
                    X_w, P_w = self.predict()
                    # Save the estimated position
                    X_results[i, k, :, :] = X_w
                    P_results[i, k, :, :, :] = P_w
                    continue
                else:
                    # Predict
                    X_w, P_w = self.predict()  
                    cost_matrix = self.compute_cost_matrix(X_w[:, 0], centroids)

                    # Handle cases where M != N by padding the cost matrix
                    max_dim = max(avg_N_components, M)
                    padded_cost_matrix = np.full((max_dim, max_dim), np.max(cost_matrix) + 999999999)  # Large penalty for unassigned
                    padded_cost_matrix[:avg_N_components, :M] = cost_matrix
                                
                    row_ind, col_ind = linear_sum_assignment(padded_cost_matrix)
                                
                    # Update
                    X_weighted = np.copy(X_w)
                    P_weighted = np.copy(P_w)
                    for l, j in zip(row_ind, col_ind):
                        if l < avg_N_components and j < M:  # Ignore padded assignments
                            # if True:
                            if padded_cost_matrix[l, j] < 50: # Test with and without this condition
                                X_weighted[l], P_weighted[l] = self.update(X_weighted[l], P_weighted[l], centroids[j])
                    self.X = np.copy(X_weighted)
                    self.P = np.copy(P_weighted)
                    
                # Save
                X_results[i, k, :, :] = X_weighted
                P_results[i, k, :, :, :] = P_weighted
        return X_results[:, :k+1], P_results[:, :k+1]
    
    @staticmethod
    def check_binary_image_background(image):
        """
        Check that image has min=0 and max=1, and that most pixels are background (intensity < 0.5).

        Raises:
            ValueError if conditions are not met.
        """
        if image.min() != 0 or image.max() != 1:
            raise ValueError(f"Image must have min=0 and max=1. Found min={image.min()}, max={image.max()}.")

        num_background = np.sum(image < 0.5)
        num_foreground = np.sum(image > 0.5)

        if num_background <= num_foreground:
            raise ValueError(f"Image must have more background (pixels < 0.5) than foreground (pixels > 0.5). "
                            f"Found {num_background} background vs {num_foreground} foreground pixels.")   

    @staticmethod
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

    @staticmethod
    # Compute Cost Matrix (Euclidean Distance)
    def compute_cost_matrix(predicted_positions, measurements):
        cost_matrix = np.abs(predicted_positions[:, None] - measurements[None, :])
        return cost_matrix