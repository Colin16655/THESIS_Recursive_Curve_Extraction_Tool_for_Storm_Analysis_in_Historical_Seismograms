import numpy as np
from scipy.optimize import linear_sum_assignment

class HungarianKalmanFilter:
    def __init__(self, A, H, Q, R):
        self.A = A  # State transition matrix
        self.H = H  # Observation matrix
        self.Q = Q  # Process noise covariance
        self.R = R  # Measurement noise covariance
    
    def predict(self):
        X, P, A, Q = self.X, self.P, self.A, self.Q
        # check if X has 1 or 2 dimension(s)
        if len(X.shape) == 1:
            self.X = A @ X
            self.P = A @ P @ A.T + Q
        else:
            # multiple states in parallel
            for o in range(X.shape[0]):
                self.X[o] = A @ X[o]
                self.P[o] = A @ P[o] @ A.T + Q
            # X = (A @ X.T).T  # Batch multiply all rows of X at once
            # P = A @ P @ A.T + Q  # Assuming P is (N, M, M), use batch multiplication
        return self.X, self.P
    
    def update(self, X, P, Z):
        H, R = self.H, self.R
        if len(X.shape) == 1:
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            X = X + K @ (Z - H @ X)
            P = P - K @ H @ P
        else:
            for o in range(X.shape[0]):
                S = H @ P[o] @ H.T + R
                K = P[o] @ H.T @ np.linalg.inv(S)
                X[o] = X[o] + K @ (Z[o] - H @ X[o])
                P[o] = P[o] - K @ H @ P[o]
        # print('end', self.X.shape)
        return X, P

    def process_sequence(self, sequence, X_0, P_0, step=5):
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

            image = batch[0].max() - batch[0]
            tresh = 0.1
            image_stepped = image[:, ::step]

            for k in range(1, image_stepped.shape[1]):
                # print(("a", self.X.shape))
                col = image_stepped[:, k]# find the non 0 value pixels, with a given treshold
                measurements = (np.where(col > tresh)[0]).astype(np.float64)
                centroids, stds = self.cluster_and_compute_stats(measurements, spacing=1)
                # centroids = measurements # !!!!
                # print("a", measurements)
                # print("b", centroids)
                M = len(centroids)
                if M == 0:
                    # Predict
                    X_w, P_w = self.predict()
                    # Save the estimated position
                    X_results[i, k, :, :] = X_w
                    P_results[i, k, :, :, :] = P_w
                    continue

                # Predict
                X_w, P_w = self.predict()  
                # print("z", X_w.shape, self.X.shape)   
                cost_matrix = self.compute_cost_matrix(X_w[:, 0], centroids)

                # Handle cases where M != N by padding the cost matrix
                max_dim = max(avg_N_components, M)
                padded_cost_matrix = np.full((max_dim, max_dim), np.max(cost_matrix) + 999999999)  # Large penalty for unassigned
                padded_cost_matrix[:avg_N_components, :M] = cost_matrix
                            
                row_ind, col_ind = linear_sum_assignment(padded_cost_matrix)
                            
                # Update
                X_weighted = np.copy(X_w)
                P_weighted = np.copy(P_w)
                # print("-----------------------")
                # print(measurements)
                for l, j in zip(row_ind, col_ind):
                    if l < avg_N_components and j < M:  # Ignore padded assignments
                        # print("l", l, centroids[j])
                        if padded_cost_matrix[l, j] < 50:
                            X_weighted[l], P_weighted[l] = self.update(X_weighted[l], P_weighted[l], centroids[j])
                self.X = np.copy(X_weighted)
                self.P = np.copy(P_weighted)
                # Save
                X_results[i, k, :, :] = X_weighted
                P_results[i, k, :, :, :] = P_weighted
        return X_results[:, :k+1], P_results[:, :k+1]
    
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