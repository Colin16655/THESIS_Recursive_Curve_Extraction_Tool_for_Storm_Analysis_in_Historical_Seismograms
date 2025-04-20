import numpy as np
from scipy.optimize import linear_sum_assignment

class HungarianExtendedKalmanFilter:
    def __init__(self, H, Q, R, dt, f_function, jacobian_function):
        """
        f_function: callable
            Nonlinear state transition function f(x, dt)
        jacobian_function: callable
            Function to compute the Jacobian A_k = df/dx evaluated at x and dt
        """
        self.H = H  # Measurement matrix
        self.Q = Q  # Process noise covariance
        self.R = R  # Measurement noise covariance
        self.dt = dt
        self.f = f_function
        self.compute_A = jacobian_function

    def predict(self, X, P):
        dt = self.dt
        if len(X.shape) == 1:
            A = self.compute_A(X, dt)
            X = self.f(X, dt)
            P = A @ P @ A.T + self.Q
            # Prevent overflows or invalid entries in covariance
            if not np.all(np.isfinite(P)):
                print("[WARN] Non-finite values in predicted covariance — resetting to identity.")
                P = np.eye(P.shape[0]) * 1e3
        else:
            for i in range(X.shape[0]):
                A = self.compute_A(X[i], dt)
                X[i] = self.f(X[i], dt)
                P[i] = A @ P[i] @ A.T + self.Q
                if not np.all(np.isfinite(P[i])):
                    print(f"[WARN] Covariance for trace {i} is invalid — resetting.")
                    P[i] = np.eye(P[i].shape[0]) * 1e3

        # Optionally clip entries to prevent explosion
        # P = np.clip(P_pred, -1e8, 1e8)
        return X, P
    
    def update(self, X, P, Z):
        H, R = self.H, self.R
        if len(X.shape) == 1:
            S = H @ P @ H.T + R
            # Avoid invalid S
            if not np.all(np.isfinite(S)) or np.linalg.cond(S) > 1e12:
                print("[WARN] Invalid S matrix — skipping update.")
                return X, P  # skip update, return prediction
            K = P @ H.T @ np.linalg.inv(S)
            X = X + K @ (Z - H @ X)
            P = P - K @ H @ P
        else:
            for o in range(X.shape[0]):
                S = H @ P[o] @ H.T + R
                # Avoid invalid S
                if not np.all(np.isfinite(S)) or np.linalg.cond(S) > 1e12:
                    print("[WARN] Invalid S matrix — skipping update.")
                    return X, P  # skip update, return prediction
                K = P[o] @ H.T @ np.linalg.inv(S)
                X[o] = X[o] + K @ (Z[o] - H @ X[o])
                P[o] = P[o] - K @ H @ P[o]

        # Clamp or clean result
        # P_new = np.clip(P_new, -1e8, 1e8)
        # P_new = np.nan_to_num(P_new, nan=1e6, posinf=1e6, neginf=1e6)
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
            X = X_0[i]
            P = P_0[i]

            image = batch[0]
            tresh = 0.5
            image_stepped = image[:, ::step]

            for k in range(1, image_stepped.shape[1]):
                col = image_stepped[:, k] # find the non 0 value pixels, with a given treshold
                measurements = (np.where(col > tresh)[0]).astype(np.float64)
                centroids, stds = self.cluster_and_compute_stats(measurements, spacing=1)
                M = len(centroids)
                
                # Predict
                X, P = self.predict(X, P)

                if len(centroids) > 0:
                    cost_matrix = self.compute_cost_matrix(X[:, 0], centroids)
                    # Validate cost matrix
                    if not np.all(np.isfinite(cost_matrix)):
                        raise ValueError("Cost matrix contains NaN or Inf. Aborting assignment.")
                    row_ind, col_ind = linear_sum_assignment(cost_matrix)
                                
                    # Update
                    for l, j in zip(row_ind, col_ind):
                        if l < avg_N_components and j < M:
                            # if padded_cost_matrix[l, j] < 50:
                            X[l], P[l] = self.update(X[l], P[l], centroids[j])
                # Save
                X_results[i, k, :, :] = X
                P_results[i, k, :, :, :] = P
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