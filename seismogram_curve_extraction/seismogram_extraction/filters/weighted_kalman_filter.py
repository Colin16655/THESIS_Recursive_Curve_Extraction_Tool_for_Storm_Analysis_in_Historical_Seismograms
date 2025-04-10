import numpy as np

class WeightedKalmanFilter:
    def __init__(self, A, H, Q, R, p_fa=1e-2):
        self.A = A  # State transition matrix
        self.H = H  # Observation matrix
        self.Q = Q  # Process noise covariance
        self.R = R  # Measurement noise covariance
        self.p_fa = p_fa  # False alarm probability
    
    def predict(self):
        X, P, A, Q = self.X, self.P, self.A, self.Q
        # check if X has 1 or 2 dimension(s)
        if len(X.shape) == 1:
            self.X = A @ X
            self.P = A @ P @ A.T + Q
        else:
            # multiple states in parallel
            for i in range(X.shape[0]):
                self.X[i] = A @ X[i]
                self.P[i] = A @ P[i] @ A.T + Q
            # X = (A @ X.T).T  # Batch multiply all rows of X at once
            # P = A @ P @ A.T + Q  # Assuming P is (N, M, M), use batch multiplication
        return self.X, self.P
    
    def weighted_update(self, Z):
        """
        Perform the Kalman filter update step using the JPDA approach for multiple traces.

        Args:
            X (np.ndarray): Shape (N, n) - Predicted state estimates for N traces.
            P (np.ndarray): Shape (N, n, n) - Predicted covariance matrices for N traces.
            Z (np.ndarray): Shape (M, m) - Measurements for M detections, m=1.
            H (np.ndarray): Shape (m, n) - Observation matrix.
            R (np.ndarray): Shape (m, m) - Measurement noise covariance, m=1.

        Returns:
            X_updated (np.ndarray): Shape (N, n) - Updated state estimates.
            P_updated (np.ndarray): Shape (N, n, n) - Updated covariance matrices.
        """
        if len(Z) == 0:
            return self.X, self.P, np.zeros((self.X.shape[0], 0))
        X, P, H, R, P_fa = self.X, self.P, self.H, self.R, self.p_fa
        N, n = X.shape  # Number of traces, state dimension
        M = Z.shape[0]      # Number of measurements
        # m = Z.shape[1]      # Measurement dimension
        m = 1
        
        X_updated = np.copy(X)
        P_updated = np.copy(P)
        
        # Compute innovation covariance and Kalman gains
        S = np.zeros((N, m, m))  # Innovation covariance for each trace
        K = np.zeros((N, n, m))  # Kalman gain for each trace
        for i in range(N):
            S[i] = H @ P[i] @ H.T + R               # Innovation covariance
            K[i] = P[i] @ H.T @ np.linalg.inv(S[i]) # Kalman gain

        # Compute association probabilities β_ij using Mahalanobis distance
        beta = np.zeros((N, M))
        likelihoods = np.zeros((N, M))
        for i in range(N):
            for j in range(M):
                residual = Z[j] - H @ X[i]  # Innovation (measurement residual)
                mahalanobis_dist = residual.T @ np.linalg.inv(S[i]) @ residual  # Mahalanobis distance
                likelihoods[i, j] = np.exp(-0.5 * mahalanobis_dist) / np.sqrt((2*np.pi)**m*np.linalg.det(S[i]))

        # Normalize probabilities
        for i in range(N):
            beta[i, :] = likelihoods[i, :] / (np.sum(likelihoods[i, :]) + P_fa)  # Avoid division by zero

        # Update states using weighted innovation
        for i in range(N):
            weighted_innovation = np.zeros((m, 1))
            for j in range(M):
                weighted_innovation += beta[i, j] * (Z[j] - H @ X[i])
            # Update state and covariance
            beta_i = np.sum(beta[i, :])
            X_updated[i] = X[i] + (K[i] @ weighted_innovation).ravel() # ! Check if this is correct
            temp = (np.eye(n) - beta_i * K[i] @ H)
            P_updated[i] = temp @ P[i] @ temp.T + K[i] @ (np.sum(beta[i, :]**2) * R) @ K[i].T

        self.X = X_updated
        self.P = P_updated

        return X_updated, P_updated, beta

    def process_sequence(self, sequence, X_0, P_0, step=None):
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
        
        for i, batch in enumerate(sequence):
            self.X = X_0[i]
            self.P = P_0[i]

            image = batch[0].max() - batch[0]
            tresh = 0.1

            for k in range(1, image.shape[1]):
                col = image[:, k]
                # find the non 0 value pixels, with a given treshold
                measurements = (np.where(col > tresh)[0]).astype(np.float64)

                # Predict
                X_w, P_w = self.predict()
                # Weighted Update
                X_weighted, P_weighted, _ = self.weighted_update(measurements)

                # Save
                X_results[i, k, :, :] = X_weighted
                P_results[i, k, :, :, :] = P_weighted

        return X_results, P_results
