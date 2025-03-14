import numpy as np

class WeightedKalmanFilter:
    def __init__(self, A, H, Q, R, P, p_fa=1e-2, initial_state=None):
        self.A = A  # State transition matrix
        self.H = H  # Observation matrix
        self.Q = Q  # Process noise covariance
        self.R = R  # Measurement noise covariance
        self.P = P  # Initial covariance
        self.p_fa = p_fa  # False alarm probability
        self.X = initial_state if initial_state is not None else np.zeros((A.shape[0], 1))  # Initial state
    
    def predict(self, X):
        P, A, Q = self.P, self.A, self.Q
        # check if X has 1 or 2 dimension(s)
        if len(X.shape) == 1:
            X = A @ X
            self.P = A @ P @ A.T + Q
        else:
            # multiple states in parallel
            for i in range(X.shape[0]):
                X[i] = A @ X[i]
                self.P[i] = A @ P[i] @ A.T + Q
            # X = (A @ X.T).T  # Batch multiply all rows of X at once
            # P = A @ P @ A.T + Q  # Assuming P is (N, M, M), use batch multiplication
        return X, P
    
    def weighted_update(self, X, Z):
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
        P, H, R, P_fa = self.P, self.H, self.R, self.p_fa
        # print("X : ", type(X[0,0]))
        # print("P : ", type(P[0,0,0]))
        # print("Z : ", type(Z[0]))
        # print("H : ", type(H[0,0]))
        # print("R : ", type(R[0,0]))
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
            # print(i, S[i], K[i])
            K[i] = P[i] @ H.T @ np.linalg.inv(S[i]) # Kalman gain

        # Compute association probabilities β_ij using Mahalanobis distance
        beta = np.zeros((N, M))
        likelihoods = np.zeros((N, M))
        for i in range(N):
            for j in range(M):
                residual = Z[j] - H @ X[i]  # Innovation (measurement residual)
                # print('res',i, j, residual)
                mahalanobis_dist = residual.T @ np.linalg.inv(S[i]) @ residual  # Mahalanobis distance
                # print(i, j, mahalanobis_dist)
                likelihoods[i, j] = np.exp(-0.5 * mahalanobis_dist) / np.sqrt((2*np.pi)**m*np.linalg.det(S[i]))
                # print(i, j, likelihoods[i, j],"-",  np.exp(-0.5 * mahalanobis_dist),"-", -0.5*mahalanobis_dist )

        # Normalize probabilities
        # print("lik", likelihoods)
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

        self.P = P_updated

        return X_updated, P_updated, beta

    def process_sequence(self, sequence):
        """
        Processes a sequence of inputs as if using an RNN. This function mimics an RNN behavior,
        where each time step input updates the state.
        """
        X_results = []
        P_results = []
        X_state = self.X  # Initial state
        
        X_results.append(X_state.copy)
        P_results.append(self.P.copy())

        for Z in sequence:
            X_pred, P_pred = self.predict(X_state)
            X_state, P, _ = self.weighted_update(X_pred, Z)
            X_results.append(X_state.copy())
            P_results.append(P.copy())
        return np.array(X_results), np.array(P_results)
