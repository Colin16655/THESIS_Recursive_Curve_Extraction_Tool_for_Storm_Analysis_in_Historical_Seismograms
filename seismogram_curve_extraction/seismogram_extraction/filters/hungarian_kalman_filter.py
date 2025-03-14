import numpy as np

class HungarianKalmanFilter:
    def __init__(self, A, H, Q, R, P, p_fa=1e-2):
        self.A = A  # State transition matrix
        self.H = H  # Observation matrix
        self.Q = Q  # Process noise covariance
        self.R = R  # Measurement noise covariance
        self.P = P  # Initial covariance
        self.p_fa = p_fa  # False alarm probability
    
    def predict(self, X):
        X = self.A @ X
        self.P = self.A @ self.P @ self.A.T + self.Q
        return X
    
    def weighted_update(self, X, Z):
        N, n = X.shape  # Number of traces, state dimension
        M = Z.shape[0]  # Number of measurements
        m = 1

        X_updated = np.copy(X)
        P_updated = np.copy(self.P)

        S = np.zeros((N, m, m))
        K = np.zeros((N, n, m))
        for i in range(N):
            S[i] = self.H @ self.P @ self.H.T + self.R
            K[i] = self.P @ self.H.T @ np.linalg.inv(S[i])
        
        beta = np.zeros((N, M))
        likelihoods = np.zeros((N, M))
        for i in range(N):
            for j in range(M):
                residual = Z[j] - self.H @ X[i]
                mahalanobis_dist = residual.T @ np.linalg.inv(S[i]) @ residual
                likelihoods[i, j] = np.exp(-0.5 * mahalanobis_dist) / np.sqrt((2*np.pi)**m * np.linalg.det(S[i]))
        
        for i in range(N):
            beta[i, :] = likelihoods[i, :] / (np.sum(likelihoods[i, :]) + self.p_fa)
        
        for i in range(N):
            weighted_innovation = np.zeros((m, 1))
            for j in range(M):
                weighted_innovation += beta[i, j] * (Z[j] - self.H @ X[i])
            
            beta_i = np.sum(beta[i, :])
            X_updated[i] = X[i] + (K[i] @ weighted_innovation).ravel()
            temp = (np.eye(n) - beta_i * K[i] @ self.H)
            P_updated[i] = temp @ self.P @ temp.T + K[i] @ (np.sum(beta[i, :]**2) * self.R) @ K[i].T

        self.P = P_updated
        return X_updated

    def process_sequence(self, sequence):
        """
        Processes a sequence of inputs as if using an RNN. This function mimics an RNN behavior,
        where each time step input updates the state.
        """
        results = []
        X_state = np.zeros((2, 2))  # Initial state
        for Z in sequence:
            X_pred = self.predict(X_state)
            X_state = self.weighted_update(X_pred, Z)
            results.append(X_state.copy())
        return np.array(results)
