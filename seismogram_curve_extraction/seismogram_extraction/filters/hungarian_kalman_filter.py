import numpy as np
from scipy.optimize import linear_sum_assignment

class HungarianKalmanFilter:
    def __init__(self, A, H, Q, max_gap=20):
        self.A = A  # State transition matrix
        self.H = H  # Observation matrix
        self.Q = Q  # Process noise covariance
        self.max_gap = max_gap
    
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
    
    def update(self, X, P, Z, R):
        H = self.H
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
        return X, P

    def process_sequence(self, sequence, X_0, P_0):
        """
        Processes a sequence of inputs as if using an RNN. This function mimics an RNN behavior,
        where each time step input updates the state.

        Note that max_gap (int) is the maximum gap between predicted and measured positions to consider them as the same object.

        Args:
            sequence (np.ndarray): Shape (batch_size, 1, height, width) - Sequence of batch_size-dimensional inputs.
            X_0 (np.ndarray): Shape (batch_size, N_traces, N_states) - Initial state estimates for N_states traces.
            P_0 (np.ndarray): Shape (batch_size, N_traces, N_states, N_states) - Initial covariance matrices for N_states traces.

        Returns:
            X_results (np.ndarray): Shape (batch_size, width, N_traces, N_states) - Updated state estimates for each time step (width).
            P_results (np.ndarray): Shape (batch_size, width, N_traces, N_states, N_states) - Updated covariance matrices for each time step.
        """
        # invert all images
        sequence = sequence.max() - sequence

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
            tresh = 0.1

            for k in range(1, image.shape[1]):
                col = image[:, k]# find the non 0 value pixels, with a given treshold
                measurements_temp = (np.where(col > tresh)[0]).astype(np.float64)
                measurements, measurements_std = self.compute_measurements(measurements_temp)
                M = len(measurements)
                measurements = np.array(measurements)
                if M == 0:
                    print("boo")
                    # Predict
                    self.predict()
                    # Save the estimated position
                    X_results[i, k, :, :] = self.X
                    P_results[i, k, :, :, :] = self.P
                    continue

                # Predict
                # if M < avg_N_components:
                #     print("M < avg_N_components")
                #     print("X before predict", self.X)
                #     # print("P before predict", self.P)
                #     print("measurements", measurements)
                self.predict()  
                # if M < avg_N_components:
                #     print("X after predict", self.X)
                #     # print("P after predict", self.P)
                cost_matrix = self.compute_cost_matrix(self.X[:, 0], measurements)
                            
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                # if M < avg_N_components:
                #     print("M < avg_N_components")
                #     print("cost_matrix", cost_matrix)
                #     print("row_ind", row_ind)
                    
                            
                # Update
                # if M < avg_N_components:
                #     print("X before update", self.X)
                #     # print("P before update", self.P)
                #     print("measurements", measurements)
                X_temp = np.copy(self.X)
                P_temp = np.copy(self.P)
                for l, j in zip(row_ind, col_ind):
                    if l < avg_N_components and j < M:  # Ignore padded assignments
                        if np.abs(self.X[l, 0] - measurements[j]) > self.max_gap: 
                            mmmm = 0
                            # print('baaa k', k)
                            # print("X", self.X)
                            # print("measurements", measurements)
                            # print("measurements_temp", measurements_temp)
                            # print("cost_matrix", cost_matrix)
                            # print("row_ind", row_ind)
                            # print("col_ind", col_ind)
                        else:
                            R = np.array([[measurements_std[j] ** 2]]) # Update R with the measurement noise
                            X_temp[l], P_temp[l] = self.update(X_temp[l], P_temp[l], measurements[j], R)
                    else: print("biii")
                self.X = np.copy(X_temp)
                self.P = np.copy(P_temp)
                # Save
                X_results[i, k, :, :] = X_temp
                P_results[i, k, :, :, :] = P_temp

                # if M < avg_N_components:
                #     print("X after update", self.X)
                #     # print("P after update", self.P)

        return X_results, P_results

    @staticmethod
    # Compute Cost Matrix (Euclidean Distance)
    def compute_cost_matrix(predicted_positions, measurements):
        cost_matrix = np.abs(predicted_positions[:, None] - measurements[None, :])
        return cost_matrix

    @staticmethod
    def find_contiguous_subsets(positions, n=2):
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

    @staticmethod
    def compute_measurements(positions):
        subsets = HungarianKalmanFilter.find_contiguous_subsets(positions)
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

# Example usage
# positions = np.array([1, 2, 3, 7, 8, 9, 10, 15, 16, 20])
# measurements, R = HungarianKalmanFilter.compute_measurements(positions)

# print("Measurements (Mean Positions):", measurements)
# print("Covariance Matrix R:")
# print(R)
