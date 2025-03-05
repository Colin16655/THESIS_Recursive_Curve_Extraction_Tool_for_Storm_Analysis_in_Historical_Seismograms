import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
from itertools import product

@dataclass
class KalmanParameters:
    """Parameters for the Kalman filter."""
    dt: float  # Time step
    measurement_variance: np.ndarray  # R matrix parameter
    process_variance: np.ndarray  # Q matrix parameters
    initial_states: np.ndarray # Initial state vector
    initial_covariance: np.ndarray # Initial state covariance matrix

def get_state_transition_matrix(degree, Delta):
    """
    Generates the state transition matrix A for a given system degree.
    
    Parameters:
        degree (int): The number of state variables (e.g., 3 for acceleration, 5 for snap).
        Delta (float): Time step.
    
    Returns:
        np.array: State transition matrix of shape (degree, degree).
    """
    A = np.zeros((degree, degree))
    for i in range(degree):
        for j in range(i + 1):
            A[j,i] = (Delta ** (i - j)) / factorial(i - j)
    
    return A

class SeismicTraceKalmanFilter:
    def __init__(self, params: KalmanParameters, degree=2):
        """
        Initialize Kalman Filter for tracking seismic trace.
        State vector: [position, velocity, acceleration, ...]
        """
        self.dt = params.dt
        
        # State transition matrix (F)
        self.A = get_state_transition_matrix(degree, self.dt)
        
        # Measurement matrix (H)
        # We only measure position
        self.H = np.zeros((1, degree))
        self.H[0, 0] = 1
        
        # Measurement noise covariance (R)
        self.R = params.measurement_variance

        # Process noise covariance (Q)
        # Using continuous white noise acceleration model
        self.Q = params.process_variance
        
        # Initial state covariance (P)
        self.P = params.initial_covariance
        
        # Initial state
        self.x = params.initial_states

    def predict(self):
        """Predict next state."""
        # check if X has 1 or 2 dimension(s)
        if len(self.X.shape) == 1:
            self.X = self.A @ self.X
            self.P = self.A @ self.P @ self.A.T + self.Q
        else:
            # multiple states in parallel
            for i in range(self.X.shape[0]):
                self.X[i] = self.A @ self.X[i]
                self.P[i] = self.A @ self.P[i] @ self.A.T + self.Q
            # X = (A @ X.T).T  # Batch multiply all rows of X at once
            # P = A @ P @ A.T + Q  # Assuming P is (N, M, M), use batch multiplication
        return self.X, self.P
    
    def weighted_update(self, P_fa=1e-2):
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
        X, P, Z, H, R = self.X, self.P, self.Z, self.H, self.R
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


        return X_updated, P_updated, beta                   
    
    def get_state(self):
        """Return current state estimate."""
        return self.x.copy()

def reconstruct_trace(image, params):
    """
    Reconstruct the trace from the image using Kalman filter.
    Returns the reconstructed positions and velocities.

    args:
        image (np.ndarray): The input image.
        params (KalmanParameters): Parameters for the Kalman filter.
    """    
    # Initialize Kalman filter parameters
    params = KalmanParameters(
        dt=1.0,  # One pixel width is one time step
        measurement_variance=meas_variance,
        process_variance=0.1,  # Adjust based on expected acceleration changes
        initial_position_variance=100.0,
        initial_velocity_variance=10.0,
        initial_acceleration_variance=1.0
    )
    
    # Create Kalman filter
    kf = SeismicTraceKalmanFilter(params)
    
    # Arrays to store results
    num_steps = len(measurements)
    positions = np.zeros(num_steps)
    velocities = np.zeros(num_steps)
    accelerations = np.zeros(num_steps)
    
    # Process each measurement
    for t in range(num_steps):
        # Predict
        kf.predict()
        
        # Update with measurement
        state = kf.update(measurements[t])
        
        # Store results
        positions[t] = state[0]
        velocities[t] = state[1]
        accelerations[t] = state[2]
    
    return positions, velocities, accelerations

# Example usage and visualization
if __name__ == "__main__":
    ### Parameters
    Dts = [0.5] # np.linspace(0.5, 5, 2) # Time step related to the state transition matrix A, ! different than sampling rate dt of signal s

    degrees = [2] # np.arange(2, 5)  # Order of the model (e.g., 2 for constant velocity, 3 for constant acceleration, etc.)

    # Assuming no process noise
    sigma_ps = [0.01] # np.linspace(1e-2, 2, 2) 
    sigma_vs = [2] # np.linspace(1e-2, 2, 2)

    # Assuming no measurement noise
    sigma_zs = [0.25] # np.linspace(1e-6, 1, 5)

    p_fas = [0.0001] # np.linspace(1e-4, 1, 5)
    ###

    for Dt, sigma_p, sigma_v, sigma_z, p_fa in product(Dts, sigma_ps, sigma_vs, sigma_zs, p_fas):
        Q = np.zeros((degrees, degrees))
        Q[0, 0] = sigma_s
        Q[1, 1] = sigma_v

        params = KalmanParameters(
            dt = Dt,
            measurement_variance = np.array([[sigma_z]]),
            process_variance = Q,
            initial_states = np.array([0, 0]),
        process_variance: n  # Q matrix parameters
        initial_states: np.ndarray # Initial state vector
        initial_covariance: np.ndarray # Initial state covariance matrix
        )
    
    # Generate sample image
    image, ground_truths = generator.generate()
    
    # Reconstruct the trace
    positions, velocities, accelerations = reconstruct_trace(image)
    
    # Create visualization
    plt.figure(figsize=(15, 12))
    
    # Plot 1: Original image with reconstructed curve overlay
    plt.subplot(211)
    plt.imshow(image, cmap='gray', aspect='auto')
    plt.plot(positions, 'r-', linewidth=1.5, alpha=0.7, label='Reconstructed')
    plt.legend()
    plt.title('Original Seismogram with Reconstruction Overlay')
    
    # Plot 2: State estimates
    plt.subplot(234)
    plt.plot(velocities)
    plt.title('Estimated Velocity')
    plt.grid(True)
    
    plt.subplot(235)
    plt.plot(accelerations)
    plt.title('Estimated Acceleration')
    plt.grid(True)
    
    # Plot 3: Error analysis
    plt.subplot(236)
    # Extract original signal from ground truth for comparison
    original_signal = np.array(ground_truths[0]['signal'])
    # Scale and center the original signal to match the pixel coordinates
    original_signal = original_signal * (image.shape[0] * 0.4) + image.shape[0]/2
    
    # Calculate and plot error
    error = positions - original_signal
    plt.plot(error, 'g-', label='Error')
    plt.title('Reconstruction Error')
    plt.grid(True)
    plt.legend()
    
    # Add some statistics
    plt.figtext(0.02, 0.02, 
                f'RMS Error: {np.sqrt(np.mean(error**2)):.2f} pixels\n'
                f'Max Error: {np.max(np.abs(error)):.2f} pixels', 
                fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    # Print some statistics
    print("\nReconstruction Statistics:")
    print(f"RMS Error: {np.sqrt(np.mean(error**2)):.2f} pixels")
    print(f"Max Absolute Error: {np.max(np.abs(error)):.2f} pixels")
    print(f"Mean Absolute Error: {np.mean(np.abs(error)):.2f} pixels")
    print(f"Standard Deviation of Error: {np.std(error):.2f} pixels")