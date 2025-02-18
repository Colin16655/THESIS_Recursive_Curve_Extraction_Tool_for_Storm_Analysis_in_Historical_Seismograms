import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# === Initialization ===
np.random.seed(42)

# Number of traces (targets)
N = 3  
T = 50  # Number of time steps

# State transition matrix (constant velocity model in 2D)
dt = 1  # Time step
F = np.array([[1, dt, 0,  0],
              [0,  1, 0,  0],
              [0,  0, 1, dt],
              [0,  0, 0,  1]])

# Observation matrix (we observe only positions, not velocities)
H = np.array([[1, 0, 0, 0],
              [0, 0, 1, 0]])

# Process and measurement noise covariances
Q = np.eye(4) * 0.1  # Process noise
R = np.eye(2) * 2.0  # Measurement noise

# Initial state: [x, vx, y, vy] (starting at different locations)
X_true = np.array([[0, 1, 0, 1], 
                   [5, 0.5, 10, -0.5], 
                   [10, -1, 20, -1]])  

# Initial state covariance
P_init = np.eye(4) * 10

# Store true states and noisy measurements
true_states = []
measurements = []

for t in range(T):
    # Update true states using motion model + noise
    X_true = X_true @ F.T + np.random.multivariate_normal([0,0,0,0], Q, N)
    true_states.append(X_true[:, [0, 2]])  # Store true positions
    
    # Generate noisy measurements
    noisy_measurements = X_true[:, [0, 2]] + np.random.multivariate_normal([0, 0], R, N)
    measurements.append(noisy_measurements)

true_states = np.array(true_states)
measurements = np.array(measurements)

# === JPDA Kalman Filter Implementation ===
class JPDAKalmanFilter:
    def __init__(self, N, F, H, Q, R, P_init):
        self.N = N  # Number of traces
        self.F = F  # State transition matrix
        self.H = H  # Observation matrix
        self.Q = Q  # Process noise covariance
        self.R = R  # Measurement noise covariance
        self.x = np.zeros((N, 4, 1))  # State estimates
        self.P = np.array([P_init] * N)  # Covariances

    def initialize(self, init_states):
        self.x = np.expand_dims(init_states, axis=2)

    def predict(self):
        for i in range(self.N):
            self.x[i] = self.F @ self.x[i]
            self.P[i] = self.F @ self.P[i] @ self.F.T + self.Q

    def update(self, measurements):
        likelihoods = np.zeros((self.N, self.N))
        betas = np.zeros((self.N, self.N))

        # Compute measurement likelihoods
        for i in range(self.N):
            z_pred = self.H @ self.x[i]  # Predicted measurement
            S = self.H @ self.P[i] @ self.H.T + self.R  # Innovation covariance
            for j in range(self.N):
                residual = measurements[j] - z_pred.flatten()
                likelihoods[i, j] = multivariate_normal.pdf(residual, mean=np.zeros(S.shape[0]), cov=S)

        # Normalize association probabilities
        beta_sum = likelihoods.sum(axis=1, keepdims=True) + 1e-6
        betas = likelihoods / beta_sum

        # Update each trace using weighted measurements
        for i in range(self.N):
            K = self.P[i] @ self.H.T @ np.linalg.inv(self.H @ self.P[i] @ self.H.T + self.R)
            innovation = (measurements - self.H @ self.x[i]).T  # Ensure (2, 1)
            self.x[i] += K @ np.sum(betas[i, :, None] * innovation, axis=0, keepdims=True)
            self.P[i] = (np.eye(self.P[i].shape[0]) - K @ self.H) @ self.P[i]

# === Run JPDA Kalman Filter ===
tracker = JPDAKalmanFilter(N, F, H, Q, R, P_init)
tracker.initialize(X_true)

tracked_states = []

for t in range(T):
    tracker.predict()
    tracker.update(measurements[t])
    tracked_states.append(tracker.x[:, [0, 2], 0])  # Store estimated positions

tracked_states = np.array(tracked_states)

# === Visualization ===
plt.figure(figsize=(10, 6))

colors = ['r', 'g', 'b']
for i in range(N):
    plt.plot(true_states[:, i, 0], true_states[:, i, 1], f'{colors[i]}-', label=f'True Trace {i+1}')
    plt.plot(tracked_states[:, i, 0], tracked_states[:, i, 1], f'{colors[i]}--', label=f'JPDA Estimate {i+1}')
    plt.scatter(measurements[:, i, 0], measurements[:, i, 1], color=colors[i], alpha=0.4, marker='o')

plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('JPDA Kalman Filter Tracking Multiple Traces')
plt.legend()
plt.grid()
plt.show()
