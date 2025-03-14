import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import norm

# ---------------------------
# 1. Simulate a Basic Dynamical System
# ---------------------------
def simulate_dynamics(T=50, dt=1.0, process_noise_std=0.1, measurement_noise_std=1.0):
    A = np.array([[1, dt], [0, 1]])
    H = np.array([[1, 0]])  # Only observe position

    x = np.zeros((T, 2))
    y = np.zeros((T, 1))

    # Initial state
    x[0] = [0, 1]  # Start at position 0 with velocity 1

    for t in range(1, T):
        process_noise = np.random.randn(2) * process_noise_std
        x[t] = A.dot(x[t-1]) + process_noise

    for t in range(T):
        measurement_noise = np.random.randn(1) * measurement_noise_std
        y[t] = H.dot(x[t]) + measurement_noise

    return x, y, A, H

T = 50
true_states, measurements, A, H = simulate_dynamics(T=T)

# ---------------------------
# 2. Classical Kalman Filter Implementation
# ---------------------------
def kalman_filter(measurements, A, H, Q, R, init_x, init_P):
    T = len(measurements)
    n = init_x.shape[0]
    x_estimates = np.zeros((T, n))
    P_estimates = np.zeros((T, n, n))  # Store covariance matrices
    x_est = init_x
    P = init_P

    for t in range(T):
        # Prediction step
        x_pred = A.dot(x_est)
        P_pred = A.dot(P).dot(A.T) + Q

        # Update step
        S = H.dot(P_pred).dot(H.T) + R
        K = P_pred.dot(H.T).dot(np.linalg.inv(S))
        y_t = measurements[t]
        x_est = x_pred + K.dot(y_t - H.dot(x_pred))
        P = (np.eye(n) - K.dot(H)).dot(P_pred)

        x_estimates[t] = x_est.reshape(-1)
        P_estimates[t] = P  # Store uncertainty for plotting

    return x_estimates, P_estimates

# Define noise covariances
Q = np.array([[0.01, 0],
              [0, 0.01]])
R = np.array([[1.0]])
init_x = np.array([[0.0],
                   [1.0]])
init_P = np.eye(2)

kf_estimates, P_estimates = kalman_filter(measurements, A, H, Q, R, init_x, init_P)

# ---------------------------
# 3. Plotting with Horizontal Time Axis, Gaussian Uncertainty, and Velocity Arrows
# ---------------------------
plt.figure(figsize=(12, 6))

# Plot true position, measurements, and Kalman estimate
plt.plot(range(T), true_states[:, 0], label="True Position", linewidth=2, color='black')
plt.scatter(range(T), measurements[:, 0], label="Noisy Measurements", alpha=0.4, color='red')
plt.plot(range(T), kf_estimates[:, 0], label="Kalman Filter Estimate", linestyle='--', color='blue')

# Select evenly spaced time steps for Gaussian visualization & velocity arrows
indices = np.linspace(0, T-1, num=10, dtype=int)

for i in indices:
    mu = kf_estimates[i, 0]  # Estimated position
    sigma = np.sqrt(P_estimates[i, 0, 0])  # Standard deviation in position
    x_vals = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    y_vals = norm.pdf(x_vals, mu, sigma)  # Gaussian distribution
    y_vals = y_vals / max(y_vals) * 2  # Scale for visibility

    # Draw Gaussian as a shaded region (vertical for position uncertainty)
    plt.fill_betweenx(x_vals, i, i + y_vals, alpha=0.2, color='blue')

    # Draw velocity vector tangent to trajectory (pointing in x direction)
    velocity = kf_estimates[i, 1]  # Estimated velocity
    plt.arrow(i, mu, 0.0, velocity*7, head_width=0.3, head_length=0.5, fc='green', ec='green')

plt.xlabel("Time Step")
plt.ylabel("Position")
plt.legend()
plt.title("Object Tracking: Position Estimation with Uncertainty (Gaussians) and Velocity (Arrows)")
plt.show()
