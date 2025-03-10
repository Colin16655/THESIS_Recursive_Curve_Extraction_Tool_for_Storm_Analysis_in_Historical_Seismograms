import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

# Simulated Kalman Filter Predictions (Positions Only)
def predict_positions(states):
    return np.array([state[0] for state in states])  # Extract positions

# Simulated True Trajectories from Sine Waves
def generate_true_trajectories(time_steps, num_objects):
    return np.array([np.sin(0.5 * time_steps*i/3) * (i + 1) + i*6 for i in range(num_objects)])

# Simulated Measurements from True Trajectories
def generate_measurements_sine(true_positions, num_measurements, noise_std=0.5):
    replace = num_measurements > len(true_positions)  # Handle cases where M > N
    if replace:
        indices_0 = np.random.choice(len(true_positions), len(true_positions), replace=False)
        indices_1 = np.random.choice(len(true_positions), num_measurements - len(true_positions), replace=True)
        indices = np.concatenate([indices_0, indices_1])
    else:
        indices = np.random.choice(len(true_positions), num_measurements, replace=replace)  # Select M random measurements
    return true_positions[indices] + np.random.normal(0, noise_std, size=(num_measurements,))


# Compute Cost Matrix (Euclidean Distance)
def compute_cost_matrix(predicted_positions, measurements):
    cost_matrix = np.abs(predicted_positions[:, None] - measurements[None, :])
    return cost_matrix

# Kalman Filter State Update (Basic)
def kalman_update(state, measurement, R=0.5):
    p, v = state
    K = 1 / (1 + R)  # Simplified Kalman Gain
    p_new = p + K * (measurement - p)
    return np.array([p_new, v])

# Number of Objects (Tracks) and Measurements
N = 5  # Number of objects
M = 6  # Can be greater or smaller than N
T = 30  # Increased number of time steps

time_steps = np.arange(T)
true_trajectories = generate_true_trajectories(time_steps, N)

# Initialize States (Position, Velocity)
states = np.ones((N, 2))  # Initial states
states[:, 0] = true_trajectories[:, 0]  # Initial positions

time_series = []
pred_series = []
meas_series = []
assignments_series = []

# Time Loop for Visualization
for t in range(T):
    predicted_positions = predict_positions(states)
    measurements = generate_measurements_sine(true_trajectories[:, t], M)
    print("X", predicted_positions.shape, "measurements", measurements.shape)
    cost_matrix = compute_cost_matrix(predicted_positions, measurements)
    
    # Handle cases where M != N by padding the cost matrix
    max_dim = max(N, M)
    padded_cost_matrix = np.full((max_dim, max_dim), np.max(cost_matrix) + 999999999)  # Large penalty for unassigned
    padded_cost_matrix[:N, :M] = cost_matrix
    
    row_ind, col_ind = linear_sum_assignment(padded_cost_matrix)
    
    for i, j in zip(row_ind, col_ind):
        if i < N and j < M:  # Ignore padded assignments
            states[i] = kalman_update(states[i], measurements[j])
    
    time_series.append(t)
    pred_series.append(predicted_positions.copy())
    meas_series.append(measurements.copy())
    assignments_series.append(list(zip(row_ind, col_ind)))

# Convert to numpy arrays for plotting
time_series = np.array(time_series)
pred_series = np.array(pred_series)
meas_series = np.array(meas_series, dtype=object)  # Variable-length measurements

# Plot Results
plt.figure(figsize=(10, 5))
for i in range(N):
    plt.plot(time_series, true_trajectories[i, :], '--', label=f"True Trajectory {i}")
    plt.plot(time_series, pred_series[:, i], 'o-', label=f"Predicted {i}")
for t in range(T):
    plt.scatter([time_series[t]]*len(meas_series[t]), meas_series[t], marker='x', color='red', label='Measurements' if t == 0 else "")
plt.xlabel("Time")
plt.ylabel("Position")
plt.title("Kalman Filter State Tracking with Hungarian Assignment (Handling M != N)")
plt.show()
