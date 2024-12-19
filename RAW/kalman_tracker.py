import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
import matplotlib.pyplot as plt
from seismogram_generator import SeismogramGenerator

@dataclass
class KalmanParameters:
    """Parameters for the Kalman filter."""
    dt: float  # Time step
    measurement_variance: float  # R matrix parameter
    process_variance: float  # Q matrix base parameter
    initial_position_variance: float
    initial_velocity_variance: float
    initial_acceleration_variance: float

class SeismicTraceKalmanFilter:
    def __init__(self, params: KalmanParameters):
        """
        Initialize Kalman Filter for tracking seismic trace.
        State vector: [position, velocity, acceleration]
        """
        self.dt = params.dt
        
        # State transition matrix (F)
        self.F = np.array([
            [1, self.dt, 0.5 * self.dt**2],
            [0, 1, self.dt],
            [0, 0, 1]
        ])
        
        # Measurement matrix (H)
        # We only measure position
        self.H = np.array([[1, 0, 0]])
        
        # Measurement noise covariance (R)
        self.R = np.array([[params.measurement_variance]])
        
        # Process noise covariance (Q)
        # Using continuous white noise acceleration model
        q = params.process_variance
        self.Q = q * np.array([
            [self.dt**4/4, self.dt**3/2, self.dt**2/2],
            [self.dt**3/2, self.dt**2, self.dt],
            [self.dt**2/2, self.dt, 1]
        ])
        
        # Initial state covariance (P)
        self.P = np.diag([
            params.initial_position_variance,
            params.initial_velocity_variance,
            params.initial_acceleration_variance
        ])
        
        # Initial state
        self.x = np.zeros(3)
        
    def predict(self) -> np.ndarray:
        """Predict next state."""
        # State prediction
        self.x = self.F @ self.x
        
        # Covariance prediction
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return self.x
    
    def update(self, measurement: float) -> np.ndarray:
        """Update state based on measurement."""
        # Innovation / measurement residual
        y = measurement - self.H @ self.x
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # State update
        self.x = self.x + K @ y
        
        # Covariance update
        I = np.eye(3)
        self.P = (I - K @ self.H) @ self.P
        
        return self.x
    
    def get_state(self) -> np.ndarray:
        """Return current state estimate."""
        return self.x.copy()

def extract_measurements_from_image(image: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Extract measurements from image column by column.
    Returns measurements and estimated measurement variance.
    """
    height, width = image.shape
    measurements = np.zeros(width)
    
    for x in range(width):
        # Find the black pixels in this column
        black_pixels = np.where(image[:, x] < 128)[0]
        if len(black_pixels) > 0:
            # Take the average position of black pixels
            measurements[x] = np.mean(black_pixels)
        else:
            # If no black pixels found, use the last known position
            measurements[x] = measurements[x-1] if x > 0 else height/2
    
    # Estimate measurement variance from the data
    # Using the variance of differences between consecutive measurements
    measurement_variance = np.var(np.diff(measurements))
    
    return measurements, measurement_variance

def reconstruct_trace(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct the seismic trace from the image using Kalman filter.
    Returns the reconstructed positions and velocities.
    """
    # Extract measurements from image
    measurements, meas_variance = extract_measurements_from_image(image)
    
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
    # Set the random seed once at the start
    np.random.seed(42)

    # Create a sample seismogram
    generator = SeismogramGenerator(
        width=800,
        height=400,
        trace_thickness=1,
        num_traces=2,
        amplitude_factor=1.0
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