from typing import List, Tuple, Dict
import numpy as np
import matplotlib.pyplot as plt
from seismogram_generator import SeismogramGenerator
from kalman_tracker import KalmanParameters, SeismicTraceKalmanFilter

class SeismicAnalyzer:
    def __init__(
        self,
        width: int = 800,
        height: int = 1200,
        num_traces: int = 4,
        trace_spacing: int = 200,
        amplitude_factor: float = 1.2,
    ):
        """Initialize SeismicAnalyzer with configuration parameters."""
        self.width = width
        self.height = height
        self.num_traces = num_traces
        self.trace_spacing = trace_spacing
        self.amplitude_factor = amplitude_factor
        
        self.generator = SeismogramGenerator(
            width=width,
            height=height,
            trace_thickness=1,
            num_traces=num_traces,
            trace_spacing=trace_spacing,
            amplitude_factor=amplitude_factor,
            min_freq=0.5,
            max_freq=5.0,
            num_components=5,
            noise_level=0.05
        )

    def extract_multiple_traces_measurements(
        self, 
        image: np.ndarray
    ) -> List[Tuple[np.ndarray, float]]:
        """
        Extract measurements for multiple traces from image.
        
        Args:
            image: Input seismogram image array
            
        Returns:
            List of (measurements, variance) tuples for each trace
        
        Raises:
            ValueError: If image dimensions don't match initialization parameters
        """
        height, width = image.shape
        if width != self.width or height != self.height:
            raise ValueError(f"Image dimensions {image.shape} don't match expected {(self.height, self.width)}")
            
        trace_measurements = []
        
        for trace_idx in range(self.num_traces):
            expected_center = (trace_idx + 1) * self.trace_spacing
            window_size = int(self.trace_spacing * 0.8)
            window_start = max(0, expected_center - window_size//2)
            window_end = min(height, expected_center + window_size//2)
            
            measurements = np.zeros(width)
            
            # Process each column
            for x in range(width):
                column_slice = image[window_start:window_end, x]
                black_pixels = np.where(column_slice < 128)[0]
                
                if len(black_pixels) > 0:
                    measurements[x] = window_start + np.mean(black_pixels)
                else:
                    measurements[x] = measurements[x-1] if x > 0 else expected_center
            
            measurement_variance = np.var(np.diff(measurements))
            trace_measurements.append((measurements, measurement_variance))
        
        return trace_measurements

    def reconstruct_traces(
        self, 
        image: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Reconstruct multiple seismic traces using Kalman filters.
        
        Args:
            image: Input seismogram image array
            
        Returns:
            List of (positions, velocities, accelerations) for each trace
        """
        trace_measurements = self.extract_multiple_traces_measurements(image)
        reconstructed_traces = []
        
        for measurements, meas_variance in trace_measurements:
            params = KalmanParameters(
                dt=1.0,
                measurement_variance=meas_variance,
                process_variance=0.1,
                initial_position_variance=100.0,
                initial_velocity_variance=10.0,
                initial_acceleration_variance=1.0
            )
            
            kf = SeismicTraceKalmanFilter(params)
            
            num_steps = len(measurements)
            positions = np.zeros(num_steps)
            velocities = np.zeros(num_steps)
            accelerations = np.zeros(num_steps)
            
            for t in range(num_steps):
                kf.predict()
                state = kf.update(measurements[t])
                
                positions[t] = state[0]
                velocities[t] = state[1]
                accelerations[t] = state[2]
            
            reconstructed_traces.append((positions, velocities, accelerations))
        
        return reconstructed_traces

    def analyze_and_plot(
        self,
        save_plots: bool = False,
        output_prefix: str = "seismic_analysis"
    ) -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray, np.ndarray]], List[Dict]]:
        """
        Analyze synthetic seismic data and create visualization plots.
        
        Args:
            save_plots: Whether to save the plots to files
            output_prefix: Prefix for output files if saving
            
        Returns:
            Tuple of (image array, reconstructed traces, error statistics)
        """
        print("Generating synthetic seismogram...")
        image, ground_truths = self.generator.generate()
        
        print("Reconstructing traces...")
        reconstructed_traces = self.reconstruct_traces(image)
        
        print("Creating visualizations...")
        self._create_visualization(
            image, 
            reconstructed_traces, 
            ground_truths, 
            save_plots, 
            output_prefix
        )
        
        error_stats = self._calculate_error_stats(
            reconstructed_traces, 
            ground_truths
        )
        
        return image, reconstructed_traces, error_stats

    def _create_visualization(
        self,
        image: np.ndarray,
        reconstructed_traces: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
        ground_truths: List[Dict],
        save_plots: bool,
        output_prefix: str
    ) -> None:
        """Create and optionally save visualization plots."""
        plt.figure(figsize=(15, 15))
        colors = ['r', 'g', 'b', 'y', 'm', 'c']
        
        # Plot 1: Original image with reconstructed curves
        plt.subplot(211)
        plt.imshow(image, cmap='gray', aspect='auto')
        
        for i, (positions, _, _) in enumerate(reconstructed_traces):
            plt.plot(positions, colors[i % len(colors)],
                    linewidth=1.5, alpha=0.7,
                    label=f'Reconstructed Trace {i+1}')
        
        plt.legend()
        plt.title('Original Seismogram with Reconstruction Overlay')
        
        # Plot 2: Velocities
        plt.subplot(223)
        for i, (_, velocities, _) in enumerate(reconstructed_traces):
            plt.plot(velocities, colors[i % len(colors)],
                    alpha=0.7, label=f'Trace {i+1}')
        plt.title('Estimated Velocities')
        plt.grid(True)
        plt.legend()
        
        # Plot 3: Accelerations
        plt.subplot(224)
        for i, (_, _, accelerations) in enumerate(reconstructed_traces):
            plt.plot(accelerations, colors[i % len(colors)],
                    alpha=0.7, label=f'Trace {i+1}')
        plt.title('Estimated Accelerations')
        plt.grid(True)
        plt.legend()
        
        error_stats = self._calculate_error_stats(
            reconstructed_traces, 
            ground_truths
        )
        
        stats_text = self._format_error_stats(error_stats)
        plt.figtext(0.02, 0.02, stats_text,
                   fontsize=8,
                   bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f"{output_prefix}_analysis.png", 
                       dpi=300, bbox_inches='tight')
            plt.imsave(f"{output_prefix}_seismogram.png", 
                      image, cmap='gray')
            
            with open(f"{output_prefix}_statistics.txt", 'w') as f:
                f.write(stats_text)

    def _calculate_error_stats(
        self,
        reconstructed_traces: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
        ground_truths: List[Dict]
    ) -> List[Dict]:
        """Calculate error statistics for reconstructed traces."""
        error_stats = []
        
        for i, ((positions, _, _), truth) in enumerate(zip(reconstructed_traces, ground_truths)):
            original_signal = np.array(truth['signal'])
            original_signal = original_signal * (self.trace_spacing * 0.4) + (i + 1) * self.trace_spacing
            
            error = positions - original_signal
            error_stats.append({
                'trace': i+1,
                'rms': np.sqrt(np.mean(error**2)),
                'max': np.max(np.abs(error)),
                'mean': np.mean(np.abs(error))
            })
            
            print(f"\nTrace {i+1} Statistics:")
            print(f"RMS Error: {error_stats[-1]['rms']:.2f} pixels")
            print(f"Max Error: {error_stats[-1]['max']:.2f} pixels")
            print(f"Mean Error: {error_stats[-1]['mean']:.2f} pixels")
        
        return error_stats

    @staticmethod
    def _format_error_stats(error_stats: List[Dict]) -> str:
        """Format error statistics for display."""
        stats_text = "Error Statistics:\n"
        for stat in error_stats:
            stats_text += f"\nTrace {stat['trace']}:\n"
            stats_text += f"RMS Error: {stat['rms']:.2f} pixels\n"
            stats_text += f"Max Error: {stat['max']:.2f} pixels\n"
            stats_text += f"Mean Error: {stat['mean']:.2f} pixels"
        return stats_text

if __name__ == "__main__":
    # Example usage
    analyzer = SeismicAnalyzer(
        width=800,
        height=1200,
        num_traces=4,
        trace_spacing=200,
        amplitude_factor=1.2
    )
    
    image, traces, stats = analyzer.analyze_and_plot(
        save_plots=True,
        output_prefix="seismic_analysis"
    )