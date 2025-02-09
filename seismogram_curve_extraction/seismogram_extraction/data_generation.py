import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
import json
import os
from dataclasses import dataclass
from stat_analysis import SeismogramAnalysis, sanitize_filename

@dataclass
class SeismogramGT:
    """
    A class representing the ground truth (GT) of a seismogram.

    The temporal signal is constructed as:
        signal = sum(A_k * cos(2 * pi * f_k * t) + B_k * sin(2 * pi * f_k * t)) (1)

    Attributes:
        f: list of f_k in (1).
        A: list of A_k in (1) corresponding to each frequency.
        B: list of B_k in (1) corresponding to each frequency.
        image: A 2D NumPy array representing the raster image of the seismogram.
        signal: A 1D NumPy array containing the temporal signal values.
        init: A boolean indicating whether the initialization is complete.
        meta: A dictionary storing additional metadata about the seismogram, including:
            - width: The width of the raster image in pixels.
            - height: The height of the raster image in pixels.
            - trace_thickness: The thickness of each seismic trace in pixels.
            - num_traces: The number of seismic traces in the image.
            - trace_spacing: The vertical spacing between traces in pixels.
            - noise_level: The amount of noise added to the signal (range 0-1).
            - overlap_level: The amount of overlap between traces (range 0-1).
    """
    f = None
    A = None
    B = None
    image = None
    t = None
    signal = None
    init = False
    meta = {}

class SeismogramGenerator:
    def __init__(self,
                 width=240,
                 height= 120,
                 trace_thickness=8,
                 num_traces=5,
                 trace_spacing=48,
                 amplitude_factor=1.0,
                 min_freq=0.5,
                 max_freq=5.0,
                 num_components=5,
                 noise_level=0.05,
                 network="BE", 
                 station="UCC", 
                 location="",
                 channel="HHE", 
                 start_time="2024-01-01T00:06:00", 
                 end_time="2024-01-14T00:12:00", 
                 batch_length=int(86400/4),
                 bandwidth_0=1,
                 bandwidth=0.1):
        """
        Initialize the seismogram generator.

        Args:
            width: Width of the image in pixels
            height: Height of the image in pixels
            trace_thickness: Thickness of the seismic trace in pixels
            num_traces: Number of traces to generate
            trace_spacing: Vertical spacing between traces in pixels
            num_components: Number of sine/cosine components to use
            noise_level: Amount of noise to add (0-1), std of the White Gaussian noise
            network, station, location, channel, start_time, end_time, batch_length, bandwidth_0, bandwidth: Parameters for the SeismogramAnalysis
        """
        self.seismo_gt = SeismogramGT()

        # Define the file path to save or load the results
        filepath = sanitize_filename(r"seismogram_curve_extraction\results\{}_{}_{}_{}_{}_{}_{}_{}_{}\mseed_files_PDFs".format(network, 
                                                                                                        station, 
                                                                                                        location, 
                                                                                                        channel, 
                                                                                                        start_time, 
                                                                                                        end_time, 
                                                                                                        batch_length, 
                                                                                                        bandwidth_0,
                                                                                                        bandwidth))
        
        if os.path.exists(filepath):
            print(f"File {filepath} exists. Loading precomputed PDFs...")

            self.analysis = SeismogramAnalysis.load_analysis(filepath)
        else:
            print(f"File {filepath} not found. Computing PDFs...")

            self.analysis = SeismogramAnalysis(network=network, 
                                          station=station, 
                                          location=location, 
                                          channel=channel, 
                                          start_time=start_time, 
                                          end_time=end_time,
                                          batch_length=batch_length) # 86400 for 1 day batch length
            # Process the seismogram in batches and compute the PDFs
            self.analysis.process_batches(bandwidth_0=bandwidth_0, bandwidth=bandwidth)
            
            # Save the results for future use
            self.analysis.save_analysis(filepath)    

            print(f"PDFs computed and saved to {filepath}")

        self.width = width
        self.height = height
        self.trace_thickness = max(1, int(trace_thickness))
        self.num_traces = num_traces
        self.trace_spacing = trace_spacing
        self.num_components = num_components
        self.noise_level = noise_level
        self.seismo_gt.meta['width'] = width 
        self.seismo_gt.meta['height'] = height
        self.seismo_gt.meta['trace_thickness'] = max(1, int(trace_thickness))
        self.seismo_gt.meta['num_traces'] = num_traces
        self.seismo_gt.meta['trace_spacing'] = trace_spacing
        self.seismo_gt.meta['num_components'] = num_components
        self.seismo_gt.meta['noise_level'] = noise_level

    def resample_signal(self, n_samples, dt, T, seed=42):
        """Resample the signal in the Fourier domain from the PDFs of A_k and B_k.

        Args:
            n_samples: Number of samples to generate
            dt: Time step of the signal
            T: length of the signal, len(signal) = T
            seed: Random seed for reproducibility
        """
        # Set the random seed
        np.random.seed(seed)

        # Generate the frequencies f_k
        self.seismo_gt.f = self.analysis.frequencies
        # Generate the A_k and B_k coefficients
        self.seismo_gt.A = np.array([pdf.sample(n_samples).flatten() for pdf in self.analysis.PDFs['A']])
        self.seismo_gt.B = np.array([pdf.sample(n_samples).flatten() for pdf in self.analysis.PDFs['B']])

        # Time vector
        self.seismo_gt.t = np.arange(0, T, dt)
        self.seismo_gt.signal = self.analysis.reconstruct_signal(self.seismo_gt.A, 
                                                              self.seismo_gt.B, 
                                                              self.seismo_gt.f, 
                                                              self.seismo_gt.t, 
                                                              n_samples)

        return self.seismo_gt.signal

    def draw_thick_line(self, image: np.ndarray, x1: int, y1: int, x2: int, y2: int):
        """Draw a thick line between two points."""
        if abs(y2 - y1) <= 1:
            y_range = [y1]
        else:
            y_range = np.arange(min(y1, y2), max(y1, y2) + 1)

        for y in y_range:
            for thickness_offset in range(-(self.trace_thickness // 2),
                                          (self.trace_thickness + 1) // 2):
                y_pos = y + thickness_offset
                if 0 <= y_pos < self.height:
                    image[y_pos, x1] = 0

    def create_image(self, signals: List[np.ndarray]):
        """Create an image from multiple signals."""
        image = np.ones((self.height, self.width)) * 255
        for trace_idx, signal in enumerate(signals):
            baseline = (trace_idx + 1) * self.trace_spacing
            scaled_signal = signal * (self.trace_spacing * 0.4) # NO
            centered_signal = baseline + scaled_signal
            centered_signal = np.clip(centered_signal, 0, self.height - 1) # NO

            for x in range(self.width - 1):
                y1, y2 = int(centered_signal[x]), int(centered_signal[x + 1])
                self.draw_thick_line(image, x, y1, x + 1, y2)
        return image.astype(np.uint8)

    def generate(self):
        """Generate multiple traces and return the image and ground truth parameters."""
        signals, ground_truths = [], []
        for _ in range(self.num_traces):
            signal = self.resample_signal(n_samples=1, dt=0.1, T=86400*0.01)
        return self.create_image(signals), ground_truths


def save_example(filename_prefix: str, image: np.ndarray, ground_truths: List[Dict]):
    """Save the generated image and ground truth to the correct folders."""
    # Set the base directory to 'seismogram_curve_extraction' folder
    base_dir = os.path.join(os.getcwd(), 'seismogram_curve_extraction')
    
    # Ensure the necessary directories exist inside the base directory
    os.makedirs(os.path.join(base_dir, "data/raw"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "data/ground_truth"), exist_ok=True)

    # Save the image to 'data/raw'
    image_filename = os.path.join(base_dir, "data/raw", f"{filename_prefix}_image.png")
    plt.imsave(image_filename, image, cmap='gray')

    # Save the ground truth as a JSON file to 'data/ground_truth'
    gt_filename = os.path.join(base_dir, "data/ground_truth", f"{filename_prefix}_truth.json")
    with open(gt_filename, 'w') as f:
        json.dump(ground_truths, f, indent=2)

    print(f"Example saved: {image_filename} and {gt_filename}")

# Example usage
if __name__ == "__main__":
    # Set the random seed once at the start
    np.random.seed(42)

    

    # Create generator with custom parameters
    generator = SeismogramGenerator(
        amplitude_factor=1.2,
        min_freq=0.5,
        max_freq=5.0,
        num_components=5,
        noise_level=0.05
    )

    signals = generator.resample_signal(n_samples=5, dt=0.1, T=86400*0.01)

    # Display the signals
    plt.figure(figsize=(15, 5))
    for signal in signals:
        plt.plot(signal)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.show()

    # # Generate a sample
    # image, ground_truths = generator.generate()

    # # Display the image
    # plt.figure(figsize=(15, 15))
    # plt.imshow(image, cmap='gray')
    # plt.axis('off')
    # plt.title("Generated Multi-Trace Seismogram")
    # plt.show()

    # # Print ground truth parameters for each trace
    # print("\nGround Truth Parameters:")
    # for i, truth in enumerate(ground_truths):
    #     print(f"\nTrace {i + 1}:")
    #     print(f"Frequencies: {truth['frequencies']}")
    #     print(f"Amplitudes: {truth['amplitudes']}")
    #     print(f"Use Sine: {truth['use_sine']}")

    # # Save example
    # save_example("sample_multi_seismogram", image, ground_truths)
