import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
import json
import os
from dataclasses import dataclass

@dataclass
class SeismogramGT:
    """
    A class representing the ground truth (GT) of a seismogram.

    The temporal signal is constructed as:
        signal = sum(A_k * cos(2 * pi * f_k * t) + B_k * sin(2 * pi * f_k * t))

    Attributes:
        f: A list of frequencies (Hz) for each signal component.
        A: A list of amplitudes corresponding to each frequency.
        B: A list of phase indicators (True for sine, False for cosine).
        image: A 2D NumPy array representing the raster image of the seismogram.
        signal: A 1D NumPy array containing the temporal signal values.
        init: A boolean indicating whether the initialization is complete.
        meta: A dictionary storing additional metadata about the seismogram, including:
            - width: The width of the raster image in pixels.
            - height: The height of the raster image in pixels.
            - trace_thickness: The thickness of each seismic trace in pixels.
            - num_traces: The number of seismic traces in the image.
            - trace_spacing: The vertical spacing between traces in pixels.
            - num_components: The number of sine/cosine components used in the signal.
            - noise_level: The amount of noise added to the signal (range 0-1).
    """
    f: np.ndarray = None
    A: np.ndarray = None
    B: np.ndarray = None
    image: np.ndarray = None
    signal: np.ndarray = None
    init: bool = False
    meta: Dict = None

class SeismogramGenerator:
    def __init__(self,
                 width: int = 240,
                 height: int = 120,
                 trace_thickness: int = 8,
                 num_traces: int = 5,
                 trace_spacing: int = 48,
                 amplitude_factor: float = 1.0,
                 min_freq: float = 0.5,
                 max_freq: float = 5.0,
                 num_components: int = 5,
                 noise_level: float = 0.05):
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
        """
        self.seismo_gt = SeismogramGT()
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

    def generate_random_parameters(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate random parameters for a single trace."""
        frequencies = np.random.uniform(self.min_freq, self.max_freq, self.num_components)
        amplitudes = np.random.random(self.num_components) # random float in [0.0, 1.0]
        amplitudes /= amplitudes.sum()
        use_sine = np.random.choice([True, False], size=self.num_components)
        return frequencies, amplitudes, use_sine

    def generate_signal(self,
                        frequencies: List[float],
                        amplitudes: List[float],
                        use_sine: List[bool]) -> np.ndarray:
        """Generate a single trace signal using the given parameters."""
        t = np.linspace(0, 10, self.width)
        signal = np.zeros(self.width)

        for freq, amp, is_sine in zip(frequencies, amplitudes, use_sine):
            if is_sine:
                signal += amp * np.sin(2 * np.pi * freq * t)
            else:
                signal += amp * np.cos(2 * np.pi * freq * t)

        # Add noise
        signal += np.random.normal(0, self.noise_level, self.width)
        signal *= self.amplitude_factor
        return signal

    def draw_thick_line(self, image: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> None:
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

    def create_image(self, signals: List[np.ndarray]) -> np.ndarray:
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

    def generate(self) -> Tuple[np.ndarray, List[Dict]]:
        """Generate multiple traces and return the image and ground truth parameters."""
        signals, ground_truths = [], []
        for _ in range(self.num_traces):
            frequencies, amplitudes, use_sine = self.generate_random_parameters()
            signal = self.generate_signal(frequencies, amplitudes, use_sine)
            signals.append(signal)
            ground_truths.append({
                'frequencies': frequencies,
                'amplitudes': amplitudes,
                'use_sine': use_sine,
                'signal': signal.tolist()
            })
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

    # Generate a sample
    image, ground_truths = generator.generate()

    # Display the image
    plt.figure(figsize=(15, 15))
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.title("Generated Multi-Trace Seismogram")
    plt.show()

    # Print ground truth parameters for each trace
    print("\nGround Truth Parameters:")
    for i, truth in enumerate(ground_truths):
        print(f"\nTrace {i + 1}:")
        print(f"Frequencies: {truth['frequencies']}")
        print(f"Amplitudes: {truth['amplitudes']}")
        print(f"Use Sine: {truth['use_sine']}")

    # Save example
    save_example("sample_multi_seismogram", image, ground_truths)
