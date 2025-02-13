import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
import json
import os
from dataclasses import dataclass
from stat_analysis import SeismogramAnalysis, sanitize_filename
import cv2
import pickle

@dataclass
class SeismogramGT:
    """
    A class representing the ground truth (GT) of a seismogram.

    The temporal signal is constructed as:
        signal = sum(A_k * cos(2 * pi * f_k * t) + B_k * sin(2 * pi * f_k * t)) (1)

    Attributes:
        f (List[float]): List of frequencies f_k in the signal equation.
        A (List[float]): List of coefficients A_k for each frequency.
        B (List[float]): List of coefficients B_k for each frequency.
        image (np.ndarray): A 2D array representing the raster image of the seismogram.
        signal (np.ndarray): A 1D array containing the temporal signal values.
        init (bool): A flag indicating whether the initialization is complete.
        meta (Dict[str, float]): A dictionary storing metadata such as image dimensions, trace information, and noise level.
            - width: Width of the raster image in pixels.
            - height: Height of the raster image in pixels.
            - line_thickness: Thickness of the seismic trace in pixels.
            - noise_level: The amount of noise added to the signal.
            - l_margin: Left margin of the image in pixels.
            - r_margin: Right margin of the image in pixels.
            - t_margin: Top margin of the image in pixels.
            - b_margin: Bottom margin of the image in pixels.
            - num_traces: The number of traces in the seismogram.
            - spacing: Vertical spacing between traces in pixels.
            - overlap_percentage: The percentage of overlap between traces.
            - color_mode: The color mode of the raster image ('rgb' or 'bw').
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
    """
        A class for generating synthetic seismograms and processing them using Fourier-based signal generation.
    """
    def __init__(self,
                 num_traces=5,
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
            num_traces (int): Number of traces to be included in the seismogram.
            network, station, location, channel, start_time, end_time (str): Parameters for the seismogram's metadata.
            batch_length (int): Length of the time window for each batch of data (in seconds).
            bandwidth_0, bandwidth (float): Parameters for the frequency analysis.

        Attributes:
            seismo_gt (SeismogramGT): An instance of the SeismogramGT class, representing the ground truth of the seismogram.
            analysis (SeismogramAnalysis): An instance of SeismogramAnalysis to perform signal analysis and reconstruction.
            width, height, line_thickness, noise_level, l_margin, r_margin, t_margin, b_margin, num_traces, spacing, 
            overlap_percentage, color_mode (int/str): Configurable parameters for image generation.
        """
        self.num_traces = num_traces
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

    def resample_signal(self, dt, T, seed=42):
        """Resample the signal in the Fourier domain from the PDFs of A_k and B_k.

        Args:
            dt: Time step of the signal
            T: length of the signal, len(signal) = T
            seed: Random seed for reproducibility

        Returns:
            np.ndarray: The resampled signal in the time domain.
        """
        # Number of samples
        n_samples = self.num_traces

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
    
    def generate_seismogram_raster(self, 
                                   width=800, 
                                   height=400, 
                                   line_thickness=2, 
                                   noise_level=0.1, 
                                   l_margin=50, 
                                   r_margin=50, 
                                   t_margin=50, 
                                   b_margin=50,
                                   overlap_percentage=0.2, 
                                   color_mode='rgb',
                                   color_flag=False):
        """
        Generate a raster-style artificial old paper seismogram from multiple given signals.

        Args:
            width (int): Width of the generated seismogram image (in pixels).
            height (int): Height of the generated seismogram image (in pixels).
            line_thickness (int): Thickness of the seismic traces (in pixels).
            noise_level (float): Amount of noise to add to the signal (standard deviation of Gaussian noise).
            l_margin, r_margin, t_margin, b_margin (int): Margins around the image (in pixels).
            spacing (int): Vertical spacing between the traces (in pixels).
            overlap_percentage (float): Percentage of overlap between consecutive traces.
            color_mode (str): Color mode for the raster image, either 'rgb' or 'bw'.
            color_flag (bool): Flag to apply different colors for each signal.

        Returns:
            np.ndarray: A 2D array representing the generated raster-style seismogram.
        """      

        self.seismo_gt.meta['width'] = width 
        self.seismo_gt.meta['height'] = height
        self.seismo_gt.meta['line_thickness'] = max(1, int(line_thickness))
        self.seismo_gt.meta['noise_level'] = noise_level
        self.seismo_gt.meta['l_margin'] = l_margin
        self.seismo_gt.meta['r_margin'] = r_margin
        self.seismo_gt.meta['t_margin'] = t_margin
        self.seismo_gt.meta['b_margin'] = b_margin
        self.seismo_gt.meta['num_traces'] = self.num_traces
        self.seismo_gt.meta['overlap_percentage'] = overlap_percentage
        self.seismo_gt.meta['color_mode'] = color_mode

        signals = self.seismo_gt.signal
        
        # Create blank canvas with aged paper color (beige in RGB)
        if color_mode == 'rgb':
            background = np.full((height, width, 3), (235, 215, 180), dtype=np.uint8)
        else:  # 'bw' mode
            background = np.full((height, width), 235, dtype=np.uint8)  # Grayscale background
        
        num_signals = len(signals)
        
        # Ensure that the signals fit within the image bounds while respecting the margins
        available_height = height - t_margin - b_margin  # Space available for signals
        max_amplitude = (available_height / num_signals) / (1 - overlap_percentage)  # Control overlap
        
        # Create vertical offsets while considering margins and overlap
        vertical_offsets = np.linspace(t_margin + max_amplitude / 2, height - b_margin - max_amplitude / 2, num_signals)
        
        # Horizontal shifting
        available_width = width - l_margin - r_margin
        horizontal_offsets = l_margin # np.linspace(l_margin, l_margin + (num_signals - 1) * spacing, num_signals).astype(int)
        
        # If color_flag is set, generate a color palette with enough distinct colors
        if color_flag:
            # Generate a set of distinct colors using a colormap (e.g., the "tab20" colormap can generate up to 20 distinct colors)
            cmap = plt.get_cmap("tab20")  # Using "tab20" for up to 20 distinct colors
            color_palette = [tuple((np.array(cmap(i)) * 255).astype(int)[:3]) for i in range(num_signals)]
        else:
            color_palette = [(0, 0, 0)] * num_signals  # All black if no color_flag

        for i, signal in enumerate(signals):
            # Normalize and scale signal
            signal = (signal - np.mean(signal)) / np.max(np.abs(signal))  # Center and normalize
            scaled_signal = (max_amplitude / 2) * signal + vertical_offsets[i]  # Rescale amplitude to fit within frame
            
            # Create x-coordinates for plotting with horizontal offset
            x_coords = np.linspace(0, available_width - 1, len(signal)).astype(np.int32) + horizontal_offsets
            y_coords = scaled_signal.astype(np.int32)  # Ensure y-coordinates stay in bounds
            
            # Select color for the signal
            color = color_palette[i]  # Pick the color from the palette

            # Ensure x-coordinates remain within image bounds
            # x_coords = np.clip(x_coords, l_margin, width - r_margin - 1)
            
            # Draw waveform (in black for both RGB and BW modes)
            for j in range(1, len(x_coords)):
                if color_mode == 'rgb':
                    cv2.line(background, (x_coords[j-1], y_coords[j-1]), (x_coords[j], y_coords[j]), color, line_thickness)
                else:  # 'bw' mode
                    cv2.line(background, (x_coords[j-1], y_coords[j-1]), (x_coords[j], y_coords[j]), color, line_thickness)
        
        # Add noise for old paper effect
        noise = np.random.normal(0, noise_level * 255, background.shape).astype(np.int32)
        noisy_image = np.clip(background + noise, 0, 255).astype(np.uint8)
        
        # Apply slight blur for authenticity
        final_image = cv2.GaussianBlur(noisy_image, (3, 3), 0)

        print("image type : ", type(final_image))
        self.seismo_gt.image = final_image
        
        return final_image
    
    def save_analysis(self, filepath):
        """
        Save the entire SeismogramGT object to a file.

        Parameters:
            filepath (str): Path to save the object.
        """
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, "wb") as file:
            pickle.dump(self.seismo_gt, file)
        print(f"SeismogramGT object saved to {filepath}.")

        # save the image in pdf format, dpi=300
        plt.imsave(filepath + ".pdf", self.seismo_gt.image, format='pdf', dpi=300)

    @staticmethod
    def load_analysis(filepath):
        """
        Load a SeismogramGT object from a file.

        Parameters:
            filepath (str): Path to load the object from.

        Returns:
            SeismogramAnalysis: The loaded analysis object.
        """
        with open(filepath, "rb") as file:
            seismo_gt = pickle.load(file)
        print(f"SeismogramGT object loaded from {filepath}.")
        return seismo_gt
    

# Example usage
if __name__ == "__main__":
    # Set the random seed once at the start
    np.random.seed(42)

    plt.rcParams['xtick.labelsize'] = 11
    plt.rcParams['ytick.labelsize'] = 11
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['legend.fontsize'] = 11
    plt.rcParams['legend.loc'] = 'upper right'
    plt.rcParams['axes.titlesize'] = 12

    network="BE", 
    station="UCC", 
    location="",
    channel="HHE", 
    start_time="2024-01-01T00:06:00", 
    end_time="2024-01-14T00:12:00", 
    batch_length=1000
    bandwidth_0 = 1
    bandwidth = 0.1  # Set the bandwidth for KDE (None for automatic selection) 

    l_margin = 50
    r_margin = 50
    t_margin = 50
    b_margin = 50
    color_mode = 'rgb'

    dt = 0.1
    T = 86400*0.01

    for i, overlap in enumerate([0.0, 0.5, 0.6]):  # Different overlap levels
        overlap_percentage = 'vary'

        # Define the file path to save or load the results
        filepath = sanitize_filename(r"seismogram_curve_extraction\results\{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}\mseed_files_PDFs".format(network, 
                                                                                                      station, 
                                                                                                      location, 
                                                                                                      channel, 
                                                                                                      start_time, 
                                                                                                      end_time, 
                                                                                                      batch_length, 
                                                                                                      bandwidth_0,
                                                                                                      bandwidth,
                                                                                                      l_margin,
                                                                                                      r_margin,
                                                                                                      t_margin,
                                                                                                      b_margin,
                                                                                                      overlap_percentage,
                                                                                                      color_mode,
                                                                                                      dt,
                                                                                                      T))
    
        if os.path.exists(filepath):
            print(f"File {filepath} exists. Loading precomputed PDFs...")

            seismo_gt = SeismogramGenerator.load_analysis(filepath)
        else:
            print(f"File {filepath} not found. Computing PDFs...")

            # Create generator with custom parameters
            generator = SeismogramGenerator()

            signals = generator.resample_signal(dt=dt, T=T)

            seismogram_image = generator.generate_seismogram_raster(l_margin=l_margin, r_margin=r_margin, t_margin=t_margin, b_margin=b_margin, 
                                                                overlap_percentage=overlap, color_mode=color_mode)
        
            # Save the results for future use
            generator.save_analysis(filepath)    
