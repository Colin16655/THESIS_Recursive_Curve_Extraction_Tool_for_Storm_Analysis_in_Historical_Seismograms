import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
import json
import os
from dataclasses import dataclass
import sys 
from tqdm import tqdm

# Get the absolute path of the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
# Add the project root to sys.path
sys.path.append(project_root)

from seismogram_extraction.data_generation.stat_analysis import SeismogramAnalysis, sanitize_filename
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
        filepath = sanitize_filename(r"seismogram_extraction\data_generation\results\{}_{}_{}_{}_{}_{}_{}_{}_{}\mseed_files_PDFs".format(network, 
                                                                                                        station, 
                                                                                                        location, 
                                                                                                        channel, 
                                                                                                        start_time, 
                                                                                                        end_time, 
                                                                                                        batch_length, 
                                                                                                        bandwidth_0,
                                                                                                        bandwidth)) + ".pkl"
        
        if os.path.exists(filepath):
            print(f"\nFile {filepath} exists. Loading precomputed PDFs...")

            self.analysis = SeismogramAnalysis.load_analysis(filepath)
        else:
            print(f"\nFile {filepath} not found. Computing PDFs...")

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

            print(f"\nPDFs computed and saved to {filepath}")

    def resample_signal(self, dt, T, seed=42):
        """Resample the signal in the Fourier domain from the PDFs of A_k and B_k.

        Args:
            dt: Time step of the signal
            T: length of the signal, len(signal) = T
            seed: Random seed for reproducibility

        Returns:
            np.ndarray: The resampled signal in the time domain.
        """
        n_samples = self.num_traces

        # Generate frequency array
        self.seismo_gt.f = self.analysis.frequencies

        # Extract PDFs once to reduce attribute access overhead
        pdfs_A, pdfs_B = self.analysis.PDFs['A'], self.analysis.PDFs['B']

        # Efficiently sample A_k and B_k
        self.seismo_gt.A = np.vstack([pdf.sample(n_samples).flatten() for pdf in pdfs_A])
        self.seismo_gt.B = np.vstack([pdf.sample(n_samples).flatten() for pdf in pdfs_B])

        # Generate time vector
        self.seismo_gt.t = np.arange(0, T, dt)

        # Compute the resampled signal
        self.seismo_gt.signal = self.analysis.reconstruct_signal(
            self.seismo_gt.A, self.seismo_gt.B, self.seismo_gt.f, self.seismo_gt.t, n_samples
        )

        return self.seismo_gt.signal

    def resample_signal_sines(self, dt, T, seed=42):
        """Resample the signal in the Fourier domain from the PDFs of A_k and B_k.

        Args:
            dt: Time step of the signal
            T: length of the signal, len(signal) = T
            seed: Random seed for reproducibility

        Returns:
            np.ndarray: The resampled signal in the time domain.
        """
        
        num_signals = self.num_traces

        # Generate frequency array

        # Generate time vector
        self.seismo_gt.t = np.arange(0, T, dt) / T * np.pi
        t = self.seismo_gt.t

        num_samples = len(self.seismo_gt.t)

        # Compute the resampled signal
        self.seismo_gt.signal = np.zeros((num_signals, num_samples))
        
        # Randomly sample frequencies and phase shifts
        frequencies = np.random.uniform(1, 20, num_signals)  # Random frequencies between 1 and 20 Hz
        phase_shifts = np.random.uniform(0, 2 * np.pi, num_signals)  # Random phase shifts between 0 and 2*pi


        # Generate sine waves and store them in the pre-allocated array
        for i in range(num_signals):
            self.seismo_gt.signal[i] = np.sin(frequencies[i] * t + phase_shifts[i]) 

        return self.seismo_gt.signal
    
    def generate_seismogram_raster(self, width=800, height=400, line_thickness=2, noise_level=0.1,
                                l_margin=50, r_margin=50, t_margin=50, b_margin=50, 
                                overlap_percentage=0.2, color_mode='rgb', color_flag=False):
        """
        Generate a raster-style artificial old paper seismogram from multiple given signals.

        Args:
            width (int): Width of the generated seismogram image (in pixels).
            height (int): Height of the generated seismogram image (in pixels).
            line_thickness (int): Thickness of the seismic traces (in pixels).
            noise_level (float): Amount of noise to add to the signal.
            l_margin, r_margin, t_margin, b_margin (int): Margins around the image.
            overlap_percentage (float): Percentage of overlap between consecutive traces.
            color_mode (str): 'rgb' for color or 'bw' for grayscale.
            color_flag (bool): If True, different colors are used for each signal.

        Returns:
            np.ndarray: A 2D array representing the generated seismogram.
        """      
        # Store metadata efficiently
        meta_keys = ['width', 'height', 'line_thickness', 'noise_level', 'l_margin', 'r_margin',
                    't_margin', 'b_margin', 'num_traces', 'overlap_percentage', 'color_mode']
        meta_values = [width, height, max(1, int(line_thickness)), noise_level, l_margin, r_margin, 
                    t_margin, b_margin, self.num_traces, overlap_percentage, color_mode]
        self.seismo_gt.meta.update(dict(zip(meta_keys, meta_values)))

        signals = self.seismo_gt.signal
        num_signals = len(signals)

        # Initialize background
        background_color = (235, 215, 180) if color_mode == 'rgb' else 235
        background = np.full((height, width, 3), background_color, dtype=np.uint8) if color_mode == 'rgb' \
                    else np.full((height, width), background_color, dtype=np.uint8)
        # Compute available height and max amplitude
        available_height = height - t_margin - b_margin
        max_amplitude = (available_height / num_signals) / (1 - overlap_percentage)

        # Precompute vertical offsets
        vertical_offsets = np.linspace(t_margin + max_amplitude / 2, height - b_margin - max_amplitude / 2, num_signals)

        # Precompute horizontal offset
        horizontal_offsets = l_margin
        available_width = width - l_margin - r_margin

        # Generate color palette efficiently
        if color_flag:
            cmap = plt.get_cmap("tab20")
            color_palette = (np.array([cmap(i)[:3] for i in range(num_signals)]) * 255).astype(int)
        else:
            color_palette = [(0, 0, 0)] * num_signals  # Default to black

        # Prepare ground truth storage
        self.GTs = np.zeros((num_signals, background.shape[1]))

        for i, signal in enumerate(signals):
            # Normalize and scale signal
            signal_mean = np.mean(signal)
            signal_max = np.max(np.abs(signal))
            signal = (signal - signal_mean) / signal_max
            scaled_signal = (max_amplitude / 2) * signal + vertical_offsets[num_signals - 1 - i]

            # Generate coordinates efficiently
            x_coords = np.linspace(0, available_width - 1, background.shape[1]).astype(np.int32) + horizontal_offsets
            y_coords = np.clip(scaled_signal.astype(np.int32), 0, height - 1)  # Ensure bounds

            # Crop x_coords and y_coords to background.shape[1]
            y_coords = y_coords[:background.shape[1]]

            self.GTs[i] = height - y_coords

            # Convert to OpenCV polyline format
            pts = np.column_stack((x_coords, height - y_coords))

            # Draw waveform using polylines for efficiency
            color = tuple(map(int, color_palette[i])) if color_mode == 'rgb' else 0
            cv2.polylines(background, [pts], isClosed=False, color=color, thickness=line_thickness)

        # Add noise efficiently
        # noise = np.random.normal(0, noise_level * 255, background.shape).astype(np.int32)
        # final_image = np.clip(background + noise, 0, 255).astype(np.uint8)
        final_image = background

        self.seismo_gt.image = final_image
        return final_image
    
    def save_analysis(self, filepath_object, filepath_image=None, filepath_npy=None, save_object=True):
        """
        Save the entire SeismogramGT object to a file.

        Parameters:
            filepath (str): Path to save the object.
        """
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filepath_object), exist_ok=True)
        os.makedirs(os.path.dirname(filepath_image), exist_ok=True)
        
        if save_object:
            with open(filepath_object, "wb") as file:
                pickle.dump(self.seismo_gt, file)
            print(f"\nSeismogramGT object saved to {filepath_object}.")

        if filepath_image is not None:
            # save the image in pdf format
            if self.seismo_gt.meta['color_mode'] == 'bw':
                plt.imsave(filepath_image, self.seismo_gt.image, dpi=300, cmap='gray')
            else:
                plt.imsave(filepath_image, self.seismo_gt.image, dpi=300)

        if filepath_npy is not None:
            # save the image in npy format
            np.save(filepath_npy, self.GTs)

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
        print(f"\nSeismogramGT object loaded from {filepath}.")
        return seismo_gt
    

# Example usage
if __name__ == "__main__":
    # Set the random seed once at the start
    # np.random.seed(42)

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

    ### USER
    option = 0 # 0: Generate seismogram from sine and cosine waves, 1: Generate seismogram from PDFs
    ###

    l_margin = 0 # must be 0 otherwise GT are not corect 
    r_margin = 0 # must be 0 otherwise GT are not corect 
    t_margin = 50
    b_margin = 50
    color_mode = 'bw'

    dt = 0.3
    T = 86400*0.003 / 4

    num_images = 1000

    # Create generator with custom parameters
    generator = SeismogramGenerator()

    for i, overlap in enumerate(np.linspace(0.0, 0.5, 10)):  # Different overlap levels
        print(f"\n\nOverlap percentage: {overlap}")
        num = 0
        overlap_percentage = overlap

        # Define the file path to save or load the results
        directory = r"seismogram_extraction\data_generation\results"
        
        if option == 0:
            folder_path = sanitize_filename(r"\sines")
            directory_data = r"data\sines"
        if option == 1:
            folder_path = sanitize_filename(r"\{}_{}_{}_{}_{}_{}_{}_{}_{}".format(network,
                                                                    station,
                                                                    location,
                                                                    channel,
                                                                    start_time,
                                                                    end_time,
                                                                    batch_length,
                                                                    bandwidth_0,
                                                                    bandwidth))
            directory_data = r"data\resampled" + folder_path
        os.makedirs(folder_path, exist_ok=True)
        os.makedirs(directory_data, exist_ok=True)
        filepath = sanitize_filename(directory + folder_path + r"\{}_{}_{}_{}_{}_{}_{}_{}\seismo_gt".format(l_margin,
                                                                                                      r_margin,
                                                                                                      t_margin,
                                                                                                      b_margin,
                                                                                                      overlap_percentage,
                                                                                                      color_mode,
                                                                                                      dt,
                                                                                                      T)) + ".pkl"
        
        filepath_data = sanitize_filename(directory_data + r"\overlap_{:.2f}".format(overlap_percentage))

        for j in tqdm(range(num_images), desc="Generating images"):
            filepath_image = filepath_data + r"\signals\image_{:05d}.jpg".format(num)
            filepath_npy = filepath_data + r"\ground_truth\sample_{:05d}.npy".format(num)
            os.makedirs(filepath_data + r"\ground_truth", exist_ok=True)
            
            if option == 0:
                signals = generator.resample_signal_sines(dt=dt, T=T)
            elif option == 1:
                signals = generator.resample_signal(dt=dt, T=T)
            else:
                raise ValueError("Invalid option. Choose 0 or 1.")

            seismogram_image = generator.generate_seismogram_raster(width=50, height=200, l_margin=l_margin, r_margin=r_margin, t_margin=t_margin, b_margin=b_margin, 
                                                                overlap_percentage=overlap_percentage, color_mode=color_mode)
        
            # Save the results for future use
            generator.save_analysis(filepath, filepath_image=filepath_image, filepath_npy=filepath_npy, save_object=(j==(num_images-1))) 
            num += 1   
