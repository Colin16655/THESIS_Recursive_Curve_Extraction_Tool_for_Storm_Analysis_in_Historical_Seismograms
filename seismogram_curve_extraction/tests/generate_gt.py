import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Get the absolute path of the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the project root to sys.path
sys.path.append(project_root)

from seismogram_extraction.data_generation import SeismogramGenerator
from seismogram_extraction import stat_analysis
from seismogram_extraction.stat_analysis import SeismogramAnalysis

# np.random.seed(42)

plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['legend.loc'] = 'upper right'
plt.rcParams['axes.titlesize'] = 12

# Example usage
def generate_raster_images(N=1000):   
    network="BE", 
    station="UCC", 
    location="",
    channel="HHE", 
    start_time="2024-01-01T00:06:00", 
    end_time="2024-01-14T00:12:00", 
    batch_length=1000
    bandwidth_0 = 1
    bandwidth = 0.1  # Set the bandwidth for KDE (None for automatic selection) 

    # Define the parameters for the seismogram generation
    l_margin = 50
    r_margin = 50
    t_margin = 50
    b_margin = 50
    color_mode = 'bw'

    dt = 0.1
    T = 86400*0.01

    # signals = [np.sin(np.linspace(0, 10 * np.pi, 1000)), 
    #        np.cos(np.linspace(0, 10 * np.pi, 1000)),
    #        np.sin(np.linspace(0, 15 * np.pi, 1000) + 1)]  # Simulated sine and cosine waves
    
    # Create generator with custom parameters
    generator = SeismogramGenerator()

    for i, overlap in enumerate([0.0, 0.2, 0.5]):  # Different overlap levels
        print(f"\n\nOverlap percentage: {overlap}")
        overlap_percentage = overlap

        folder_path = r"seismogram_curve_extraction/data/ground_truths/overlap_{}_".format(int(100*overlap_percentage))

        for j in range(4, N):
            filepath = folder_path + f"gt/{j}.npy"
            filepath_image = folder_path + f"images/{j}.jpg"

            signals = generator.resample_signal(dt=dt, T=T)

            seismogram_image = generator.generate_seismogram_raster(l_margin=l_margin, r_margin=r_margin, t_margin=t_margin, b_margin=b_margin, 
                                                                    overlap_percentage=overlap_percentage, color_mode=color_mode)
            
            # Save the results for future use
            generator.save_analysis(filepath, filepath_image=filepath_image, filepath_npy=filepath, save_object=False)    


if __name__ == "__main__":
    generate_raster_images(15)