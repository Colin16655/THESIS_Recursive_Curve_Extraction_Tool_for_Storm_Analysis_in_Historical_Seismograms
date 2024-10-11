import numpy as np
import matplotlib.pyplot as plt
from obspy import UTCDateTime
from obspy.clients.syngine import Client
from obspy.clients.base import ClientHTTPException

def generate_raster_seismogram(data, hours=44, large_gap_every=1, small_gap_every=1/60, line_spacing=1.0, overwrite_threshold=0.5, filename=None):
    """
    Generate a raster-style seismogram where each line represents 1 hour of data, with gaps inserted to mark minutes and hours.
    
    Args:
        data (np.array): Seismogram data to be displayed.
        hours (float): Total number of hours to display.
        large_gap_every (int): Insert a large gap every 'n' hours (e.g., 1 hour).
        small_gap_every (float): Insert small gaps every 'n' minutes (e.g., 1/60 for 1 minute).
        line_spacing (float): Controls the vertical spacing between lines.
        overwrite_threshold (float): Threshold to determine when to overwrite neighboring lines with large events.
        filename (str): If specified, save the plot to this filename.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    samples_per_hour = len(data) // int(hours)
    small_gap_size = int(samples_per_hour * small_gap_every)
    large_gap_size = int(samples_per_hour * large_gap_every)

    # Raster plot generation
    for i in range(int(hours)):
        start_idx = i * samples_per_hour
        end_idx = (i + 1) * samples_per_hour
        line_data = data[start_idx:end_idx]

        # Add small gaps for each minute
        for j in range(0, samples_per_hour, small_gap_size):
            line_data[j:j+small_gap_size//2] = np.nan  # Simulate small gaps
        
        # Overwrite large events (if data exceeds threshold, overwrite lines)
        if np.max(line_data) > overwrite_threshold:
            ax.plot(line_data + i * line_spacing, color='black', lw=0.75, zorder=10)  # Plot large event with overwrite
        else:
            # Add background dashed line for hourly intervals
            ax.plot(line_data + i * line_spacing, color='black', lw=0.75, linestyle='--')  # Plot dashed line for background
        
        # Insert large gap between hours
        if i > 0:
            ax.plot([np.nan] * large_gap_size, color='k', lw=0.5)  # Large gap every hour

    # Hide axis for cleaner image
    ax.axis('off')

    # Save or display the image
    if filename:
        plt.savefig(filename, bbox_inches='tight', pad_inches=0.1, dpi=100)
    else:
        plt.show()
    plt.close(fig)

# Generate sample seismic data (this will simulate 44 hours)
client = Client()
t0 = UTCDateTime("2020-01-01T00:00:00")

try:
    # Fetching seismic data for 44 hours using the 'ak135f_5s' model
    seismogram = client.get_waveforms(
        model="ak135f_5s",  
        sourcelatitude=10, 
        sourcelongitude=20,
        sourcedepthinmeters=10000, 
        receiverlatitude=30, 
        receiverlongitude=40,
        origintime=t0
    )
    trace = seismogram[0] #Get the z-component of the seismogram
    # Get start and end times
    start_time = trace.stats.starttime
    end_time = trace.stats.endtime

    # Calculate the duration in seconds
    duration_seconds = end_time - start_time

    # Convert duration to hours
    duration_hours = duration_seconds / 3600
    print(duration_hours)

    data = trace.data.astype(float)

    # Normalize the data for plotting
    data = (data - np.min(data)) / (np.max(data) - np.min(data)) * 2 - 1  # Normalize to [-1, 1]

    # Generate the raster seismogram plot
    generate_raster_seismogram(data, hours=duration_hours, line_spacing=1.2, overwrite_threshold=0.75, filename="raster_seismogram_corrected.png")

except ClientHTTPException as e:
    print(f"Error retrieving seismogram data: {e}")
