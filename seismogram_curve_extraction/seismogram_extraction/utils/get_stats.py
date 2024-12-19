import numpy as np
import matplotlib.pyplot as plt
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
from scipy.signal import stft


def analyze_seismogram_amplitude_phase(
    network="IU", station="ANMO", location="*", channel="BHZ",
    starttime="2024-01-01T00:00:00", endtime="2024-01-01T01:00:00",
    nperseg=1024, provider="IRIS"
):
    """
    Analyze the amplitude and phase probability density functions (PDFs) 
    from a seismogram using Short-Time Fourier Transform (STFT).

    Parameters:
    - network (str): Seismic network code (e.g., "IU").
    - station (str): Station code (e.g., "ANMO").
    - location (str): Location code (e.g., "*").
    - channel (str): Channel code (e.g., "BHZ").
    - starttime (str): Start time in UTC (e.g., "2024-01-01T00:00:00").
    - endtime (str): End time in UTC (e.g., "2024-01-01T01:00:00").
    - nperseg (int): Number of samples per segment for STFT.
    - provider (str): Data provider (e.g., "IRIS").

    Outputs:
    - Plots of amplitude and phase PDFs.
    - A spectrogram of amplitude vs. time and frequency.
    """
    # Step 1: Fetch Seismogram Data
    client = Client(provider)
    st = client.get_waveforms(network, station, location, channel, UTCDateTime(starttime), UTCDateTime(endtime))
    trace = st[0]
    data = trace.data  # Time-series data
    fs = trace.stats.sampling_rate  # Sampling frequency

    # Step 2: Perform Short-Time Fourier Transform (STFT)
    window = "hann"  # Window type
    f, t, Zxx = stft(data, fs=fs, window=window, nperseg=nperseg)

    # Extract amplitudes and phases
    amplitudes = np.abs(Zxx)  # Magnitudes (amplitudes)
    phases = np.angle(Zxx)    # Phases (in radians)

    # Step 3: Compute Empirical PDFs
    # Flatten amplitude and phase arrays
    amplitude_flat = amplitudes.flatten()
    phase_flat = phases.flatten()

    # Amplitude PDF
    amplitude_hist, amplitude_bins = np.histogram(amplitude_flat, bins=50, density=True)
    amplitude_centers = (amplitude_bins[:-1] + amplitude_bins[1:]) / 2

    # Phase PDF
    phase_hist, phase_bins = np.histogram(phase_flat, bins=50, density=True)
    phase_centers = (phase_bins[:-1] + phase_bins[1:]) / 2

    # Step 4: Plot Results
    # Plot Amplitude PDF
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(amplitude_centers, amplitude_hist, color="blue", label="Amplitude PDF")
    plt.xlabel("Amplitude")
    plt.ylabel("Probability Density")
    plt.title("Amplitude Probability Density Function")
    plt.legend()

    # Plot Phase PDF
    plt.subplot(1, 2, 2)
    plt.plot(phase_centers, phase_hist, color="orange", label="Phase PDF")
    plt.xlabel("Phase (radians)")
    plt.ylabel("Probability Density")
    plt.title("Phase Probability Density Function")
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Step 5: Visualize Time-Frequency Spectrogram
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(t, f, amplitudes, shading="gouraud", cmap="viridis")
    plt.colorbar(label="Amplitude")
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (s)")
    plt.title("STFT Amplitude Spectrogram")
    plt.show()


if __name__ == "__main__":
    # Call the function with example parameters
    analyze_seismogram_amplitude_phase(
        network="IU", station="ANMO", location="*", channel="BHZ",
        starttime="2024-01-01T00:00:00", endtime="2024-01-01T01:00:00",
        nperseg=1024, provider="IRIS"
    )
