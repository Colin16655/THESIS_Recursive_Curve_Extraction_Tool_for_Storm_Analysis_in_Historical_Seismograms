from obspy import UTCDateTime
from obspy.clients.fdsn import Client
import numpy as np
import matplotlib.pyplot as plt

class SeismogramAnalysis:
    """
    A class for performing seismogram analysis including waveform data retrieval,
    Fourier transform, and signal reconstruction.

    Attributes:
        network (str): Network code.
        station (str): Station code.
        location (str): Location code (leave blank if unknown).
        channel (str): Channel code.
        start_time (UTCDateTime): Start time of the signal to be analyzed.
        end_time (UTCDateTime): End time of the signal to be analyzed.
        signal (Stream): ObsPy Stream object containing the waveform data.
        A (array): Vector of A_k coefficients for Fourier transform.
        B (array): Vector of B_k coefficients for Fourier transform.
        PDFs (array): Probability density functions (to be implemented).
    """

    def __init__(self, network="BE", 
                       station="UCC", 
                       location="", 
                       channel="HHZ", 
                       start_time="2024-01-01T00:06:00", 
                       end_time="2024-02-03T00:12:00"):
        """
        Initialize the seismogram analysis tool.

        Parameters:
            network (str): Network code.
            station (str): Station code.
            location (str): Location code (leave blank if unknown).
            channel (str): Channel code.
            start_time (str): Start time in ISO8601 format.
            end_time (str): End time in ISO8601 format.
        """
        self.network = network
        self.station = station
        self.location = location
        self.channel = channel
        self.start_time = UTCDateTime(start_time)
        self.end_time = UTCDateTime(end_time)
        self.f = None
        self.A = None
        self.B = None
        self.signal = None
        self.PDFs = None

    def read_MSEED(self):
        """
        Read waveform data from a FDSN web service and return the stream.

        Returns:
            Stream: ObsPy Stream object containing the waveform data.
        """
        # Set up the FDSN client (use 'ORFEUS' or 'IRIS' depending on data availability)
        client = Client("ORFEUS")

        # Fetch the waveform data
        try:
            self.signal = client.get_waveforms(self.network, self.station, self.location, self.channel, self.start_time, self.end_time)
            print("Data successfully retrieved!")
        except Exception as e:
            print(f"Error fetching data: {e}")
        
        return self.signal
    
    def compute_dft(self, signal):
        """
        Compute the DFT of a real signal and calculate A_k and B_k for positive frequencies.

        Parameters:
            signal (array): Input seismogram signal.

        Returns:
            frequencies (array): Positive frequency values.
            A (array): Vector of A_k coefficients.
            B (array): Vector of B_k coefficients.
        """
        N = len(signal)
        sampling_rate = 1  # Assuming the sampling rate is 1 Hz for simplicity

        # Perform DFT using numpy's FFT
        X_k = np.fft.rfft(signal)  # Real FFT for efficiency with real input signals

        # Compute real and imaginary parts
        Re_X_k = np.real(X_k)
        Im_X_k = np.imag(X_k)

        # Calculate A_k and B_k
        A = 2 * Re_X_k / N
        B = -2 * Im_X_k / N

        # Adjust A_0 for the DC component
        A[0] = Re_X_k[0] / N

        # Compute the corresponding frequency axis for positive frequencies
        frequencies = np.fft.rfftfreq(N, d=1/sampling_rate)

        return frequencies, A, B

    def reconstruct_signal(self, A, B, frequencies, t):
        """
        Reconstruct the signal from A_k and B_k using the Fourier series.

        Parameters:
            A (array): Vector of A_k coefficients.
            B (array): Vector of B_k coefficients.
            frequencies (array): Frequency values.
            t (array): Time values for reconstruction.

        Returns:
            reconstructed_signal (array): Reconstructed signal.
        """
        reconstructed_signal = np.zeros_like(t)
        for k in range(len(frequencies)):
            reconstructed_signal += A[k] * np.cos(2 * np.pi * frequencies[k] * t) + B[k] * np.sin(2 * np.pi * frequencies[k] * t)
        return reconstructed_signal

if __name__ == "__main__":
    # Example usage of the SeismogramAnalysis class
    analysis = SeismogramAnalysis()

    # Read data from the FDSN server
    st = analysis.read_MSEED()

    # Extract the first trace and signal data
    signal = st[0].data
    sampling_rate = st[0].stats.sampling_rate
    N = len(signal)

    # Compute the DFT and retrieve vectors A and B
    frequencies, A, B = analysis.compute_dft(signal)

    # Time array for reconstruction
    duration = N / sampling_rate
    t = np.linspace(0, duration, N, endpoint=False)

    # Reconstruct the signal
    reconstructed_signal = analysis.reconstruct_signal(A, B, frequencies, t)

    # Plot the original and reconstructed signals
    plt.figure(figsize=(12, 6))
    plt.plot(t, signal, label="Original Signal", color="blue")
    plt.plot(t, reconstructed_signal, label="Reconstructed Signal", color="red", linestyle="--")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Original and Reconstructed Signals")
    plt.legend()
    plt.show()

    # Plot the amplitude spectrum
    plt.figure(figsize=(12, 6))
    plt.semilogy(frequencies, np.abs(A), label="A_k", color="blue")
    plt.semilogy(frequencies, np.abs(B), label="B_k", color="red")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.title("Amplitude Spectrum")
    plt.legend()
    plt.show()
