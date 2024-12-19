from obspy import UTCDateTime
from obspy.clients.fdsn import Client
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

class SeismogramAnalysis:
    """
    A class for performing seismogram analysis including waveform data retrieval,
    Fourier transform, signal reconstruction, and probability distribution estimation
    of Fourier coefficients using Kernel Density Estimation (KDE).

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
        PDFs (dict): Dictionary containing KDE results for A_k and B_k.
    """

    def __init__(self, network="BE", 
                       station="UCC", 
                       location="", 
                       channel="HHZ", 
                       start_time="2024-01-01T00:06:00", 
                       end_time="2024-02-03T00:12:00", 
                       batch_length=1000):
        """
        Initialize the seismogram analysis tool.

        Parameters:
            network (str): Network code.
            station (str): Station code.
            location (str): Location code (leave blank if unknown).
            channel (str): Channel code.
            start_time (str): Start time in ISO8601 format.
            end_time (str): End time in ISO8601 format.
            batch_length (int): Number of samples in each batch for processing.
        """
        self.network = network
        self.station = station
        self.location = location
        self.channel = channel
        self.start_time = UTCDateTime(start_time)
        self.end_time = UTCDateTime(end_time)
        self.batch_length = batch_length
        self.signal = None
        self.A = None
        self.B = None
        self.PDFs = {'A': None, 'B': None}

    def read_MSEED_batch(self, start_time, end_time):
        """
        Read a batch of waveform data from a FDSN web service.

        Parameters:
            start_time (UTCDateTime): Start time for the batch.
            end_time (UTCDateTime): End time for the batch.

        Returns:
            Stream: ObsPy Stream object containing the waveform data for the batch.
        """
        client = Client("ORFEUS")

        try:
            # Retrieve the batch data from the server
            signal = client.get_waveforms(self.network, self.station, self.location, self.channel, start_time, end_time)
            print(f"Data successfully retrieved for {start_time} to {end_time}!")
        except Exception as e:
            print(f"Error fetching data for {start_time} to {end_time}: {e}")
            signal = None
        
        return signal
    
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

    def compute_pdf(self, A, B):
        """
        Compute the PDF of A_k and B_k using Kernel Density Estimation (KDE).

        Parameters:
            A (array): Vector of A_k coefficients.
            B (array): Vector of B_k coefficients.
        """
        # KDE for A_k and B_k using sklearn's KernelDensity
        kde_A = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(A[:, np.newaxis])
        kde_B = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(B[:, np.newaxis])

        # Store the KDE estimators in the PDFs attribute
        self.PDFs['A'] = kde_A
        self.PDFs['B'] = kde_B

    def process_batches(self):
        """
        Process the seismogram in batches, compute the DFT for each batch, 
        and update the PDFs of A_k and B_k.
        """
        num_samples = int((self.end_time - self.start_time) * 86400)  # Convert time difference to seconds
        batch_start = self.start_time
        batch_A = []
        batch_B = []

        # Iterate over the batches
        while batch_start < self.end_time:
            batch_end = min(batch_start + self.batch_length / 86400, self.end_time)  # Convert batch length to days

            # Fetch the data for the current batch
            batch_signal = self.read_MSEED_batch(batch_start, batch_end)
            if batch_signal is None:
                break

            # Extract the signal from the first trace
            signal = batch_signal[0].data

            # Compute the DFT for the current batch
            frequencies, A, B = self.compute_dft(signal)

            # Append the coefficients to the batch lists
            batch_A.extend(A)
            batch_B.extend(B)

            # Move to the next batch
            batch_start = batch_end

        # After processing all batches, compute the PDFs for A_k and B_k
        self.compute_pdf(np.array(batch_A), np.array(batch_B))

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
    analysis = SeismogramAnalysis(batch_length=1000)

    # Process the seismogram in batches and compute the PDFs
    analysis.process_batches()

    # Print the PDFs of A_k and B_k
    print("PDFs of A_k:")
    print(analysis.PDFs['A'])

    print("PDFs of B_k:")
    print(analysis.PDFs['B'])

    # Optionally, visualize the KDE results for A_k and B_k
    if analysis.PDFs['A'] is not None and analysis.PDFs['B'] is not None:
        A_values = np.linspace(np.min(np.array(analysis.PDFs['A'].sample(1000))), np.max(np.array(analysis.PDFs['A'].sample(1000))), 1000)
        B_values = np.linspace(np.min(np.array(analysis.PDFs['B'].sample(1000))), np.max(np.array(analysis.PDFs['B'].sample(1000))), 1000)

        plt.figure(figsize=(12, 6))
        plt.plot(A_values, np.exp(analysis.PDFs['A'].score_samples(A_values[:, np.newaxis])), label="PDF of A_k", color="blue")
        plt.plot(B_values, np.exp(analysis.PDFs['B'].score_samples(B_values[:, np.newaxis])), label="PDF of B_k", color="red")
        plt.xlabel("Amplitude")
        plt.ylabel("Probability Density")
        plt.title("PDF of Fourier Coefficients (A_k and B_k)")
        plt.legend()
        plt.show()