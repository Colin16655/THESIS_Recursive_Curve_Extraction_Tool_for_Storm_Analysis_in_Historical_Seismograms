from obspy import UTCDateTime
from obspy.clients.fdsn import Client
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import KFold
from tqdm import tqdm
import os
from obspy import read
import pickle

def sanitize_filename(name):
            """Sanitize a filename by replacing invalid characters."""
            return name.replace(":", "-").replace(".", "-").replace(" ", "_")

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
                       channel="HHE", 
                       start_time="2024-01-01T00:06:00", 
                       end_time="2024-01-14T00:12:00", 
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
        self.frequencies = None
        self.batch_A = None
        self.batch_B = None
        self.PDFs = {'A': None, 'B': None}
        self.init = False
        self.bandwidth_0 = None
        self.bandwidth = None

    def read_MSEED_batch(self, start_time, end_time, folder=r"seismogram_curve_extraction\data\mseed_files", mute=False, plot_batch=False):
        """
        Read a batch of waveform data from a local file or an FDSN web service.

        Parameters:
            start_time (UTCDateTime): Start time for the batch.
            end_time (UTCDateTime): End time for the batch.
            folder (str): Directory where MSEED files are stored.
            mute (bool): If True, suppress messages.

        Returns:
            Stream: ObsPy Stream object containing the waveform data for the batch.
        """
        # Ensure the folder exists
        if not os.path.exists(folder):
            os.makedirs(folder)
            if not mute:
                print(f"Created folder: {folder}")
        
        # Define the full file path
        filename = sanitize_filename(f"{self.station}_{self.channel}_{start_time}_{end_time}.mseed")
        filepath = os.path.join(folder, filename)
        
        # Check if the file exists
        if os.path.exists(filepath):
            if not mute:
                print(f"Loading data from local file: {filepath}")
            try:
                # Read the MSEED file
                signal = read(filepath)
            except Exception as e:
                print(f"Error reading local file {filepath}: {e}")
                signal = None
        else:
            # Fetch the data from the server
            client = Client("IRIS")
            try:
                if not mute:
                    print(f"Fetching data from server for {start_time} to {end_time}...")
                signal = client.get_waveforms(network=self.network, 
                                              station=self.station, 
                                              location=self.location,
                                              channel=self.channel, 
                                              starttime=start_time, 
                                              endtime=end_time)
                # Save the data locally for future use
                signal.write(filepath, format="MSEED")
                if not mute:
                    print(f"Data saved to local file: {filepath}")
            except Exception as e:
                print(f"Error fetching data for {start_time} to {end_time}: {e}")
                signal = None

        # plot the signal
        if plot_batch:
            signal.plot()
        
        return signal
    
    def compute_dft(self, dt, signal, plot=False):
        """
        Compute the DFT of a real signal and calculate A_k and B_k for positive frequencies.

        Parameters:
            dt (float): Sampling rate of the signal.
            signal (array): Input seismogram signal.

        Returns:
            frequencies (array): Positive frequency values.
            A (array): Vector of A_k coefficients.
            B (array): Vector of B_k coefficients.
        """
        N = len(signal)

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
        self.frequencies = np.fft.rfftfreq(N, d=dt)

        if plot:
            # Plot the DFT results
            plt.figure(figsize=(12, 6))
            plt.plot(self.frequencies, A, label="A_k", color="blue")
            plt.plot(self.frequencies, B, label="B_k", color="red")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Amplitude")
            plt.title("Fourier Coefficients (A_k and B_k)")
            plt.legend(loc="upper right")
            plt.grid()
            plt.show()

        return self.frequencies, A, B
    
    def evaluate_bandwidth(self, data, bandwidth, kf):
        """
        Evaluate a specific bandwidth using cross-validation.
        
        Parameters:
            data (array): The data to fit the KDE on.
            bandwidth (float): The bandwidth value to test.
            kf (KFold): The KFold cross-validation iterator.
            
        Returns:
            (float, float): Average log-likelihood score for the cross-validation for A_0 and for A_k, k = 1, ..., N-1.
        """
        score0 = []
        scores = []
        for train_idx, test_idx in kf.split(data.T):  # We need to split across columns (batches)
            train_data, test_data = data[:, train_idx], data[:, test_idx]

            # Fit KDE on training data (reshape to 2D)
            PDF0 = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(train_data[0, :][:, np.newaxis])
            PDFs = [KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(train_data[k, :][:, np.newaxis]) for k in range(1, train_data.shape[0])]

            # kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
            # PDFs = [kde.fit(train_data[i, :][:, np.newaxis]) for i in range(train_data.shape[0])]
            # kde.fit(train_data.T)  # Fit using the transposed data to have samples in rows

            # Compute log-likelihood on test data
            log_likelihood0 = PDF0.score(test_data[0, :][:, np.newaxis])
            log_likelihoods = [PDFs[k].score(test_data[k, :][:, np.newaxis]) for k in range(1, train_data.shape[0])]
            score0.append(log_likelihood0)
            scores.append(np.mean(log_likelihoods))
        
        return np.mean(score0), np.mean(scores)  # Averages log-likelihood over folds

    def compute_pdf(self, A, B, bandwidth_0=None, bandwidth=None):
        """
        Compute the PDF of A_k and B_k using Kernel Density Estimation (KDE), assuming that the A_k and B_k are statistically independent.
        
        Parameters:
            A (array): Matrix of A_k coefficients (num_frequencies x num_batches).
            B (array): Matrix of B_k coefficients (num_frequencies x num_batches).
            bandwidth (float or None): If None, bandwidth will be selected using 5-folds cross-validation.
        
        Returns:
            dict: PDFs for A_k and B_k coefficients.
        """
        if bandwidth is None:
            # If no bandwidth is provided, perform cross-validation to select the best one
            bandwidth_range = np.logspace(-2, 1, 10)  # Example range from 0.01 to 10
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            best_bandwidth_A0, best_bandwidth_B0 = None, None
            best_score_A0, best_score_B0 = -np.inf, -np.inf
            best_bandwidth_A, best_bandwidth_B = None, None
            best_score_A, best_score_B = -np.inf, -np.inf
            
            # Cross-validation for A
            for bw in bandwidth_range:
                score_A0, score_A = self.evaluate_bandwidth(A, bw, kf)
                print(f"Bandwidth {bw} for A_0: Log-Likelihood: {score_A0}")
                print(f"Bandwidth {bw} for A: Log-Likelihood: {score_A}")
                if score_A > best_score_A:
                    best_score_A = score_A
                    best_bandwidth_A = bw
                if score_A0 > best_score_A0:
                    best_score_A0 = score_A0
                    best_bandwidth_A0 = bw
            
            # Cross-validation for B
            for bw in bandwidth_range:
                score_B0, score_B = self.evaluate_bandwidth(B, bw, kf)
                print(f"Bandwidth {bw} for B_0: Log-Likelihood: {score_B0}")
                print(f"Bandwidth {bw} for B: Log-Likelihood: {score_B}")
                if score_B > best_score_B:
                    best_score_B = score_B
                    best_bandwidth_B = bw
                if score_B0 > best_score_B0:
                    best_score_B0 = score_B0
                    best_bandwidth_B0 = bw

            # Set the best bandwidth for both A and B
            self.bandwidth_0 = (best_bandwidth_A0 + best_bandwidth_B0) / 2
            self.bandwidth = (best_bandwidth_A + best_bandwidth_B) / 2
            print(f"Best bandwidth for A and B: {self.bandwidth} (average of {best_bandwidth_A} and {best_bandwidth_B})")

        else:
            self.bandwidth_0 = bandwidth_0 
            self.bandwidth = bandwidth

        # Store the KDE estimators in the PDFs attribute using the selected bandwidth
        print('o')
        self.PDFs['A'] = [KernelDensity(kernel='gaussian', bandwidth=self.bandwidth_0).fit(A[0, :][:, np.newaxis])]
        self.PDFs['A'].extend([KernelDensity(kernel='gaussian', bandwidth=self.bandwidth).fit(A[k, :][:, np.newaxis]) for k in range(1, A.shape[0])]) # newaxis is used to add a new dimension to the array : N => ((N, 1))
        self.PDFs['B'] = [KernelDensity(kernel='gaussian', bandwidth=self.bandwidth_0).fit(B[0, :][:, np.newaxis])]
        self.PDFs['B'].extend([KernelDensity(kernel='gaussian', bandwidth=self.bandwidth).fit(B[k, :][:, np.newaxis]) for k in range(1 ,B.shape[0])])

        # plot a couple of PDFs
        for k in [0, len(self.frequencies)//2, len(self.frequencies)-1]:
            A_k_vals = np.linspace(np.min(A[k, :])*0.9, np.max(A[k, :])*1.1, 10000)
            log_density = self.PDFs['A'][k].score_samples(A_k_vals[:, np.newaxis])
            pdf = np.exp(log_density)
            # normalize pdf
            pdf /= np.trapz(pdf, A_k_vals)

            fig, ax = plt.subplots(1, 1, figsize=(4, 3))
            ax.plot(A_k_vals, pdf, label="KDE PDF", color="blue")
            ax.hist(A[k, :], bins=30, density=True, alpha=0.5, color="gray", label="Histogram")
            fig.suptitle(f"Kernel Density Estimation (KDE) for A_k")
            ax.set_xlabel("x")
            ax.set_ylabel("Density")
            ax.legend()
            plt.show()

        print("oo")
        return self.PDFs

    def process_batches(self, downsampling_fac=10, bandwidth_0=None, bandwidth=None):
        """
        Process the seismogram in batches, compute the DFT for each batch, 
        and update the PDFs of A_k and B_k.

        Parameters: 
            downsampling_fac (int): Downsampling factor to reduce the time complexity.
        """
        batch_start = self.start_time
        num_batches = int((self.end_time - self.start_time) / self.batch_length)

        batch_end = batch_start + self.batch_length

        # Fetch the data for the current batch
        batch_signal = self.read_MSEED_batch(batch_start, batch_end)
        if batch_signal is None:
            raise ValueError("Error fetching data for the first batch!")

        # Extract the signal from the first trace
        signal = batch_signal[0].data
        dt = batch_signal[0].stats.delta * downsampling_fac
        # print("dt = ", batch_signal[0].stats.delta) # 0.01 s: 100 Hz
        # Downsampling
        signal = signal[::downsampling_fac]

        self.batch_A = np.zeros((len(signal)//2+1, num_batches))
        self.batch_B = np.zeros((len(signal)//2+1, num_batches))

        # Iterate over the batches
        for i in tqdm(range(num_batches), desc="Processing batches"):    
            # Compute the DFT for the current batch
            self.frequencies, A, B = self.compute_dft(dt, signal)

            # Append the coefficients to the batch lists
            self.batch_A[:, i] = A
            self.batch_B[:, i] = B

            # Check if reconstruct signal matches the original signal
            # reconstructed_signal = self.reconstruct_signal(A, B, self.frequencies, np.arange(len(signal)) * dt)
            # plt.plot(np.arange(len(signal)) * dt, reconstructed_signal, label="Reconstructed signal", color="red")
            # plt.plot(np.arange(len(signal)) * dt, signal, linestyle='--', label="Original signal", color="blue")
            # plt.legend()
            # plt.show()

            # Move to the next batch
            batch_start = batch_end
            batch_end = batch_start + self.batch_length

            # Fetch the data for the current batch
            batch_signal = self.read_MSEED_batch(batch_start, batch_end)
            if batch_signal is None:
                raise ValueError(f"Error fetching data for the {i}th batch!")

            # Extract the signal from the first trace
            signal = batch_signal[0].data# Downsampling
            signal = signal[::downsampling_fac]

        norm = np.zeros((2, len(self.frequencies)))
        for k in range(len(self.frequencies)):
            norm[0, k] = np.linalg.norm(self.batch_A[k, :])
            norm[1, k] = np.linalg.norm(self.batch_B[k, :])
        plt.semilogy(self.frequencies, norm[0, :]+(1e-16), label="A_k")
        plt.semilogy(self.frequencies, norm[1, :]+(1e-16), label="B_k")
        plt.legend()
        plt.show()

        plt.hist(self.batch_A[0, :], bins=30, density=True, alpha=0.5, color="gray", label="Histogram")
        plt.show()
        plt.hist(self.batch_A[10000, :], bins=30, density=True, alpha=0.5, color="gray", label="Histogram")
        plt.show()
        plt.hist(self.batch_A[-1, :], bins=30, density=True, alpha=0.5, color="gray", label="Histogram")
        plt.show()

        # After processing all batches, compute the PDFs for A_k and B_k
        return self.compute_pdf(np.array(self.batch_A), np.array(self.batch_B), bandwidth_0=bandwidth_0, bandwidth=bandwidth)

    def reconstruct_signal(self, A, B, frequencies, t, n_samples=1):
        """
        Reconstruct the signal from A_k and B_k using the Fourier series.

        Parameters:
            A (array): Matrix of A_k coefficients with shape (N, n_samples).
            B (array): Matrix of B_k coefficients with shape (N, n_samples).
            frequencies (array): Matrix of frequency values with shape (N, n_samples).
            t (array): Time values for reconstruction.
            n_samples (int): Number of samples.

        Returns:
            reconstructed_signal (array): Reconstructed signal with shape (n_samples, len(t)).
        """
        if n_samples == 1:
            reconstructed_signal = np.zeros_like(t)
            for k in range(len(frequencies)):
                reconstructed_signal += A[k] * np.cos(2 * np.pi * frequencies[k] * t) + B[k] * np.sin(2 * np.pi * frequencies[k] * t)
        
        else:
            # Ensure the output has shape (n_samples, len(t))
            reconstructed_signal = np.zeros((n_samples, len(t)))
            
            for k in tqdm(range(A.shape[0]), desc="Reconstructing signal"):
                reconstructed_signal += (A[k, :, np.newaxis] * np.cos(2 * np.pi * frequencies[k, np.newaxis] * t) +
                                        B[k, :, np.newaxis] * np.sin(2 * np.pi * frequencies[k, np.newaxis] * t))

        return reconstructed_signal  # Shape: (n_samples, len(t)) or (len(t),) if n_samples=1

    
    def save_analysis(self, filepath):
        """
        Save the entire SeismogramAnalysis object to a file.

        Parameters:
            filepath (str): Path to save the object.
        """
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, "wb") as file:
            pickle.dump(self, file)
        print(f"Analysis object saved to {filepath}.")

    @staticmethod
    def load_analysis(filepath):
        """
        Load a SeismogramAnalysis object from a file.

        Parameters:
            filepath (str): Path to load the object from.

        Returns:
            SeismogramAnalysis: The loaded analysis object.
        """
        with open(filepath, "rb") as file:
            analysis = pickle.load(file)
        print(f"Analysis object loaded from {filepath}.")
        return analysis
    

if __name__ == "__main__": 
    # Set rcParams to customize tick labels and spines
    plt.rcParams['xtick.labelsize'] = 11
    plt.rcParams['ytick.labelsize'] = 11
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.grid'] = True
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

        analysis = SeismogramAnalysis.load_analysis(filepath)
    else:
        print(f"File {filepath} not found. Computing PDFs...")

        analysis = SeismogramAnalysis(batch_length=int(86400/4)) # 86400 for 1 day batch length
        # Process the seismogram in batches and compute the PDFs
        analysis.process_batches(bandwidth_0=bandwidth_0, bandwidth=bandwidth)
        
        # Save the results for future use
        analysis.save_analysis(filepath)    

    # Plot the PDFs
    folder = r"seismogram_curve_extraction\results\plot\PDFs"
    # Ensure the directory exists
    os.makedirs(folder, exist_ok=True)
    for k in [0, 1, 50, len(analysis.frequencies)//2, len(analysis.frequencies)-1]:
        # Plot the PDF for A_k
        LO = np.min(analysis.batch_A[k, :])
        if LO < 0: LO *= 1.1
        else: LO *= 0.9

        UP = np.max(analysis.batch_A[k, :])
        if UP < 0: UP *= 0.9
        else: UP *= 1.1

        A_k_vals = np.linspace(LO, UP, 10000)
        log_density = analysis.PDFs['A'][k].score_samples(A_k_vals[:, np.newaxis])
        pdf = np.exp(log_density)

        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        ax.plot(A_k_vals, pdf, label="KDE PDF", color="blue")
        ax.hist(analysis.batch_A[k, :], bins=30, density=True, alpha=0.5, color="gray", label="Histogram")
        fig.suptitle(f"$A_{{{k}}}$, $f_{{{k}}} = {analysis.frequencies[k]:.5f}$ Hz")
        ax.set_xlabel("x")
        ax.set_ylabel("Density")
        ax.legend()
        # plt.show()
        
        fig.savefig(folder + f"/A_{k}_bandwidth_{analysis.bandwidth}.pdf", bbox_inches='tight', format='pdf', dpi=300)

        # Plot the PDF for B_k
        LO = np.min(analysis.batch_B[k, :])
        if LO < 0: LO *= 1.1
        else: LO *= 0.9

        UP = np.max(analysis.batch_B[k, :])
        if UP < 0: UP *= 0.9
        else: UP *= 1.1

        B_k_vals = np.linspace(LO, UP, 10000)
        log_density = analysis.PDFs['B'][k].score_samples(B_k_vals[:, np.newaxis])
        pdf = np.exp(log_density)

        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        ax.plot(B_k_vals, pdf, label="KDE PDF", color="red")
        ax.hist(analysis.batch_B[k, :], bins=30, density=True, alpha=0.5, color="gray", label="Histogram")
        fig.suptitle(f"$B_{{{k}}}$, $f_{{{k}}} = {analysis.frequencies[k]:.5f}$ Hz")

        ax.set_xlabel("x")
        ax.set_ylabel("Density")
        ax.legend()
        # plt.show()

        fig.savefig(folder + f"/B_{k}_bandwidth_{analysis.bandwidth}.pdf", bbox_inches='tight', format='pdf', dpi=300)


