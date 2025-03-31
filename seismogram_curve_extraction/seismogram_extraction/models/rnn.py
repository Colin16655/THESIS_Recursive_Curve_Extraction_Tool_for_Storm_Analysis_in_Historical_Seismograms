import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import linear_sum_assignment

class model_RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.image_height = image_height
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Using batch_first=True, so input and output are (batch, seq_len, feature)
        self.rnn = nn.RNN(input_size, hidden_size, bias=True, batch_first=True, nonlinearity='tanh')
        self.output_layer = nn.Linear(hidden_size, output_size, bias=True)
    
    def forward(self, measurement, hidden):
        # measurement: expected shape (batch, input_size)
        # Unsqueeze to get shape (batch, 1, input_size)
        if measurement.dim() == 2:
            measurement = measurement.unsqueeze(1)
        output_seq, hidden_new = self.rnn(measurement, hidden)
        # Use the last hidden state for prediction; shape: (batch, hidden_size)
        prediction = self.output_layer(hidden_new[-1])
        return prediction, hidden_new

    def init_hidden(self, batch_size=1):
        # Hidden shape: (num_layers, batch, hidden_size)
        return torch.zeros(1, batch_size, self.hidden_size)

class RNN:
    def __init__(self, state_dim, obs_dim, max_gap=20, lr=0.01, lambda_smooth=0.1, epochs=100, degree=2):
        self.model = model_RNN(state_dim, obs_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.max_gap = max_gap
        self.lr = lr
        self.lambda_smooth = lambda_smooth
        self.epochs = epochs
        self.degree = degree

    def train(self, X_train, Z_train, epochs=100):
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            X_pred = X_train  # Initial prediction
            X_updated = self.model(X_pred, Z_train)
            loss = self.loss_fn(X_updated, X_train)  # Compare updated state with ground truth
            loss.backward()
            self.optimizer.step()
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item()}")

    def process_sequence(self, sequence, X_0, P_0=None):
        """
        Processes a sequence of inputs using LRNN.

        Note that max_gap (int) is the maximum gap between predicted and measured positions to consider them as the same object.

        Args:
            sequence (np.ndarray): Shape (batch_size, 1, height, width) - Sequence of batch_size-dimensional inputs.
            X_0 (np.ndarray): Shape (batch_size, N_traces, N_states) - Initial state estimates for N_states traces.

        Returns:
            X_results (np.ndarray): Shape (batch_size, width, N_traces, N_states) - Updated state estimates for each time step (width).
        """

        # invert all images
        if sequence.max() != 1: print("Warning: Sequence values are not binary.")
        sequence = sequence.max() - sequence

        # get parameters
        avg_N_components = X_0.shape[1]

        seq_length = sequence.shape[-1]
        batch_size = sequence.shape[0]

        input_size = avg_N_components
        hidden_size = self.degree * avg_N_components
        output_size = self.degree * avg_N_components

        X_results = torch.full((sequence.shape[0], sequence.shape[-1], X_0.shape[1], X_0.shape[2]), float('nan'))
        X_results[:, 0, :, :] = X_0
        X_pred = X_0

        
        tresh = 0.1
        
        for t in range(1, sequence.shape[-1]):
            Z_t = sequence[:, :, :, t]          # (batch_size, obs_dim)
            X_updated = self.model(X_pred, Z_t) # (batch_size, state_dim)
            X_results[:, t, :, :] = X_updated
            X_pred = X_updated  # Use updated state as next prediction

        return X_results
    
    @staticmethod
    def compute_cost_matrix(predicted_positions, measurements):
        return np.abs(predicted_positions[:, None] - measurements[None, :])

    @staticmethod
    def find_contiguous_subsets(positions, n=2):
        positions = np.sort(np.unique(positions))  # Ensure sorted unique positions
        subsets = []
        start = positions[0]
        prev = start
        
        for pos in positions[1:]:
            if pos - prev >= n:
                subsets.append((start, prev))
                start = pos
            prev = pos
        subsets.append((start, prev))  # Append last subset
        
        return subsets

    @staticmethod
    def compute_measurements(positions):
        subsets = RNN.find_contiguous_subsets(positions)
        measurements = []
        std_devs = []
        
        for start, stop in subsets:
            subset = np.arange(start, stop + 1)
            mean_val = np.mean(subset)
            std_val = np.std(subset)
            measurements.append(mean_val)
            std_devs.append(std_val)
        
        return measurements, std_devs