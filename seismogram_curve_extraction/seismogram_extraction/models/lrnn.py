import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import linear_sum_assignment

class LRNN(nn.Module):
    def __init__(self, state_dim, obs_dim):
        super(LRNN, self).__init__()
        self.K = nn.Parameter(torch.randn(state_dim, obs_dim))  # Learnable Kalman-like gain
        self.A = nn.Parameter(torch.eye(state_dim))  # Learnable state transition
        self.H = nn.Parameter(torch.randn(obs_dim, state_dim))  # Learnable observation matrix

    def forward(self, X_pred, Z):
        """
        X_pred: (batch, state_dim) - Predicted state
        Z: (batch, obs_dim) - Observed measurement
        """
        innovation = Z - (self.H @ X_pred.T).T  # Innovation term (measurement residual)
        X_updated = X_pred + (self.K @ innovation.T).T  # LRNN update step
        return X_updated

class HungarianLRNN:
    def __init__(self, state_dim, obs_dim):
        self.model = LRNN(state_dim, obs_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.loss_fn = nn.MSELoss()

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

    def process_sequence(self, sequence, X_0):
        """
        Processes a sequence using LRNN, similar to the Kalman filter.

        Args:
            sequence: (batch, timesteps, obs_dim) - Observed noisy measurements.
            X_0: (batch, state_dim) - Initial state estimates.

        Returns:
            X_results: (batch, timesteps, state_dim) - Updated state estimates.
        """
        batch_size, timesteps, _ = sequence.shape
        X_results = torch.zeros(batch_size, timesteps, X_0.shape[1])
        X_pred = X_0

        for t in range(timesteps):
            Z_t = sequence[:, t, :]
            X_updated = self.model(X_pred, Z_t)
            X_results[:, t, :] = X_updated
            X_pred = X_updated  # Use updated state as next prediction

        return X_results