import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


class RNN_GRU(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rnn_cell = nn.GRUCell(input_dim, hidden_dim)

    def forward(self, x, z):
        return self.rnn_cell(z, x)

    def compute_relative_position(self, X):
        if len(X.shape) == 2:
            return X[:, 0] * torch.sin(X[:, -1])
        elif len(X.shape) == 3:
            return X[:, :, 0] * torch.sin(X[:, :, -1])
        else:
            raise ValueError("Invalid input shape for compute_relative_position. Expected 2D or 3D tensor.")

    def process_image(self, image, x, meanlines, device="cpu"):
        _, W = image.shape
        hidden_dim = x.shape[1]
        N = len(meanlines)

        # Store hidden state at each k
        x_output = torch.zeros((W, N, hidden_dim), dtype=torch.float32, device=device)

        for k in range(W):
            with torch.no_grad():
                col = image[:, k]
                measurements = (np.where(col > 0.5)[0]).astype(float)
                centroids, _ = self.cluster_and_compute_stats(measurements, spacing=1)
                measurements = torch.tensor(centroids, dtype=torch.float32).to(device)
                M = len(centroids)

                pred_pos = self.compute_relative_position(x) + meanlines

                Z_assigned = pred_pos.clone()
                row_ind, col_ind = [], []
                if len(measurements) > 0:
                    cost_matrix = torch.abs(pred_pos[:, None] - measurements[None, :]).cpu().numpy()
                    row_ind, col_ind = linear_sum_assignment(cost_matrix)

                    for l, j in zip(row_ind, col_ind):
                        if l < len(meanlines) and j < len(measurements):
                            Z_assigned[l] = measurements[j]
            z_input = Z_assigned.unsqueeze(1).to(dtype=torch.float32, device=device)  # shape (N, 1)
            x = self.rnn_cell(z_input, x)  # shape: (N, hidden_dim)
            x_output[k] = x.clone()

        return x_output  # shape (W, N, hidden_dim)
        
    def train_step(self, images, GT, X_0, optimizer, meanlines, device="cpu"):
        # image shape: ([batch_size, 1, H, W])
        # ground_truths shape: ([batch_size, N_traces, W])
        # X_0.shape: ([hidden_dim]])
        # meanlines shape: ([N_traces])
        self.train()
        batch_size, _, H, W = images.shape
        total_loss = 0.0
        
        optimizer.zero_grad()
        meanlines = meanlines.to(dtype=torch.float32, device=device)
        X_0 = X_0.unsqueeze(0).repeat(len(meanlines), 1).to(dtype=torch.float32, device=device) # X_0 shape: ([N_traces, hidden_dim])

        for i in range(batch_size):
            image = images[i, 0].to(dtype=torch.float32, device=device)
            true = GT[i].T.to(dtype=torch.float32, device=device) # true shape: ([N_traces, W])

            X = self.process_image(image, X_0, meanlines, device=device) # X shape: ([W, N_traces, hidden_dim])
            preds = self.compute_relative_position(X) + meanlines # preds shape: ([N_traces, W])
            
            per_curve_losses = torch.sqrt(((preds - true) ** 2).sum(dim=0)) / preds.shape[1]
            loss = per_curve_losses.mean()
            total_loss += loss

        total_loss.backward()
        optimizer.step()
        return (total_loss / batch_size).item()
    
    def eval_loss(self, dataloader, X_0, meanlines, device="cpu"):
        self.eval()
        total_loss = 0.0
        num_batches = 0
        
        meanlines = meanlines.to(dtype=torch.float32, device=device)
        X_0 = X_0.unsqueeze(0).repeat(len(meanlines), 1).to(dtype=torch.float32, device=device) # X_0 shape: ([N_traces, hidden_dim])

        with torch.no_grad():
            for images, ground_truths in dataloader:
                batch_size, _, H, W = images.shape
                for i in range(batch_size):
                    image = images[i, 0].to(dtype=torch.float32, device=device)
                    true = ground_truths[i].T.to(dtype=torch.float32, device=device)
                    X = self.process_image(image, X_0, meanlines, device=device)
                    preds = self.compute_relative_position(X) + meanlines
                    per_curve_losses = torch.sqrt(((preds - true) ** 2).sum(dim=0)) / preds.shape[1]
                    loss = per_curve_losses.mean()
                    total_loss += loss
                    num_batches += 1

        return total_loss / num_batches

    @staticmethod
    def cluster_and_compute_stats(measurements, spacing=1):
        if len(measurements) == 0:
            return np.array([]), np.array([])
        measurements = np.sort(measurements)
        cluster_splits = np.where(np.diff(measurements) > spacing)[0] + 1
        clusters = np.split(measurements, cluster_splits)
        centroids = np.array([np.mean(cluster) for cluster in clusters])
        stds = np.array([np.std(cluster) if len(cluster) > 1 else 1.0 for cluster in clusters])
        return centroids, stds

def train_rnn(model, dataloader, eval_loader, optimizer, meanlines, X_0, device="cpu", epochs=10):
    model.train()
    train_losses = []
    eval_losses = []

    for epoch in range(epochs):
        batch_losses = []
        for images, ground_truths in dataloader:
            # image shape: ([batch_size, 1, H, W])
            # ground_truths shape: ([batch_size, N_traces, W])
            # X_0.shape: ([hidden_dim]])
            # meanlines shape: ([N_traces])
            loss = model.train_step(images, ground_truths, X_0, optimizer, meanlines, device=device)
            batch_losses.append(loss)
        epoch_loss = np.mean(batch_losses)
        train_losses.append(epoch_loss)

        eval_loss = model.eval_loss(eval_loader, X_0, meanlines, device=device)
        eval_losses.append(eval_loss)

        print(f"Epoch {epoch}: Train Loss = {epoch_loss:.4f}, Eval Loss = {eval_loss:.4f}")

    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(eval_losses, label="Eval Loss")
    plt.title("Learning Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return train_losses, eval_losses