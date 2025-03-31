import os
import numpy as np
import torch
import cv2
from torch.utils.data import DataLoader, random_split
import sys
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

# Get the path of the project root
project_root = os.path.abspath(os.path.join(os.path.dirname('__file__'), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import the ImageSequenceDataset and create_dataloader from the pipeline module.
from seismogram_extraction.pipeline import ImageSequenceDataset, create_dataloader

# -------------------------------
# RNN Model Definition
# -------------------------------
class RNN(torch.nn.Module):
    def __init__(self, image_height, hidden_size, output_size):
        """
        image_height: number of pixels in an image column.
        hidden_size: dimension of the RNN hidden state.
        output_size: number of curves (i.e. predicted positions per time step).
        """
        super(RNN, self).__init__()
        self.image_height = image_height
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Feature extractor: maps an image column (vector of size image_height)
        # to a measurement vector of size output_size (typically the number of curves)
        self.feature_extractor = torch.nn.Linear(image_height, output_size)
        
        # RNN layer: processes a sequence of extracted features.
        # With batch_first=True, the expected input shape is (batch, seq_len, output_size)
        self.rnn = torch.nn.RNN(output_size, hidden_size, bias=True, batch_first=True, nonlinearity='tanh')
        
        # Output layer: maps the hidden state to the predicted curve positions.
        self.output_layer = torch.nn.Linear(hidden_size, output_size)

    def forward(self, measurement, hidden):
        """
        measurement: tensor of shape (batch, output_size); a single time step input.
        hidden: hidden state of shape (num_layers, batch, hidden_size).
        """
        # Ensure a sequence dimension exists (i.e. time step length of 1)
        if measurement.dim() == 2:
            measurement = measurement.unsqueeze(1)  # Now: (batch, 1, output_size)
        output_seq, hidden_new = self.rnn(measurement, hidden)
        prediction = self.output_layer(hidden_new[-1])  # Use last hidden state.
        return prediction, hidden_new

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)

# -------------------------------
# Main Training and Evaluation Loop
# -------------------------------
def main(image_folder, gt_folder, batch_size=4, num_epochs=250, lr=0.01, lambda_smooth=0.1):
    # Create the full dataset using your pipeline dataset.
    dataset = ImageSequenceDataset(image_folder, gt_folder)
    # Split the dataset into training (80%) and testing (20%) sets.
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Grab a sample batch to determine image dimensions and number of curves.
    sample_batch = next(iter(train_loader))
    images, ground_truths = sample_batch  # images: (batch, 1, height, width); ground_truths: (batch, N_curves, width)
    images = images.float()
    ground_truths = ground_truths.float()
    
    image_height = images.shape[2]
    image_width  = images.shape[3]
    N_curves     = ground_truths.shape[1]
    
    # Define model parameters.
    hidden_size = 2 * N_curves
    output_size = N_curves  # predicting positions for each curve.
    
    # Instantiate the model, loss function, and optimizer.
    model = RNN(image_height, hidden_size, output_size)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Hungarian assignment matrix: used to map the hidden state to predicted positions.
    pred_mat = torch.zeros((N_curves, hidden_size))
    pred_mat[:N_curves, :N_curves] = torch.eye(N_curves) * 2
    pred_mat[:, N_curves:] = -torch.eye(N_curves)
    
    train_losses = []
    test_losses = []
    test_smooth_losses = []
    
    # Training loop.
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        
        for batch_idx, (images, ground_truths) in enumerate(train_loader):
            # Ensure inputs are float32.
            images = images.float()
            ground_truths = ground_truths.float()
            
            current_batch_size = images.size(0)
            
            # Initialize the hidden state using the first column of the image.
            first_col = images[:, 0, :, 0]  # (batch, image_height)
            first_measurement = model.feature_extractor(first_col)  # (batch, N_curves)
            hidden = model.init_hidden(current_batch_size)
            hidden[0, :, :N_curves] = first_measurement
            hidden[0, :, N_curves:] = first_measurement
            
            optimizer.zero_grad()
            predicted_positions = []
            seq_len = images.shape[3]  # number of columns (time steps)
            
            # Process each image column as a time step.
            for t in range(seq_len):
                img_col = images[:, 0, :, t]  # shape: (batch, image_height)
                measurement_feature = model.feature_extractor(img_col)  # (batch, N_curves)
                
                # For each sample in the batch, reorder the measurement using Hungarian assignment.
                assigned_measurements = []
                for b in range(current_batch_size):
                    with torch.no_grad():
                        hidden_b = hidden[0, b]  # (hidden_size,)
                        predicted_state = (pred_mat @ hidden_b.unsqueeze(1)).cpu().numpy()  # (N_curves, 1)
                        meas_b = measurement_feature[b].unsqueeze(1).cpu().numpy()  # (N_curves, 1)
                        cost_matrix = np.abs(predicted_state - meas_b)
                        row_ind, col_ind = linear_sum_assignment(cost_matrix)
                        assigned = np.zeros(N_curves)
                        for r, c in zip(row_ind, col_ind):
                            assigned[r] = measurement_feature[b, c].item()
                        assigned_measurements.append(torch.tensor(assigned, dtype=torch.float32))
                assigned_measurement_batch = torch.stack(assigned_measurements, dim=0)
                
                # Forward pass through the RNN.
                prediction, hidden = model(assigned_measurement_batch, hidden)
                predicted_positions.append(prediction)
            
            # Stack predictions to form a sequence: (batch, seq_len, N_curves)
            predicted_seq = torch.stack(predicted_positions, dim=1)
            # Rearrange ground truths from (batch, N_curves, width) to (batch, width, N_curves)
            target_seq = ground_truths.permute(0, 2, 1)
            
            mse_loss = criterion(predicted_seq, target_seq)
            # Compute smoothness loss to penalize large differences between adjacent time steps.
            diff = predicted_seq[:, 1:, :] - predicted_seq[:, :-1, :]
            smoothness_loss = torch.mean(diff ** 2)
            loss = mse_loss + lambda_smooth * smoothness_loss
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # -------------------------------
        # Evaluation on the Test Set
        # -------------------------------
        model.eval()
        total_test_mse = 0.0
        total_test_smooth = 0.0
        with torch.no_grad():
            for batch_idx, (images, ground_truths) in enumerate(test_loader):
                images = images.float()
                ground_truths = ground_truths.float()
                
                current_batch_size = images.size(0)
                first_col = images[:, 0, :, 0]
                first_measurement = model.feature_extractor(first_col)
                hidden = model.init_hidden(current_batch_size)
                hidden[0, :, :N_curves] = first_measurement
                hidden[0, :, N_curves:] = first_measurement
                
                predicted_positions = []
                for t in range(seq_len):
                    img_col = images[:, 0, :, t]
                    measurement_feature = model.feature_extractor(img_col)
                    assigned_measurements = []
                    for b in range(current_batch_size):
                        hidden_b = hidden[0, b]
                        predicted_state = (pred_mat @ hidden_b.unsqueeze(1)).cpu().numpy()
                        meas_b = measurement_feature[b].unsqueeze(1).cpu().numpy()
                        cost_matrix = np.abs(predicted_state - meas_b)
                        row_ind, col_ind = linear_sum_assignment(cost_matrix)
                        assigned = np.zeros(N_curves)
                        for r, c in zip(row_ind, col_ind):
                            assigned[r] = measurement_feature[b, c].item()
                        assigned_measurements.append(torch.tensor(assigned, dtype=torch.float32))
                    assigned_measurement_batch = torch.stack(assigned_measurements, dim=0)
                    prediction, hidden = model(assigned_measurement_batch, hidden)
                    predicted_positions.append(prediction)
                
                predicted_seq = torch.stack(predicted_positions, dim=1)
                target_seq = ground_truths.permute(0, 2, 1)
                mse_val = criterion(predicted_seq, target_seq)
                diff_val = predicted_seq[:, 1:, :] - predicted_seq[:, :-1, :]
                smooth_val = torch.mean(diff_val ** 2)
                total_test_mse += mse_val.item()
                total_test_smooth += smooth_val.item()
        
        avg_test_mse = total_test_mse / len(test_loader)
        avg_test_smooth = total_test_smooth / len(test_loader)
        test_losses.append(avg_test_mse)
        test_smooth_losses.append(avg_test_smooth)
        
        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {avg_train_loss:.6f}, Test MSE: {avg_test_mse:.6f}, Test Smooth: {avg_test_smooth:.6f}")
        
        # -------------------------------
        # Display Sample Predictions and Loss Curves Every 20 Epochs
        # -------------------------------
        if (epoch + 1) % 20 == 0:
            # For visualization, use the last computed batch.
            sample_pred = predicted_seq[0].detach().cpu().numpy()  # (seq_len, N_curves)
            sample_target = target_seq[0].detach().cpu().numpy()     # (seq_len, N_curves)
            time_steps = np.arange(seq_len)
            
            plt.figure(figsize=(14, 5))
            plt.subplot(1, 2, 1)
            plt.plot(time_steps, sample_target[:, 0], label='True Curve 1', marker='o')
            plt.plot(time_steps, sample_target[:, 1], label='True Curve 2', marker='o')
            plt.plot(time_steps, sample_pred[:, 0], label='Predicted Curve 1', marker='x')
            plt.plot(time_steps, sample_pred[:, 1], label='Predicted Curve 2', marker='x')
            plt.xlabel('Pixel Column (Time Step)')
            plt.ylabel('Curve Position (Pixel)')
            plt.title(f"Epoch {epoch+1}")
            plt.legend()
            plt.grid(True)
            
            plt.subplot(1, 2, 2)
            plt.plot(train_losses, label='Train Loss')
            plt.plot(test_losses, label='Test MSE')
            plt.plot(test_smooth_losses, label='Test Smooth Loss')
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Loss Curves")
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.show()
    
    # -------------------------------
    # Final Loss Plot
    # -------------------------------
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test MSE')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Final Loss Curves")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    # Replace these paths with your image and ground truth folders.
    image_folder = r'D:\Courses\Uclouvain\thesis\code\thesis_Colin\seismogram_curve_extraction\data\sines\overlap_0-0\signals'
    gt_folder = r'D:\Courses\Uclouvain\thesis\code\thesis_Colin\seismogram_curve_extraction\data\sines\overlap_0-0\ground_truth'

    main(image_folder, gt_folder, batch_size=4)
