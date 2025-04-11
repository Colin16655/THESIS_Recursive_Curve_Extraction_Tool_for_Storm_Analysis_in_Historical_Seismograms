import numpy as np
import matplotlib.pyplot as plt
import os, sys
import cv2
from scipy.signal import argrelextrema
from tqdm import tqdm
from itertools import product

import torch
from torch.utils.data import Dataset, DataLoader
# from sklearn.metrics import mean_squared_error

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp

class ImageSequenceDataset(Dataset):
    def __init__(self, image_folder_path, gt_folder_path, transform=None):
        self.image_folder_path = image_folder_path
        self.gt_folder_path = gt_folder_path
        
        # Collect image files
        self.image_files = sorted([os.path.join(image_folder_path, f) for f in os.listdir(image_folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        # Collect ground truth files
        self.gt_files = sorted([os.path.join(gt_folder_path, f) for f in os.listdir(gt_folder_path) if f.endswith('.npy')])
        
        self.transform = transform
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load the image
        image = cv2.imread(self.image_files[idx], cv2.IMREAD_GRAYSCALE)
        image = image.astype(np.float32) / 255.0  # Normalize to [0, 1]
        
        # Load the ground truth
        gt = np.load(self.gt_files[idx])
        
        if self.transform:
            image = self.transform(image)
        
        # Return both image and ground truth as tensors
        return torch.tensor(image).unsqueeze(0), torch.tensor(gt)

def create_dataloader(image_folder_path, gt_folder_path, batch_size=16, shuffle=True, num_workers=4):
    dataset = ImageSequenceDataset(image_folder_path, gt_folder_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

def invert_image(image):
    return image.max() - image

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

# Get the absolute path of the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add the project root to sys.path
sys.path.append(project_root)

from seismogram_extraction.filters.weighted_kalman_filter import WeightedKalmanFilter
# from seismogram_extraction.filters.hungarian_kalman_filter_kalman_filter import HungarianKalmanFilter
# from seismogram_extraction.models.lrnn import LinearRNN

def compute_rmse(pred, true):
    """
    Compute RMSE between predicted and true values.
    """
    return np.sqrt(np.mean((pred - true) ** 2))

def evaluate_filter(images_folder_path, gt_s_folder_path, output_folder_path,
                    processing_method, P_0, batch_size=4, save=True, step=1):
    """
    Evaluate the performance of a processing method on a dataset.

    Parameters
    ----------
    images_folder_path : str
        Path to the folder containing the images.
    gt_s_folder_path : str
        Path to the folder containing the ground truth seismograms.
    processing_method : object
        Object that implements the process_sequence method.
    P_0 : np.ndarray, shape (len(processing_method.A), len(processing_method.A)))
        Initial state covariance matrix.
    batch_size : int
        Number of samples per batch.
    save : bool
        Whether to save plots of predictions.
    step : int
        Downsampling step for evaluation.

    Returns
    -------
    average_rmse : float
        Average RMSE across all batches.
    rmse_std : float
        Standard deviation of batch-wise RMSEs.
    """
    dataloader = create_dataloader(images_folder_path, gt_s_folder_path, batch_size=batch_size)
    RMSEs = np.full(len(dataloader), np.nan)
    all_batch_rmse = []

    plot_counter = 0

    for batch_idx, (images, ground_truths) in enumerate(dataloader):
        N_traces = ground_truths.shape[1]
        batch_size = len(images)

        # Artificially provide the initial state, and the number of states
        # This will be automatised for the seismograms images
        X_0 = np.zeros((len(images), N_traces, len(processing_method.A)))
        X_0[:, :, 0] = ground_truths[:, :, 0].numpy()
        P_0_extended = np.tile(P_0, (len(images), N_traces, 1, 1))

        # Run filtering
        X_batch_pred, P_batch_pred = processing_method.process_sequence(images.numpy(), X_0, P_0_extended, step=step)
        ground_truths = ground_truths.numpy().transpose(0, 2, 1)

        # Compute RMSE for each sample in the batch
        batch_rmse_total = 0

        for i in range(batch_size):
            pred_positions = X_batch_pred[i, :, :, 0]
            pred_velocities = X_batch_pred[i, :, :, 1]
            pred_accelerations = X_batch_pred[i, :, :, 2]

            std_positions = np.sqrt(P_batch_pred[i, :, :, 0, 0])
            std_velocities = np.sqrt(P_batch_pred[i, :, :, 1, 1])
            std_accelerations = np.sqrt(P_batch_pred[i, :, :, 2, 2])

            true_positions = ground_truths[i, ::step, :]  # shape (T, N_traces)

            sample_rmse = 0

            if save and plot_counter < 5:
                plt.imshow(images[i, 0], cmap='gray')  # Plot background once per sample

            for j in range(pred_positions.shape[1]):
                rmse = compute_rmse(pred_positions[:, j], true_positions[:, j])
                sample_rmse += rmse

                if save and plot_counter < 5:
                    plt.scatter(np.arange(0, images.shape[-1])[::step], pred_positions[:, j], s=1,
                                label=f"Trace {j+1} RMSE: {rmse:.2f}")

            sample_rmse /= pred_positions.shape[1]
            batch_rmse_total += sample_rmse

            if save and plot_counter < 5:
                plt.legend(markerscale=5)
                plt.savefig(f"{output_folder_path}/output_{batch_idx}_{i}.pdf",
                            format='pdf', bbox_inches='tight', dpi=300)
                plt.close()

                # Save position, velocity, acceleration subplots
                T = pred_positions.shape[0]
                t = np.arange(T)

                fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

                for j in range(N_traces):
                    axes[0].plot(t, pred_positions[:, j], label=f'Trace {j+1}')
                    axes[0].fill_between(t,
                                         pred_positions[:, j] - std_positions[:, j],
                                         pred_positions[:, j] + std_positions[:, j],
                                         alpha=0.3)
                    axes[1].plot(t, pred_velocities[:, j], label=f'Trace {j+1}')
                    axes[1].fill_between(t,
                                         pred_velocities[:, j] - std_velocities[:, j],
                                         pred_velocities[:, j] + std_velocities[:, j],
                                         alpha=0.3)
                    axes[2].plot(t, pred_accelerations[:, j], label=f'Trace {j+1}')
                    axes[2].fill_between(t,
                                         pred_accelerations[:, j] - std_accelerations[:, j],
                                         pred_accelerations[:, j] + std_accelerations[:, j],
                                         alpha=0.3)

                axes[0].set_ylabel('Position')
                axes[0].legend()
                axes[1].set_ylabel('Velocity')
                axes[2].set_ylabel('Acceleration')
                axes[2].set_xlabel('Time step')
                fig.suptitle(f'Filtered State - Sample {batch_idx}_{i}')

                plt.tight_layout()
                plt.savefig(f"{output_folder_path}/states_{batch_idx}_{i}.pdf",
                            format='pdf', bbox_inches='tight', dpi=300)
                plt.close()

                plot_counter += 1

        batch_rmse_avg = batch_rmse_total / batch_size
        all_batch_rmse.append(batch_rmse_avg)

    average_rmse = np.mean(all_batch_rmse)
    rmse_std = np.std(all_batch_rmse)

    return average_rmse, rmse_std
    