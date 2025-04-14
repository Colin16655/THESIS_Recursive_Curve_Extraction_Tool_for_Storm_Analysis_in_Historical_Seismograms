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

    Returns
    -------
    average_rmse : float
        Average RMSE across all curves of all images.
    rmse_std : float
        Standard deviation of the per-curve RMSEs.
    all_curve_rmses : list
        List of RMSEs for each curve in each image.
    """
    dataloader = create_dataloader(images_folder_path, gt_s_folder_path, batch_size=batch_size)
    all_curve_rmses = []

    plot_counter = 0

    for batch_idx, (images, ground_truths) in enumerate(dataloader):
        N_traces = ground_truths.shape[1]
        batch_size = len(images)

        X_0 = np.zeros((batch_size, N_traces, len(processing_method.A)))
        X_0[:, :, 0] = ground_truths[:, :, 0].numpy()
        P_0_extended = np.tile(P_0, (batch_size, N_traces, 1, 1))

        X_batch_pred, P_batch_pred = processing_method.process_sequence(images.numpy(), X_0, P_0_extended, step=step)
        ground_truths = ground_truths.numpy().transpose(0, 2, 1)

        for i in range(batch_size):
            pred_positions = X_batch_pred[i, :, :, 0]
            pred_velocities = X_batch_pred[i, :, :, 1]
            pred_accelerations = X_batch_pred[i, :, :, 2]

            std_positions = np.sqrt(P_batch_pred[i, :, :, 0, 0])
            std_velocities = np.sqrt(P_batch_pred[i, :, :, 1, 1])
            std_accelerations = np.sqrt(P_batch_pred[i, :, :, 2, 2])

            true_positions = ground_truths[i, ::step, :]  # shape (T, N_traces)

            if save and plot_counter < 5:
                fig, ax = plt.subplots(figsize=(8, 4))
                t_steps = np.arange(0, images.shape[-1])[::step]
                ax.imshow(images[i, 0], cmap='gray')

            for j in range(pred_positions.shape[1]):
                rmse = compute_rmse(pred_positions[:, j], true_positions[:, j])
                all_curve_rmses.append(rmse)

                if save and plot_counter < 5:
                    ax.plot(t_steps, pred_positions[:, j], label=f'Trace {j+1} RMSE: {rmse:.2f}')
                    ax.fill_between(t_steps,
                                    pred_positions[:, j] - std_positions[:, j],
                                    pred_positions[:, j] + std_positions[:, j],
                                    alpha=0.3)

            if save and plot_counter < 5:
                ax.legend(markerscale=5)
                fig.tight_layout()
                fig.savefig(f"{output_folder_path}/output_{batch_idx}_{i}.pdf",
                            format='pdf', bbox_inches='tight', dpi=300)
                plt.close(fig)

                T = pred_positions.shape[0]
                t = np.arange(T)
                fig, axes = plt.subplots(2, 1, figsize=(8, 4), sharex=True)

                for j in range(N_traces):
                    axes[0].plot(t, pred_velocities[:, j], label=f'Trace {j+1}')
                    axes[0].fill_between(t,
                                         pred_velocities[:, j] - std_velocities[:, j],
                                         pred_velocities[:, j] + std_velocities[:, j],
                                         alpha=0.3)
                    axes[1].plot(t, pred_accelerations[:, j], label=f'Trace {j+1}')
                    axes[1].fill_between(t,
                                         pred_accelerations[:, j] - std_accelerations[:, j],
                                         pred_accelerations[:, j] + std_accelerations[:, j],
                                         alpha=0.3)

                axes[0].set_xlim(0, T)
                axes[1].set_xlim(0, T)
                axes[0].set_ylabel('Velocity')
                axes[1].set_ylabel('Acceleration')
                axes[1].set_xlabel('Time step')
                axes[0].legend(markerscale=5)
                axes[1].legend(markerscale=5)

                fig.tight_layout()
                fig.savefig(f"{output_folder_path}/states_{batch_idx}_{i}.pdf",
                            format='pdf', bbox_inches='tight', dpi=300)
                plt.close(fig)

                plot_counter += 1

    average_rmse = np.mean(all_curve_rmses)
    rmse_std = np.std(all_curve_rmses)

    return average_rmse, rmse_std, all_curve_rmses
    