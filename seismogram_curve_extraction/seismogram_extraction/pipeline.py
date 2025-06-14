import numpy as np
import matplotlib.pyplot as plt
import os, sys
import cv2
from scipy.signal import argrelextrema
from tqdm import tqdm
from itertools import product

import torch
from torch.utils.data import Dataset, DataLoader, random_split
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
        # self.gt_files = sorted([os.path.join(gt_folder_path, f) for f in os.listdir(gt_folder_path) if f.endswith('.npy')])
        self.gt_files = sorted([
            os.path.join(gt_folder_path, f)
            for f in os.listdir(gt_folder_path)
            if f.endswith('.npy') and not f.endswith('vel.npy') and not f.endswith('acc.npy')
        ])

        self.transform = transform
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load the image
        image = cv2.imread(self.image_files[idx], cv2.IMREAD_GRAYSCALE)
        image = image.astype(np.float32) / 255.0  # Normalize to [0, 1]
        # invert the image
        image = invert_image(image)
        
        
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


def create_dataloader(image_folder_path, gt_folder_path, batch_size=16, shuffle=True, num_workers=4,
                      split=False, val_ratio=0.2):
    dataset = ImageSequenceDataset(image_folder_path, gt_folder_path)

    if split:
        val_size = int(len(dataset) * val_ratio)
        train_size = len(dataset) - val_size

        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return train_loader, val_loader

    # Default behavior: return a single dataloader
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

def compute_rmse(pred, true):
    """
    Compute RMSE between predicted and true values.
    """
    return np.sqrt(np.mean((pred - true) ** 2))

def evaluate_filter(images_folder_path, gt_s_folder_path, output_folder_path,
                    processing_method, P_0, batch_size=4, save=True, step=1,
                    labels=["Velocity", "Acceleration"], a_0=None, omega_0=None, phi_0=None, forum=False):
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
        if forum:
            meanlines = np.array(ground_truths.mean(axis=2).mean(axis=0))
            # print("meanlines", meanlines)#ground_truths.shape, np.array(ground_truths.mean(axis=2)))
        N_traces = ground_truths.shape[1]
        batch_size = len(images)

        X_0 = np.zeros((batch_size, N_traces, processing_method.Q.shape[-1]))
        X_0[:, :, 0] = ground_truths[:, :, 0].numpy()  # position (p)
        # if sine aware HEKF
        if forum:
            if a_0 == None:
                X_0[:, :, 0] = ground_truths[:, :, 0].numpy() - meanlines
            else:
                X_0[:, :, 0] = a_0
            X_0[:, :, 1] = omega_0
            X_0[:, :, 2] = phi_0

        P_0_extended = np.tile(P_0, (batch_size, N_traces, 1, 1))
        if forum:         
            X_batch_pred, P_batch_pred = processing_method.process_sequence(images.numpy(), X_0, P_0_extended, meanlines, step=step)
        else:
            X_batch_pred, P_batch_pred = processing_method.process_sequence(images.numpy(), X_0, P_0_extended, step=step)
        ground_truths = ground_truths.numpy().transpose(0, 2, 1)

        for i in range(batch_size):
            if forum:
                pred_positions = X_batch_pred[i, :, :, 0] * np.sin(X_batch_pred[i, :, :, 2]) 
                for p in range(len(meanlines)):
                    pred_positions[:, p] += meanlines[p]
                std_positions = np.zeros_like(X_batch_pred[i, :, :, 0])
            else:
                pred_positions = X_batch_pred[i, :, :, 0]
                std_positions = np.sqrt(P_batch_pred[i, :, :, 0, 0])
            true_positions = ground_truths[i, ::step, :]  # shape (T, N_traces)

            if save and plot_counter < 5:
                fig, ax = plt.subplots(figsize=(4.5, 2.5))
                t_steps = np.arange(0, images.shape[-1])[::step]
                ax.imshow(images[i, 0].max() - images[i, 0], cmap='gray', origin='lower')

            for j in range(pred_positions.shape[1]):
                rmse = compute_rmse(pred_positions[:, j], true_positions[:, j])
                all_curve_rmses.append(rmse)

                if save and plot_counter < 5:
                    if j == 0:
                        ax.plot(t_steps, pred_positions[:, j], label=f'RMSE: {rmse:.2f}')
                    else:
                        ax.plot(t_steps, pred_positions[:, j], label=f'{rmse:.2f}')
                    ax.fill_between(t_steps,
                                    pred_positions[:, j] - std_positions[:, j],
                                    pred_positions[:, j] + std_positions[:, j],
                                    alpha=0.3)

            if save and plot_counter < 5:
                ax.set_xlabel(r"time $k$ [pixel]")
                ax.set_ylabel(r"position $p$ [pixel]")
                ax.legend(markerscale=5, loc='upper center', bbox_to_anchor=(0.5, 1.4), ncol=3)
                ax.grid(False)
                fig.tight_layout(rect=[0, 0.1, 1, 1])
                fig.savefig(f"{output_folder_path}/output_{batch_idx}_{i}.pdf",
                            format='pdf', bbox_inches='tight', dpi=300)
                plt.close(fig)

                T = pred_positions.shape[0]
                t = np.arange(T)
                fig, axes = plt.subplots(len(labels), 1, figsize=(6, 3), sharex=True)
                for ll in range(len(labels)):
                    for j in range(N_traces):
                        if forum: l = ll - 1
                        else: l = ll
                        pred = X_batch_pred[i, :, :, l+1]
                        std_pred = np.sqrt(P_batch_pred[i, :, :, l+1, l+1])

                        axes[l].plot(t, pred[:, j], label=f'Trace {j+1}')
                        axes[l].fill_between(t,
                                            pred[:, j] - std_pred[:, j],
                                            pred[:, j] + std_pred[:, j],
                                            alpha=0.3)

                    axes[l].set_xlim(0, T)
                    axes[l].legend(markerscale=5)
                    axes[l].set_ylabel(labels[l])
                axes[-1].set_xlabel('Time step')

                fig.tight_layout()
                fig.savefig(f"{output_folder_path}/states_{batch_idx}_{i}.pdf",
                            format='pdf', bbox_inches='tight', dpi=300)
                plt.close(fig)

                plot_counter += 1

    average_rmse = np.mean(all_curve_rmses)
    rmse_std = np.std(all_curve_rmses)

    return average_rmse, rmse_std, all_curve_rmses
    