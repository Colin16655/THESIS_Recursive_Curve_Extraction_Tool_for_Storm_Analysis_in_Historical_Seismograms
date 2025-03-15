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

def mean_squared_error(y_true, y_pred):
    """
    Compute the mean squared error between the ground truth and the predictions.
    
    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values.
    y_pred : np.ndarray
        Predicted values.
    
    Returns
    -------
    mse : float
        Mean squared error between the ground truth and the predictions.
    """
    MSEs = np.full((y_true.shape[0], y_true.shape[2]), np.nan)
    for i in range(y_true.shape[0]):
        for j in range(y_true.shape[2]):
            MSEs[i, j] = np.mean((y_true[i, :, j] - y_pred[i, :, j])**2)
    print(MSEs)
    # print(y_true.shape)
    # temp1 = (y_true - y_pred)**2
    # print(temp1.shape)
    # print(np.mean(temp1, axis=2).shape)
    return np.mean(MSEs)

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
from seismogram_extraction.models.lrnn import LinearRNN

def evaluate_filter(images_folder_path, gt_s_folder_path, output_folder_path, processing_method, batch_size=4):
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
    
    Returns
    -------
    mse : float
        Mean squared error between the ground truth and the predictions.
    error_mean : np.ndarray
        Mean error of the predictions.
    error_std : np.ndarray
        Standard deviation of the error of the predictions.
    """
    dataloader = create_dataloader(images_folder_path, gt_s_folder_path, batch_size=batch_size)
    
    MSEs = np.full(len(dataloader), np.nan)
    all_MSEs = []

    for batch_idx, (images, ground_truths) in tqdm(enumerate(dataloader), desc="Batches", total=len(dataloader)):
        N_traces = ground_truths.shape[1]

        # Artificially provide the initial state, and the number of states
        # This will be automatised for the seismograms images
        X_0 = np.zeros((len(images), N_traces, len(processing_method.A)))
        X_0_temp = ground_truths[:, :, 0].numpy()
        X_0[:, :, 0] = X_0_temp
        P_0 = np.zeros((len(images), N_traces, len(processing_method.A), len(processing_method.A)))
        P_0[:, :, 1, 1] = 10
        
        X_batch_pred, P_batch_pred = processing_method.process_sequence(images.numpy(), X_0, P_0)

        ground_truths = ground_truths.numpy().transpose(0, 2, 1)

        total_MSE = 0
        for i, batch in enumerate(images):
            X_pos_batch_pred = X_batch_pred[i, :, :, 0]
            X_vel_batch_pred = X_batch_pred[i, :, :, 1]
            P_pos_batch_pred = P_batch_pred[i, :, :, 0, 0]
            P_vel_batch_pred = P_batch_pred[i, :, :, 1, 1]
            gr = ground_truths[i]

            plt.imshow(batch[0], cmap='gray')
            temp_total_MSE = 0
            for j in range(X_pos_batch_pred.shape[1]):
                MSE = np.mean((X_pos_batch_pred[:, j] - gr[:, j])**2)
                temp_total_MSE += MSE
                plt.scatter(np.arange(0, len(X_pos_batch_pred[:, j])), X_pos_batch_pred[:, j], s=1, label=r"MSE: {:.2f}".format(MSE))
            plt.legend(markerscale=5)
            # Save the plot
            plt.savefig(output_folder_path + f"/output_{batch_idx}_{i}.pdf", format='pdf', bbox_inches='tight', dpi=300)
            plt.close()

            temp_total_MSE /= X_pos_batch_pred.shape[1]
            total_MSE += temp_total_MSE
        total_MSE /= len(images)

        all_MSEs.append(total_MSE)

        MSEs[batch_idx] = total_MSE
        print(f"\nMSE of the batch {batch_idx}: {MSEs[batch_idx]}")
    
    MSEs_std = np.std(all_MSEs, axis=0)
    print("\n-------------------------------\n")
    print(f"Total Average MSE: {np.mean(MSEs)}")
    print(f"Std: {MSEs_std}")
    print("\n-------------------------------\n")
    