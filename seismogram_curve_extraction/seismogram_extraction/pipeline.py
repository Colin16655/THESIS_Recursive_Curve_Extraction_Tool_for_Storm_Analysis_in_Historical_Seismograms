import numpy as np
import matplotlib.pyplot as plt
import os, sys
import cv2
from scipy.signal import argrelextrema
from tqdm import tqdm
from itertools import product

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

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

def evaluate_filter(images_folder_path, gt_s_folder_path, processing_method):
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
    dataloader = create_dataloader(images_folder_path, gt_s_folder_path, batch_size=16, shuffle=False, num_workers=4)
    
    MSEs = np.full(len(dataloader), np.nan)
    for batch_idx, (images, ground_truths) in enumerate(dataloader):

        # Artificially provide the initial state, and the number of states
        # This will be automatised for the seismograms images
        # x_0 = all_x_0[1]
        # X_weighted = np.array([[x_0[0], 0], [x_0[1], 0]]).astype(np.float64)
        # P_weighted = np.array([np.copy(P), np.copy(P)]).astype(np.float64)

        batch_pred = processing_method.process_sequence(images.numpy())
        MSEs[batch_idx] = mean_squared_error(ground_truths.numpy(), batch_pred)
    
    # error = ground_truth - predictions
    # error_mean = np.mean(error, axis=0)
    # error_std = np.std(error, axis=0)
    
    # print(f"MSE: {mse}")
    # print(f"Mean Error: {error_mean}")
    # print(f"Std Dev of Error: {error_std}")
    
    # return mse, error_mean, error_std

### Parameters
Dt = 0.5 # np.linspace(0.5, 5, 2) # Time step related to the state transition matrix A, ! different than sampling rate dt of signal s

# Assuming no process noise
sigma_p = 0.01 # np.linspace(1e-2, 2, 2) 
sigma_v = 2 # np.linspace(1e-2, 2, 2)

# Assuming no measurement noise
sigma_z = 0.25 # np.linspace(1e-6, 1, 5)

p_fa = 0.0001 # np.linspace(1e-4, 1, 5)
###

A = np.array([[1, Dt],
              [0, 1]]).astype(np.float64)
                    
H = np.array([[1, 0]]).astype(np.float64)

Q = np.array([[sigma_p**2, 0],
              [0, sigma_v**2]])

R = np.array([[sigma_z**2]])

# Initial state covariance given all_x_0
P = np.zeros((2, 2))
P[1, 1] = 10

image_folder_path = r"D:\Courses\Uclouvain\thesis\code\thesis_Colin\seismogram_curve_extraction\data\sines\overlap_0-0\signals"
GTs_folder_path = r"D:\Courses\Uclouvain\thesis\code\thesis_Colin\seismogram_curve_extraction\data\sines\overlap_0-0\ground_truth"

evaluate_filter(image_folder_path, GTs_folder_path, WeightedKalmanFilter(A, H, Q, R, P, p_fa))
