from skimage.io import imread
from skimage.transform import resize
from skimage.filters import threshold_otsu, threshold_local, try_all_threshold

import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import math
import cv2

class RasterImage:
    def __init__(self, file_path, batch_size=(100, 100)):
        """
        Initialize the RasterImage object.

        Args:
            file_path (str): Path to the image file.
            batch_size (tuple[int, int]): Tuple specifying the size (height, width) of the image batches.
        """
        self.file_path = file_path
        self.image = None
        self.sliced_image = None
        self.binary_image = None # same shape as sliced_image
        self.batch_size = batch_size  # Expected as (height, width)
        self.current_row = 0
        self.current_col = 0

    def __iter__(self):
        """Initialize the iterator by setting the row and column to 0."""
        self.current_row = 0
        self.current_col = 0
        return self

    def __next__(self):
        """Generate the next batch (rectangular image) in row-wise order."""
        if self.sliced_image is None:
            raise ValueError("Sliced image not available. Please downsample the image first.")
        
        # Image dimensions
        img_height, img_width = self.sliced_image.shape[:2]
        batch_height, batch_width = self.batch_size

        # Check if we have reached the end of the image
        if self.current_row >= img_height:
            raise StopIteration

        # Calculate the row and column indices for the batch
        start_row = self.current_row
        start_col = self.current_col
        end_row = min(start_row + batch_height, img_height)
        end_col = min(start_col + batch_width, img_width)

        # Get the current batch
        batch = self.sliced_image[start_row:end_row, start_col:end_col]

        # Update column and row indices for the next batch
        self.current_col += batch_width
        if self.current_col >= img_width:
            self.current_col = 0
            self.current_row += batch_height

        return batch

    def num_batches(self):
        """Calculate the number of batches required to cover the entire image."""
        img_height, img_width = self.sliced_image.shape[:2]
        batch_height, batch_width = self.batch_size

        num_rows = math.ceil(img_height / batch_height)
        num_cols = math.ceil(img_width / batch_width)
        return num_rows * num_cols

    def __len__(self):
        """Return the total number of batches in the image."""
        return self.num_batches()

    def load_image(self) -> None:
        """Load the image from the specified file path."""
        self.image = imread(self.file_path)
        shape = self.image.shape
        if shape[0] > shape[1]: self.image = self.image.T  # Transpose if necessary
        print(f"Loaded image shape: {self.image.shape}")
        print(f"Original min intensity: {np.min(self.image)}, max intensity: {np.max(self.image)}")

    def slice_downsample(self, factor: int) -> None:
        """
        Resize the image to the specified output shape.

        Args:
            output_shape (tuple): Desired shape of the output image (height, width).
        """
        if self.image is None:
            raise ValueError("Image not loaded. Please load the image first.")
        self.sliced_image = self.image[::factor, ::factor]
        print(f"sliced image shape: {self.sliced_image.shape}")

    def discard_original_image(self) -> None:
        """Discard the original image to save memory."""
        self.image = None

    # def image_to_tensor(self) -> torch.Tensor:
    #     """
    #     Convert the sliced image to a PyTorch tensor.

    #     Returns:
    #         torch.Tensor: The image as a PyTorch tensor with appropriate dimensions.
    #     """
    #     if self.sliced_image is None:
    #         raise ValueError("sliced image not available. Please resize the image first.")
        
    #     tensor_image = torch.tensor(self.sliced_image, dtype=torch.float32)
        
    #     # If the image has a single channel (grayscale), add a channel dimension
    #     if tensor_image.ndimension() == 2:
    #         tensor_image = tensor_image.unsqueeze(0)
        
    #     # If the image has multiple channels, rearrange dimensions to match PyTorch's format (C, H, W)
    #     elif tensor_image.ndimension() == 3:
    #         tensor_image = tensor_image.permute(2, 0, 1)
        
    #     return tensor_image

    def apply_otsu_thresholding(self, mode='global', params_local=None, sens_analysis=False, save=False, directory=".", show=False, save_name="global_otsu_all"):
        """
        Apply Otsu's thresholding globally or locally to the sliced image.

        Args:
            mode (str): Either 'global' or 'local'. 
                        'global' applies Otsu thresholding to the whole image.
                        'local' applies Otsu thresholding to each grid cell.
            cell_size (tuple): Size of each grid cell in pixels (height, width) for local thresholding.
                               Required if mode is 'local'.

        Raises:
            ValueError: If the sliced image is not available or invalid arguments are provided.
        """
        if self.sliced_image is None:
            raise ValueError("Sliced image not available. Please downsample the image first.")

        # Convert to grayscale if it's a color image
        if self.sliced_image.ndim == 3:
            self.sliced_image = self.sliced_image.mean(axis=2)

        if sens_analysis:
            # Display the effect of different block sizes on local thresholding
            block_sizes = [5, 15, 25, 35] # USER
            offsets = [5, 10, 20] # USER
            fig, axes = plt.subplots(len(block_sizes), len(offsets), figsize=(3*len(block_sizes), 5*len(offsets)))
            for i, offset in enumerate(offsets):
                for j, block_size in enumerate(block_sizes):
                    ax = axes[j, i]
                    local_thresh = threshold_local(self.sliced_image, block_size, offset=offset)
                    binary_local = self.sliced_image > local_thresh
                    ax.imshow(binary_local, cmap='gray')
                    ax.axis('off')
                    ax.set_title(f'Block Size: {block_size}, offset: {offset}')

            # Optionally save the figure as a PDF
            if save:
                os.makedirs(directory, exist_ok=True)  # Create directory if it doesn't exist
                full_path = os.path.join(directory, "local_otsu_sens_an.pdf")
                fig.savefig(full_path, format='pdf', bbox_inches='tight', pad_inches=0.1, dpi=300)  # Save as PDF with tight layout
                print(f"Figure saved as '{full_path}'.")

            if show: plt.show()
            plt.close(fig)

            fig, ax = try_all_threshold(self.sliced_image, figsize=(10, 8), verbose=False)

            # Optionally save the figure as a PDF
            if save:
                os.makedirs(directory, exist_ok=True)  # Create directory if it doesn't exist
                full_path = os.path.join(directory, save_name)
                fig.savefig(full_path, format='pdf', bbox_inches='tight', pad_inches=0.1, dpi=300)  # Save as PDF with tight layout
                print(f"Figure saved as '{full_path}'.")

            if show: plt.show()
            plt.close(fig)

        if mode == 'global':
            # Apply global Otsu thresholding
            thresh = threshold_otsu(self.sliced_image)
            self.binary_image = (self.sliced_image > thresh).astype(np.uint8) * 255
            print(f"Global Otsu's threshold: {thresh}")
            if save:
                # plt.imshow(self.binary_image, cmap='gray')
                # plt.show()
                os.makedirs(directory, exist_ok=True)  # Create directory if it doesn't exist
                full_path = os.path.join(directory, save_name)
                cv2.imwrite(full_path, self.binary_image)
                print(f"Figure saved as '{full_path}'.")

        elif mode == 'local':
            if params_local is None:
                raise ValueError("params_local must be specified for local thresholding.")
                                    
            # Apply local thresholding
            block_size = params_local['block_size']
            block_size = (block_size | 1)  # Ensure block size is odd
            offset = params_local['offset']
            thresh_local = threshold_local(self.sliced_image, block_size, offset=offset)
            self.binary_image = (self.sliced_image > thresh_local).astype(int)

        else:
            raise ValueError("Invalid mode. Choose either 'global' or 'local'.")

    def display_image(self, show=False, image_type: str = 'sliced_image', save: bool = False, filename: str = "image.pdf", directory: str = ".") -> None:
        """Display a specified image using Matplotlib and optionally save it to a PDF file.

        Args:
            image_type (str): Type of image to display ('image', 'sliced_image', 'binary_image').
            save (bool): If True, saves the figure to a file. Default is False.
            filename (str): The filename to save the figure as a PDF. Default is "image.pdf".
            directory (str): The directory where the file will be saved. Default is the current directory.

        Raises:
            ValueError: If the specified image is not available.
        """
        # Select the image based on the image_type
        if image_type == 'image':
            img_to_display = self.image
        elif image_type == 'sliced_image':
            img_to_display = self.sliced_image
        elif image_type == 'binary_image':
            img_to_display = self.binary_image
        else:
            raise ValueError("Invalid image type. Choose from 'image', 'sliced_image', or 'binary_image'.")

        if img_to_display is None:
            raise ValueError(f"{image_type} not available. Please ensure it has been created.")

        # Create a figure and axis
        fig, ax = plt.subplots()
        
        # Display the selected image
        ax.imshow(img_to_display, cmap='gray')
        ax.set_title(f"{os.path.basename(self.file_path)} - {image_type}")
        ax.axis('off')  # Hide the axis

        # Optionally save the figure as a PDF
        if save:
            os.makedirs(directory, exist_ok=True)  # Create directory if it doesn't exist
            full_path = os.path.join(directory, filename)
            fig.savefig(full_path, format='pdf', bbox_inches='tight', pad_inches=0.1, dpi=300)  # Save as PDF with tight layout
            print(f"Figure saved as '{full_path}'.")

        # Show the figure
        if show: plt.show()
        plt.close(fig)

    def plot_histogram(self, show=False, image_type: str = 'sliced_image', save: bool = False, filename: str = "histogram.png", directory: str = ".") -> None:
        """Plot the intensity histogram of the specified image and optionally save it.

        Args:
            image_type (str): Type of image to plot ('image', 'sliced_image', 'binary_image').
            save (bool): If True, saves the histogram to a file. Default is False.
            filename (str): The filename to save the histogram. Default is "histogram.png".
            directory (str): The directory where the file will be saved. Default is the current directory.

        Raises:
            ValueError: If the specified image is not available.
        """
        # Select the image based on the image_type
        if image_type == 'image':
            img_to_plot = self.image
        elif image_type == 'sliced_image':
            img_to_plot = self.sliced_image
        elif image_type == 'binary_image':
            img_to_plot = self.binary_image
        else:
            raise ValueError("Invalid image type. Choose from 'image', 'sliced_image', or 'binary_image'.")

        if img_to_plot is None:
            raise ValueError(f"{image_type} not available. Please ensure it has been created.")

        # Create a figure and axis
        fig, ax = plt.subplots()  # Create figure and axis
        
        # Plot the histogram on the axis
        ax.hist(img_to_plot.ravel(), bins=np.max(img_to_plot) + 1,
                range=(np.min(img_to_plot), np.max(img_to_plot)))
        ax.set_title(f'Intensity Histogram of {os.path.basename(self.file_path)}')
        ax.set_xlabel('Intensity Value')
        ax.set_ylabel('Frequency')

        # Optionally save the histogram
        if save:
            os.makedirs(directory, exist_ok=True)  # Create directory if it doesn't exist
            full_path = os.path.join(directory, filename)
            fig.savefig(full_path, bbox_inches='tight', pad_inches=0.1, dpi=300)  # Save as PNG with tight layout
            print(f"Histogram saved as '{full_path}'.")

        # Show the histogram
        if show: plt.show()
        plt.close(fig) 


# Usage example
if __name__ == "__main__":
    # File paths of the images
    source_directories = [r'data\raw']

    # source_directories = [r"D:/Courses/Uclouvain/thesis/code/these_Colin/seismogram_curve_extraction/results/sBE_UCC__HHE_2024-01-01T00-06-00_2024-01-14T00-12-00_1000_1_0-1_50_50_50_50_0-0_bw_0-1_864-0/",
    #                       r"D:/Courses/Uclouvain/thesis/code/these_Colin/seismogram_curve_extraction/results/sBE_UCC__HHE_2024-01-01T00-06-00_2024-01-14T00-12-00_1000_1_0-1_50_50_50_50_0-5_bw_0-1_864-0/",
    #                       r"D:/Courses/Uclouvain/thesis/code/these_Colin/seismogram_curve_extraction/results/sBE_UCC__HHE_2024-01-01T00-06-00_2024-01-14T00-12-00_1000_1_0-1_50_50_50_50_0-6_bw_0-1_864-0/"]



    # image_files = [
    #     r'seismogram_curve_extraction/data/ground_truths/overlap_0_images/4.jpg',
    #     r'seismogram_curve_extraction/data/ground_truths/overlap_0_images/5.jpg',
    #     r'seismogram_curve_extraction/data/ground_truths/overlap_0_images/6.jpg',
    #     r'seismogram_curve_extraction/data/ground_truths/overlap_0_images/7.jpg'
    # ]

    # Hyperparameters
    factor = 10  # Downsample sclicing factor

    # Load, resize, and display each image
    for source_directory in source_directories:
        for image_file in os.listdir(source_directory):
            if image_file.endswith('.tif') and not image_file.endswith('_binary.jpg'):
            # if True:
                image_file = os.path.join(source_directory, image_file)

                raster_image = RasterImage(image_file, batch_size=(400, 800))
                raster_image.load_image()
                raster_image.slice_downsample(factor)
                raster_image.discard_original_image()
                
                # Display the sliced image
                raster_image.display_image(save=True, filename="sliced.pdf", directory=f"output/{os.path.basename(image_file).removesuffix('.tif')}")
                # Plot the intensity histogram
                raster_image.plot_histogram(save=True, filename="sliced_histogram.pdf", directory=f"output/{os.path.basename(image_file).removesuffix('.tif')}")

                # Apply global Otsu thresholding
                raster_image.apply_otsu_thresholding(save=True, directory=source_directory, save_name=os.path.basename(image_file).removesuffix('.tif') + "_binary.jpg")

                # save the image
                # fig.savefig(full_path, format='pdf', bbox_inches='tight', pad_inches=0.1, dpi=300)  # Save as PDF with tight layout


                # Display the binary image
                raster_image.display_image(image_type='binary_image', save=True, 
                                        filename="binary_global_otsu.pdf", directory=f"output/{os.path.basename(image_file).removesuffix('.tif')}")
                # Plot the intensity histogram of the binary image
                # raster_image.plot_histogram(image_type='binary_image', save=True, 
                                            # filename="binary_global_otsu_histogram.pdf", directory=f"output/{os.path.basename(image_file).removesuffix('.tif')}")  
                
                # # Apply local thresholding
                params_local = {
                    'block_size' : 15,
                    'offset' : 10 
                }
                raster_image.apply_otsu_thresholding(mode='local', params_local=params_local, sens_analysis=True,
                                                    save=True, directory=f"output/{os.path.basename(image_file).removesuffix('.tif')}")

                # Display the binary image
                raster_image.display_image(image_type='binary_image', save=True, 
                                        filename=f"binary_local_block_size_{params_local['block_size']}_offset_{params_local['offset']}.pdf", directory=f"output/{os.path.basename(image_file).removesuffix('.tif')}")
                # Plot the intensity histogram of the binary image
                raster_image.plot_histogram(image_type='binary_image', save=True, 
                                            filename=f"binary_local_histogram_{params_local['block_size']}_offset_{params_local['offset']}.pdf", directory=f"output/{os.path.basename(image_file).removesuffix('.tif')}")
