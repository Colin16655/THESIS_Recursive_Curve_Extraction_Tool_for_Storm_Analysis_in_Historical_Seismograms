from raster_image import RasterImage
import matplotlib.pyplot as plt

image_file = r'images/corruptions/UCC19540112Gal_E_0750.tif'

# Hyperparameters
factor = 10  # Downsample sclicing factor

# Load, resize, and display each image
raster_image = RasterImage(image_file, batch_size=(100, 200))
raster_image.load_image()
raster_image.slice_downsample(factor)
raster_image.discard_original_image()


# Get the number of batches
print(f"Number of batches: {len(raster_image)}")

# Iterate over the image batches
for batch in raster_image:
    # Do something with each batch, e.g., display or process
    plt.imshow(batch, cmap='gray')
    plt.show()