import numpy as np
import matplotlib.pyplot as plt
import cv2

def generate_seismogram_raster(signals, 
                               width=800, 
                               height=400, 
                               line_thickness=2, 
                               noise_level=0.1, 
                                l_margin=50, 
                                r_margin=50, 
                                t_margin=50, 
                                b_margin=50, 
                                spacing=10, 
                                overlap_percentage=0.2, 
                                color_mode='rgb'):
    """Generate a raster-style artificial old paper seismogram from multiple given signals with adjustable margins, spacing, overlap, and color mode."""
    # Create blank canvas with aged paper color (beige in RGB)
    if color_mode == 'rgb':
        background = np.full((height, width, 3), (235, 215, 180), dtype=np.uint8)
    else:  # 'bw' mode
        background = np.full((height, width), 235, dtype=np.uint8)  # Grayscale background
    
    num_signals = len(signals)
    
    # Ensure that the signals fit within the image bounds while respecting the margins
    available_height = height - t_margin - b_margin  # Space available for signals
    max_amplitude = (available_height / num_signals) / (1 - overlap_percentage)  # Control overlap
    
    # Create vertical offsets while considering margins and overlap
    vertical_offsets = np.linspace(t_margin + max_amplitude / 2, height - b_margin - max_amplitude / 2, num_signals)
    
    # Horizontal shifting
    available_width = width - l_margin - r_margin
    horizontal_offsets = l_margin # np.linspace(l_margin, l_margin + (num_signals - 1) * spacing, num_signals).astype(int)
    
    for i, signal in enumerate(signals):
        # Normalize and scale signal
        signal = (signal - np.mean(signal)) / np.max(np.abs(signal))  # Center and normalize
        scaled_signal = (max_amplitude / 2) * signal + vertical_offsets[i]  # Rescale amplitude to fit within frame
        
        # Create x-coordinates for plotting with horizontal offset
        x_coords = np.linspace(0, available_width - 1, len(signal)).astype(np.int32) + horizontal_offsets
        y_coords = scaled_signal.astype(np.int32)  # Ensure y-coordinates stay in bounds
        
        # Ensure x-coordinates remain within image bounds
        # x_coords = np.clip(x_coords, l_margin, width - r_margin - 1)
        
        # Draw waveform (in black for both RGB and BW modes)
        for j in range(1, len(x_coords)):
            if color_mode == 'rgb':
                cv2.line(background, (x_coords[j-1], y_coords[j-1]), (x_coords[j], y_coords[j]), (0, 0, 0), line_thickness)
            else:  # 'bw' mode
                cv2.line(background, (x_coords[j-1], y_coords[j-1]), (x_coords[j], y_coords[j]), 0, line_thickness)
    
    # Add noise for old paper effect
    noise = np.random.normal(0, noise_level * 255, background.shape).astype(np.int32)
    noisy_image = np.clip(background + noise, 0, 255).astype(np.uint8)
    
    # Apply slight blur for authenticity
    final_image = cv2.GaussianBlur(noisy_image, (3, 3), 0)
    
    return final_image

# Example usage with different overlap levels and color modes:
signals = [np.sin(np.linspace(0, 10 * np.pi, 1000)), 
           np.cos(np.linspace(0, 10 * np.pi, 1000)),
           np.sin(np.linspace(0, 15 * np.pi, 1000) + 1)]  # Simulated sine and cosine waves

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, overlap in enumerate([0.0, 0.2, 0.5]):  # Different overlap levels
    seismogram_image_rgb = generate_seismogram_raster(signals, l_margin=50, r_margin=50, t_margin=50, b_margin=50, 
                                                      spacing=20, overlap_percentage=overlap, color_mode='rgb')
    seismogram_image_bw = generate_seismogram_raster(signals, l_margin=50, r_margin=50, t_margin=50, b_margin=50, 
                                                     spacing=20, overlap_percentage=overlap, color_mode='bw')
    
    # RGB image
    axes[i].imshow(seismogram_image_rgb)
    axes[i].set_title(f'Overlap: {overlap * 100:.0f}% (RGB)')
    axes[i].axis('off')

plt.show()

