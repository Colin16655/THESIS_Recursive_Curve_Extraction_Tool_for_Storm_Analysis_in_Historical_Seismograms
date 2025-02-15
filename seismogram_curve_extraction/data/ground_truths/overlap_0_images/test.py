import cv2
import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter

def read_and_binarize(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    return binary

def detect_traces(binary_img):
    vertical_band = np.sum(binary_img, axis=1)
    trace_indices = np.where(vertical_band > np.max(vertical_band) * 0.1)[0]
    
    traces = []
    last_idx = -1
    for idx in trace_indices:
        if last_idx == -1 or idx - last_idx > 10:
            traces.append(idx)
        last_idx = idx
    
    return traces

def apply_kalman_filter(trace_y_positions, binary_img):
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = np.array([trace_y_positions[0], 0])
    kf.F = np.array([[1, 1], [0, 1]])
    kf.H = np.array([[1, 0]])
    kf.P *= 10
    kf.R = 1
    kf.Q = np.array([[0.1, 0], [0, 0.1]])
    
    filtered_trace = []
    for y in trace_y_positions:
        kf.predict()
        kf.update(y)
        filtered_trace.append(kf.x[0])
    
    return filtered_trace

def extract_traces(binary_img, traces):
    all_traces = []
    for trace_idx in traces:
        trace_y_positions = np.where(binary_img[trace_idx, :] > 0)[0]
        if len(trace_y_positions) > 0:
            filtered_trace = apply_kalman_filter(trace_y_positions, binary_img)
            all_traces.append(filtered_trace)
    return all_traces

# Load and process image
image_path = 'seismogram_curve_extraction\data\ground_truths\overlap_0_images\0.pdf'
binary_img = read_and_binarize(image_path)
traces = detect_traces(binary_img)
extracted_traces = extract_traces(binary_img, traces)

# Plot results
plt.imshow(binary_img, cmap='gray')
for trace in extracted_traces:
    plt.plot(trace, range(len(trace)), color='red')
plt.show()
