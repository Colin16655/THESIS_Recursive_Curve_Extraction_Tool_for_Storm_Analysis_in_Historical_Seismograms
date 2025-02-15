"""
Seismogram Curve Extraction Package

This package provides tools for:
- Generating ground truth data
- Preprocessing seismograms
- Implementing filtering and machine learning models
- Visualizing and evaluating extracted curves
"""

# Import key modules for convenient access
from .stat_analysis import sanitize_filename, SeismogramAnalysis
from .data_generation import SeismogramGenerator
# from .preprocessing import preprocess_seismogram
# from .pipeline import run_pipeline

# # Import specific filters and models
# from .filters.kalman_filter import KalmanFilter
# from .filters.extended_kalman import ExtendedKalmanFilter
# from .filters.unscented_kalman import UnscentedKalmanFilter
# from .filters.particle_filter import ParticleFilter
# from .models.rnn import RNN
# from .models.lstm import LSTM

# Define the public API of the package
# __all__ is a list of strings specifying which symbols (functions, classes, variables) are considered public.
# It controls what is imported when a user does `from seismogram_extraction import *`.
# Only names listed in __all__ will be accessible in such cases.
__all__ = [
    "sanitize_filename",
    "SeismogramAnalysis",
    "SeismogramGenerator"
]
    # "preprocess_seismogram",
    # "run_pipeline",
    # "KalmanFilter",
    # "ExtendedKalmanFilter",
    # "UnscentedKalmanFilter",
    # "ParticleFilter",
    # "RNN",
    # "LSTM",
# ]
