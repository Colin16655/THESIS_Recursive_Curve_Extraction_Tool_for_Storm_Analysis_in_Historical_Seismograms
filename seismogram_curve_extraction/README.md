## Project Structure

The project is organized as follows:

```plaintext
seismogram_curve_extraction/
│
├── data/                        # Data storage
│   ├── raw/                     # Raw full historical seismograms
│   ├── sines/                   # Artificial data obtained from simple sinusoidal signals
│   ├── resampled/               # Artificial data obtained from Inverse Fourier Transform
│   ├── seismograms/             # Subsampled raw historical seismograms
│   │   ├── preprocessed/               # Preprocessed data
│   └── results/                 # Outputs from the processing pipeline
│
├── seismogram_extraction/       # Main package
│   ├── __init__.py              # Package initialization
│   ├── data_generation.py       # Code for generating ground truth data (sines and resampled)
│   ├── raster_image.py          # Code for subsampling the raw full historical seismograms
│   ├── preprocessing.py         # Code for preprocessing existing seismograms
│   ├── stat_analysis.py         # Code for preprocessing existing seismograms
│   ├── filters/                 # Filter implementations
│   │   ├── kalman_filter.py     # Base Kalman filter implementation
│   ├── models/                  # Neural network implementations
│   │   ├── rnn.py               # Recurrent Neural Network implementation
│   │   └── lstm.py              # Long Short-Term Memory implementation
│   ├── utils/                   # Utility functions
│   │   ├── visualization.py     # Code for plotting and visualizing curves
│   │   └── metrics.py           # Evaluation metrics (e.g., RMSE, accuracy)
│   └── pipeline.py              # Script orchestrating the entire pipeline
│
├── tests/                       # Tests for the project
├── notebooks/                   # Jupyter notebooks for exploration and experiments
├── docs/                        # Documentation
├── config/                      # Configuration files
├── requirements.txt             # Dependencies
└── README.md                    # Project overview
