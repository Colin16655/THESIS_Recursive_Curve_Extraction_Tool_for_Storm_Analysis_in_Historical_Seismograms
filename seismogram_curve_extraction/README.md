## Project Structure

The project is organized as follows:

```plaintext
seismogram_curve_extraction/
│
├── data/                        # Data storage
│   ├── raw/                     # Raw historical seismograms
│   ├── processed/               # Preprocessed data
│   ├── ground_truth/            # Ground truth data
│   └── results/                 # Outputs from the processing pipeline
│
├── seismogram_extraction/       # Main package
│   ├── __init__.py              # Package initialization
│   ├── data_generation.py       # Code for generating ground truth data
│   ├── preprocessing.py         # Code for preprocessing existing seismograms
│   ├── filters/                 # Filter implementations
│   │   ├── kalman_filter.py     # Base Kalman filter implementation
│   │   ├── extended_kalman.py   # Extended Kalman filter version
│   │   ├── unscented_kalman.py  # Unscented Kalman filter version
│   │   └── particle_filter.py   # Particle filter implementation
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
├── scripts/                     # Standalone scripts or tools
├── docs/                        # Documentation
├── config/                      # Configuration files
├── requirements.txt             # Dependencies
└── README.md                    # Project overview
