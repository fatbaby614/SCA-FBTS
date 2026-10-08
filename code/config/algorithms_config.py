import os
from datetime import datetime


# Get project root directory (code/config/algorithms_config.py -> project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_PATH = os.path.join(PROJECT_ROOT, 'data')
RESULTS_PATH = os.path.join(PROJECT_ROOT, 'results')
MODELS_PATH = os.path.join(PROJECT_ROOT, 'models')


def get_timestamped_filename(base_filename, extension=None):
    """
    Generate a filename with timestamp to avoid overwriting previous results.
    
    Args:
        base_filename: Base name of the file (without extension)
        extension: File extension (e.g., 'csv', 'json', 'png')
    
    Returns:
        Filename with timestamp (e.g., 'results_20250306_143052.csv')
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if extension:
        return f"{base_filename}_{timestamp}.{extension}"
    return f"{base_filename}_{timestamp}"


def get_results_path(filename=None):
    """
    Get the path to the results directory or a specific file in it.
    
    Args:
        filename: Optional filename to append to the results path
    
    Returns:
        Path to results directory or specific file
    """
    os.makedirs(RESULTS_PATH, exist_ok=True)
    if filename:
        return os.path.join(RESULTS_PATH, filename)
    return RESULTS_PATH


os.makedirs(RESULTS_PATH, exist_ok=True)
os.makedirs(MODELS_PATH, exist_ok=True)


# ===== Sleep EEG Analysis Configuration =====
FS = 100  # Typical sleep EEG sampling rate
LOW_FREQ = 0.5  # Lower frequency for sleep analysis (delta band)
HIGH_FREQ = 30  # Higher frequency for sleep analysis (beta band)
TMIN = 0.0  # Start of epoch
TMAX = 30.0  # End of epoch (30-second epochs for AASM standard)

SLEEP_BANDS = [
    (0.5, 4, 'delta'),
    (4, 8, 'theta'),
    (8, 12, 'alpha'),
    (12, 16, 'sigma'),
    (16, 30, 'beta'),
]

# ===== Motor Imagery Configuration (kept for reference) =====
MI_FS = 250  
MI_LOW_FREQ = 4  
MI_HIGH_FREQ = 40  
MI_TMIN = 0.5
MI_TMAX = 3.0


RANDOM_STATE = 42
N_SPLITS = 5  
