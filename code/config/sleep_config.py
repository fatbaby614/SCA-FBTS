"""Configuration for sleep EEG analysis pipeline."""
from pathlib import Path

# EEG acquisition parameters
CHANNEL_LABELS = [
    "C3",
    "C4",
    "Cz",
    "F3",
    "F4",
    "Fz",
    "O1",
    "O2",
]
EXPECTED_CHANNEL_COUNT = len(CHANNEL_LABELS)
SAMPLE_RATE_HZ = 100  # Common sleep EEG sampling rate
CHUNK_LENGTH_SEC = 30.0  # 30-second epochs (AASM standard)

# Data collection / storage
DATA_ROOT = Path("data")
MAT_FILE_TEMPLATE = "subject_{subject_id}_session_{session_id}.mat"

# Sleep stage labels (AASM standard)
SLEEP_STAGE_LABELS = {
    0: "W",    # Wake
    1: "N1",   # N1 sleep
    2: "N2",   # N2 sleep
    3: "N3",   # N3 sleep (deep sleep)
    4: "REM",  # REM sleep
}
SLEEP_STAGE_CUES = {
    0: "Wake",
    1: "N1",
    2: "N2",
    3: "N3",
    4: "REM",
}

# Epoch parameters for sleep analysis
EPOCH_DURATION_SEC = 30.0  # AASM standard 30-second epochs
PRE_EVENT_MARGIN_SEC = 0.0
BASELINE_DURATION_SEC = 0.0

# Filter bank parameters optimized for sleep EEG
FILTER_BANKS = [
    (0.5, 4),   # Delta wave (slow wave sleep)
    (4, 8),     # Theta wave (drowsiness, N1)
    (8, 12),    # Alpha wave (relaxed wakefulness)
    (12, 14),   # Sigma wave (sleep spindles, N2)
    (14, 20),   # Low beta
    (20, 30),   # High beta
]
CSP_COMPONENTS_PER_BAND = 6
SVM_KERNEL = "rbf"
SVM_C = 1.0
SVM_CLASS_WEIGHT = "balanced"  # Important for imbalanced sleep stages

# Training / evaluation
CROSS_VALIDATION_FOLDS = 5
MODEL_OUTPUT_DIR = Path("models")
MODEL_ARTIFACT_BASENAME = "sleep_fbts_svm_model"

# Real-time decoding (if needed)
SLIDING_WINDOW_SEC = 30.0
WINDOW_STEP_SEC = 30.0
MAJORITY_VOTE_WINDOW = 5
CONFIDENCE_THRESHOLD = 0.5
IDLE_STAGE = "W"

# UI parameters (if needed)
SCREEN_SIZE = (1200, 900)
REFRESH_RATE_HZ = 30
FONT_NAME = "Arial"
BACKGROUND_COLOR = (10, 10, 40)
CURSOR_COLOR = (255, 200, 0)
RESOURCE_ROOT = Path("res")
