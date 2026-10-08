import numpy as np
import pickle
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from scipy import signal
from sklearn.feature_selection import SelectKBest, f_classif

# Set MNE log level to suppress LEDOIT_WOLF messages
import mne
mne.set_log_level('warning')  # Only show warnings and errors

# Import pyriemann for Riemannian geometry methods
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

# Import random state from config
from config.algorithms_config import RANDOM_STATE

def apply_bandpass_filter(data, low_freq, high_freq, fs):
    """Apply bandpass filter to EEG data.
    
    Args:
        data: Input data of shape (n_times,)
        low_freq: Lower frequency bound
        high_freq: Upper frequency bound
        fs: Sampling frequency
        
    Returns:
        Filtered data of shape (n_times,)
    """
    nyquist = 0.5 * fs
    low = low_freq / nyquist
    high = high_freq / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    filtered_data = signal.filtfilt(b, a, data, axis=-1)
    return filtered_data


class SCA_FBTS:
    """SCA-FBTS: Sleep Context-Aware Filter Bank Tangent Space algorithm for EEG sleep staging.
    
    This algorithm combines multi-band filtering with Riemannian tangent space and
    temporal smoothing to improve sleep staging performance, especially for N1 detection.
    
    Key features:
    1. Multi-band filtering for capturing different sleep rhythms
    2. Riemannian tangent space projection for robust feature extraction
    3. Temporal smoothing with N1 protection for context-aware classification
    4. Feature selection to avoid overfitting
    
    Reference: Based on Filter Bank Tangent Space with added temporal smoothing module
    """
    def __init__(self, n_bands=6, estimator='oas', metric='riemann', 
                 classifier='svm', n_features=100, fs=100, 
                 freq_bands=None, temporal_smoothing=True, smoothing_window=3,
                 causal_smoothing=False):
        """Initialize SCA-FBTS classifier.
        
        Args:
            n_bands: Number of frequency bands (default: 6)
            estimator: Covariance matrix estimator ('oas', 'lwf', 'scm', 'cov', 'corr')
            metric: Metric for tangent space projection ('riemann', 'euclid', 'logeuclid')
            classifier: Classifier to use ('lda', 'svm', 'rf')
            n_features: Number of features to select (default: 100)
            fs: Sampling frequency (default: 100)
            freq_bands: Custom frequency bands (list of tuples), overrides n_bands
            temporal_smoothing: Whether to apply temporal smoothing (default: True for SCA-FBTS)
            smoothing_window: Window size for temporal smoothing (default: 3)
            causal_smoothing: If False (default), a centered (zero-phase, i.e.
                non-causal) moving average is used, which also uses future epochs.
                If True, the window only includes the current and past epochs
                (suitable for real-time monitoring). Setting causal_smoothing=True
                implies temporal_smoothing=True.
        """
        self.n_bands = n_bands
        self.estimator = estimator
        self.metric = metric
        self.fs = fs
        self.n_features = n_features
        self.classifier_name = classifier
        self.temporal_smoothing = temporal_smoothing or causal_smoothing
        self.smoothing_window = smoothing_window
        self.causal_smoothing = causal_smoothing
        
        if freq_bands is not None:
            self.freq_bands = freq_bands
        else:
            # Sleep EEG optimized frequency bands (Fine-grained for N1/Alpha transition)
            self.freq_bands = [
                (0.5, 4),    # Delta (deep sleep, N3)
                (4, 6),      # Low Theta (N1 onset)
                (6, 8),      # High Theta 
                (8, 10),     # Low Alpha (wakefulness to N1)
                (10, 12),    # High Alpha
                (12, 14),    # Sigma (sleep spindles, N2)
                (14, 20),    # Low beta
                (20, 30),    # High beta
            ]
        
        self.cov_estimators = []
        self.ts_transformers = []
        self.feature_selector = None
        self.classifier = None
    
    def fit(self, X, y):
        """Fit the SCA-FBTS classifier.
        
        Args:
            X: Input data of shape (n_samples, n_channels, n_times)
            y: Target labels of shape (n_samples,)
        
        Returns:
            self: Fitted model
        """
        n_samples = X.shape[0]
        
        features_list = []
        self.cov_estimators = []
        self.ts_transformers = []
        
        for low, high in self.freq_bands:
            print(f"    Processing band {low}-{high}Hz...")
            
            # Step 1: Filter Bank
            X_band = np.array([apply_bandpass_filter(trial, low, high, self.fs) for trial in X])
            
            # Step 2: Compute covariance matrices
            cov_estimator = Covariances(estimator=self.estimator)
            cov_matrices = cov_estimator.fit_transform(X_band)
            self.cov_estimators.append(cov_estimator)
            
            # Step 3: Project to tangent space
            ts_transformer = TangentSpace(metric=self.metric)
            ts_features = ts_transformer.fit_transform(cov_matrices, y)
            self.ts_transformers.append(ts_transformer)
            
            features_list.append(ts_features)
        
        # Step 4: Concatenate features from all bands
        X_combined = np.hstack(features_list)
        print(f"    Combined feature dimension: {X_combined.shape[1]}")
        
        # Step 5: Feature selection (CRITICAL to avoid overfitting)
        if self.n_features is not None:
            print(f"    Selecting top {self.n_features} features...")
            self.feature_selector = SelectKBest(f_classif, k=min(self.n_features, X_combined.shape[1]))
            X_selected = self.feature_selector.fit_transform(X_combined, y)
            print(f"    Selected feature dimension: {X_selected.shape[1]}")
        else:
            print(f"    No feature selection (using all {X_combined.shape[1]} features)...")
            self.feature_selector = None
            X_selected = X_combined
        
        # Step 6: Train classifier
        if self.classifier_name == 'lda':
            self.classifier = LDA()
        elif self.classifier_name == 'svm':
            self.classifier = SVC(kernel='rbf', C=1.0, probability=True, class_weight='balanced', random_state=RANDOM_STATE)
        elif self.classifier_name == 'rf':
            self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unknown classifier: {self.classifier_name}")
        
        self.classifier.fit(X_selected, y)
        self.classes_ = self.classifier.classes_
        return self
    
    def predict(self, X):
        """Predict labels for new data with temporal smoothing.
        
        Args:
            X: Input data of shape (n_samples, n_channels, n_times)
            
        Returns:
            y_pred: Predicted labels of shape (n_samples,)
        """
        features_list = []
        
        for i, (low, high) in enumerate(self.freq_bands):
            # Step 1: Filter Bank
            X_band = np.array([apply_bandpass_filter(trial, low, high, self.fs) for trial in X])
            
            # Step 2: Compute covariance matrices
            cov_matrices = self.cov_estimators[i].transform(X_band)
            
            # Step 3: Project to tangent space
            ts_features = self.ts_transformers[i].transform(cov_matrices)
            
            features_list.append(ts_features)
        
        # Step 4: Concatenate features from all bands
        X_combined = np.hstack(features_list)
        
        # Step 5: Apply feature selection
        if self.feature_selector is not None:
            X_selected = self.feature_selector.transform(X_combined)
        else:
            X_selected = X_combined
        
        # Step 6: Predict with temporal smoothing if enabled
        if getattr(self, 'temporal_smoothing', False):
            # For SCA-FBTS, get smoothed probabilities first, then argmax
            if hasattr(self.classifier, "predict_proba"):
                y_proba = self.classifier.predict_proba(X_selected)
                if len(y_proba) > 1:
                    from scipy.ndimage import uniform_filter1d
                    # Centered (non-causal) window by default; causal window (only
                    # current + past epochs) if causal_smoothing is enabled.
                    origin = (self.smoothing_window - 1) // 2 if self.causal_smoothing else 0
                    mode = 'nearest' if self.causal_smoothing else 'reflect'
                    y_proba_smoothed = uniform_filter1d(y_proba, size=self.smoothing_window,
                                                        axis=0, origin=origin, mode=mode)
                    
                    # Adaptive Smoothing: Rescue N1 (protect brief transitions)
                    # Find N1 class index based on label characteristics
                    n1_idx = -1
                    # Try to identify N1 by common label patterns
                    for i, c in enumerate(self.classes_):
                        c_str = str(c).lower()
                        if 'n1' in c_str or 'sleep stage 1' in c_str or '1' == c_str.strip():
                            n1_idx = i
                            break
                    # If not found, try by label index (common: 1 or 0)
                    if n1_idx == -1 and len(self.classes_) >= 2:
                        # Common N1 indices: 1 (W=0, N1=1, N2=2, N3=3, REM=4)
                        # or 0 in some datasets
                        if 1 < len(self.classes_):
                            n1_idx = 1
                    if n1_idx != -1 and n1_idx < len(self.classes_):
                        raw_preds = np.argmax(y_proba, axis=1)
                        is_n1 = (raw_preds == n1_idx)
                        # Keep original probabilities where the model is confident it's N1
                        y_proba_smoothed = np.where(is_n1[:, None], y_proba, y_proba_smoothed)
                        
                    return self.classes_[np.argmax(y_proba_smoothed, axis=1)]
                
        return self.classifier.predict(X_selected)
    
    def predict_proba(self, X):
        """Predict class probabilities for new data with temporal smoothing.
        
        Args:
            X: Input data of shape (n_samples, n_channels, n_times)
            
        Returns:
            y_proba: Predicted probabilities of shape (n_samples, n_classes)
        """
        features_list = []
        
        for i, (low, high) in enumerate(self.freq_bands):
            # Step 1: Filter Bank
            X_band = np.array([apply_bandpass_filter(trial, low, high, self.fs) for trial in X])
            
            # Step 2: Compute covariance matrices
            cov_matrices = self.cov_estimators[i].transform(X_band)
            
            # Step 3: Project to tangent space
            ts_features = self.ts_transformers[i].transform(cov_matrices)
            
            features_list.append(ts_features)
        
        # Step 4: Concatenate features from all bands
        X_combined = np.hstack(features_list)
        
        # Step 5: Apply feature selection
        if self.feature_selector is not None:
            X_selected = self.feature_selector.transform(X_combined)
        else:
            X_selected = X_combined
        
        # Step 6: Predict probabilities
        y_proba = self.classifier.predict_proba(X_selected)
        
        # Step 7: Temporal Smoothing if enabled (SCA-FBTS)
        if getattr(self, 'temporal_smoothing', False) and len(y_proba) > 1:
            from scipy.ndimage import uniform_filter1d
            # Apply moving average filter along the time axis (axis 0)
            origin = (self.smoothing_window - 1) // 2 if self.causal_smoothing else 0
            mode = 'nearest' if self.causal_smoothing else 'reflect'
            y_proba_smoothed = uniform_filter1d(y_proba, size=self.smoothing_window,
                                                axis=0, origin=origin, mode=mode)
            
            # Adaptive Smoothing: Rescue N1 (protect brief transitions)
            # Find N1 class index based on label characteristics
            n1_idx = -1
            # Try to identify N1 by common label patterns
            for i, c in enumerate(self.classes_):
                c_str = str(c).lower()
                if 'n1' in c_str or 'sleep stage 1' in c_str or '1' == c_str.strip():
                    n1_idx = i
                    break
            # If not found, try by label index (common: 1 or 0)
            if n1_idx == -1 and len(self.classes_) >= 2:
                # Common N1 indices: 1 (W=0, N1=1, N2=2, N3=3, REM=4)
                # or 0 in some datasets
                if 1 < len(self.classes_):
                    n1_idx = 1
            if n1_idx != -1 and n1_idx < len(self.classes_):
                raw_preds = np.argmax(y_proba, axis=1)
                is_n1 = (raw_preds == n1_idx)
                # Keep original probabilities where the model is confident it's N1
                y_proba_smoothed = np.where(is_n1[:, None], y_proba, y_proba_smoothed)
                
            y_proba_smoothed = y_proba_smoothed / y_proba_smoothed.sum(axis=1, keepdims=True)
            return y_proba_smoothed
            
        return y_proba
    
    def save_model(self, path):
        """Save the model to a file.
        
        Args:
            path: Path to save the model
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load_model(cls, path):
        """Load the model from a file.
        
        Args:
            path: Path to load the model from
            
        Returns:
            Loaded model instance
        """
        with open(path, 'rb') as f:
            return pickle.load(f)
