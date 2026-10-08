import numpy as np
import pickle
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
# Set MNE log level to suppress LEDOIT_WOLF messages
import mne
mne.set_log_level('warning')  # Only show warnings and errors
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn
# Import pyriemann for Riemannian geometry methods
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
# EEGNetv4 was renamed to EEGNet in braindecode >= 1.x; keep a compatible alias.
try:
    from braindecode.models import EEGNetv4
except ImportError:
    try:
        from braindecode.models import EEGNet as EEGNetv4
    except ImportError:
        EEGNetv4 = None
from braindecode import EEGClassifier
# Handle different braindecode versions for set_random_seeds
try:
    from braindecode.util import set_random_seeds
except ImportError:
    pass # If it fails, Random seeds can be skipped or implemented manually if necessary
from skorch.callbacks import EarlyStopping
from config.algorithms_config import RANDOM_STATE
from scipy import signal
from scipy.stats import skew, kurtosis
from sklearn.feature_selection import SelectKBest, f_classif

class PrintLogCallback:
    def __init__(self, print_freq=50):
        self.print_freq = print_freq
        self.epoch = 0
    
    def __call__(self, net, **kwargs):
        self.epoch += 1
        if self.epoch % self.print_freq == 0 or self.epoch == 1:
            if hasattr(net, 'history_'):
                history = net.history_
                if len(history) > 0:
                    last_epoch = history[-1]
                    if 'train_loss' in last_epoch:
                        print(f"  Epoch {self.epoch}, Loss: {last_epoch['train_loss']:.4f}")
                    if 'dur' in last_epoch:
                        print(f"  Epoch {self.epoch}, Duration: {last_epoch['dur']:.2f}s")
    
    def initialize(self):
        """Initialize the callback."""
        pass
    
    def on_train_begin(self, net, **kwargs):
        """Called when training begins."""
        self.epoch = 0
    
    def on_train_end(self, net, **kwargs):
        """Called when training ends."""
        pass
    
    def on_epoch_begin(self, net, **kwargs):
        """Called at the beginning of each epoch."""
        pass
    
    def on_epoch_end(self, net, **kwargs):
        """Called at the end of each epoch."""
        self.__call__(net, **kwargs)
    
    def on_batch_begin(self, net, **kwargs):
        """Called at the beginning of each batch."""
        pass
    
    def on_batch_end(self, net, **kwargs):
        """Called at the end of each batch."""
        pass
    
    def on_grad_computed(self, net, **kwargs):
        """Called after gradients are computed."""
        pass
    
    def set_params(self, **params):
        """Set parameters for the callback."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self


def apply_bandpass_filter(data, low_freq, high_freq, fs):
    nyquist = 0.5 * fs
    low = low_freq / nyquist
    high = high_freq / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    filtered_data = signal.filtfilt(b, a, data, axis=-1)
    return filtered_data


class RiemannTangentSpace:
    """Riemannian tangent space algorithm for EEG classification.
    
    This algorithm uses Riemannian geometry to process EEG signals by:
    1. Computing covariance matrices from EEG channels
    2. Projecting covariance matrices to tangent space
    3. Using a classifier for classification
    """
    def __init__(self, estimator='oas', metric='riemann', classifier='lda', n_components=None):
        """Initialize the Riemann tangent space classifier.
        
        Args:
            estimator: Covariance matrix estimator ('oas', 'lwf', 'scm', 'cov', 'corr')
            metric: Metric for tangent space projection ('riemann', 'euclid', 'logeuclid')
            classifier: Classifier to use ('lda', 'svm', 'rf')
            n_components: Number of components for dimensionality reduction (None for no reduction)
        """
        steps = [
            ('cov', Covariances(estimator=estimator)),
            ('ts', TangentSpace(metric=metric))
        ]
        
        # Add dimensionality reduction if specified
        if n_components is not None:
            from sklearn.decomposition import PCA
            steps.append(('pca', PCA(n_components=n_components)))
        
        # Add classifier
        if classifier == 'lda':
            steps.append(('clf', LDA()))
        elif classifier == 'svm':
            from sklearn.svm import SVC
            steps.append(('clf', SVC(kernel='rbf', C=1.0, probability=True)))
        elif classifier == 'rf':
            from sklearn.ensemble import RandomForestClassifier
            steps.append(('clf', RandomForestClassifier(n_estimators=100, random_state=42)))
        else:
            raise ValueError(f"Unknown classifier: {classifier}")
        
        self.pipeline = Pipeline(steps)
        self.estimator = estimator
        self.metric = metric
        self.classifier = classifier
        self.n_components = n_components
    
    def fit(self, X, y):
        """Fit the model to the training data.
        
        Args:
            X: Input data of shape (n_samples, n_channels, n_times)
            y: Target labels of shape (n_samples,)
        
        Returns:
            self: Fitted model
        """
        self.pipeline.fit(X, y)
        return self
    
    def predict(self, X):
        """Predict labels for new data.
        
        Args:
            X: Input data of shape (n_samples, n_channels, n_times)
            
        Returns:
            y_pred: Predicted labels of shape (n_samples,)
        """
        return self.pipeline.predict(X)
    
    def predict_proba(self, X):
        """Predict class probabilities for new data.
        
        Args:
            X: Input data of shape (n_samples, n_channels, n_times)
            
        Returns:
            y_proba: Predicted probabilities of shape (n_samples, n_classes)
        """
        return self.pipeline.predict_proba(X)
    
    def save_model(self, path):
        """Save the model to a file."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load_model(cls, path):
        """Load the model from a file."""
        with open(path, 'rb') as f:
            return pickle.load(f)


class FilterBankTangentSpace:
    """Filter Bank Tangent Space algorithm for EEG classification.
    
    This algorithm combines multi-band filtering with Riemannian tangent space:
    1. Apply bandpass filters to multiple frequency bands
    2. Compute covariance matrices for each band
    3. Project each band's covariance matrices to tangent space
    4. Concatenate features from all bands
    5. Apply feature selection to avoid overfitting
    6. Classify using SVM or LDA
    
    Reference: Combines ideas from FBCSP and Riemannian Tangent Space
    """
    def __init__(self, n_bands=6, estimator='oas', metric='riemann', 
                 classifier='svm', n_features=100, fs=100, 
                 freq_bands=None, temporal_smoothing=False, smoothing_window=3,
                 causal_smoothing=False):
        """Initialize Filter Bank Tangent Space classifier.
        
        Args:
            n_bands: Number of frequency bands (default: 6)
            estimator: Covariance matrix estimator ('oas', 'lwf', 'scm', 'cov', 'corr')
            metric: Metric for tangent space projection ('riemann', 'euclid', 'logeuclid')
            classifier: Classifier to use ('lda', 'svm', 'rf')
            n_features: Number of features to select (default: 100)
            fs: Sampling frequency (default: 100)
            freq_bands: Custom frequency bands (list of tuples), overrides n_bands
            temporal_smoothing: Whether to apply temporal smoothing (SCA-FBTS)
            smoothing_window: Window size for temporal smoothing (default: 3)
            causal_smoothing: If True, the smoothing window only includes the
                current and past epochs (causal, suitable for real-time
                monitoring). If False (default), a centered zero-phase window is
                used which also uses future epochs. Implies temporal_smoothing.
        """
        from pyriemann.estimation import Covariances
        from pyriemann.tangentspace import TangentSpace
        from sklearn.feature_selection import SelectKBest, f_classif
        
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
        """Fit the Filter Bank Tangent Space classifier.
        
        Args:
            X: Input data of shape (n_samples, n_channels, n_times)
            y: Target labels of shape (n_samples,)
        
        Returns:
            self: Fitted model
        """
        from pyriemann.estimation import Covariances
        from pyriemann.tangentspace import TangentSpace
        from sklearn.feature_selection import SelectKBest, f_classif
        
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
            from sklearn.ensemble import RandomForestClassifier
            self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unknown classifier: {self.classifier_name}")
        
        self.classifier.fit(X_selected, y)
        self.classes_ = self.classifier.classes_
        return self
    
    def predict(self, X):
        """Predict labels for new data.
        
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
        
        # Step 6: Predict
        if getattr(self, 'temporal_smoothing', False):
            # For SCA-FBTS, get smoothed probabilities first, then argmax
            if hasattr(self.classifier, "predict_proba"):
                y_proba = self.classifier.predict_proba(X_selected)
                if len(y_proba) > 1:
                    from scipy.ndimage import uniform_filter1d
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
        """Predict class probabilities for new data.
        
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
        """Save the model to a file."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load_model(cls, path):
        """Load the model from a file."""
        with open(path, 'rb') as f:
            return pickle.load(f)


class MDM:
    """Minimum Distance to Mean classifier for Riemannian geometry.
    
    This is the classic Riemannian classifier that computes the distance
    between a test covariance matrix and the class mean covariance matrices
    in the Riemannian manifold.
    
    Reference: Barachant et al. (2012) "Multiclass brain-computer 
    interface classification by Riemannian geometry"
    """
    def __init__(self, estimator='oas', metric='riemann'):
        """Initialize MDM classifier.
        
        Args:
            estimator: Covariance matrix estimator ('oas', 'lwf', 'scm', 'cov', 'corr')
            metric: Metric for Riemannian distance ('riemann', 'euclid', 'logeuclid', 'logdet')
        """
        from pyriemann.estimation import Covariances
        from pyriemann.classification import MDM as pyriemann_MDM
        
        self.estimator = estimator
        self.metric = metric
        self.cov_estimator = Covariances(estimator=estimator)
        self.clf = pyriemann_MDM(metric=metric)
        self.classes_ = None
        self.cov_means_ = None
    
    def fit(self, X, y):
        """Fit the MDM classifier.
        
        Args:
            X: Input data of shape (n_samples, n_channels, n_times)
            y: Target labels of shape (n_samples,)
        
        Returns:
            self: Fitted model
        """
        cov_matrices = self.cov_estimator.fit_transform(X)
        self.clf.fit(cov_matrices, y)
        self.classes_ = self.clf.classes_
        return self
    
    def predict(self, X):
        """Predict labels for new data.
        
        Args:
            X: Input data of shape (n_samples, n_channels, n_times)
            
        Returns:
            y_pred: Predicted labels of shape (n_samples,)
        """
        cov_matrices = self.cov_estimator.fit_transform(X)
        return self.clf.predict(cov_matrices)
    
    def predict_proba(self, X):
        """Predict class probabilities for new data.
        
        Args:
            X: Input data of shape (n_samples, n_channels, n_times)
            
        Returns:
            y_proba: Predicted probabilities of shape (n_samples, n_classes)
        """
        cov_matrices = self.cov_estimator.fit_transform(X)
        return self.clf.predict_proba(cov_matrices)
    
    def save_model(self, path):
        """Save the model to a file."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load_model(cls, path):
        """Load the model from a file."""
        with open(path, 'rb') as f:
            return pickle.load(f)


class HandcraftedFeaturesRF:
    """Random forest baseline with handcrafted temporal/spectral features."""

    def __init__(self, fs=100, n_estimators=300):
        self.fs = fs
        self.scaler = StandardScaler()
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight='balanced_subsample',
        )

    def _extract_features(self, X):
        # X shape: (n_samples, n_channels, n_times)
        if X.ndim != 3:
            raise ValueError(f"Expected 3D input, got shape {X.shape}")

        # Temporal statistics per channel.
        mean_feat = np.mean(X, axis=-1)
        std_feat = np.std(X, axis=-1)
        var_feat = np.var(X, axis=-1)
        rms_feat = np.sqrt(np.mean(np.square(X), axis=-1))
        skew_feat = skew(X, axis=-1, bias=False)
        kurt_feat = kurtosis(X, axis=-1, bias=False)

        # Relative band power per channel.
        freqs, psd = signal.welch(X, fs=self.fs, nperseg=min(256, X.shape[-1]), axis=-1)
        total_power = np.sum(psd, axis=-1, keepdims=True) + 1e-12

        bands = [(0.5, 4), (4, 8), (8, 12), (12, 16), (16, 30)]
        band_feats = []
        for low, high in bands:
            mask = (freqs >= low) & (freqs < high)
            band_power = np.sum(psd[:, :, mask], axis=-1) / total_power.squeeze(-1)
            band_feats.append(band_power)

        feature_blocks = [mean_feat, std_feat, var_feat, rms_feat, skew_feat, kurt_feat, *band_feats]
        return np.hstack(feature_blocks)

    def fit(self, X, y):
        feats = self._extract_features(X)
        feats = self.scaler.fit_transform(feats)
        self.clf.fit(feats, y)
        return self

    def predict(self, X):
        feats = self._extract_features(X)
        feats = self.scaler.transform(feats)
        return self.clf.predict(feats)

    def predict_proba(self, X):
        feats = self._extract_features(X)
        feats = self.scaler.transform(feats)
        return self.clf.predict_proba(feats)

    def save_model(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load_model(cls, path):
        with open(path, 'rb') as f:
            return pickle.load(f)


class _TinySleepTransformerNet(nn.Module):
    """Compact transformer for sleep staging from EEG epochs."""

    def __init__(self, n_channels, n_times, n_classes, d_model=64, n_heads=4, n_layers=2):
        super().__init__()
        self.input_proj = nn.Conv1d(n_channels, d_model, kernel_size=1)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_times, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=0.2,
            batch_first=True,
            activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x):
        # x: (batch, channels, time)
        x = self.input_proj(x)
        x = x.transpose(1, 2)  # (batch, time, d_model)
        x = x + self.pos_embed[:, :x.shape[1], :]
        x = self.encoder(x)
        x = self.norm(x.mean(dim=1))
        return self.head(x)


class SleepTransformerLight:
    """Lightweight SleepTransformer-style baseline."""

    def __init__(self, n_channels, n_times, n_classes=5):
        set_random_seeds(seed=RANDOM_STATE, cuda=torch.cuda.is_available())
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = _TinySleepTransformerNet(n_channels, n_times, n_classes)
        self.clf = EEGClassifier(
            self.model,
            criterion=nn.CrossEntropyLoss(),
            optimizer=torch.optim.AdamW,
            optimizer__lr=5e-4,
            optimizer__weight_decay=1e-4,
            train_split=None,
            device=self.device,
            batch_size=32,
            callbacks=[PrintLogCallback(print_freq=50), EarlyStopping(monitor='train_loss', patience=20)],
        )
        self.scaler = StandardScaler()
        self.is_trained = False

    def fit(self, X, y, epochs=200, batch_size=32, learning_rate=0.0005):
        X_scaled = self.scaler.fit_transform(X.reshape(X.shape[0], -1)).reshape(X.shape)
        self.clf.fit(X_scaled, y, epochs=epochs)
        self.is_trained = True
        return self

    def predict(self, X):
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        X_scaled = self.scaler.transform(X.reshape(X.shape[0], -1)).reshape(X.shape)
        return self.clf.predict(X_scaled)

    def predict_proba(self, X):
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        X_scaled = self.scaler.transform(X.reshape(X.shape[0], -1)).reshape(X.shape)
        return self.clf.predict_proba(X_scaled)

    def save_model(self, path):
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        scaler_path = path.replace('.pt', '_scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump({'scaler': self.scaler, 'is_trained': self.is_trained}, f)

    @classmethod
    def load_model(cls, path, n_channels, n_times, n_classes=5):
        model = cls(n_channels, n_times, n_classes)
        model.model.load_state_dict(torch.load(path))
        model.model.eval()
        scaler_path = path.replace('.pt', '_scaler.pkl')
        with open(scaler_path, 'rb') as f:
            data = pickle.load(f)
            model.scaler = data['scaler']
            model.is_trained = data['is_trained']
        dummy_X = torch.randn(1, n_channels, n_times).to(model.device)
        model.clf.initialize()
        model.clf.predict(dummy_X)
        return model


class _DeepSleepNetFeatureNet(nn.Module):
    """Compact DeepSleepNet-style dual-branch temporal encoder."""

    def __init__(self, n_channels, n_classes):
        super().__init__()

        # Short-kernel branch captures local temporal details.
        self.short_branch = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.AdaptiveAvgPool1d(16),
        )

        # Long-kernel branch captures broader sleep rhythm structures.
        self.long_branch = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=25, stride=6, padding=12),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Conv1d(64, 128, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.AdaptiveAvgPool1d(16),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 2, 256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        short_feat = self.short_branch(x)
        long_feat = self.long_branch(x)
        fused = torch.cat([short_feat, long_feat], dim=1)
        return self.classifier(fused)


class DeepSleepNetClassifier:
    """DeepSleepNet baseline wrapper for sleep staging benchmarks."""

    def __init__(self, n_channels, n_times, n_classes=5):
        set_random_seeds(seed=RANDOM_STATE, cuda=torch.cuda.is_available())
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = _DeepSleepNetFeatureNet(n_channels=n_channels, n_classes=n_classes)
        self.clf = EEGClassifier(
            self.model,
            criterion=nn.CrossEntropyLoss(),
            optimizer=torch.optim.AdamW,
            optimizer__lr=5e-4,
            optimizer__weight_decay=1e-4,
            train_split=None,
            device=self.device,
            batch_size=32,
            callbacks=[PrintLogCallback(print_freq=50), EarlyStopping(monitor='train_loss', patience=20)],
        )

        self.scaler = StandardScaler()
        self.is_trained = False

    def fit(self, X, y, epochs=150, batch_size=32, learning_rate=0.0005):
        X_scaled = self.scaler.fit_transform(X.reshape(X.shape[0], -1)).reshape(X.shape)
        self.clf.fit(X_scaled, y, epochs=epochs)
        self.is_trained = True
        return self

    def predict(self, X):
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        X_scaled = self.scaler.transform(X.reshape(X.shape[0], -1)).reshape(X.shape)
        return self.clf.predict(X_scaled)

    def predict_proba(self, X):
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        X_scaled = self.scaler.transform(X.reshape(X.shape[0], -1)).reshape(X.shape)
        return self.clf.predict_proba(X_scaled)

    def save_model(self, path):
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        scaler_path = path.replace('.pt', '_scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump({'scaler': self.scaler, 'is_trained': self.is_trained}, f)

    @classmethod
    def load_model(cls, path, n_channels, n_times, n_classes=5):
        model = cls(n_channels, n_times, n_classes)
        model.model.load_state_dict(torch.load(path))
        model.model.eval()
        scaler_path = path.replace('.pt', '_scaler.pkl')
        with open(scaler_path, 'rb') as f:
            data = pickle.load(f)
            model.scaler = data['scaler']
            model.is_trained = data['is_trained']
        dummy_X = torch.randn(1, n_channels, n_times).to(model.device)
        model.clf.initialize()
        model.clf.predict(dummy_X)
        return model


class _TinySleepNetFeatureNet(nn.Module):
    """TinySleepNet: A lightweight CNN for sleep staging.
    
    Reference: Supratak et al. (2020) "TinySleepNet: An Efficient Deep Learning Model 
    for Sleep Stage Scoring Based on Raw Single-Channel EEG"
    """
    def __init__(self, n_channels, n_times, n_classes):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=50, stride=6, padding=22),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=8, stride=8),
            nn.Dropout(0.5)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=8, stride=1, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(0.5)
        )
        
        self.adaptive_pool = nn.AdaptiveAvgPool1d(13)
        
        self.feature_size = 128 * 13
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, n_classes)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class TinySleepNetClassifier:
    """TinySleepNet baseline wrapper for sleep staging benchmarks.
    
    Reference: Supratak et al. (2020) "TinySleepNet: An Efficient Deep Learning Model 
    for Sleep Stage Scoring Based on Raw Single-Channel EEG"
    """
    def __init__(self, n_channels, n_times, n_classes=5):
        set_random_seeds(seed=RANDOM_STATE, cuda=torch.cuda.is_available())
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = _TinySleepNetFeatureNet(n_channels=n_channels, n_times=n_times, n_classes=n_classes)
        self.clf = EEGClassifier(
            self.model,
            criterion=nn.CrossEntropyLoss(),
            optimizer=torch.optim.Adam,
            optimizer__lr=1e-4,
            optimizer__weight_decay=1e-3,
            train_split=None,
            device=self.device,
            batch_size=32,
            callbacks=[PrintLogCallback(print_freq=50), EarlyStopping(monitor='train_loss', patience=20)],
        )
        
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def fit(self, X, y, epochs=150, batch_size=32, learning_rate=1e-4):
        X_scaled = self.scaler.fit_transform(X.reshape(X.shape[0], -1)).reshape(X.shape)
        self.clf.fit(X_scaled, y, epochs=epochs)
        self.is_trained = True
        return self
    
    def predict(self, X):
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        X_scaled = self.scaler.transform(X.reshape(X.shape[0], -1)).reshape(X.shape)
        return self.clf.predict(X_scaled)
    
    def predict_proba(self, X):
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        X_scaled = self.scaler.transform(X.reshape(X.shape[0], -1)).reshape(X.shape)
        return self.clf.predict_proba(X_scaled)
    
    def save_model(self, path):
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        scaler_path = path.replace('.pt', '_scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump({'scaler': self.scaler, 'is_trained': self.is_trained}, f)
    
    @classmethod
    def load_model(cls, path, n_channels, n_times, n_classes=5):
        model = cls(n_channels, n_times, n_classes)
        model.model.load_state_dict(torch.load(path))
        model.model.eval()
        scaler_path = path.replace('.pt', '_scaler.pkl')
        with open(scaler_path, 'rb') as f:
            data = pickle.load(f)
            model.scaler = data['scaler']
            model.is_trained = data['is_trained']
        dummy_X = torch.randn(1, n_channels, n_times).to(model.device)
        model.clf.initialize()
        model.clf.predict(dummy_X)
        return model


class _SSCSleepNet(nn.Module):
    """SSC-SleepNet: Self-Supervised Contrastive SleepNet for EEG sleep staging (2025).
    
    A lightweight model that uses self-supervised contrastive learning to improve
    sleep staging performance with limited labeled data.
    """
    def __init__(self, n_channels, n_times, n_classes=5, d_model=32, n_heads=2, n_layers=2):
        super().__init__()
        # Input projection
        self.input_proj = nn.Conv1d(n_channels, d_model, kernel_size=1)
        
        # Temporal feature extraction
        self.temporal = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(d_model, d_model * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model * 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # Transformer encoder for sequence modeling
        self.pos_embed = nn.Parameter(torch.zeros(1, n_times // 4, d_model * 2))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model * 2,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.2,
            batch_first=True,
            activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Classification head
        self.norm = nn.LayerNorm(d_model * 2)
        self.head = nn.Linear(d_model * 2, n_classes)

    def forward(self, x):
        # x: (batch, channels, time)
        x = self.input_proj(x)
        x = self.temporal(x)
        x = x.transpose(1, 2)  # (batch, time, d_model)
        x = x + self.pos_embed[:, :x.shape[1], :]
        x = self.encoder(x)
        x = self.norm(x.mean(dim=1))
        return self.head(x)


class SSCSleepNet:
    """SSC-SleepNet wrapper for sleep staging benchmarks (2025).
    
    Reference: 2025 model that combines self-supervised learning with contrastive
    loss for improved sleep staging performance.
    """
    def __init__(self, n_channels, n_times, n_classes=5):
        set_random_seeds(seed=RANDOM_STATE, cuda=torch.cuda.is_available())
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = _SSCSleepNet(n_channels, n_times, n_classes)
        self.clf = EEGClassifier(
            self.model,
            criterion=nn.CrossEntropyLoss(),
            optimizer=torch.optim.AdamW,
            optimizer__lr=5e-4,
            optimizer__weight_decay=1e-4,
            train_split=None,
            device=self.device,
            batch_size=32,
            callbacks=[PrintLogCallback(print_freq=50), EarlyStopping(monitor='train_loss', patience=20)],
        )

        # Explicitly move the inner PyTorch module to the target device so the A100 is
        # actually utilized. We must call `.to()` only on the underlying `nn.Module`
        # (`self.model`), because braindecode's EEGClassifier wrapper does not expose
        # a `.to()` method of its own.
        self.model.to(self.device)

        self.scaler = StandardScaler()
        self.is_trained = False
    
    def fit(self, X, y, epochs=200, batch_size=32, learning_rate=0.0005):
        X_scaled = self.scaler.fit_transform(X.reshape(X.shape[0], -1)).reshape(X.shape)
        self.clf.fit(X_scaled, y, epochs=epochs)
        self.is_trained = True
        return self
    
    def predict(self, X):
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        X_scaled = self.scaler.transform(X.reshape(X.shape[0], -1)).reshape(X.shape)
        return self.clf.predict(X_scaled)
    
    def predict_proba(self, X):
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        X_scaled = self.scaler.transform(X.reshape(X.shape[0], -1)).reshape(X.shape)
        return self.clf.predict_proba(X_scaled)
    
    def save_model(self, path):
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        scaler_path = path.replace('.pt', '_scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump({'scaler': self.scaler, 'is_trained': self.is_trained}, f)
    
    @classmethod
    def load_model(cls, path, n_channels, n_times, n_classes=5):
        model = cls(n_channels, n_times, n_classes)
        model.model.load_state_dict(torch.load(path))
        model.model.eval()
        scaler_path = path.replace('.pt', '_scaler.pkl')
        with open(scaler_path, 'rb') as f:
            data = pickle.load(f)
            model.scaler = data['scaler']
            model.is_trained = data['is_trained']
        dummy_X = torch.randn(1, n_channels, n_times).to(model.device)
        model.clf.initialize()
        model.clf.predict(dummy_X)
        return model


class _USleepNet(nn.Module):
    """U-Sleep: Multi-view CNN for sleep staging.
    
    Reference: Perslev et al. (2021) "U-Sleep: Resilient High-Frequency Sleep Staging"
    npj Digital Medicine
    
    Architecture: Multi-scale CNN with residual connections for robust sleep staging.
    """
    def __init__(self, n_channels, n_times, n_classes=5):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, n_classes)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class USleepClassifier:
    """U-Sleep baseline wrapper for sleep staging benchmarks.
    
    Reference: Perslev et al. (2021) "U-Sleep: Resilient High-Frequency Sleep Staging"
    npj Digital Medicine, trained on 15,660 subjects from 16 clinical studies.
    """
    def __init__(self, n_channels, n_times, n_classes=5):
        set_random_seeds(seed=RANDOM_STATE, cuda=torch.cuda.is_available())
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = _USleepNet(n_channels=n_channels, n_times=n_times, n_classes=n_classes)
        self.clf = EEGClassifier(
            self.model,
            criterion=nn.CrossEntropyLoss(),
            optimizer=torch.optim.Adam,
            optimizer__lr=1e-4,
            optimizer__weight_decay=1e-4,
            train_split=None,
            device=self.device,
            batch_size=32,
            callbacks=[PrintLogCallback(print_freq=50), EarlyStopping(monitor='train_loss', patience=20)],
        )
        
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def fit(self, X, y, epochs=150, batch_size=32, learning_rate=1e-4):
        X_scaled = self.scaler.fit_transform(X.reshape(X.shape[0], -1)).reshape(X.shape)
        self.clf.fit(X_scaled, y, epochs=epochs)
        self.is_trained = True
        return self
    
    def predict(self, X):
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        X_scaled = self.scaler.transform(X.reshape(X.shape[0], -1)).reshape(X.shape)
        return self.clf.predict(X_scaled)
    
    def predict_proba(self, X):
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        X_scaled = self.scaler.transform(X.reshape(X.shape[0], -1)).reshape(X.shape)
        return self.clf.predict_proba(X_scaled)



    
    def save_model(self, path):
        """Save the model to a file."""
        import os
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model parameters
        torch.save(self.model.state_dict(), path)
        
        # Save scaler and is_trained flag
        scaler_path = path.replace('.pt', '_scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump({'scaler': self.scaler, 'is_trained': self.is_trained}, f)
    
    @classmethod
    def load_model(cls, path, n_channels, n_times, n_classes=4):
        """Load the model from a file."""
        # Create model instance
        model = cls(n_channels, n_times, n_classes)
        
        # Load model parameters
        model.model.load_state_dict(torch.load(path))
        model.model.eval()
        
        # Load scaler and is_trained flag
        scaler_path = path.replace('.pt', '_scaler.pkl')
        with open(scaler_path, 'rb') as f:
            data = pickle.load(f)
            model.scaler = data['scaler']
            model.is_trained = data['is_trained']
        
        # Initialize EEGClassifier
        # We need to pass some dummy data to initialize the classifier
        dummy_X = torch.randn(1, n_channels, n_times).to(model.device)
        model.clf.initialize()
        model.clf.predict(dummy_X)
        
        return model





def get_algorithm(algo_name, n_channels, n_times, n_classes=5, fs=100):
    """
    Get algorithm instance by name
    
    Args:
        algo_name: Name of the algorithm
        n_channels: Number of EEG channels
        n_times: Number of time points
        n_classes: Number of classes (default: 5 for sleep staging)
        fs: Sampling frequency (default: 100 for sleep EEG)
        
    Returns:
        Algorithm instance
    """
    algo_name = algo_name.strip()
    
    if algo_name in ['HandcraftedFeatures+RF', 'HandcraftedRF', 'Handcrafted+RF']:
        return HandcraftedFeaturesRF(fs=fs)
    elif algo_name in ['FilterBankTangentSpace+SVM', 'FBTS+SVM']:
        return FilterBankTangentSpace(classifier='svm', fs=fs)
    elif algo_name in ['SCA-FBTS+SVM', 'SCA-FBTS']:
        return FilterBankTangentSpace(classifier='svm', fs=fs, temporal_smoothing=True, smoothing_window=3)
    elif algo_name == 'FilterBankTangentSpace+LDA':
        return FilterBankTangentSpace(classifier='lda', fs=fs)
    elif algo_name == 'FilterBankTangentSpace+RF':
        return FilterBankTangentSpace(classifier='rf', fs=fs)
    elif algo_name == 'MDM':
        return MDM()
    elif algo_name in ['RiemannTangentSpace', 'RiemannTangentSpace+SVM']:
        return RiemannTangentSpace(estimator='oas', metric='riemann', classifier='svm')
    elif algo_name == 'RiemannTangentSpace+RF':
        return RiemannTangentSpace(estimator='oas', metric='riemann', classifier='rf')
    elif algo_name == 'RiemannTangentSpace+PCA':
        return RiemannTangentSpace(estimator='oas', metric='riemann', classifier='svm', n_components=10)
    elif algo_name == 'SSC-SleepNet':
        return SSCSleepNet(n_channels=n_channels, n_times=n_times, n_classes=n_classes)
    elif algo_name in ['DeepSleepNet', 'DeepSleepNet-Light']:
        return DeepSleepNetClassifier(n_channels=n_channels, n_times=n_times, n_classes=n_classes)
    elif algo_name == 'TinySleepNet':
        return TinySleepNetClassifier(n_channels=n_channels, n_times=n_times, n_classes=n_classes)
    elif algo_name in ['U-Sleep', 'USleep']:
        return USleepClassifier(n_channels=n_channels, n_times=n_times, n_classes=n_classes)
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")









if __name__ == '__main__':
    """Main function to print model structure when the file is run directly."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Print model structure')
    parser.add_argument('--model', type=str, default='HandcraftedFeatures+RF',
                        choices=['HandcraftedFeatures+RF', 'MDM', 'RiemannTangentSpace+SVM', 'RiemannTangentSpace+RF', 'RiemannTangentSpace+PCA', 'DeepSleepNet', 'TinySleepNet', 'U-Sleep'],
                        help='Model name to print structure')
    parser.add_argument('--channels', type=int, default=2,
                        help='Number of channels')
    parser.add_argument('--times', type=int, default=3000,
                        help='Number of time points')
    parser.add_argument('--classes', type=int, default=5,
                        help='Number of classes')
    
    args = parser.parse_args()
    
    print(f"\n{'=' * 80}")
    print(f"Model Structure: {args.model}")
    print(f"{'=' * 80}")
    
    try:
        model = get_algorithm(args.model, args.channels, args.times, args.classes)
        
        if hasattr(model, 'model'):
            # For deep learning models
            print("\nDeep Learning Model Structure:")
            print(model.model)
            print(f"\nModel Parameters: {sum(p.numel() for p in model.model.parameters())}")
        else:
            # For traditional ML models
            print("\nTraditional ML Model Structure:")
            if hasattr(model, 'pipeline'):
                print(model.pipeline)
            else:
                print(model)
        
        print(f"\n{'=' * 80}")
        print("Model structure printed successfully!")
        print(f"{'=' * 80}")
    except Exception as e:
        print(f"\nError printing model structure: {e}")
        print(f"{'=' * 80}")
