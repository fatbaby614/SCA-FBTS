#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
FBTS Algorithm Ablation Study

This script performs ablation experiments to evaluate the contribution of each component:
- FBTS (Full): FilterBank + Covariance + Riemannian Tangent Space + SVM
- w/o FilterBank: Single-band + Riemannian Tangent Space + SVM
- w/o TangentSpace: FilterBank + Covariance + MDM (direct classification)
- w/o Riemannian: FilterBank + PSD features + SVM

Usage:
    python run_fbts_ablation.py --data-path /path/to/physionet --dataset sleep_edf
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import numpy as np
import pandas as pd
import time
import sys
import os
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from scipy.signal import welch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.algorithms_config import RESULTS_PATH, RANDOM_STATE, get_timestamped_filename
import data_loader_sleep as data_loader
from sca_fbts import SCA_FBTS


SLEEP_BANDS = [
    (0.5, 4, 'delta'),
    (4, 8, 'theta'),
    (8, 12, 'alpha'),
    (12, 16, 'sigma'),
    (16, 30, 'beta'),
]


def bandpass_filter(data, low_freq, high_freq, fs, order=4):
    from scipy.signal import butter, filtfilt
    nyq = fs / 2
    low = max(low_freq / nyq, 0.001)
    high = min(high_freq / nyq, 0.999)
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=-1)


def compute_covariance_matrix(epochs):
    n_epochs, n_channels, n_times = epochs.shape
    covs = np.zeros((n_epochs, n_channels, n_channels))
    
    for i in range(n_epochs):
        epoch = epochs[i]
        cov = np.cov(epoch)
        if np.any(np.isnan(cov)) or np.any(np.isinf(cov)):
            cov = np.eye(n_channels)
        covs[i] = cov
    
    return covs


def tangent_space_projection(covs, C_ref=None, ref_metric='riemann'):
    from pyriemann.utils.mean import mean_riemann, mean_covariance
    from pyriemann.utils.tangentspace import tangent_space
    
    try:
        if C_ref is None:
            if ref_metric == 'riemann':
                C_ref = mean_riemann(covs)
            else:
                C_ref = mean_covariance(covs, metric='euclidean')
        
        features = tangent_space(covs, C_ref)
        return features, C_ref
    except Exception as e:
        print(f"Warning: Tangent space projection failed: {e}")
        features = np.array([np.log(np.diag(c)) for c in covs])
        return features, None


def extract_psd_features(epochs, fs, bands):
    n_epochs, n_channels, n_times = epochs.shape
    n_bands = len(bands)
    features = np.zeros((n_epochs, n_channels * n_bands))
    
    for i in range(n_epochs):
        epoch = epochs[i]
        feat_list = []
        
        for ch in range(n_channels):
            signal = epoch[ch]
            freqs, psd = welch(signal, fs=fs, nperseg=min(256, n_times))
            
            for low, high, _ in bands:
                mask = (freqs >= low) & (freqs < high)
                if np.any(mask):
                    band_power = np.mean(psd[mask])
                else:
                    band_power = 0
                feat_list.append(np.log(band_power + 1e-10))
        
        features[i] = feat_list
    
    return features


class FBTSFull:
    def __init__(self, fs=100):
        self.fs = fs
        self.classifier = SVC(kernel='rbf', C=1.0, random_state=RANDOM_STATE)
        self.scaler = StandardScaler()
        self.C_ref = None
        
    def fit(self, X, y, epochs=100):
        n_epochs, n_channels, n_times = X.shape
        
        all_features = []
        for low, high, _ in SLEEP_BANDS:
            filtered = bandpass_filter(X, low, high, self.fs)
            covs = compute_covariance_matrix(filtered)
            all_features.append(covs)
        
        combined_covs = np.mean(all_features, axis=0)
        features, self.C_ref = tangent_space_projection(combined_covs)
        
        features_scaled = self.scaler.fit_transform(features)
        self.classifier.fit(features_scaled, y)
        
        return self
    
    def predict(self, X):
        n_epochs, n_channels, n_times = X.shape
        
        all_features = []
        for low, high, _ in SLEEP_BANDS:
            filtered = bandpass_filter(X, low, high, self.fs)
            covs = compute_covariance_matrix(filtered)
            all_features.append(covs)
        
        combined_covs = np.mean(all_features, axis=0)
        features, _ = tangent_space_projection(combined_covs)
        
        features_scaled = self.scaler.transform(features)
        return self.classifier.predict(features_scaled)


class FBTSSingleBand:
    def __init__(self, fs=100):
        self.fs = fs
        self.classifier = SVC(kernel='rbf', C=1.0, random_state=RANDOM_STATE)
        self.scaler = StandardScaler()
        self.C_ref = None
        
    def fit(self, X, y, epochs=100):
        n_epochs, n_channels, n_times = X.shape
        
        covs = compute_covariance_matrix(X)
        features, self.C_ref = tangent_space_projection(covs)
        
        features_scaled = self.scaler.fit_transform(features)
        self.classifier.fit(features_scaled, y)
        
        return self
    
    def predict(self, X):
        covs = compute_covariance_matrix(X)
        features, _ = tangent_space_projection(covs)
        
        features_scaled = self.scaler.transform(features)
        return self.classifier.predict(features_scaled)


class FBTSwoTangentSpace:
    def __init__(self, fs=100):
        self.fs = fs
        from pyriemann.classification import MDM
        self.classifier = MDM(metric='riemann')
        
    def fit(self, X, y, epochs=100):
        n_epochs, n_channels, n_times = X.shape
        
        all_covs = []
        for low, high, _ in SLEEP_BANDS:
            filtered = bandpass_filter(X, low, high, self.fs)
            covs = compute_covariance_matrix(filtered)
            all_covs.append(covs)
        
        combined_covs = np.mean(all_covs, axis=0)
        self.classifier.fit(combined_covs, y)
        
        return self
    
    def predict(self, X):
        all_covs = []
        for low, high, _ in SLEEP_BANDS:
            filtered = bandpass_filter(X, low, high, self.fs)
            covs = compute_covariance_matrix(filtered)
            all_covs.append(covs)
        
        combined_covs = np.mean(all_covs, axis=0)
        return self.classifier.predict(combined_covs)


class FBTSwoRiemannian:
    def __init__(self, fs=100):
        self.fs = fs
        self.classifier = SVC(kernel='rbf', C=1.0, random_state=RANDOM_STATE)
        self.scaler = StandardScaler()
        
    def fit(self, X, y, epochs=100):
        features = extract_psd_features(X, self.fs, SLEEP_BANDS)
        features_scaled = self.scaler.fit_transform(features)
        self.classifier.fit(features_scaled, y)
        return self
    
    def predict(self, X):
        features = extract_psd_features(X, self.fs, SLEEP_BANDS)
        features_scaled = self.scaler.transform(features)
        return self.classifier.predict(features_scaled)


def run_fbts_ablation(X, y, n_splits=5, subject_indices=None):
    """
    Run SCA-FBTS ablation study.
    
    Args:
        X: EEG data of shape (n_epochs, n_channels, n_times)
        y: Labels
        n_splits: Number of cross-validation folds
        subject_indices: Array mapping each epoch to its subject index (for subject-wise CV)
        
    Returns:
        DataFrame with results
    """
    # Create SCA-FBTS variants for ablation
    def create_sca_fbts_full():
        return SCA_FBTS(temporal_smoothing=True, smoothing_window=3)
    
    def create_sca_fbts_wo_filterbank():
        # Single band instead of filter bank
        return SCA_FBTS(freq_bands=[(0.5, 30)], temporal_smoothing=True, smoothing_window=3)
    
    def create_sca_fbts_wo_tangentspace():
        # Using MDM instead of tangent space
        from pyriemann.classification import MDM
        class FBTSwoTangentSpace:
            def __init__(self):
                from pyriemann.estimation import Covariances
                self.cov_estimator = Covariances(estimator='oas')
                self.classifier = MDM(metric='riemann')
            
            def fit(self, X, y):
                # Apply filter bank
                freq_bands = [(0.5, 4), (4, 6), (6, 8), (8, 10), (10, 12), (12, 14), (14, 20), (20, 30)]
                all_covs = []
                for low, high in freq_bands:
                    from sca_fbts import apply_bandpass_filter
                    X_band = np.array([apply_bandpass_filter(trial, low, high, 100) for trial in X])
                    covs = self.cov_estimator.fit_transform(X_band)
                    all_covs.append(covs)
                combined_covs = np.mean(all_covs, axis=0)
                self.classifier.fit(combined_covs, y)
                return self
            
            def predict(self, X):
                # Apply filter bank
                freq_bands = [(0.5, 4), (4, 6), (6, 8), (8, 10), (10, 12), (12, 14), (14, 20), (20, 30)]
                all_covs = []
                for low, high in freq_bands:
                    from sca_fbts import apply_bandpass_filter
                    X_band = np.array([apply_bandpass_filter(trial, low, high, 100) for trial in X])
                    covs = self.cov_estimator.transform(X_band)
                    all_covs.append(covs)
                combined_covs = np.mean(all_covs, axis=0)
                return self.classifier.predict(combined_covs)
        return FBTSwoTangentSpace()
    
    def create_sca_fbts_wo_riemannian():
        # Using PSD features instead of Riemannian
        class FBTSwoRiemannian:
            def __init__(self):
                from sklearn.svm import SVC
                from sklearn.preprocessing import StandardScaler
                self.classifier = SVC(kernel='rbf', C=1.0, random_state=RANDOM_STATE)
                self.scaler = StandardScaler()
            
            def fit(self, X, y):
                # Extract PSD features
                features = extract_psd_features(X, 100, SLEEP_BANDS)
                features_scaled = self.scaler.fit_transform(features)
                self.classifier.fit(features_scaled, y)
                return self
            
            def predict(self, X):
                features = extract_psd_features(X, 100, SLEEP_BANDS)
                features_scaled = self.scaler.transform(features)
                return self.classifier.predict(features_scaled)
        return FBTSwoRiemannian()
    
    def create_sca_fbts_wo_temporal_smoothing():
        return SCA_FBTS(temporal_smoothing=False)
    
    models = {
        'SCA-FBTS (Full)': create_sca_fbts_full,
        'w/o FilterBank': create_sca_fbts_wo_filterbank,
        'w/o TangentSpace': create_sca_fbts_wo_tangentspace,
        'w/o Riemannian': create_sca_fbts_wo_riemannian,
        'w/o Temporal Smoothing': create_sca_fbts_wo_temporal_smoothing,
    }
    
    # Use GroupKFold for subject-wise cross-validation, otherwise use StratifiedKFold
    if subject_indices is not None:
        from sklearn.model_selection import GroupKFold
        kf = GroupKFold(n_splits=n_splits)
        # Store splits as a list so it can be reused for each model
        splits = list(kf.split(X, y, groups=subject_indices))
        print(f"Using subject-wise cross-validation with {len(np.unique(subject_indices))} subjects")
    else:
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        # Store splits as a list so it can be reused for each model
        splits = list(kf.split(X, y))
        print(f"Using stratified cross-validation")
    
    results = []
    
    for model_name, model_creator in models.items():
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*60}")
        
        fold_results = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            print(f"  Fold {fold_idx + 1}/{n_splits}...", end=" ")
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            model = model_creator()
            
            start_time = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start_time
            
            start_time = time.time()
            y_pred = model.predict(X_test)
            inference_time = (time.time() - start_time) * 1000 / len(X_test)
            
            acc = accuracy_score(y_test, y_pred)
            kappa = cohen_kappa_score(y_test, y_pred)
            f1_macro = f1_score(y_test, y_pred, average='macro')
            
            fold_results.append({
                'fold': fold_idx + 1,
                'accuracy': acc,
                'kappa': kappa,
                'f1_macro': f1_macro,
                'train_time': train_time,
                'inference_time_ms': inference_time,
            })
            
            print(f"Acc: {acc:.4f}, Kappa: {kappa:.4f}, F1: {f1_macro:.4f}")
        
        df = pd.DataFrame(fold_results)
        
        results.append({
            'model': model_name,
            'accuracy_mean': df['accuracy'].mean(),
            'accuracy_std': df['accuracy'].std(),
            'kappa_mean': df['kappa'].mean(),
            'kappa_std': df['kappa'].std(),
            'f1_macro_mean': df['f1_macro'].mean(),
            'f1_macro_std': df['f1_macro'].std(),
            'train_time_mean': df['train_time'].mean(),
            'inference_time_ms_mean': df['inference_time_ms'].mean(),
        })
    
    return pd.DataFrame(results)


def generate_latex_table(results_df):
    latex = "\\begin{table}[htbp]\n"
    latex += "\\centering\n"
    latex += "\\caption{FBTS Ablation Study Results}\n"
    latex += "\\label{tab:fbts_ablation}\n"
    latex += "\\begin{tabular}{lccc}\n"
    latex += "\\toprule\n"
    latex += "Model & Accuracy & Kappa & Macro-F1 \\\\\n"
    latex += "\\midrule\n"
    
    for _, row in results_df.iterrows():
        model = row['model']
        acc = f"{row['accuracy_mean']:.3f} $\\pm$ {row['accuracy_std']:.3f}"
        kappa = f"{row['kappa_mean']:.3f} $\\pm$ {row['kappa_std']:.3f}"
        f1 = f"{row['f1_macro_mean']:.3f} $\\pm$ {row['f1_macro_std']:.3f}"
        
        latex += f"{model} & {acc} & {kappa} & {f1} \\\\\n"
    
    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}"
    
    return latex


def main():
    parser = argparse.ArgumentParser(description='SCA-FBTS Algorithm Ablation Study')
    parser.add_argument('--data-path', type=str,
                        default=("E:/datasets/Sleep/sleep-edf-database-expanded-1.0.0" if os.name == 'nt'
                                 else "/mnt/data1/home/tanhuang/datasets/sleep-edf-database-expanded-1.0.0"),
                        help='Path to Sleep-EDF dataset')
    parser.add_argument('--isruc-path', type=str, default=None,
                        help='Path to ISRUC-Sleep dataset')
    parser.add_argument('--dataset', type=str, default='sleep_edf',
                        choices=['sleep_edf', 'isruc', 'combined'],
                        help='Dataset to use')
    parser.add_argument('--subjects', type=str, default='all',
                        help='Subjects to use (comma-separated or "all")')
    parser.add_argument('--n-splits', type=int, default=5,
                        help='Number of cross-validation folds')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for results')
    parser.add_argument('--cv-mode', type=str, default='subject',
                        choices=['subject', 'stratified'],
                        help='Cross-validation mode: subject-wise or stratified. Default: subject (to ensure consistency with main evaluation)')
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = RESULTS_PATH
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("SCA-FBTS Algorithm Ablation Study")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Data path: {args.data_path}")
    
    print("\nLoading data...")
    
    if args.dataset in ['sleep_edf']:
        if args.subjects == 'all':
            subjects = None
        else:
            # Handle range format like "0~1"
            if '~' in args.subjects:
                start, end = args.subjects.split('~')
                subjects = list(range(int(start), int(end) + 1))
            else:
                subjects = [int(s) for s in args.subjects.split(',')]
        X, y, meta = data_loader.load_sleep_edf(
            data_path=args.data_path,
            subjects=subjects,
            select_n_channels=2,
        )
    elif args.dataset == 'isruc':
        if args.isruc_path is None:
            raise ValueError("--isruc-path is required for ISRUC dataset")
        subjects = None if args.subjects == 'all' else [int(s) for s in args.subjects.split(',')]
        X, y, meta = data_loader.load_isruc_sleep(
            data_path=args.isruc_path,
            subjects=subjects,
            select_n_channels=2,
        )
    elif args.dataset == 'combined':
        if args.isruc_path is None:
            raise ValueError("--isruc-path is required for combined dataset")
        X, y, meta = data_loader.load_combined_datasets(
            sleep_edf_path=args.data_path,
            isruc_path=args.isruc_path,
            select_n_channels=2,
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    print(f"Loaded {len(X)} epochs with {len(np.unique(y))} classes")
    
    # Extract subject indices for subject-wise cross-validation
    subject_indices = None
    if args.cv_mode == 'subject' and meta is not None and 'subject' in meta.columns:
        subject_indices = meta['subject'].values
        n_subjects = len(np.unique(subject_indices))
        print(f"Using {n_subjects} unique subjects for cross-validation")
        
        # Adjust n_splits if it exceeds number of subjects for GroupKFold
        if args.n_splits > n_subjects:
            print(f"Warning: n_splits ({args.n_splits}) > number of subjects ({n_subjects}). Adjusting n_splits to {n_subjects}.")
            args.n_splits = n_subjects
    
    results_df = run_fbts_ablation(X, y, n_splits=args.n_splits, subject_indices=subject_indices)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    csv_path = os.path.join(args.output_dir, f'sca_fbts_ablation_results_{timestamp}.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    latex_table = generate_latex_table(results_df)
    latex_path = os.path.join(args.output_dir, f'sca_fbts_ablation_table_{timestamp}.tex')
    with open(latex_path, 'w') as f:
        f.write(latex_table)
    print(f"LaTeX table saved to: {latex_path}")
    
    print("\n" + "=" * 60)
    print("SCA-FBTS Ablation Study Results Summary")
    print("=" * 60)
    print(results_df.to_string(index=False))
    
    print("\n" + "=" * 60)
    print("Key Findings:")
    print("=" * 60)
    
    full_acc = results_df[results_df['model'] == 'SCA-FBTS (Full)']['accuracy_mean'].values[0]
    for _, row in results_df.iterrows():
        if row['model'] != 'SCA-FBTS (Full)':
            diff = full_acc - row['accuracy_mean']
            print(f"  {row['model']}: -{diff:.3f} accuracy drop")
    
    print("\n" + "=" * 60)
    print("Ablation study completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
