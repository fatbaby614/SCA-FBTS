#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Parameter Sensitivity Analysis for SCA-FBTS Sleep Staging
Tests the impact of key hyperparameters:
- Number of filter banks (3 / 5 / 7 bands)
- Feature dimension (50 / 100 / 150 / 200)
- Epoch window length (20s / 30s)

This is a pure algorithm experiment - no additional subjects or ethical approval needed.
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
import time

from config.algorithms_config import RESULTS_PATH, RANDOM_STATE, FS
import data_loader_sleep as data_loader
from sca_fbts import SCA_FBTS


def create_fbts_with_params(n_channels, n_times, n_classes, fs, 
                            freq_bands=None, n_features=100):
    """
    Create SCA-FBTS model with custom parameters.
    
    Args:
        n_channels: Number of EEG channels
        n_times: Number of time points per epoch
        n_classes: Number of classes
        fs: Sampling frequency
        freq_bands: List of (low, high) frequency bands
        n_features: Number of features after tangent space projection
    
    Returns:
        Configured SCA-FBTS model
    """
    return SCA_FBTS(
        freq_bands=freq_bands,
        n_features=n_features,
        fs=fs,
        temporal_smoothing=True,
        smoothing_window=3
    )


def get_freq_bands_config(n_bands):
    """
    Get frequency band configurations for different numbers of bands.
    
    Args:
        n_bands: Number of frequency bands (3, 5, or 7)
    
    Returns:
        List of (low, high) frequency tuples
    """
    configs = {
        3: [
            (0.5, 4),    # Delta
            (4, 12),     # Theta + Alpha combined
            (12, 30),    # Sigma + Beta combined
        ],
        5: [
            (0.5, 4),    # Delta
            (4, 8),      # Theta
            (8, 12),     # Alpha
            (12, 16),    # Sigma
            (16, 30),    # Beta
        ],
        7: [
            (0.5, 2),    # Slow delta
            (2, 4),      # Fast delta
            (4, 8),      # Theta
            (8, 12),     # Alpha
            (12, 16),    # Sigma
            (16, 24),    # Low beta
            (24, 30),    # High beta
        ],
    }
    return configs.get(n_bands, configs[5])


def run_filter_bank_sensitivity(
    dataset_name='sleep_edf',
    data_path=None,
    n_folds=5,
    n_bands_list=[3, 5, 7],
    output_dir=None,
):
    """
    Test sensitivity to number of filter banks.
    
    Args:
        dataset_name: Dataset name
        data_path: Path to dataset
        n_folds: Number of CV folds
        n_bands_list: List of filter bank counts to test
        output_dir: Output directory
    
    Returns:
        results_df: DataFrame with results
    """
    print("\n" + "=" * 80)
    print("Filter Bank Sensitivity Analysis")
    print("=" * 80)
    
    if output_dir is None:
        output_dir = RESULTS_PATH
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nLoading dataset: {dataset_name}")
    X, y, meta = data_loader.load_sleep_dataset(dataset_name, data_path=data_path)
    
    n_samples, n_channels, n_times = X.shape
    n_classes = len(np.unique(y))
    features_per_band = n_channels * (n_channels + 1) // 2
    n_bands_ref = 5
    total_features = features_per_band * n_bands_ref
    
    print(f"  Samples: {n_samples}, Channels: {n_channels}")
    print(f"  Features per band: {features_per_band}, Total ({n_bands_ref} bands): {total_features}")
    
    results = []
    
    for n_bands in n_bands_list:
        print(f"\n  Testing {n_bands} filter banks...")
        
        freq_bands = get_freq_bands_config(n_bands)
        print(f"    Bands: {freq_bands}")
        
        fold_results = []
        
        # Use subject-wise CV by default (consistent with main experiment)
        if isinstance(meta, pd.DataFrame) and 'subject' in meta.columns:
            groups = meta['subject'].astype(str).to_numpy()
            unique_subjects = np.unique(groups)
            n_splits = min(n_folds, len(unique_subjects))
            kf = GroupKFold(n_splits=n_splits)
            splits = list(kf.split(X, y, groups))
            print(f"    Using subject-wise CV with {len(unique_subjects)} subjects, {n_splits} splits")
        else:
            kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
            splits = list(kf.split(X, y))
            print(f"    Using stratified CV")
        
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            model = create_fbts_with_params(
                n_channels, n_times, n_classes, FS,
                freq_bands=freq_bands, n_features=100
            )
            
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                accuracy = accuracy_score(y_test, y_pred)
                kappa = cohen_kappa_score(y_test, y_pred)
                macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
                
                fold_results.append({
                    'accuracy': accuracy,
                    'kappa': kappa,
                    'macro_f1': macro_f1,
                })
                
            except Exception as e:
                print(f"    Fold {fold_idx + 1} failed: {e}")
                continue
        
        if fold_results:
            mean_acc = np.mean([r['accuracy'] for r in fold_results])
            std_acc = np.std([r['accuracy'] for r in fold_results])
            mean_kappa = np.mean([r['kappa'] for r in fold_results])
            mean_f1 = np.mean([r['macro_f1'] for r in fold_results])
            
            print(f"    Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
            
            results.append({
                'parameter': 'n_filter_banks',
                'value': n_bands,
                'accuracy_mean': mean_acc,
                'accuracy_std': std_acc,
                'kappa_mean': mean_kappa,
                'macro_f1_mean': mean_f1,
                'n_folds': len(fold_results),
            })
    
    return pd.DataFrame(results)


def run_feature_dimension_sensitivity(
    dataset_name='sleep_edf',
    data_path=None,
    n_folds=5,
    n_features_list=[50, 100, 150, 200],
    output_dir=None,
):
    """
    Test sensitivity to feature dimension.
    
    Args:
        dataset_name: Dataset name
        data_path: Path to dataset
        n_folds: Number of CV folds
        n_features_list: List of feature dimensions to test
        output_dir: Output directory
    
    Returns:
        results_df: DataFrame with results
    """
    print("\n" + "=" * 80)
    print("Feature Dimension Sensitivity Analysis")
    print("=" * 80)
    
    if output_dir is None:
        output_dir = RESULTS_PATH
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nLoading dataset: {dataset_name}")
    X, y, meta = data_loader.load_sleep_dataset(dataset_name, data_path=data_path)
    
    n_samples, n_channels, n_times = X.shape
    n_classes = len(np.unique(y))
    features_per_band = n_channels * (n_channels + 1) // 2
    n_bands_ref = 5
    total_features = features_per_band * n_bands_ref
    
    print(f"  Samples: {n_samples}, Channels: {n_channels}")
    print(f"  Features per band: {features_per_band}, Total ({n_bands_ref} bands): {total_features}")
    print(f"  NOTE: n_features > {total_features} selects ALL features (no actual reduction)")
    
    freq_bands = get_freq_bands_config(5)
    
    # Auto-scale to actual feature dimension
    n_features_test = sorted(set([5, int(features_per_band), min(20, total_features), min(50, total_features), total_features]))
    n_features_test = [v for v in n_features_test if v > 0]
    print(f"  Testing n_features (auto-scaled): {n_features_test}")
    
    results = []
    
    for n_features in n_features_test:
        print(f"\n  Testing n_features={n_features} (effective max={total_features})...")
        
        fold_results = []
        
        # Use subject-wise CV by default (consistent with main experiment)
        if isinstance(meta, pd.DataFrame) and 'subject' in meta.columns:
            groups = meta['subject'].astype(str).to_numpy()
            unique_subjects = np.unique(groups)
            n_splits = min(n_folds, len(unique_subjects))
            kf = GroupKFold(n_splits=n_splits)
            splits = list(kf.split(X, y, groups))
        else:
            kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
            splits = list(kf.split(X, y))
        
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            model = create_fbts_with_params(
                n_channels, n_times, n_classes, FS,
                freq_bands=freq_bands, n_features=n_features
            )
            
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                accuracy = accuracy_score(y_test, y_pred)
                kappa = cohen_kappa_score(y_test, y_pred)
                macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
                
                fold_results.append({
                    'accuracy': accuracy,
                    'kappa': kappa,
                    'macro_f1': macro_f1,
                })
                
            except Exception as e:
                print(f"    Fold {fold_idx + 1} failed: {e}")
                continue
        
        if fold_results:
            mean_acc = np.mean([r['accuracy'] for r in fold_results])
            std_acc = np.std([r['accuracy'] for r in fold_results])
            mean_kappa = np.mean([r['kappa'] for r in fold_results])
            mean_f1 = np.mean([r['macro_f1'] for r in fold_results])
            
            print(f"    Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
            
            results.append({
                'parameter': 'n_features',
                'value': n_features,
                'accuracy_mean': mean_acc,
                'accuracy_std': std_acc,
                'kappa_mean': mean_kappa,
                'macro_f1_mean': mean_f1,
                'n_folds': len(fold_results),
            })
    
    return pd.DataFrame(results)


def run_window_length_sensitivity(
    dataset_name='sleep_edf',
    data_path=None,
    n_folds=5,
    window_lengths=[20, 30],
    output_dir=None,
    meta=None,
):
    """
    Test sensitivity to epoch window length.
    
    Args:
        dataset_name: Dataset name
        data_path: Path to dataset
        n_folds: Number of CV folds
        window_lengths: List of window lengths in seconds
        output_dir: Output directory
    
    Returns:
        results_df: DataFrame with results
    """
    print("\n" + "=" * 80)
    print("Window Length Sensitivity Analysis")
    print("=" * 80)
    
    if output_dir is None:
        output_dir = RESULTS_PATH
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nLoading dataset: {dataset_name}")
    X_orig, y_orig, meta = data_loader.load_sleep_dataset(dataset_name, data_path=data_path)
    
    n_samples, n_channels, n_times_orig = X_orig.shape
    n_classes = len(np.unique(y_orig))
    orig_window = n_times_orig / FS
    
    print(f"  Original samples: {n_samples}")
    print(f"  Original window: {orig_window}s")
    
    freq_bands = get_freq_bands_config(5)
    
    results = []
    
    for window_length in window_lengths:
        print(f"\n  Testing {window_length}s window...")
        
        if window_length == int(orig_window):
            X = X_orig
            n_times = n_times_orig
        else:
            ratio = window_length / orig_window
            n_times = int(n_times_orig * ratio)
            
            if ratio < 1:
                start = (n_times_orig - n_times) // 2
                X = X_orig[:, :, start:start+n_times]
            else:
                X = np.zeros((n_samples, n_channels, n_times))
                for i in range(n_samples):
                    for j in range(n_channels):
                        X[i, j] = np.interp(
                            np.linspace(0, n_times_orig-1, n_times),
                            np.arange(n_times_orig),
                            X_orig[i, j]
                        )
        
        print(f"    Reshaped to: {X.shape}")
        
        fold_results = []
        
        # Use subject-wise CV by default
        if isinstance(meta, pd.DataFrame) and 'subject' in meta.columns:
            groups = meta['subject'].astype(str).to_numpy()
            unique_subjects = np.unique(groups)
            n_splits = min(n_folds, len(unique_subjects))
            kf = GroupKFold(n_splits=n_splits)
            splits = list(kf.split(X, y_orig, groups))
            print(f"    Using subject-wise CV with {len(unique_subjects)} subjects")
        else:
            kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
            splits = list(kf.split(X, y_orig))
            print(f"    Using stratified CV")
        
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y_orig[train_idx], y_orig[test_idx]
            
            model = create_fbts_with_params(
                n_channels, X.shape[2], n_classes, FS,
                freq_bands=freq_bands, n_features=100
            )
            
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                accuracy = accuracy_score(y_test, y_pred)
                kappa = cohen_kappa_score(y_test, y_pred)
                macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
                
                fold_results.append({
                    'accuracy': accuracy,
                    'kappa': kappa,
                    'macro_f1': macro_f1,
                })
                
            except Exception as e:
                print(f"    Fold {fold_idx + 1} failed: {e}")
                continue
        
        if fold_results:
            mean_acc = np.mean([r['accuracy'] for r in fold_results])
            std_acc = np.std([r['accuracy'] for r in fold_results])
            mean_kappa = np.mean([r['kappa'] for r in fold_results])
            mean_f1 = np.mean([r['macro_f1'] for r in fold_results])
            
            print(f"    Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
            
            results.append({
                'parameter': 'window_length',
                'value': window_length,
                'accuracy_mean': mean_acc,
                'accuracy_std': std_acc,
                'kappa_mean': mean_kappa,
                'macro_f1_mean': mean_f1,
                'n_folds': len(fold_results),
            })
    
    return pd.DataFrame(results)


def plot_sensitivity_results(all_results, output_dir, timestamp):
    """Generate publication-ready plots for sensitivity analysis."""
    
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.linewidth'] = 1.2
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    param_names = ['n_filter_banks', 'n_features', 'window_length']
    param_labels = ['Number of Filter Banks', 'Feature Dimension', 'Window Length (s)']
    
    for idx, (param, label) in enumerate(zip(param_names, param_labels)):
        ax = axes[idx]
        
        param_df = all_results[all_results['parameter'] == param].sort_values('value')
        
        x = param_df['value'].values
        y = param_df['accuracy_mean'].values
        yerr = param_df['accuracy_std'].values
        
        ax.errorbar(x, y, yerr=yerr, marker='o', linewidth=2, markersize=8, 
                   capsize=4, color='#2E86AB')
        
        ax.fill_between(x, y - yerr, y + yerr, alpha=0.2, color='#2E86AB')
        
        ax.set_xlabel(label, fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        if param == 'n_filter_banks':
            ax.set_xticks([3, 5, 7])
        elif param == 'n_features':
            ax.set_xticks([50, 100, 150, 200])
        elif param == 'window_length':
            ax.set_xticks([20, 30])
        
        y_range = max(y) - min(y)
        ax.set_ylim(min(y) - 0.05, max(y) + 0.05)
    
    plt.tight_layout()
    
    fig_path = os.path.join(output_dir, f'parameter_sensitivity_{timestamp}.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to: {fig_path}")


def generate_latex_table(all_results):
    """Generate LaTeX table for manuscript."""
    
    latex_str = """
\\begin{table}[H]
\\caption{Parameter sensitivity analysis results. Accuracy (mean ± std) across 5-fold cross-validation.}\\label{tab:sensitivity}
\\centering
\\footnotesize
\\begin{tabular}{lcc}
\\toprule
Parameter & Value & Accuracy \\\\
\\midrule
"""
    
    for param in ['n_filter_banks', 'n_features', 'window_length']:
        param_df = all_results[all_results['parameter'] == param].sort_values('value')
        
        param_labels = {
            'n_filter_banks': 'Filter Banks',
            'n_features': 'Features',
            'window_length': 'Window (s)',
        }
        
        for _, row in param_df.iterrows():
            latex_str += f"{param_labels[param]} & {int(row['value'])} & {row['accuracy_mean']:.3f} $\\pm$ {row['accuracy_std']:.3f} \\\\\n"
        
        latex_str += "\\midrule\n"
    
    latex_str = latex_str.rstrip("\\midrule\n")
    
    latex_str += """\\bottomrule
\\end{tabular}
\\normalsize
\\end{table}
"""
    
    return latex_str


def main():
    parser = argparse.ArgumentParser(description='Parameter Sensitivity Analysis')
    parser.add_argument('--dataset', type=str, default='sleep_edf',
                       help='Dataset name')
    parser.add_argument('--data-path', type=str,
                       default=("E:/datasets/Sleep/sleep-edf-database-expanded-1.0.0" if os.name == 'nt'
                                else "/mnt/data1/home/tanhuang/datasets/sleep-edf-database-expanded-1.0.0"),
                       help='Path to dataset')
    parser.add_argument('--n-folds', type=int, default=5,
                       help='Number of cross-validation folds')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory')
    
    args = parser.parse_args()
    
    output_dir = args.output_dir or RESULTS_PATH
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = []
    
    fb_results = run_filter_bank_sensitivity(
        dataset_name=args.dataset,
        data_path=args.data_path,
        n_folds=args.n_folds,
        output_dir=output_dir,
    )
    all_results.append(fb_results)
    
    feat_results = run_feature_dimension_sensitivity(
        dataset_name=args.dataset,
        data_path=args.data_path,
        n_folds=args.n_folds,
        output_dir=output_dir,
    )
    all_results.append(feat_results)
    
    window_results = run_window_length_sensitivity(
        dataset_name=args.dataset,
        data_path=args.data_path,
        n_folds=args.n_folds,
        output_dir=output_dir,
    )
    all_results.append(window_results)
    
    all_results_df = pd.concat(all_results, ignore_index=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f'parameter_sensitivity_results_{timestamp}.csv')
    all_results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    plot_sensitivity_results(all_results_df, output_dir, timestamp)
    
    print("\n" + "=" * 80)
    print("Summary:")
    print("=" * 80)
    print(all_results_df.to_string(index=False))
    
    print("\n" + "=" * 80)
    print("LaTeX Table:")
    print("=" * 80)
    print(generate_latex_table(all_results_df))


if __name__ == "__main__":
    main()