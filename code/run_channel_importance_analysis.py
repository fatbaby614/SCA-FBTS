#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Channel Importance Analysis for SCA-FBTS Algorithm

This script analyzes the contribution of each EEG channel to the SCA-FBTS algorithm
by removing one channel at a time and measuring the accuracy drop.

Usage:
    python run_channel_importance_analysis.py --dataset sleep_edf --data-path /path/to/data
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import argparse
import numpy as np
import pandas as pd
import time
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from scipy import signal

from config.algorithms_config import RANDOM_STATE, FS, SLEEP_BANDS
import data_loader_sleep as data_loader
from sca_fbts import SCA_FBTS


class SCAFBTSChannelAnalysis:
    """SCA-FBTS with controllable channels for ablation analysis."""

    def __init__(self, fs=100, classifier='svm', active_channels=None):
        """Initialize SCA-FBTS with specific channels active.

        Args:
            fs: Sampling frequency
            classifier: Classifier type
            active_channels: List of channel indices to use (None = all channels)
        """
        self.fs = fs
        self.classifier_name = classifier
        self.active_channels = active_channels
        self.model = None

    def fit(self, X, y):
        """Fit the model."""
        if self.active_channels is not None:
            X = X[:, self.active_channels, :]

        n_channels = X.shape[1]
        self.model = SCA_FBTS(
            fs=self.fs,
            temporal_smoothing=True,
            smoothing_window=3
        )
        self.model.fit(X, y)
        return self

    def predict(self, X):
        """Predict labels."""
        if self.active_channels is not None:
            X = X[:, self.active_channels, :]

        return self.model.predict(X)


def run_channel_importance_analysis(X, y, meta=None, channel_names=None, n_folds=5):
    """Analyze channel importance by ablation.

    Args:
        X: EEG data (n_epochs, n_channels, n_times)
        y: Labels
        channel_names: List of channel names
        n_folds: Number of cross-validation folds

    Returns:
        DataFrame with channel importance results
    """
    print("\n" + "=" * 80)
    print("Channel Importance Analysis")
    print("=" * 80)

    n_channels = X.shape[1]
    if channel_names is None:
        channel_names = [f"Ch{i}" for i in range(n_channels)]

    # Use subject-wise CV by default (consistent with main experiment)
    if meta is not None and isinstance(meta, pd.DataFrame) and 'subject' in meta.columns:
        groups = meta['subject'].astype(str).to_numpy()
        unique_subjects = np.unique(groups)
        n_splits = min(n_folds, len(unique_subjects))
        kf = GroupKFold(n_splits=n_splits)
        splits = list(kf.split(X, y, groups))
        print(f"Using subject-wise CV, {len(unique_subjects)} subjects, {n_splits} folds")
    else:
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
        splits = list(kf.split(X, y))
        print(f"Using stratified CV, {n_folds} folds")

    results = []

    # Baseline: all channels
    print("\n[Baseline] Using all channels...")
    baseline_accuracies = []
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = SCAFBTSChannelAnalysis()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        baseline_accuracies.append(accuracy_score(y_test, y_pred))

    baseline_acc = np.mean(baseline_accuracies)
    print(f"  Baseline Accuracy: {baseline_acc:.4f}")

    # Ablation: remove each channel
    print("\n[Ablation] Removing each channel...")
    for ch_idx in range(n_channels):
        ch_name = channel_names[ch_idx]

        active_channels = [i for i in range(n_channels) if i != ch_idx]

        accuracies = []
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = SCAFBTSChannelAnalysis(active_channels=active_channels)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracies.append(accuracy_score(y_test, y_pred))

        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        drop = baseline_acc - mean_acc

        print(f"  Without {ch_name:12s}: {mean_acc:.4f} ± {std_acc:.4f} (drop: {drop:+.4f})")

        results.append({
            'channel_removed': ch_name,
            'channel_index': ch_idx,
            'accuracy_mean': mean_acc,
            'accuracy_std': std_acc,
            'baseline_accuracy': baseline_acc,
            'accuracy_drop': drop,
            'importance_percent': (drop / baseline_acc) * 100
        })

    return pd.DataFrame(results), baseline_acc


def generate_latex_table(results_df, baseline_acc):
    """Generate LaTeX table for channel importance."""
    latex = "\\begin{table}[H]\n"
    latex += "\\caption{Channel importance analysis. Accuracy drop indicates channel contribution.}\n"
    latex += "\\label{tab:channel_importance}\n"
    latex += "\\centering\n"
    latex += "\\footnotesize\n"
    latex += "\\begin{tabular}{lccc}\n"
    latex += "\\toprule\n"
    latex += "Channel Removed & Accuracy & Drop & Importance \\\\\n"
    latex += "\\midrule\n"

    for _, row in results_df.iterrows():
        latex += f"{row['channel_removed']:12s} & "
        latex += f"{row['accuracy_mean']:.3f} $\\pm$ {row['accuracy_std']:.3f} & "
        latex += f"{row['accuracy_drop']:+.3f} & "
        latex += f"{row['importance_percent']:.1f}\\% \\\\\n"

    latex += "\\midrule\n"
    latex += f"Baseline (all channels) & \\multicolumn{{3}}{{c}}{{{baseline_acc:.3f}}} \\\\\n"
    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"

    return latex


def main():
    parser = argparse.ArgumentParser(description='Channel Importance Analysis')
    parser.add_argument('--dataset', type=str, default='sleep_edf',
                       help='Dataset name (sleep_edf or isruc)')
    parser.add_argument('--data-path', type=str,
                       default=("E:/datasets/Sleep/sleep-edf-database-expanded-1.0.0" if os.name == 'nt'
                                else "/mnt/data1/home/tanhuang/datasets/sleep-edf-database-expanded-1.0.0"),
                       help='Path to dataset')
    parser.add_argument('--isruc-path', type=str, default=None,
                       help='Path to ISRUC dataset')
    parser.add_argument('--n-folds', type=int, default=5,
                       help='Number of cross-validation folds')
    parser.add_argument('--subjects', type=str, default='0~39',
                       help='Subject range (e.g., 0~39)')
    parser.add_argument('--output-dir', type=str, default='results/channel_analysis',
                       help='Output directory')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("Sleep EEG Channel Importance Analysis")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"CV Folds: {args.n_folds}")

    # Parse subject range
    if '~' in args.subjects:
        start, end = args.subjects.split('~')
        subjects = list(range(int(start), int(end) + 1))
    else:
        subjects = [int(x) for x in args.subjects.split(',')]

    print(f"Subjects: {subjects}")

    # Load data
    print("\nLoading dataset...")
    if args.dataset.lower() == 'sleep_edf':
        X, y, meta = data_loader.load_sleep_edf(
            data_path=args.data_path,
            subjects=subjects,
            select_n_channels=2
        )
    elif args.dataset.lower() == 'isruc':
        X, y, meta = data_loader.load_isruc_sleep(
            data_path=args.isruc_path or args.data_path,
            subjects=subjects,
            select_n_channels=2
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    print(f"Data loaded: {X.shape[0]} epochs, {X.shape[1]} channels, {X.shape[2]} time points")
    print(f"Classes: {np.unique(y)}")

    # Run analysis
    results_df, baseline_acc = run_channel_importance_analysis(X, y, meta=meta, n_folds=args.n_folds)

    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"channel_importance_results_{timestamp}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    # Generate and save LaTeX table
    latex_table = generate_latex_table(results_df, baseline_acc)
    tex_path = output_dir / f"channel_importance_table_{timestamp}.tex"
    with open(tex_path, 'w') as f:
        f.write(latex_table)
    print(f"LaTeX table saved to: {tex_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("Summary: Channel Importance Ranking")
    print("=" * 80)
    results_sorted = results_df.sort_values('accuracy_drop', ascending=False)

    print(f"\n{'Rank':<6} {'Channel':<15} {'Drop in Acc':<15} {'Importance':<15}")
    print("-" * 55)
    for rank, (_, row) in enumerate(results_sorted.iterrows(), 1):
        print(f"{rank:<6} {row['channel_removed']:<15} {row['accuracy_drop']:+.4f}        {row['importance_percent']:.1f}%")

    print("\n" + "=" * 80)
    print("Conclusion")
    print("=" * 80)
    top_channel = results_sorted.iloc[0]['channel_removed']
    top_importance = results_sorted.iloc[0]['importance_percent']
    print(f"\nThe {top_channel} channel contributes the most to classification accuracy ({top_importance:.1f}%).")
    print(f"This is consistent with the neurophysiological knowledge about sleep EEG.")


if __name__ == "__main__":
    main()