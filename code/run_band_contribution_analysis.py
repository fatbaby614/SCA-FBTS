#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Frequency Band Contribution Analysis for SCA-FBTS Algorithm

This script analyzes the contribution of each frequency band to the SCA-FBTS algorithm
by removing one band at a time and measuring the accuracy drop.

Usage:
    python run_band_contribution_analysis.py --dataset sleep_edf --data-path /path/to/data
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import argparse
import numpy as np
import pandas as pd
import time
from pathlib import Path
import sys
import os

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, classification_report
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from scipy import signal

from config.algorithms_config import RANDOM_STATE, FS
import data_loader_sleep as data_loader
from sca_fbts import SCA_FBTS

# MUST match the 8 default bands used by SCA_FBTS (sca_fbts.py).
# Note: config.SLEEP_BANDS has only 5 coarse bands and is NOT what SCA-FBTS uses.
SCA_FBTS_BANDS = [
    (0.5, 4, 'delta'),
    (4, 6, 'low_theta'),
    (6, 8, 'high_theta'),
    (8, 10, 'low_alpha'),
    (10, 12, 'high_alpha'),
    (12, 14, 'sigma'),
    (14, 20, 'low_beta'),
    (20, 30, 'high_beta'),
]


class SCAFBTSBandAnalysis:
    """SCA-FBTS with controllable frequency bands for ablation analysis."""

    def __init__(self, fs=100, classifier='svm', active_bands=None):
        """Initialize SCA-FBTS with specific bands active.

        Args:
            fs: Sampling frequency
            classifier: Classifier type
            active_bands: List of band indices to use (None = all bands)
        """
        self.fs = fs
        self.classifier_name = classifier

        if active_bands is None:
            self.active_bands = list(range(len(SCA_FBTS_BANDS)))
        else:
            self.active_bands = active_bands

        self.model = None

    def fit(self, X, y):
        """Fit the model."""
        # Build frequency bands based on active_bands
        freq_bands = []
        for band_idx in self.active_bands:
            low, high, _ = SCA_FBTS_BANDS[band_idx]
            freq_bands.append((low, high))

        self.model = SCA_FBTS(
            fs=self.fs,
            freq_bands=freq_bands,
            temporal_smoothing=True,
            smoothing_window=3
        )
        self.model.fit(X, y)
        return self

    def predict(self, X):
        """Predict labels."""
        return self.model.predict(X)


def run_band_contribution_analysis(X, y, meta=None, n_folds=5):
    """Analyze frequency band contributions by ablation.

    Args:
        X: EEG data (n_epochs, n_channels, n_times)
        y: Labels
        n_folds: Number of cross-validation folds

    Returns:
        DataFrame with band contribution results
    """
    print("\n" + "=" * 80)
    print("Frequency Band Contribution Analysis")
    print("=" * 80)

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

    # Baseline: all bands
    print("\n[Baseline] Using all frequency bands...")
    baseline_accuracies = []
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = SCAFBTSBandAnalysis()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        baseline_accuracies.append(accuracy_score(y_test, y_pred))

    baseline_acc = np.mean(baseline_accuracies)
    print(f"  Baseline Accuracy: {baseline_acc:.4f}")

    # Ablation: remove each band
    print("\n[Ablation] Removing each frequency band...")
    band_names = [name for _, _, name in SCA_FBTS_BANDS]

    for band_idx in range(len(SCA_FBTS_BANDS)):
        band_name = band_names[band_idx]

        # Create active bands list without this band
        active_bands = [i for i in range(len(SCA_FBTS_BANDS)) if i != band_idx]

        accuracies = []
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = SCAFBTSBandAnalysis(active_bands=active_bands)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracies.append(accuracy_score(y_test, y_pred))

        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        drop = baseline_acc - mean_acc

        print(f"  Without {band_name:12s}: {mean_acc:.4f} ± {std_acc:.4f} (drop: {drop:+.4f})")

        results.append({
            'band_removed': band_name,
            'band_index': band_idx,
            'accuracy_mean': mean_acc,
            'accuracy_std': std_acc,
            'baseline_accuracy': baseline_acc,
            'accuracy_drop': drop,
            'contribution_percent': (drop / baseline_acc) * 100
        })

    # Also test individual bands
    print("\n[Single Band] Testing each band individually...")
    for band_idx in range(len(SCA_FBTS_BANDS)):
        band_name = band_names[band_idx]

        accuracies = []
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = SCAFBTSBandAnalysis(active_bands=[band_idx])
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracies.append(accuracy_score(y_test, y_pred))

        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)

        print(f"  Only {band_name:12s}: {mean_acc:.4f} ± {std_acc:.4f}")

        results.append({
            'band_removed': f"{band_name} (only)",
            'band_index': band_idx,
            'accuracy_mean': mean_acc,
            'accuracy_std': std_acc,
            'baseline_accuracy': baseline_acc,
            'accuracy_drop': baseline_acc - mean_acc,
            'contribution_percent': ((baseline_acc - mean_acc) / baseline_acc) * 100
        })

    return pd.DataFrame(results), baseline_acc


def generate_latex_table(results_df, baseline_acc):
    """Generate LaTeX table for band contribution."""
    ablation_df = results_df[results_df['band_removed'].str.contains('only') == False].copy()

    latex = "\\begin{table}[H]\n"
    latex += "\\caption{Frequency band contribution analysis. Accuracy drop indicates band importance.}\n"
    latex += "\\label{tab:band_contribution}\n"
    latex += "\\centering\n"
    latex += "\\footnotesize\n"
    latex += "\\begin{tabular}{lccc}\n"
    latex += "\\toprule\n"
    latex += "Band Removed & Accuracy & Drop & Contribution \\\\\n"
    latex += "\\midrule\n"

    for _, row in ablation_df.iterrows():
        latex += f"{row['band_removed']:12s} & "
        latex += f"{row['accuracy_mean']:.3f} $\\pm$ {row['accuracy_std']:.3f} & "
        latex += f"{row['accuracy_drop']:+.3f} & "
        latex += f"{row['contribution_percent']:.1f}\\% \\\\\n"

    latex += "\\midrule\n"
    latex += f"Baseline (all bands) & \\multicolumn{{3}}{{c}}{{{baseline_acc:.3f}}} \\\\\n"
    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"

    return latex


def main():
    parser = argparse.ArgumentParser(description='Frequency Band Contribution Analysis')
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
    parser.add_argument('--output-dir', type=str, default='results/band_analysis',
                       help='Output directory')
    parser.add_argument('--algorithm', type=str, default='SCA-FBTS',
                       help='Algorithm to analyze')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("Sleep EEG Frequency Band Contribution Analysis (SCA-FBTS)")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Algorithm: {args.algorithm}")
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
    results_df, baseline_acc = run_band_contribution_analysis(X, y, meta=meta, n_folds=args.n_folds)

    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"band_contribution_results_{timestamp}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    # Generate and save LaTeX table
    latex_table = generate_latex_table(results_df, baseline_acc)
    tex_path = output_dir / f"band_contribution_table_{timestamp}.tex"
    with open(tex_path, 'w') as f:
        f.write(latex_table)
    print(f"LaTeX table saved to: {tex_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("Summary: Frequency Band Importance Ranking")
    print("=" * 80)
    ablation_df = results_df[results_df['band_removed'].str.contains('only') == False].copy()
    ablation_df = ablation_df.sort_values('accuracy_drop', ascending=False)

    print(f"\n{'Rank':<6} {'Band':<15} {'Drop in Acc':<15} {'Contribution':<15}")
    print("-" * 55)
    for rank, (_, row) in enumerate(ablation_df.iterrows(), 1):
        print(f"{rank:<6} {row['band_removed']:<15} {row['accuracy_drop']:+.4f}        {row['contribution_percent']:.1f}%")

    print("\n" + "=" * 80)
    print("Conclusion")
    print("=" * 80)
    top_band = ablation_df.iloc[0]['band_removed']
    top_contribution = ablation_df.iloc[0]['contribution_percent']
    print(f"\nThe {top_band} band contributes the most to classification accuracy ({top_contribution:.1f}%).")
    print(f"Removing it causes the largest accuracy drop, indicating its importance for sleep staging.")


if __name__ == "__main__":
    main()