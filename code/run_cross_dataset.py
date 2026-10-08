#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Cross-dataset generalization experiment.
Trains models on a Source dataset (e.g. Sleep-EDF) and evaluates directly on a Target dataset (e.g. ISRUC).
Demonstrates the domain adaptation and generalization capabilities of SCA-FBTS.
"""

import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

from data_loader_sleep import load_sleep_dataset
from algorithms_collection import get_algorithm

def main():
    parser = argparse.ArgumentParser(description="Cross-dataset Generalization")
    parser.add_argument('--source-data-path', type=str, required=True, help='Source dataset path')
    parser.add_argument('--target-data-path', type=str, required=True, help='Target dataset path')
    parser.add_argument('--output-dir', type=str, default='results/common', help='Output directory')
    parser.add_argument('--quick', action='store_true', help='Use few subjects for quick test')
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # To ensure fairness, we train on parts of Sleep-EDF and test on parts of ISRUC
    if args.quick:
        source_subjects = list(range(2))
        target_subjects = list(range(1, 3))
        algorithms = ['SCA-FBTS']
    else:
        # Full mode
        source_subjects = list(range(39))  # Train on 39 subjects from EDF
        target_subjects = list(range(1, 20))  # Test on 19 subjects from ISRUC
        algorithms = ['HandcraftedFeatures+RF', 'SCA-FBTS', 'DeepSleepNet']

    print("="*80)
    print("Cross-Dataset Generalization Experiment")
    print(f"Source (Train): Sleep-EDF ({len(source_subjects)} subjects)")
    print(f"Target (Test): ISRUC ({len(target_subjects)} subjects)")
    print("="*80)

    # 1. Load Source Data
    print("\nLoading Source Data (Sleep-EDF)...")
    X_train, y_train, _ = load_sleep_dataset(
        'sleep_edf',
        data_path=args.source_data_path,
        subjects=source_subjects,
        select_n_channels=2  # Crucial: align channel dimensions
    )
    print(f"Source Data Shape: {X_train.shape}, Classes: {np.unique(y_train)}")

    # 2. Load Target Data
    print("\nLoading Target Data (ISRUC)...")
    X_test, y_test, _ = load_sleep_dataset(
        'isruc',
        data_path=args.target_data_path,
        subjects=target_subjects,
        select_n_channels=2  # Crucial: align channel dimensions
    )
    print(f"Target Data Shape: {X_test.shape}, Classes: {np.unique(y_test)}")

    # 3. Resample to align time dimensions (Sleep-EDF: 100 Hz, ISRUC: 200 Hz)
    # Use polyphase resampling with an anti-aliasing filter instead of naive
    # decimation (taking every Nth sample), which would alias high-frequency
    # content into the band of interest.
    from math import gcd
    from scipy.signal import resample_poly
    if X_train.shape[2] != X_test.shape[2]:
        n_in = X_test.shape[2]
        n_out = X_train.shape[2]
        up, down = n_out, n_in
        g = gcd(up, down)
        up, down = up // g, down // g
        print(f"\nResampling target data {n_in} -> {n_out} samples (factor {down}/{up})")
        X_test = resample_poly(X_test, up, down, axis=2)
        print(f"Target Data (after resampling): {X_test.shape}")

    # 4. Evaluate algorithms
    results = []
    print("\nStarting evaluation...")
    n_channels = X_train.shape[1]
    n_times = X_train.shape[2]

    for algo_name in algorithms:
        print("-" * 50)
        print(f"Evaluating: {algo_name}")
        model = get_algorithm(algo_name, n_channels=n_channels, n_times=n_times)

        print("  Training on Source (Sleep-EDF)...")
        try:
            model.fit(X_train, y_train)
        except Exception as e:
            print(f"  [Error] Training failed for {algo_name}: {e}")
            continue

        print("  Predicting on Target (ISRUC)...")
        try:
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            mf1 = f1_score(y_test, y_pred, average='macro')

            print(f"  >> Accuracy: {acc:.4f}, Macro-F1: {mf1:.4f}")
            results.append({'Algorithm': algo_name, 'Accuracy': acc, 'Macro-F1': mf1})
        except Exception as e:
            print(f"  [Error] Prediction failed for {algo_name}: {e}")

    if not results:
        print("No results to plot.")
        return

    # 5. Save Results and Plot
    df = pd.DataFrame(results)
    csv_path = os.path.join(args.output_dir, 'cross_dataset_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # Plot
    plt.figure(figsize=(10, 6))
    # Colors for traditional, riemann, proposed, deep
    colors = ['#9E9E9E', '#2196F3', '#FF5722', '#F44336']
    if len(algorithms) == 2:
        colors = ['#2196F3', '#FF5722']

    bars = plt.bar(df['Algorithm'], df['Accuracy'], color=colors[:len(df)])

    # Add title and labels
    plt.ylabel('Accuracy on Target Dataset (ISRUC)', fontsize=12, fontweight='bold')
    plt.title('Cross-Dataset Generalization\nTrain: Sleep-EDF $\\rightarrow$ Test: ISRUC', fontsize=14, fontweight='bold', pad=15)
    plt.xticks(rotation=15, ha='right', fontsize=11)
    plt.ylim(0, 1.0)
    plt.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    fig_path = os.path.join(args.output_dir, 'cross_dataset_accuracy.png')
    plt.savefig(fig_path, dpi=300, facecolor='white')
    print(f"Plot saved to {fig_path}")

if __name__ == '__main__':
    main()
