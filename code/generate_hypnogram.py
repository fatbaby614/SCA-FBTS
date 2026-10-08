#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate Sleep Hypnogram Figure
Plot sleep hypnogram comparing true labels vs predicted labels for a subject.

Usage:
    python generate_hypnogram.py --data-path /path/to/physionet --subject 0
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import numpy as np
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.algorithms_config import RESULTS_PATH, get_timestamped_filename
from algorithms_collection import get_algorithm
import data_loader_sleep as data_loader
import visualization


def main():
    parser = argparse.ArgumentParser(description='Generate Sleep Hypnogram Figure')
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to Sleep-EDF dataset')
    parser.add_argument('--isruc-path', type=str, default=None,
                        help='Path to ISRUC-Sleep dataset')
    parser.add_argument('--dataset', type=str, default='sleep_physionet',
                        choices=['sleep_edf', 'sleep_physionet', 'isruc', 'combined'],
                        help='Dataset to use')
    parser.add_argument('--subject', type=str, default='0',
                        help='Subject ID to visualize (e.g., 0 for Sleep-EDF, 1 for ISRUC)')
    parser.add_argument('--recording', type=int, default=1,
                        help='Recording index (1 or 2 for Sleep-EDF)')
    parser.add_argument('--algorithm', type=str, default='FilterBankTangentSpace+SVM',
                        help='Algorithm to use for prediction')
    parser.add_argument('--compare-algorithms', type=str, nargs='+', default=None,
                        help='Additional algorithms to compare (for comparison plot)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for figures')
    parser.add_argument('--n-channels', type=int, default=2,
                        help='Number of channels to use')
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = RESULTS_PATH
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Sleep Hypnogram Generator")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Subject: {args.subject}")
    print(f"Algorithm: {args.algorithm}")
    
    # Load data
    print("\nLoading data...")
    
    if args.dataset in ['sleep_edf', 'sleep_physionet']:
        subject_id = int(args.subject)
        X, y, meta = data_loader.load_sleep_edf(
            data_path=args.data_path,
            subjects=[subject_id],
            recording=[args.recording],
            select_n_channels=args.n_channels,
        )
        subject_str = f"SC4{subject_id:02d}{'E' if args.recording == 1 else 'E'}"
    elif args.dataset == 'isruc':
        if args.isruc_path is None:
            raise ValueError("--isruc-path is required for ISRUC dataset")
        subject_id = int(args.subject)
        X, y, meta = data_loader.load_isruc_sleep(
            data_path=args.isruc_path,
            subjects=[subject_id],
            select_n_channels=args.n_channels,
        )
        subject_str = f"ISRUC-{subject_id}"
    elif args.dataset == 'combined':
        if args.isruc_path is None:
            raise ValueError("--isruc-path is required for combined dataset")
        X, y, meta = data_loader.load_combined_datasets(
            sleep_edf_path=args.data_path,
            isruc_path=args.isruc_path,
            select_n_channels=args.n_channels,
        )
        subject_str = "Combined"
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    print(f"Loaded {len(X)} epochs")
    
    # Train model and predict
    n_channels = X.shape[1]
    n_times = X.shape[2]
    n_classes = len(np.unique(y))
    
    print(f"Training {args.algorithm}...")
    model = get_algorithm(args.algorithm, n_channels, n_times, n_classes, fs=100)
    # Check if model's fit method accepts epochs parameter
    import inspect
    fit_signature = inspect.signature(model.fit)
    if 'epochs' in fit_signature.parameters:
        model.fit(X, y, epochs=100)
    else:
        model.fit(X, y)
    
    print("Generating predictions...")
    y_pred = model.predict(X)
    
    # Generate single algorithm hypnogram
    print("\nGenerating hypnogram...")
    hypnogram_path = os.path.join(
        args.output_dir, 
        get_timestamped_filename(f'hypnogram_{args.dataset}_subject{args.subject}', 'png')
    )
    visualization.plot_hypnogram(
        true_labels=y,
        pred_labels=y_pred,
        subject_id=subject_str,
        algorithm_name=args.algorithm.replace('FilterBankTangentSpace+SVM', 'FBTS-SVM'),
        save_path=hypnogram_path,
    )
    print(f"Single hypnogram saved to: {hypnogram_path}")
    
    # Generate comparison hypnogram if multiple algorithms requested
    if args.compare_algorithms:
        print(f"\nGenerating comparison with: {', '.join(args.compare_algorithms)}")
        
        pred_dict = {args.algorithm.replace('FilterBankTangentSpace+SVM', 'FBTS-SVM'): y_pred}
        
        for algo_name in args.compare_algorithms:
            print(f"  Training {algo_name}...")
            model_comp = get_algorithm(algo_name, n_channels, n_times, n_classes, fs=100)
            # Check if model's fit method accepts epochs parameter
            import inspect
            fit_signature = inspect.signature(model_comp.fit)
            if 'epochs' in fit_signature.parameters:
                model_comp.fit(X, y, epochs=100)
            else:
                model_comp.fit(X, y)
            y_pred_comp = model_comp.predict(X)
            
            display_name = algo_name.replace('FilterBankTangentSpace+SVM', 'FBTS-SVM')
            display_name = display_name.replace('HandcraftedFeatures+RF', 'Handcrafted+RF')
            pred_dict[display_name] = y_pred_comp
        
        comparison_path = os.path.join(
            args.output_dir,
            get_timestamped_filename(f'hypnogram_comparison_{args.dataset}_subject{args.subject}', 'png')
        )
        visualization.plot_hypnogram_comparison(
            true_labels=y,
            pred_labels_dict=pred_dict,
            subject_id=subject_str,
            save_path=comparison_path,
        )
        print(f"Comparison hypnogram saved to: {comparison_path}")
    
    # Print accuracy
    accuracy = np.mean(y == y_pred)
    print(f"\nSubject {subject_str} accuracy: {accuracy:.1%}")
    
    print("\n" + "=" * 60)
    print("Hypnogram generation completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
