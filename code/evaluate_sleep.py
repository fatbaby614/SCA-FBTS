#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sleep EEG Algorithm Evaluation System
Evaluates FilterBankTangentSpace+SVM and other algorithms on sleep stage classification
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import argparse
import numpy as np
import pandas as pd
import time
import sys
import pickle
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    classification_report,
    f1_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
)
import os
from config.algorithms_config import RESULTS_PATH, RANDOM_STATE, N_SPLITS, get_timestamped_filename, get_results_path, FS
from algorithms_collection import get_algorithm
import data_loader_sleep as data_loader
import visualization
import torch

from pathlib import Path

DEFAULT_SLEEP_EDF_PATH = ("E:/datasets/Sleep/sleep-edf-database-expanded-1.0.0" if os.name == 'nt'
                          else "/mnt/data1/home/tanhuang/datasets/sleep-edf-database-expanded-1.0.0")


def _extract_dataset_info(meta, dataset_name, data_path, isruc_path, subjects_arg, recording_arg, 
                          select_channels_arg, select_n_channels_arg):
    """Extract detailed dataset information from metadata."""
    info = {
        'dataset_name': dataset_name,
        'data_path': str(data_path) if data_path else '',
        'isruc_path': str(isruc_path) if isruc_path else '',
        'subjects_arg': str(subjects_arg) if subjects_arg else 'all',
        'recording_arg': str(recording_arg) if recording_arg else 'all',
        'select_channels_arg': str(select_channels_arg) if select_channels_arg else 'auto',
        'select_n_channels_arg': select_n_channels_arg if select_n_channels_arg else 'all',
    }
    
    if isinstance(meta, pd.DataFrame):
        if 'dataset' in meta.columns:
            datasets_included = meta['dataset'].unique().tolist()
            info['datasets_included'] = datasets_included
            
            dataset_counts = meta.groupby('dataset')['subject'].nunique().to_dict()
            info['subjects_per_dataset'] = dataset_counts
            
            if 'record_id' in meta.columns:
                recording_counts = meta.groupby('dataset')['record_id'].nunique().to_dict()
                info['recordings_per_dataset'] = recording_counts
        
        if 'subject' in meta.columns:
            all_subjects = meta['subject'].astype(str).unique().tolist()
            info['n_subjects'] = len(all_subjects)
            info['subject_list'] = all_subjects
            
            if 'dataset' in meta.columns:
                subjects_by_dataset = {}
                for ds in meta['dataset'].unique():
                    ds_subjects = meta[meta['dataset'] == ds]['subject'].astype(str).unique().tolist()
                    subjects_by_dataset[ds] = ds_subjects
                info['subjects_by_dataset'] = subjects_by_dataset
        
        if 'night' in meta.columns:
            nights = meta['night'].astype(str).unique().tolist()
            info['nights'] = nights
        
        if 'record_id' in meta.columns:
            records = meta['record_id'].astype(str).unique().tolist()
            info['n_recordings'] = len(records)
            info['record_list'] = records
    
    return info


def _get_model_size_mb(model, algo_name):
    """Calculate model size in megabytes."""
    try:
        if algo_name in ['EEGNet-Light', 'EEGNet-Lite', 
                         'SleepTransformer-Light', 'SleepTransformer-Lite',
                         'DeepSleepNet', 'DeepSleepNet-Light',
                         'TinySleepNet', 'U-Sleep']:
            if hasattr(model, 'model') and hasattr(model.model, 'parameters'):
                param_size = sum(p.numel() * p.element_size() for p in model.model.parameters())
                buffer_size = sum(b.numel() * b.element_size() for b in model.model.buffers())
                return (param_size + buffer_size) / (1024 * 1024)
        
        model_bytes = pickle.dumps(model)
        return len(model_bytes) / (1024 * 1024)
    except Exception as e:
        print(f"    Warning: Could not calculate model size: {e}")
        return 0.0


def _measure_inference_time(model, X_test, n_runs=10):
    """Measure average inference time per epoch in milliseconds."""
    try:
        times = []
        n_samples = min(100, len(X_test))
        X_sample = X_test[:n_samples]
        
        for _ in range(3):
            _ = model.predict(X_sample)
        
        for _ in range(n_runs):
            start = time.perf_counter()
            _ = model.predict(X_sample)
            end = time.perf_counter()
            times.append((end - start) * 1000 / n_samples)
        
        return np.mean(times)
    except Exception as e:
        print(f"    Warning: Could not measure inference time: {e}")
        return 0.0


def _extract_features_for_tsne(model, X, algo_name):
    """Extract model features for t-SNE visualization."""
    try:
        if algo_name in ['HandcraftedFeatures+RF', 'HandcraftedRF', 'Handcrafted+RF']:
            feats = model._extract_features(X)
            return model.scaler.transform(feats)

        if algo_name in ['FilterBankTangentSpace+SVM', 'SCA-FBTS',
                         'FilterBankTangentSpace+LDA', 'FilterBankTangentSpace+RF']:
            from algorithms_collection import apply_bandpass_filter
            features_list = []
            for i, (low, high) in enumerate(model.freq_bands):
                X_band = np.array([apply_bandpass_filter(trial, low, high, model.fs) for trial in X])
                cov_matrices = model.cov_estimators[i].transform(X_band)
                ts_features = model.ts_transformers[i].transform(cov_matrices)
                features_list.append(ts_features)
            X_combined = np.hstack(features_list)
            if model.feature_selector is not None:
                return model.feature_selector.transform(X_combined)
            return X_combined

        if algo_name == 'MDM':
            cov_matrices = model.cov_estimator.transform(X)
            return cov_matrices.reshape(cov_matrices.shape[0], -1)

        if algo_name in ['RiemannTangentSpace', 'RiemannTangentSpace+SVM',
                         'RiemannTangentSpace+RF', 'RiemannTangentSpace+PCA']:
            cov_matrices = model.pipeline.named_steps['cov'].transform(X)
            ts_features = model.pipeline.named_steps['ts'].transform(cov_matrices)
            if 'pca' in model.pipeline.named_steps:
                ts_features = model.pipeline.named_steps['pca'].transform(ts_features)
            return ts_features

        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)
            if isinstance(proba, np.ndarray):
                return proba

    except Exception as exc:
        print(f"    Warning: feature extraction failed for {algo_name}: {exc}")

    return X.reshape(X.shape[0], -1)


def evaluate_sleep_dataset(algorithms, dataset_name='dummy', n_epochs=150, save_model=False, 
                          load_model=False, model_dir='models', data_path=DEFAULT_SLEEP_EDF_PATH, 
                          isruc_path=None, n_epochs_dummy=200, **kwargs):
    """
    Evaluate algorithms on sleep dataset
    
    Args:
        algorithms: List of algorithm names to evaluate
        dataset_name: Name of dataset ('dummy', 'sleep_edf', 'isruc', 'combined')
        n_epochs: Number of training epochs for deep learning models
        save_model: Whether to save trained models
        load_model: Whether to load pretrained models
        model_dir: Directory to save/load models
        data_path: Path to Sleep-EDF dataset
        isruc_path: Path to ISRUC-Sleep dataset (for combined/isruc)
        n_epochs_dummy: Number of epochs for dummy data
        **kwargs: Additional arguments for data loader and evaluation options
    
    Returns:
        results_df: DataFrame with detailed results
        summary_df: DataFrame with summary results
    """
    print("=" * 80)
    print("Sleep EEG Algorithm Evaluation System")
    print("=" * 80)

    cv_mode = kwargs.pop('cv_mode', 'stratified')
    tsne_enabled = kwargs.pop('tsne', False)
    tsne_use_pca = kwargs.pop('tsne_use_pca', True)
    tsne_max_samples = kwargs.pop('tsne_max_samples', 500)
    summary_view = kwargs.pop('summary_view', 'compact')
    output_dir = kwargs.pop('output_dir', 'results')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    subjects_arg = kwargs.get('subjects', None)
    recording_arg = kwargs.get('recording', (1, 2))
    select_channels_arg = kwargs.get('select_channels', None)
    select_n_channels_arg = kwargs.get('select_n_channels', None)
    dreams_database_arg = kwargs.pop('dreams_database', 'patients')
    
    print(f"\nLoading dataset: {dataset_name}")
    if dataset_name.lower() == 'dummy':
        X, y, meta = data_loader.load_dummy_sleep_data(n_epochs=n_epochs_dummy)
    elif dataset_name.lower() == 'combined':
        subjects = kwargs.pop('subjects', None)
        recording = kwargs.pop('recording', None)
        select_channels = kwargs.pop('select_channels', None)
        select_n_channels = kwargs.pop('select_n_channels', 2)
        X, y, meta = data_loader.load_combined_datasets(
            sleep_edf_path=data_path,
            isruc_path=isruc_path,
            sleep_edf_subjects=subjects,
            select_n_channels=select_n_channels,
        )
    elif dataset_name.lower() == 'isruc':
        if isruc_path is None and data_path is not None:
            isruc_path = data_path
        # ISRUC doesn't support 'recording' parameter, remove it
        isruc_kwargs = {k: v for k, v in kwargs.items() if k != 'recording'}
        X, y, meta = data_loader.load_sleep_dataset(dataset_name, data_path=isruc_path, **isruc_kwargs)
    elif dataset_name.lower() == 'dreams':
        # DREAMS doesn't support 'recording' parameter, remove it
        dreams_kwargs = {k: v for k, v in kwargs.items() if k != 'recording'}
        
        # Default to select_n_channels = 2 if not provided safely, as patient channels vary
        if dreams_kwargs.get('select_n_channels') is None and dreams_kwargs.get('select_channels') is None:
            dreams_kwargs['select_n_channels'] = 2
            
        X, y, meta = data_loader.load_sleep_dataset(
            dataset_name,
            data_path=data_path,
            database=dreams_database_arg,
            **dreams_kwargs,
        )
    else:
        X, y, meta = data_loader.load_sleep_dataset(dataset_name, data_path=data_path, **kwargs)
    
    n_samples, n_channels, n_times = X.shape
    n_classes = len(np.unique(y))
    
    dataset_info = _extract_dataset_info(meta, dataset_name, data_path, isruc_path, 
                                          subjects_arg, recording_arg, select_channels_arg, 
                                          select_n_channels_arg)
    
    print(f"\nData loaded:")
    print(f"  Samples: {n_samples}")
    print(f"  Channels: {n_channels}")
    print(f"  Time points: {n_times}")
    print(f"  Classes: {n_classes}")
    print(f"  Class distribution: {np.bincount(y)}")
    if dataset_info.get('n_subjects'):
        print(f"  Total subjects: {dataset_info['n_subjects']}")
    if dataset_info.get('n_recordings'):
        print(f"  Total recordings: {dataset_info['n_recordings']}")
    if dataset_info.get('datasets_included'):
        print(f"  Datasets included: {', '.join(dataset_info['datasets_included'])}")

    cv_splits = _build_cv_splits(X, y, meta, cv_mode=cv_mode)
    
    results = []
    subject_results = []
    
    for algo_name in algorithms:
        print(f"\n{'=' * 60}")
        print(f"Evaluating algorithm: {algo_name}")
        print(f"{'=' * 60}")
        print(f"  CV mode: {cv_mode} | Total folds: {len(cv_splits)}")

        # Wrap each algorithm in its own try/except so a single failing algorithm
        # (e.g. an OOM or initialization bug) does not abort the entire run and
        # we still get a partial summary CSV for the successful algorithms.
        try:
            for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):
                print(f"  [Fold {fold_idx+1}/{len(cv_splits)}] "
                      f"Train samples: {len(train_idx)} | Test samples: {len(test_idx)}")
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                meta_test = meta.iloc[test_idx].reset_index(drop=True) if isinstance(meta, pd.DataFrame) else None

                train_subjects = None
                test_subjects = None
                if isinstance(meta, pd.DataFrame) and 'subject' in meta.columns:
                    train_subjects = sorted(meta.iloc[train_idx]['subject'].astype(str).unique().tolist())
                    test_subjects = sorted(meta.iloc[test_idx]['subject'].astype(str).unique().tolist())
            
                # Model save/load path
                model_path = os.path.join(
                    model_dir,
                    f'sleep_{dataset_name}_{cv_mode}_{algo_name}_fold{fold_idx+1}.pt',
                )
            
                if load_model and os.path.exists(model_path):
                    print(f"  Loading model from {model_path}")
                    model = _load_sleep_model(model_path, algo_name, n_channels, n_times, n_classes)
                    train_time = 0
                else:
                    # SCA-FBTS uses temporal smoothing consistently across all CV modes
                    model = get_algorithm(algo_name, n_channels, n_times, n_classes, fs=FS)
                
                    start_time = time.time()
                    # Check if model's fit method accepts epochs parameter
                    import inspect
                    import gc
                    fit_signature = inspect.signature(model.fit)
                    try:
                        if 'epochs' in fit_signature.parameters:
                            model.fit(X_train, y_train, epochs=n_epochs)
                        else:
                            model.fit(X_train, y_train)
                    except RuntimeError as e:
                        if 'out of memory' in str(e).lower() or 'cuda' in str(e).lower():
                            import torch
                            print(f"\n[!] ⚠️ GPU Out of Memory (OOM) during fold {fold_idx+1}. Cleaning up and trying CPU...")
                            torch.cuda.empty_cache()
                            gc.collect()
                            # Fallback to CPU if supported by the model interface
                            if hasattr(model, 'device') or hasattr(model, 'set_params'):
                                try:
                                    if hasattr(model, 'device'):
                                        model.device = 'cpu'
                                    else:
                                        model.set_params(device='cpu')
                                
                                    print(f"Successfully recovered. Fitting on CPU for fold {fold_idx+1}...")
                                    if 'epochs' in fit_signature.parameters:
                                        model.fit(X_train, y_train, epochs=n_epochs)
                                    else:
                                        model.fit(X_train, y_train)
                                except Exception as cpu_e:
                                    print(f"CPU Fallback failed: {cpu_e}")
                                    raise e
                            else:
                                print("Model does not support dynamic device fallback. Raising error.")
                                raise e
                        else:
                            raise e
                    train_time = time.time() - start_time
                    print(f"  [Fold {fold_idx+1}] Training finished in {train_time:.1f}s")
                
                    # Save model
                    if save_model:
                        os.makedirs(model_dir, exist_ok=True)
                        print(f"  Saving model to {model_path}")
                        model.save_model(model_path)
            
                try:
                    y_pred = model.predict(X_test)
                except Exception as pred_exc:
                    import traceback
                    print(f"\n[!] ERROR during prediction: algorithm={algo_name}, "
                          f"fold={fold_idx+1}/{len(cv_splits)}, X_test shape={X_test.shape}")
                    traceback.print_exc()
                    raise pred_exc
            
                inference_time_ms = _measure_inference_time(model, X_test)
                model_size_mb = _get_model_size_mb(model, algo_name)
            
                accuracy = accuracy_score(y_test, y_pred)
                kappa = cohen_kappa_score(y_test, y_pred)
                macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
                weighted_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                balanced_acc = balanced_accuracy_score(y_test, y_pred)
                macro_precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
                macro_recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
                cm = confusion_matrix(y_test, y_pred, labels=list(range(n_classes))).tolist()
            
                print(f"  Fold {fold_idx + 1}/{len(cv_splits)}")
                print(
                    "    Accuracy: {:.4f}, Kappa: {:.4f}, Macro-F1: {:.4f}, "
                    "Balanced-Acc: {:.4f}, Weighted-F1: {:.4f}".format(
                        accuracy,
                        kappa,
                        macro_f1,
                        balanced_acc,
                        weighted_f1,
                    )
                )
                if cv_mode == 'subject' and test_subjects is not None:
                    print(f"    Test subjects: {', '.join(test_subjects)}")
            
                results.append({
                    'dataset': dataset_name,
                    'algorithm': algo_name,
                    'fold': fold_idx + 1,
                    'cv_mode': cv_mode,
                    'accuracy': accuracy,
                    'kappa': kappa,
                    'macro_f1': macro_f1,
                    'weighted_f1': weighted_f1,
                    'balanced_accuracy': balanced_acc,
                    'macro_precision': macro_precision,
                    'macro_recall': macro_recall,
                    'train_time': train_time,
                    'inference_time_ms': inference_time_ms,
                    'model_size_mb': model_size_mb,
                    'confusion_matrix': cm,
                    'n_train_samples': len(train_idx),
                    'n_test_samples': len(test_idx),
                    'train_subjects': ';'.join(train_subjects) if train_subjects is not None else '',
                    'test_subjects': ';'.join(test_subjects) if test_subjects is not None else '',
                    'n_channels': n_channels,
                    'n_times': n_times,
                    'n_classes': n_classes,
                    'datasets_included': ';'.join(dataset_info.get('datasets_included', [])),
                    'subjects_per_dataset': str(dataset_info.get('subjects_per_dataset', {})),
                    'recordings_per_dataset': str(dataset_info.get('recordings_per_dataset', {})),
                    'data_path': dataset_info.get('data_path', ''),
                    'isruc_path': dataset_info.get('isruc_path', ''),
                    'subjects_arg': dataset_info.get('subjects_arg', ''),
                    'recording_arg': dataset_info.get('recording_arg', ''),
                    'select_channels_arg': dataset_info.get('select_channels_arg', ''),
                    'select_n_channels_arg': str(dataset_info.get('select_n_channels_arg', '')),
                })
            
                # Compute per-subject metrics (independent statistical units for subject-based analysis,
                # instead of replicating fold-level metrics across all test subjects)
                if isinstance(meta_test, pd.DataFrame) and 'subject' in meta_test.columns:
                    y_test_arr = np.asarray(y_test)
                    y_pred_arr = np.asarray(y_pred)
                    subject_col = meta_test['subject'].astype(str).to_numpy()
                    for subj in np.unique(subject_col):
                        mask = subject_col == subj
                        ys = y_test_arr[mask]
                        yp = y_pred_arr[mask]
                        n_unique = len(np.unique(ys))
                        subject_results.append({
                            'dataset': dataset_name,
                            'algorithm': algo_name,
                            'fold': fold_idx + 1,
                            'cv_mode': cv_mode,
                            'subject': subj,
                            'n_epochs': int(mask.sum()),
                            'accuracy': accuracy_score(ys, yp),
                            'kappa': cohen_kappa_score(ys, yp) if n_unique > 1 else np.nan,
                            'macro_f1': f1_score(ys, yp, average='macro', zero_division=0),
                            'weighted_f1': f1_score(ys, yp, average='weighted', zero_division=0),
                            'balanced_accuracy': balanced_accuracy_score(ys, yp),
                        })
        
            # Print classification report for last fold
            print(f"\n  Classification report (last fold):")
            class_names = ['W', 'N1', 'N2', 'N3', 'REM'][:n_classes]
            class_labels = list(range(n_classes))
            print(classification_report(
                y_test,
                y_pred,
                labels=class_labels,
                target_names=class_names,
                zero_division=0,
            ))

            if tsne_enabled:
                X_vis = X_test
                y_vis = y_test
                if tsne_max_samples is not None and len(X_vis) > tsne_max_samples:
                    rng = np.random.default_rng(RANDOM_STATE)
                    idx = rng.choice(len(X_vis), size=tsne_max_samples, replace=False)
                    X_vis = X_vis[idx]
                    y_vis = y_vis[idx]

                try:
                    features = _extract_features_for_tsne(model, X_vis, algo_name)
                    tsne_name = get_timestamped_filename(
                        f"sleep_tsne_{dataset_name}_{cv_mode}_{algo_name.replace('+', 'plus')}",
                        'png'
                    )
                
                    # Ensure output directory exists before saving plots
                    os.makedirs(output_dir, exist_ok=True)
                
                    tsne_path = os.path.join(output_dir, tsne_name)
                    visualization.plot_tsne_visualization(
                        features=features,
                        labels=y_vis,
                        algorithm_name=algo_name,
                        subject_id=0,
                        dataset_name=f"{dataset_name}-{cv_mode}",
                        save_path=tsne_path,
                        use_pca=tsne_use_pca,
                    )
                    if meta_test is not None:
                        if len(meta_test) != len(X_test):
                            meta_vis = meta_test.iloc[:len(X_vis)].reset_index(drop=True)
                        elif len(X_vis) != len(X_test):
                            meta_vis = meta_test.iloc[idx].reset_index(drop=True)
                        else:
                            meta_vis = meta_test

                        traj_name = get_timestamped_filename(
                            f"sleep_tsne_trajectory_{dataset_name}_{cv_mode}_{algo_name.replace('+', 'plus')}",
                            'png'
                        )
                        traj_path = os.path.join(output_dir, traj_name)
                        os.makedirs(output_dir, exist_ok=True)
                        visualization.plot_sleep_trajectory_tsne(
                            features=features,
                            labels=y_vis,
                            meta=meta_vis,
                            algorithm_name=algo_name,
                            dataset_name=f"{dataset_name}-{cv_mode}",
                            save_path=traj_path,
                            use_pca=tsne_use_pca,
                        )

                        stab_name = get_timestamped_filename(
                            f"sleep_tsne_stability_{dataset_name}_{cv_mode}_{algo_name.replace('+', 'plus')}",
                            'png'
                        )
                        stab_path = os.path.join(output_dir, stab_name)
                        os.makedirs(output_dir, exist_ok=True)
                        visualization.plot_sleep_stability_tsne(
                            features=features,
                            labels=y_vis,
                            meta=meta_vis,
                            algorithm_name=algo_name,
                            dataset_name=f"{dataset_name}-{cv_mode}",
                            save_path=stab_path,
                            use_pca=tsne_use_pca,
                        )

                        panel_name = get_timestamped_filename(
                            f"sleep_tsne_{dataset_name}_{cv_mode}_{algo_name.replace('+', 'plus')}",
                            'png'
                        )
                        panel_path = os.path.join(output_dir, panel_name)
                        os.makedirs(output_dir, exist_ok=True)
                        visualization.plot_sleep_tsne(
                            features=features,
                            labels=y_vis,
                            meta=meta_vis,
                            algorithm_name=algo_name,
                            dataset_name=f"{dataset_name}-{cv_mode}",
                            save_path=panel_path,
                            use_pca=tsne_use_pca,
                        )
                    print(f"  t-SNE saved: {tsne_path}")
                except Exception as exc:
                    print(f"  Warning: t-SNE generation failed for {algo_name}: {exc}")

        except Exception as algo_exc:
            # An algorithm failed (init, training, or prediction). Log and skip to
            # the next algorithm so that the remaining ones can still produce
            # partial summary CSVs.
            import traceback
            print(f"\n[!] ERROR in algorithm '{algo_name}': {algo_exc}")
            print(f"[!] Traceback:")
            traceback.print_exc()
            print(f"[!] Skipping remaining folds for '{algo_name}' and continuing with next algorithm.")
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
            import gc
            gc.collect()
            continue
    
    # Generate summaries
    results_df = pd.DataFrame(results)
    summary_df = generate_sleep_summary(results_df)
    
    # Print summary
    print("\n" + "=" * 80)
    print("Performance Summary:")
    print("=" * 80)
    if summary_view == 'compact':
        _print_compact_summary(summary_df)
    else:
        print(summary_df.to_string(index=False))
    
    # Save results
    dataset_suffix = dataset_name.lower().replace('_', '')
    results_filename = get_timestamped_filename(f'sleep_evaluation_results_{dataset_suffix}', 'csv')
    summary_filename = get_timestamped_filename(f'sleep_evaluation_summary_{dataset_suffix}', 'csv')
    subjects_filename = get_timestamped_filename(f'sleep_evaluation_subjects_{dataset_suffix}', 'csv')
    
    # Ensure output directory exists before saving
    os.makedirs(output_dir, exist_ok=True)
    
    results_df.to_csv(os.path.join(output_dir, results_filename), index=False)
    summary_df.to_csv(os.path.join(output_dir, summary_filename), index=False)
    
    # Save subject-wise results if available (true per-subject metrics, not fold-level copies)
    if subject_results:
        subject_df = pd.DataFrame(subject_results)
        subject_df.to_csv(os.path.join(output_dir, subjects_filename), index=False)
        print(f"  Subject-wise results: {os.path.join(output_dir, subjects_filename)}")
    
    efficiency_df = summary_df[['algorithm', 'model_size_mb_mean', 'inference_time_ms_mean', 'train_time_mean']].copy()
    efficiency_df.columns = ['Algorithm', 'Model Size (MB)', 'Inference Time (ms/epoch)', 'Training Time (s)']
    efficiency_filename = get_timestamped_filename(f'model_efficiency_results_{dataset_suffix}', 'csv')
    efficiency_df.to_csv(os.path.join(output_dir, efficiency_filename), index=False)
    print(f"  Model efficiency: {os.path.join(output_dir, efficiency_filename)}")
    
    config_filename = get_timestamped_filename(f'sleep_evaluation_config_{dataset_suffix}', 'json')
    config_path = os.path.join(output_dir, config_filename)
    import json
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\nResult files saved:")
    print(f"  Detailed results: {os.path.join(output_dir, results_filename)}")
    print(f"  Summary: {os.path.join(output_dir, summary_filename)}")
    print(f"  Configuration: {config_path}")
    
    # Generate publication-ready figures
    print("\nGenerating publication-ready figures...")
    try:
        fig2_path = os.path.join(output_dir, get_timestamped_filename('algorithm_comparison', 'png'))
        visualization.plot_algorithm_comparison_bars(summary_df, save_path=fig2_path)
    except Exception as e:
        print(f"Warning: Could not generate algorithm comparison figure: {e}")
    
    try:
        fig3_path = os.path.join(output_dir, get_timestamped_filename('confusion_matrices', 'png'))
        visualization.plot_confusion_matrices_grid(results_df, save_path=fig3_path)
    except Exception as e:
        print(f"Warning: Could not generate confusion matrices figure: {e}")
    
    try:
        fig4_path = os.path.join(output_dir, get_timestamped_filename('per_stage_f1', 'png'))
        visualization.plot_per_stage_f1_scores(results_df, save_path=fig4_path)
    except Exception as e:
        print(f"Warning: Could not generate Figure 4: {e}")
    
    try:
        latex_table = visualization.generate_per_stage_f1_table(results_df)
        latex_path = os.path.join(output_dir, get_timestamped_filename('per_stage_f1_table', 'tex'))
        with open(latex_path, 'w') as f:
            f.write(latex_table)
        print(f"Per-stage F1 LaTeX table saved to {latex_path}")
    except Exception as e:
        print(f"Warning: Could not generate per-stage F1 table: {e}")
    
    return results_df, summary_df


def _load_sleep_model(model_path, algo_name, n_channels, n_times, n_classes):
    """Helper function to load sleep models"""
    if algo_name in ['HandcraftedFeatures+RF', 'HandcraftedRF', 'Handcrafted+RF']:
        from algorithms_collection import HandcraftedFeaturesRF
        return HandcraftedFeaturesRF.load_model(model_path)
    elif algo_name in ['FilterBankTangentSpace+SVM', 
                      'FilterBankTangentSpace+LDA', 'FilterBankTangentSpace+RF']:
        from algorithms_collection import FilterBankTangentSpace
        return FilterBankTangentSpace.load_model(model_path)
    elif algo_name == 'MDM':
        from algorithms_collection import MDM
        return MDM.load_model(model_path)
    elif algo_name == 'RiemannTangentSpace':
        from algorithms_collection import RiemannTangentSpace
        return RiemannTangentSpace.load_model(model_path)
    elif algo_name in ['EEGNet-Light', 'EEGNet-Lite']:
        from algorithms_collection import EEGNetLight
        return EEGNetLight.load_model(model_path, n_channels, n_times, n_classes)
    elif algo_name == 'SSC-SleepNet':
        from algorithms_collection import SSCSleepNet
        return SSCSleepNet.load_model(model_path, n_channels, n_times, n_classes)
    elif algo_name in ['SleepTransformer-Light', 'SleepTransformer-Lite']:
        from algorithms_collection import SleepTransformerLight
        return SleepTransformerLight.load_model(model_path, n_channels, n_times, n_classes)
    elif algo_name in ['DeepSleepNet', 'DeepSleepNet-Light']:
        from algorithms_collection import DeepSleepNetClassifier
        return DeepSleepNetClassifier.load_model(model_path, n_channels, n_times, n_classes)
    elif algo_name == 'TinySleepNet':
        from algorithms_collection import TinySleepNetClassifier
        return TinySleepNetClassifier.load_model(model_path, n_channels, n_times, n_classes)
    elif algo_name in ['U-Sleep', 'USleep']:
        from algorithms_collection import USleepClassifier
        return USleepClassifier.load_model(model_path, n_channels, n_times, n_classes)
    else:
        raise ValueError(f"Model loading not implemented for {algo_name}")


def generate_sleep_summary(results_df):
    """Generate summary of sleep evaluation results"""
    agg_dict = {
        'accuracy': ['mean', 'std', 'min', 'max'],
        'kappa': ['mean', 'std', 'min', 'max'],
        'macro_f1': ['mean', 'std', 'min', 'max'],
        'weighted_f1': ['mean', 'std', 'min', 'max'],
        'balanced_accuracy': ['mean', 'std', 'min', 'max'],
        'macro_precision': ['mean', 'std', 'min', 'max'],
        'macro_recall': ['mean', 'std', 'min', 'max'],
        'train_time': ['mean', 'std'],
        'inference_time_ms': ['mean', 'std'],
        'model_size_mb': ['mean'],
    }

    agg_dict = {k: v for k, v in agg_dict.items() if k in results_df.columns}

    summary = results_df.groupby(['dataset', 'cv_mode', 'algorithm']).agg(agg_dict).reset_index()
    
    summary.columns = ['_'.join(col).strip('_') for col in summary.columns.values]
    summary = summary.rename(columns={'algorithm_': 'algorithm'})
    
    return summary


def _print_compact_summary(summary_df):
    """Print compact console summary for readability."""
    preferred_cols = [
        'dataset',
        'cv_mode',
        'algorithm',
        'accuracy_mean',
        'kappa_mean',
        'macro_f1_mean',
        'train_time_mean',
        'inference_time_ms_mean',
        'model_size_mb_mean',
    ]
    cols = [c for c in preferred_cols if c in summary_df.columns]
    compact = summary_df[cols].copy()
    print(compact.to_string(index=False))


def check_gpu():
    """Check GPU availability"""
    print("=" * 80)
    print("GPU Check")
    print("=" * 80)
    print(f"\n1. PyTorch version: {torch.__version__}")
    print(f"2. CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"3. CUDA version: {torch.version.cuda}")
        print(f"4. Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"\n   GPU {i}:")
            print(f"     Name: {torch.cuda.get_device_name(i)}")
            print(f"     Total memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
            print(f"     Compute capability: {torch.cuda.get_device_capability(i)[0]}.{torch.cuda.get_device_capability(i)[1]}")
            print(f"     Multi-processor count: {torch.cuda.get_device_properties(i).multi_processor_count}")
        
        print(f"\n5. Current device: {torch.cuda.current_device()}")
        print(f"   Current device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        
        print("\nTesting GPU computation...")
        try:
            x = torch.randn(3, 3).cuda()
            y = torch.randn(3, 3).cuda()
            z = x + y
            print("✅ GPU computation test successful!")
        except Exception as e:
            print(f"❌ GPU computation test failed: {e}")
    print("\n" + "=" * 80)


def _build_cv_splits(X, y, meta, cv_mode='stratified'):
    """Build CV splits for epoch-wise or subject-wise evaluation."""
    cv_mode = cv_mode.lower()
    if cv_mode == 'stratified':
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        splits = list(skf.split(X, y))
        print(f"  CV mode: stratified ({len(splits)} folds)")
        return splits

    if cv_mode == 'subject':
        if not isinstance(meta, pd.DataFrame) or 'subject' not in meta.columns:
            raise ValueError("Subject-wise CV requires meta with a 'subject' column")

        groups = meta['subject'].astype(str).to_numpy()
        unique_subjects = np.unique(groups)
        n_subjects = len(unique_subjects)
        if n_subjects < 2:
            raise ValueError("Subject-wise CV requires at least 2 unique subjects")

        n_splits = min(N_SPLITS, n_subjects)
        if n_splits < 2:
            raise ValueError("Unable to create subject-wise folds with current data")

        gkf = GroupKFold(n_splits=n_splits)
        splits = list(gkf.split(X, y, groups))
        print(f"  CV mode: subject ({len(splits)} folds, {n_subjects} subjects)")
        return splits

    raise ValueError(f"Unknown cv_mode: {cv_mode}. Use 'stratified' or 'subject'.")


def _parse_subjects_arg(subject_tokens):
    """Parse subject tokens like ['0', '1~4', '7'] into a list of ints."""
    if subject_tokens is None:
        return None

    subjects = []
    seen = set()

    for token in subject_tokens:
        token = str(token).strip()
        if not token:
            continue

        if '~' in token:
            parts = token.split('~')
            if len(parts) != 2:
                raise ValueError(f"Invalid subject range: {token}. Use start~end, e.g. 0~4")
            start = int(parts[0])
            end = int(parts[1])
            step = 1 if end >= start else -1
            values = range(start, end + step, step)
        else:
            values = [int(token)]

        for val in values:
            if val not in seen:
                seen.add(val)
                subjects.append(val)

    return subjects


def main():
    parser = argparse.ArgumentParser(description='Sleep EEG Algorithm Evaluation')
    parser.add_argument('--algorithms', type=str, nargs='+', 
                        default=['FilterBankTangentSpace+SVM', 'SCA-FBTS', 'HandcraftedFeatures+RF', 'RiemannTangentSpace', 'MDM', 'TinySleepNet', 'DeepSleepNet', 'SSC-SleepNet'],
                        help='List of algorithms to evaluate')
    parser.add_argument('--dataset', type=str, default='dummy',
                        choices=['dummy', 'sleep_edf', 'isruc', 'dreams', 'combined'], 
                        help='Dataset to use (default: dummy)')
    parser.add_argument('--data-path', type=str, default=DEFAULT_SLEEP_EDF_PATH,
                        help='Sleep-EDF dataset root path')
    parser.add_argument('--isruc-path', type=str, default=None,
                        help='ISRUC-Sleep dataset path (required for combined/isruc)')
    parser.add_argument('--dreams-database', type=str, default='patients',
                        choices=['patients'],
                        help='DREAMS subset to evaluate (default: patients)')
    parser.add_argument('--subjects', type=str, nargs='+', default=None,
                        help='Sleep PhysioNet subject IDs or ranges, e.g. --subjects 0 1 2 3 or --subjects 0~4')
    parser.add_argument('--recording', type=int, nargs='+', default=[1, 2],
                        help='Sleep PhysioNet recording indices, e.g. --recording 1 or --recording 1 2 (default: both nights)')
    parser.add_argument('--select-channels', type=str, nargs='+', default=None,
                        help='Explicit EEG channels to use, e.g. --select-channels "EEG Fpz-Cz" "EEG Pz-Oz"')
    parser.add_argument('--select-n-channels', type=int, default=None,
                        help='Use first N EEG channels after filtering, e.g. --select-n-channels 4')
    parser.add_argument('--cv-mode', type=str, default='subject',
                        choices=['stratified', 'subject'],
                        help='Cross-validation mode: stratified (epoch-wise) or subject (subject-wise). Default: subject')
    parser.add_argument('--tsne', action='store_true',
                        help='Generate t-SNE feature visualization for each algorithm')
    parser.add_argument('--tsne-use-pca', action='store_true',
                        help='Apply PCA preprocessing before t-SNE')
    parser.add_argument('--tsne-max-samples', type=int, default=800,
                        help='Maximum samples for each algorithm t-SNE plot')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of training epochs for deep learning (default: 300)')
    parser.add_argument('--save-model', action='store_true',
                        help='Save trained models')
    parser.add_argument('--load-model', action='store_true',
                        help='Load trained models instead of training from scratch')
    parser.add_argument('--model-dir', type=str, default='models',
                        help='Directory to save/load models (default: models)')
    parser.add_argument('--check-gpu', action='store_true',
                        help='Check GPU availability and exit')
    parser.add_argument('--n-epochs-dummy', type=int, default=200,
                        help='Number of epochs for dummy data (default: 200)')
    parser.add_argument('--summary-view', type=str, default='compact', choices=['compact', 'full'],
                        help='Console summary view: compact or full (default: compact)')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Directory to save results (default: results)')
    
    args = parser.parse_args()

    args.subjects = _parse_subjects_arg(args.subjects)
    
    if args.check_gpu:
        check_gpu()
        return
    
    print("=" * 80)
    print("Configuration:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Algorithms: {', '.join(args.algorithms)}")
    print(f"  Training: {args.epochs} epochs (deep learning)")
    print(f"  CV mode: {args.cv_mode}")
    print(f"  t-SNE: {'on' if args.tsne else 'off'}")
    print(f"  Summary view: {args.summary_view}")
    if args.dataset == 'sleep_edf':
        print(f"  Subjects: {args.subjects if args.subjects is not None else '[default]'}")
        print(f"  Recording: {args.recording}")
        if args.select_channels:
            print(f"  Selected channels: {args.select_channels}")
        if args.select_n_channels is not None:
            print(f"  Selected first N EEG channels: {args.select_n_channels}")
    print("=" * 80)
    
    # Evaluate
    evaluate_sleep_dataset(
        algorithms=args.algorithms,
        dataset_name=args.dataset,
        n_epochs=args.epochs,
        save_model=args.save_model,
        load_model=args.load_model,
        model_dir=args.model_dir,
        data_path=args.data_path,
        isruc_path=args.isruc_path,
        dreams_database=args.dreams_database,
        n_epochs_dummy=args.n_epochs_dummy,
        subjects=args.subjects,
        recording=args.recording,
        select_channels=args.select_channels,
        select_n_channels=args.select_n_channels,
        cv_mode=args.cv_mode,
        summary_view=args.summary_view,
        tsne=args.tsne,
        tsne_use_pca=args.tsne_use_pca,
        tsne_max_samples=args.tsne_max_samples,
        output_dir=args.output_dir,
    )


if __name__ == '__main__':
    main()
