#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Noise Robustness Experiment for Sleep EEG Classification
Tests algorithm performance under simulated noise conditions:
- Powerline interference (50 Hz)
- EMG artifacts
- Electrode noise

This is a pure simulation experiment - no additional subjects or ethical approval needed.
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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
import time

from config.algorithms_config import RESULTS_PATH, RANDOM_STATE, FS
from algorithms_collection import get_algorithm
import data_loader_sleep as data_loader


def add_noise_at_snr(X, fs, noise_type, snr_db, seed=None, frequency=49.5):
    """Add a single noise type at an explicitly controlled SNR (in dB).

    SNR (dB) = 10*log10(signal_power / noise_power), so:
      +5 dB  -> signal power is ~3.16x the noise power (signal STRONGER than noise)
      +10 dB -> signal power is ~10x the noise power
      +20 dB -> signal power is ~100x the noise power

    Note on the powerline frequency: with fs=100 Hz the nominal 50 Hz component
    sits exactly on the Nyquist frequency, where sin(2*pi*50*k/100)=0 for every
    sampled time point k/100, i.e. a pure 50 Hz sinusoid is invisible to the
    sampler. We therefore simulate mains interference at 49.5 Hz (grid drift),
    which is standard for 100 Hz-sampled EEG and still lies outside the 0.5-30 Hz
    preprocessing band.

    Args:
        X: EEG data of shape (n_epochs, n_channels, n_times)
        fs: Sampling frequency
        noise_type: 'powerline' | 'emg' | 'electrode'
        snr_db: Target SNR in decibels
        seed: Random seed for reproducible noise realizations
        frequency: Powerline frequency (49.5 Hz for 100 Hz-sampled EEG; 50/60 Hz
                   only usable when fs > 2*frequency)

    Returns:
        EEG data with injected noise at the requested SNR
    """
    rng = np.random.default_rng(seed)
    X_noisy = X.copy()
    n_epochs, n_channels, n_times = X.shape
    t = np.arange(n_times) / fs

    for epoch_idx in range(n_epochs):
        for ch_idx in range(n_channels):
            signal = X[epoch_idx, ch_idx]
            signal_power = np.mean(signal ** 2)
            noise_power = signal_power / (10.0 ** (snr_db / 10.0))

            if noise_type == 'powerline':
                noise = np.sin(2 * np.pi * frequency * t)
            elif noise_type == 'emg':
                noise = rng.standard_normal(n_times)
                noise = np.convolve(noise, np.ones(5) / 5, mode='same')
            elif noise_type == 'electrode':
                drift_freq = rng.uniform(0.1, 0.5)
                slow_drift = np.sin(2 * np.pi * drift_freq * t)
                n_jumps = rng.integers(0, 3)
                jumps = np.zeros(n_times)
                for _ in range(n_jumps):
                    jump_idx = rng.integers(0, n_times)
                    jumps[jump_idx:] += rng.choice([-1.0, 1.0])
                noise = slow_drift + jumps
            else:
                noise = rng.standard_normal(n_times)

            if np.mean(noise ** 2) > 0:
                noise = noise * np.sqrt(noise_power / np.mean(noise ** 2))
            X_noisy[epoch_idx, ch_idx] += noise

    return X_noisy


def add_combined_noise_at_snr(X, fs, snr_db, seed=None):
    """Add combined noise (powerline + EMG + electrode) at a target overall SNR."""
    X_noisy = add_noise_at_snr(X, fs, 'powerline', snr_db + 3, seed=seed)
    X_noisy = add_noise_at_snr(X_noisy, fs, 'emg', snr_db + 3, seed=seed)
    X_noisy = add_noise_at_snr(X_noisy, fs, 'electrode', snr_db, seed=seed)
    return X_noisy


def add_powerline_noise(X, fs, amplitude, frequency=49.5):
    """
    Add powerline interference to EEG signals.
    
    Args:
        X: EEG data of shape (n_epochs, n_channels, n_times)
        fs: Sampling frequency
        amplitude: Noise amplitude relative to signal (0.0 to 1.0)
        frequency: Powerline frequency. Default 49.5 Hz: at fs=100 Hz the nominal
                   50 Hz sits on the Nyquist frequency and a pure 50 Hz sinusoid
                   vanishes after sampling (sin(2*pi*50*k/100)=0), so grid drift
                   is simulated instead.
    
    Returns:
        Noisy EEG data
    """
    n_epochs, n_channels, n_times = X.shape
    t = np.arange(n_times) / fs
    
    noise = amplitude * np.sin(2 * np.pi * frequency * t)
    noise = noise[np.newaxis, np.newaxis, :]
    
    signal_power = np.mean(X ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power > 0:
        scale = np.sqrt(signal_power * amplitude / noise_power)
        noise = noise * scale
    
    return X + noise


def add_emg_artifacts(X, fs, amplitude, burst_duration_range=(0.1, 0.3)):
    """
    Add EMG (muscle) artifacts to EEG signals.
    EMG artifacts are characterized by high-frequency bursts.
    
    Args:
        X: EEG data of shape (n_epochs, n_channels, n_times)
        fs: Sampling frequency
        amplitude: Artifact amplitude relative to signal (0.0 to 1.0)
        burst_duration_range: Range of burst durations in seconds
    
    Returns:
        EEG data with EMG artifacts
    """
    n_epochs, n_channels, n_times = X.shape
    X_noisy = X.copy()
    
    np.random.seed(RANDOM_STATE)
    
    for epoch_idx in range(n_epochs):
        for ch_idx in range(n_channels):
            signal = X[epoch_idx, ch_idx]
            signal_power = np.std(signal)
            
            n_bursts = np.random.randint(1, 4)
            
            for _ in range(n_bursts):
                burst_duration = np.random.uniform(*burst_duration_range)
                burst_samples = int(burst_duration * fs)
                
                start_idx = np.random.randint(0, n_times - burst_samples)
                end_idx = start_idx + burst_samples
                
                emg_noise = np.random.randn(burst_samples)
                emg_noise = np.convolve(emg_noise, np.ones(5)/5, mode='same')
                
                emg_noise = emg_noise * amplitude * signal_power
                
                X_noisy[epoch_idx, ch_idx, start_idx:end_idx] += emg_noise
    
    return X_noisy


def add_electrode_noise(X, fs, amplitude, drift_amplitude=0.1):
    """
    Add electrode noise (contact instability, impedance fluctuations) to EEG signals.
    Includes slow drifts and sudden jumps.
    
    Args:
        X: EEG data of shape (n_epochs, n_channels, n_times)
        fs: Sampling frequency
        amplitude: Noise amplitude relative to signal (0.0 to 1.0)
        drift_amplitude: Amplitude of slow drifts
    
    Returns:
        EEG data with electrode noise
    """
    n_epochs, n_channels, n_times = X.shape
    X_noisy = X.copy()
    
    np.random.seed(RANDOM_STATE)
    
    for epoch_idx in range(n_epochs):
        for ch_idx in range(n_channels):
            signal = X[epoch_idx, ch_idx]
            signal_power = np.std(signal)
            
            drift_freq = np.random.uniform(0.1, 0.5)
            t = np.arange(n_times) / fs
            slow_drift = drift_amplitude * signal_power * np.sin(2 * np.pi * drift_freq * t)
            
            n_jumps = np.random.randint(0, 3)
            jump_noise = np.zeros(n_times)
            for _ in range(n_jumps):
                jump_idx = np.random.randint(0, n_times)
                jump_size = np.random.choice([-1, 1]) * amplitude * signal_power * np.random.uniform(0.5, 1.5)
                jump_noise[jump_idx:] += jump_size
            
            X_noisy[epoch_idx, ch_idx] += slow_drift + jump_noise
    
    return X_noisy


def add_combined_noise(X, fs, noise_level='low'):
    """
    Add combined noise to EEG signals.
    
    Args:
        X: EEG data of shape (n_epochs, n_channels, n_times)
        fs: Sampling frequency
        noise_level: 'low', 'medium', or 'high'
    
    Returns:
        Noisy EEG data
    """
    noise_configs = {
        'low': {
            'powerline': 0.05,
            'emg': 0.05,
            'electrode': 0.03,
        },
        'medium': {
            'powerline': 0.15,
            'emg': 0.15,
            'electrode': 0.10,
        },
        'high': {
            'powerline': 0.30,
            'emg': 0.30,
            'electrode': 0.20,
        },
    }
    
    config = noise_configs[noise_level]
    
    X_noisy = X.copy()
    X_noisy = add_powerline_noise(X_noisy, fs, config['powerline'])
    X_noisy = add_emg_artifacts(X_noisy, fs, config['emg'])
    X_noisy = add_electrode_noise(X_noisy, fs, config['electrode'])
    
    return X_noisy


def run_noise_robustness_experiment(
    algorithms,
    dataset_name='sleep_edf',
    data_path=None,
    isruc_path=None,
    n_folds=5,
    output_dir=None,
    cv_mode='subject',
    n_repeats=1,
):
    """
    Run noise robustness experiment comparing algorithms under different noise levels.

    Design: models are trained ONCE per CV fold on CLEAN training data, then
    evaluated on the held-out test epochs corrupted at each noise level
    (powerline + EMG + electrode at controlled SNR). This measures inference-
    time robustness, matching the portable-device deployment scenario (model
    trained offline, deployed on noisy wearables).

    Addresses JBHI reviewer comment R3-6: multiple independent noise realizations
    (n_repeats) are used so that mean +/- SD / CI can be reported, and SNR is
    explicitly controlled in dB (correct power relationship: +5 dB => signal power
    ~3.16x noise power).

    Args:
        algorithms: List of algorithm names to test
        dataset_name: Dataset name
        data_path: Path to Sleep-EDF dataset
        isruc_path: Path to ISRUC-Sleep dataset (for combined)
        n_folds: Number of cross-validation folds
        output_dir: Output directory for results
        cv_mode: Cross-validation mode: 'subject' or 'stratified'
        n_repeats: Number of independent noise realizations per noise level

    Returns:
        results_df: DataFrame with results (mean/std across folds x repeats)
    """
    if output_dir is None:
        output_dir = RESULTS_PATH
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("Noise Robustness Experiment")
    print("=" * 80)
    print(f"  CV mode: {cv_mode}")
    print(f"  Noise repeats: {n_repeats}")
    
    print(f"\nLoading dataset: {dataset_name}")
    if dataset_name.lower() == 'combined':
        X, y, meta = data_loader.load_combined_datasets(
            sleep_edf_path=data_path,
            isruc_path=isruc_path,
            sleep_edf_subjects=['0~19'],
            select_n_channels=2,
        )
    else:
        X, y, meta = data_loader.load_sleep_dataset(dataset_name, data_path=data_path)
    
    n_samples, n_channels, n_times = X.shape
    n_classes = len(np.unique(y))
    
    print(f"  Samples: {n_samples}")
    print(f"  Channels: {n_channels}")
    print(f"  Time points: {n_times}")
    print(f"  Classes: {n_classes}")
    
    # Build CV splits
    if cv_mode == 'subject' and isinstance(meta, pd.DataFrame) and 'subject' in meta.columns:
        from sklearn.model_selection import GroupKFold
        groups = meta['subject'].astype(str).to_numpy()
        unique_subjects = np.unique(groups)
        n_splits = min(n_folds, len(unique_subjects))
        kf = GroupKFold(n_splits=n_splits)
        splits = list(kf.split(X, y, groups))
        print(f"  Using subject-wise CV with {len(unique_subjects)} subjects")
    else:
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
        splits = list(kf.split(X, y))
        print(f"  Using stratified CV")
    
    # Noise levels mapped to explicit SNR values (dB).
    # Positive SNR => signal STRONGER than noise (correct power relationship).
    noise_levels = ['clean', 'low', 'medium', 'high']
    snr_map = {'clean': None, 'low': 20.0, 'medium': 10.0, 'high': 5.0}

    # ---------------------------------------------------------------------------
    # Experiment design (Plan A): TRAIN ONCE ON CLEAN DATA, EVALUATE ON NOISY
    # TEST SETS.
    #
    # For each algorithm and each CV fold the model is trained exactly once on
    # the clean training split (matching deployment: the model is trained once
    # offline, then faces noisy recordings at inference time). Each noise level
    # is then applied ONLY to the held-out test epochs, with `n_repeats`
    # independent noise realizations (reviewer R3-6). This measures inference-
    # time robustness and reduces training cost by ~4x compared to retraining
    # per noise level.
    # ---------------------------------------------------------------------------

    # (algorithm, noise_level) -> list of per-(fold, repeat) metrics
    metrics_by_key = {}

    for algo_name in algorithms:
        print(f"\n{'=' * 60}")
        print(f"Algorithm: {algo_name} (train once per fold on CLEAN data)")
        print(f"{'=' * 60}")

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            print(f"\n  [Fold {fold_idx + 1}/{len(splits)}] "
                  f"train={len(train_idx)} test={len(test_idx)} ...", flush=True)

            model = get_algorithm(algo_name, n_channels, n_times, n_classes, fs=FS)

            try:
                import inspect
                fold_start = time.time()
                fit_signature = inspect.signature(model.fit)
                if 'epochs' in fit_signature.parameters:
                    model.fit(X_train, y_train, epochs=300)
                else:
                    model.fit(X_train, y_train)
                fit_elapsed = time.time() - fold_start
                print(f"    Training done in {fit_elapsed:.1f}s", flush=True)
            except Exception as e:
                print(f"    [!] Fold {fold_idx + 1} training failed for "
                      f"'{algo_name}': {e}")
                import traceback
                traceback.print_exc()
                continue

            for noise_level in noise_levels:
                # 'clean' is deterministic -> evaluate once (repeat 0 only).
                repeats_for_level = 1 if noise_level == 'clean' else n_repeats

                for repeat_idx in range(repeats_for_level):
                    if noise_level == 'clean':
                        X_test_eval = X_test
                    else:
                        X_test_eval = add_combined_noise_at_snr(
                            X_test, FS, snr_map[noise_level],
                            seed=RANDOM_STATE + repeat_idx,
                        )

                    try:
                        pred_start = time.time()
                        y_pred = model.predict(X_test_eval)
                        pred_elapsed = time.time() - pred_start

                        accuracy = accuracy_score(y_test, y_pred)
                        kappa = cohen_kappa_score(y_test, y_pred)
                        macro_f1 = f1_score(y_test, y_pred, average='macro',
                                            zero_division=0)

                        metrics_by_key.setdefault((algo_name, noise_level), []).append({
                            'accuracy': accuracy,
                            'kappa': kappa,
                            'macro_f1': macro_f1,
                        })

                        print(f"    {noise_level.upper():>6} "
                              f"(repeat {repeat_idx + 1}/{repeats_for_level}) "
                              f"predict {pred_elapsed:.1f}s: "
                              f"acc={accuracy:.4f} kappa={kappa:.4f} "
                              f"f1={macro_f1:.4f}", flush=True)
                    except Exception as e:
                        print(f"    [!] {noise_level} (repeat {repeat_idx + 1}) "
                              f"evaluation failed for '{algo_name}': {e}")
                        import traceback
                        traceback.print_exc()
                        continue

        # Free GPU/CPU memory between algorithms
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        import gc
        gc.collect()

    results = []
    for algo_name in algorithms:
        for noise_level in noise_levels:
            repeat_metrics = metrics_by_key.get((algo_name, noise_level), [])
            if not repeat_metrics:
                print(f"  [!] No results for {algo_name} @ {noise_level}")
                continue

            mean_acc = np.mean([r['accuracy'] for r in repeat_metrics])
            std_acc = np.std([r['accuracy'] for r in repeat_metrics])
            mean_kappa = np.mean([r['kappa'] for r in repeat_metrics])
            std_kappa = np.std([r['kappa'] for r in repeat_metrics])
            mean_f1 = np.mean([r['macro_f1'] for r in repeat_metrics])
            std_f1 = np.std([r['macro_f1'] for r in repeat_metrics])

            print(f"  {algo_name} @ {noise_level}: "
                  f"acc {mean_acc:.4f} ± {std_acc:.4f} (n={len(repeat_metrics)})")

            results.append({
                'noise_level': noise_level,
                'snr_db': snr_map[noise_level],
                'algorithm': algo_name,
                'accuracy_mean': mean_acc,
                'accuracy_std': std_acc,
                'kappa_mean': mean_kappa,
                'kappa_std': std_kappa,
                'macro_f1_mean': mean_f1,
                'macro_f1_std': std_f1,
                'n_observations': len(repeat_metrics),
                'n_repeats': 1 if noise_level == 'clean' else n_repeats,
            })

    results_df = pd.DataFrame(results)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f'noise_robustness_results_{timestamp}.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    plot_noise_robustness(results_df, output_dir, timestamp)
    
    return results_df


def plot_noise_robustness(results_df, output_dir, timestamp):
    """Generate publication-ready plots for noise robustness experiment."""
    
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.linewidth'] = 1.2
    
    noise_levels = ['clean', 'low', 'medium', 'high']
    noise_labels = ['Clean', 'Low', 'Medium', 'High']
    
    algorithms = results_df['algorithm'].unique()
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(algorithms)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
    
    for idx, (metric, title) in enumerate([
        ('accuracy_mean', 'Accuracy'),
        ('kappa_mean', 'Kappa'),
        ('macro_f1_mean', 'Macro-F1')
    ]):
        ax = axes[idx]
        
        for i, algo in enumerate(algorithms):
            algo_data = results_df[results_df['algorithm'] == algo]
            
            x_pos = [noise_levels.index(nl) for nl in algo_data['noise_level']]
            y_vals = [algo_data[algo_data['noise_level'] == nl][metric].values[0] 
                     if nl in algo_data['noise_level'].values else np.nan 
                     for nl in noise_levels]
            y_std = [algo_data[algo_data['noise_level'] == nl]['accuracy_std'].values[0] 
                    if nl in algo_data['noise_level'].values else 0 
                    for nl in noise_levels]
            
            # Highlight SCA-FBTS with bold line and vivid color
            if algo == 'SCA-FBTS':
                ax.errorbar(range(4), y_vals, yerr=y_std, 
                           label='SCA-FBTS (Ours)', color='#00008B', marker=markers[i % len(markers)],
                           linewidth=3, markersize=10, capsize=4, alpha=0.9)
            else:
                # Make other algorithms lighter and use dashed lines
                ax.errorbar(range(4), y_vals, yerr=y_std, 
                           label=algo, color=colors[i], marker=markers[i % len(markers)],
                           linewidth=1.5, markersize=6, capsize=3, alpha=0.7, linestyle='--')
        
        ax.set_xlabel('Noise Level', fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        ax.set_xticks(range(4))
        ax.set_xticklabels(noise_labels)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.3, 3.3)
    
    axes[0].legend(loc='lower left', fontsize=9, ncol=2)
    
    plt.tight_layout()
    
    fig_path = os.path.join(output_dir, f'noise_robustness_curve_{timestamp}.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to: {fig_path}")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    pivot_df = results_df.pivot(index='algorithm', columns='noise_level', values='accuracy_mean')
    pivot_df = pivot_df[['clean', 'low', 'medium', 'high']]
    
    degradation = (pivot_df['clean'] - pivot_df['high']) / pivot_df['clean'] * 100
    
    x = np.arange(len(pivot_df.index))
    width = 0.18
    
    for i, noise_level in enumerate(['clean', 'low', 'medium', 'high']):
        bars = ax.bar(x + i * width, pivot_df[noise_level], width, 
                     label=noise_level.capitalize(), alpha=0.8)
    
    ax.set_xlabel('Algorithm', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_xticks(x + 1.5 * width)
    # Replace algorithm names for better display
    xticklabels = []
    for label in pivot_df.index:
        if label == 'SCA-FBTS':
            xticklabels.append('SCA-FBTS (Ours)')
        elif label == 'FilterBankTangentSpace+SVM':
            xticklabels.append('FBTS-SVM')
        else:
            xticklabels.append(label)
    ax.set_xticklabels(xticklabels, rotation=45, ha='right')
    ax.legend(title='Noise Level', loc='lower left', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    bar_path = os.path.join(output_dir, f'noise_robustness_bars_{timestamp}.png')
    plt.savefig(bar_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Bar chart saved to: {bar_path}")


def generate_latex_table(results_df):
    """Generate LaTeX table for manuscript."""
    
    pivot_df = results_df.pivot(index='algorithm', columns='noise_level', 
                                values=['accuracy_mean', 'accuracy_std'])
    
    latex_str = """
\\begin{table}[H]
\\caption{Noise robustness results: Classification accuracy under different noise levels. Values show mean ± standard deviation across 5 folds.}\\label{tab:noise}
\\centering
\\footnotesize
\\begin{tabular}{lcccc}
\\toprule
Algorithm & Clean & Low Noise & Medium Noise & High Noise \\\\
\\midrule
"""
    
    for algo in pivot_df.index:
        row = f"{algo}"
        for noise in ['clean', 'low', 'medium', 'high']:
            mean = pivot_df.loc[algo, ('accuracy_mean', noise)]
            std = pivot_df.loc[algo, ('accuracy_std', noise)]
            row += f" & {mean:.3f} $\\pm$ {std:.3f}"
        row += " \\\\\n"
        latex_str += row
    
    latex_str += """\\bottomrule
\\end{tabular}
\\normalsize
\\end{table}
"""
    
    return latex_str


def main():
    parser = argparse.ArgumentParser(description='Noise Robustness Experiment')
    parser.add_argument('--algorithms', nargs='+', 
                       default=['CSP+LDA', 'Handcrafted+RF', 'MDM', 'RiemannTangentSpace',
                               'SCA-FBTS', 'DeepSleepNet', 'TinySleepNet', 'SSC-SleepNet'],
                       help='Algorithms to test')
    parser.add_argument('--dataset', type=str, default='sleep_edf',
                       help='Dataset name')
    parser.add_argument('--data-path', type=str,
                       default=("E:/datasets/Sleep/sleep-edf-database-expanded-1.0.0" if os.name == 'nt'
                                else "/mnt/data1/home/tanhuang/datasets/sleep-edf-database-expanded-1.0.0"),
                       help='Path to Sleep-EDF dataset')
    parser.add_argument('--isruc-path', type=str, default=None,
                       help='Path to ISRUC-Sleep dataset (required for combined)')
    parser.add_argument('--n-folds', type=int, default=5,
                       help='Number of cross-validation folds')
    parser.add_argument('--n-repeats', type=int, default=1,
                       help='Number of independent noise realizations per noise level (e.g. 3-5 for SD/CI)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory')
    parser.add_argument('--cv-mode', type=str, default='subject',
                       choices=['subject', 'stratified'],
                       help='Cross-validation mode: subject-wise or stratified. Default: subject (to ensure consistency with main evaluation)')
    
    args = parser.parse_args()
    
    results_df = run_noise_robustness_experiment(
        algorithms=args.algorithms,
        dataset_name=args.dataset,
        data_path=args.data_path,
        isruc_path=args.isruc_path,
        n_folds=args.n_folds,
        output_dir=args.output_dir,
        cv_mode=args.cv_mode,
        n_repeats=args.n_repeats,
    )
    
    print("\n" + "=" * 80)
    print("Summary Table:")
    print("=" * 80)
    print(results_df.pivot(index='algorithm', columns='noise_level', values='accuracy_mean').round(4))
    
    print("\n" + "=" * 80)
    print("LaTeX Table:")
    print("=" * 80)
    print(generate_latex_table(results_df))


if __name__ == "__main__":
    main()
