#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Causal vs Non-causal Temporal Smoothing Ablation (JBHI reviewer R3-8)

The default SCA-FBTS uses a centered (zero-phase, non-causal) moving-average
window over class probabilities (w=3), i.e. it also uses future epochs. This is
appropriate for offline scoring but raises a question for real-time monitoring.

This script compares three variants on the same subject-wise CV folds:
  - SCA-FBTS (non-causal, w=3)   : current paper setting (uses future epochs)
  - SCA-FBTS-Causal (w=3)        : window only contains current + past epochs
  - SCA-FBTS-NoSmooth            : no temporal smoothing at all

The goal is to quantify the accuracy / per-stage (especially N1) cost of the
causal constraint, so the portability claim can be stated precisely.

Usage:
    python run_causal_smoothing_ablation.py --data-path <Sleep-EDF Expanded>
        [--subjects 0~9] [--n-folds 5] [--output-dir <dir>]
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.algorithms_config import RESULTS_PATH, RANDOM_STATE
from algorithms_collection import get_algorithm, FilterBankTangentSpace
import data_loader_sleep as data_loader
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import GroupKFold, StratifiedKFold


STAGE_NAMES = ["W", "N1", "N2", "N3", "REM"]


def create_variants(n_channels, n_times, n_classes, fs):
    """Return dict of {variant_name: factory} for the three smoothing settings."""
    def make_noncausal():
        return get_algorithm('SCA-FBTS', n_channels, n_times, n_classes, fs=fs)

    def make_causal():
        return FilterBankTangentSpace(
            classifier='svm', fs=fs,
            temporal_smoothing=True, smoothing_window=3, causal_smoothing=True)

    def make_nosmooth():
        return FilterBankTangentSpace(
            classifier='svm', fs=fs, temporal_smoothing=False)

    return {
        'SCA-FBTS (non-causal)': make_noncausal,
        'SCA-FBTS-Causal': make_causal,
        'SCA-FBTS-NoSmooth': make_nosmooth,
    }


def parse_subjects(spec):
    """Parse '0~9' / '0 1 2' / 'SC-00 SC-01' into a subjects list for the loader."""
    if spec is None or spec.strip() == '':
        return None
    if '~' in spec:
        parts = spec.split('~')
        if len(parts) == 2:
            return list(range(int(parts[0]), int(parts[1]) + 1))
        raise ValueError(f"Invalid subjects range spec: {spec}")
    return [s for s in spec.split() if s]


def main():
    parser = argparse.ArgumentParser(description='Causal vs non-causal temporal smoothing ablation')
    parser.add_argument('--dataset', type=str, default='sleep_edf')
    parser.add_argument('--data-path', type=str,
                        default=("E:/datasets/Sleep/sleep-edf-database-expanded-1.0.0" if os.name == 'nt'
                                 else "/mnt/data1/home/tanhuang/datasets/sleep-edf-database-expanded-1.0.0"))
    parser.add_argument('--subjects', type=str, default='0~9',
                        help="Subject spec, e.g. '0~9' or 'SC-00 SC-01'")
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--cv-mode', type=str, default='subject', choices=['subject', 'stratified'])
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else RESULTS_PATH
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Causal vs Non-causal Temporal Smoothing Ablation (R3-8)")
    print("=" * 80)

    # 1. Load data (Expanded direct scan)
    subjects_list = parse_subjects(args.subjects)
    X, y, meta = data_loader.load_sleep_dataset(
        args.dataset, data_path=args.data_path, subjects=subjects_list)
    n_samples, n_channels, n_times = X.shape
    n_classes = len(np.unique(y))
    print(f"\nData: {X.shape} | subjects: {meta['subject'].nunique()} | classes: {n_classes}")
    print(f"Class distribution: {np.bincount(y)}")

    # 2. CV splits (same folds for all variants)
    if args.cv_mode == 'subject' and 'subject' in meta.columns:
        groups = meta['subject'].astype(str).to_numpy()
        unique_subjects = np.unique(groups)
        n_splits = min(args.n_folds, len(unique_subjects))
        kf = GroupKFold(n_splits=n_splits)
        splits = list(kf.split(X, y, groups))
        print(f"Subject-wise CV, {len(unique_subjects)} subjects, {n_splits} folds")
    else:
        kf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=RANDOM_STATE)
        splits = list(kf.split(X, y))
        print(f"Stratified CV, {args.n_folds} folds")

    # 3. Evaluate variants
    variants = create_variants(n_channels, n_times, n_classes, fs=100)
    rows = []

    for variant_name, factory in variants.items():
        print(f"\n{'=' * 60}")
        print(f"Evaluating: {variant_name}")
        print(f"{'=' * 60}")

        fold_metrics = []
        stage_f1_folds = []

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = factory()
            t0 = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - t0

            t0 = time.time()
            y_pred = model.predict(X_test)
            infer_time = time.time() - t0

            acc = accuracy_score(y_test, y_pred)
            kappa = cohen_kappa_score(y_test, y_pred)
            mf1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
            stage_f1 = f1_score(y_test, y_pred, average=None, labels=sorted(np.unique(y)),
                                zero_division=0)

            fold_metrics.append({'fold': fold_idx + 1, 'accuracy': acc, 'kappa': kappa,
                                 'macro_f1': mf1, 'train_time': train_time,
                                 'inference_time': infer_time})
            stage_f1_folds.append(stage_f1)
            print(f"  Fold {fold_idx + 1}/{len(splits)} | Acc {acc:.4f} | Kappa {kappa:.4f} | "
                  f"Macro-F1 {mf1:.4f} | stage F1 {np.round(stage_f1, 3)}")

        # Aggregate across folds (mean +/- std)
        fm = pd.DataFrame(fold_metrics)
        stage_mean = np.mean(stage_f1_folds, axis=0)
        stage_std = np.std(stage_f1_folds, axis=0)
        stage_names = [STAGE_NAMES[c] if c < len(STAGE_NAMES) else str(c) for c in sorted(np.unique(y))]

        row = {
            'variant': variant_name,
            'n_folds': len(fold_metrics),
            'accuracy_mean': fm['accuracy'].mean(),
            'accuracy_std': fm['accuracy'].std(),
            'kappa_mean': fm['kappa'].mean(),
            'kappa_std': fm['kappa'].std(),
            'macro_f1_mean': fm['macro_f1'].mean(),
            'macro_f1_std': fm['macro_f1'].std(),
            'train_time_mean': fm['train_time'].mean(),
            'inference_time_mean': fm['inference_time'].mean(),
        }
        for name, m, s in zip(stage_names, stage_mean, stage_std):
            row[f'f1_{name}_mean'] = m
            row[f'f1_{name}_std'] = s
        rows.append(row)

        print(f"\n  SUMMARY {variant_name}: Acc {row['accuracy_mean']:.4f} +/- {row['accuracy_std']:.4f} | "
              f"Macro-F1 {row['macro_f1_mean']:.4f} | N1 F1 {row.get('f1_N1_mean', float('nan')):.4f}")

    # 4. Save results
    df = pd.DataFrame(rows)
    csv_path = output_dir / 'causal_smoothing_ablation_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # LaTeX table
    tex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Causal vs.\ non-causal temporal smoothing (Sleep-EDF Expanded, "
        r"subject-wise CV). Values are mean $\pm$ std over folds.}",
        r"\label{tab:causal_smoothing}",
        r"\small",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Variant & Acc & Kappa & Macro-F1 & N1-F1 & N2-F1 & REM-F1 \\",
        r"\midrule",
    ]
    for _, r in df.iterrows():
        def fmt(name):
            m = r.get(f'{name}_mean')
            s = r.get(f'{name}_std')
            return f"{m:.3f} $\\pm$ {s:.3f}" if m == m else "--"
        tex_lines.append(
            f"{r['variant']} & {fmt('accuracy')} & {fmt('kappa')} & {fmt('macro_f1')} "
            f"& {fmt('f1_N1')} & {fmt('f1_N2')} & {fmt('f1_REM')} \\\\")
    tex_lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    tex_path = output_dir / 'causal_smoothing_ablation_table.tex'
    tex_path.write_text("\n".join(tex_lines), encoding='utf-8')
    print(f"LaTeX table saved to {tex_path}")

    # 5. Bar chart
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
        variants_list = df['variant'].tolist()
        x = np.arange(len(variants_list))
        width = 0.55
        bars = ax.bar(x, df['accuracy_mean'], width, yerr=df['accuracy_std'],
                      capsize=4, color=['#FF5722', '#2196F3', '#9E9E9E'],
                      edgecolor='black', linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(variants_list, rotation=10, ha='right', fontsize=10)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Causal vs Non-causal Temporal Smoothing\n'
                     'Sleep-EDF Expanded (subject-wise CV)', fontsize=12)
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                    f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        plt.tight_layout()
        fig_path = output_dir / 'causal_smoothing_ablation.png'
        plt.savefig(fig_path, dpi=300, facecolor='white')
        print(f"Figure saved to {fig_path}")
    except Exception as exc:
        print(f"Warning: could not plot figure: {exc}")

    print("\nDone.")


if __name__ == '__main__':
    main()
