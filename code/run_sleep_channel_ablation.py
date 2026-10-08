#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Run channel-count ablation study for sleep staging and save summary curves."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate_sleep import evaluate_sleep_dataset, _parse_subjects_arg  # noqa: E402
from config.algorithms_config import RESULTS_PATH, get_timestamped_filename  # noqa: E402


SLEEP_EDF_CHANNELS = [
    "EEG Fpz-Cz",
    "EEG Pz-Oz",
]

ISRUC_CHANNELS = [
    "F3",
    "F4",
    "C3",
    "C4",
    "O1",
    "O2",
]

DEFAULT_CANDIDATE_CHANNELS = SLEEP_EDF_CHANNELS


def _plot_channel_ablation(summary_df: pd.DataFrame, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    for algo, group in summary_df.groupby("algorithm"):
        g = group.sort_values("channel_count")
        ax.plot(g["channel_count"], g["accuracy_mean"], marker="o", linewidth=2, label=f"{algo} (Acc)")

    ax.set_xlabel("Number of channels")
    ax.set_ylabel("Accuracy")
    ax.set_title("Channel Ablation: Accuracy vs Number of Channels")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    ax.set_ylim(0, 1)
    
    unique_channel_counts = sorted(summary_df["channel_count"].unique())
    ax.set_xticks(unique_channel_counts)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sleep channel-count ablation")
    parser.add_argument("--dataset", type=str, default="sleep_edf", choices=["sleep_edf", "isruc", "combined"])
    parser.add_argument("--data-path", type=str,
                        default=("E:/datasets/Sleep/sleep-edf-database-expanded-1.0.0" if os.name == "nt"
                                 else "/mnt/data1/home/tanhuang/datasets/sleep-edf-database-expanded-1.0.0"))
    parser.add_argument("--isruc-path", type=str, default=None,
                        help="Path to ISRUC-Sleep dataset (required for combined/isruc)")
    parser.add_argument("--subjects", type=str, nargs="+", default=["0~19"],
                        help="Subject IDs or ranges, e.g. --subjects 0~19")
    parser.add_argument("--recording", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--algorithms", type=str, nargs="+", default=["SCA-FBTS", "HandcraftedFeatures+RF"])
    parser.add_argument("--cv-mode", type=str, default="subject", choices=["subject", "stratified"])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--channel-counts", type=int, nargs="+", default=[2, 1],
                        help="Channel counts to evaluate. Sleep-EDF has 2 EEG channels, so use [2, 1]")
    parser.add_argument("--channel-mode", type=str, default="first-n", choices=["first-n", "by-name"],
                        help="first-n: take first N EEG channels; by-name: take first N from candidate list")
    parser.add_argument("--candidate-channels", type=str, nargs="+", default=DEFAULT_CANDIDATE_CHANNELS,
                        help="Ordered candidate channels for subsetting")
    parser.add_argument("--tsne", action="store_true", help="Enable t-SNE during ablation (off by default)")
    parser.add_argument("--tsne-use-pca", action="store_true")
    parser.add_argument("--tsne-max-samples", type=int, default=800)
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: RESULTS_PATH)")

    args = parser.parse_args()

    subjects = _parse_subjects_arg(args.subjects)

    if args.dataset == "sleep_edf":
        default_channels = SLEEP_EDF_CHANNELS
    elif args.dataset in ["isruc", "combined"]:
        default_channels = ISRUC_CHANNELS
    else:
        default_channels = SLEEP_EDF_CHANNELS

    if args.candidate_channels == DEFAULT_CANDIDATE_CHANNELS:
        args.candidate_channels = default_channels
        print(f"Auto-detected dataset '{args.dataset}', using {len(default_channels)} channels: {default_channels}")

    all_results = []
    all_summary = []

    counts = []
    for c in args.channel_counts:
        if c <= 0:
            continue
        if c > len(args.candidate_channels):
            print(f"Warning: skip channel_count={c}, only {len(args.candidate_channels)} candidate channels provided")
            continue
        counts.append(c)

    for count in counts:
        selected_channels = args.candidate_channels[:count] if args.channel_mode == "by-name" else None
        select_n_channels = count if args.channel_mode == "first-n" else None
        print("=" * 80)
        if args.channel_mode == "first-n":
            print(f"Running ablation with first {count} EEG channels")
        else:
            print(f"Running ablation with {count} named channels: {selected_channels}")
        print("=" * 80)

        results_df, summary_df = evaluate_sleep_dataset(
            algorithms=args.algorithms,
            dataset_name=args.dataset,
            n_epochs=args.epochs,
            data_path=args.data_path,
            isruc_path=args.isruc_path,
            subjects=subjects,
            recording=args.recording,
            cv_mode=args.cv_mode,
            select_channels=selected_channels,
            select_n_channels=select_n_channels,
            tsne=args.tsne,
            tsne_use_pca=args.tsne_use_pca,
            tsne_max_samples=args.tsne_max_samples,
        )

        results_df = results_df.copy()
        summary_df = summary_df.copy()
        results_df["channel_count"] = count
        summary_df["channel_count"] = count
        results_df["selected_channels"] = "|".join(selected_channels) if selected_channels else "AUTO:first-n"
        summary_df["selected_channels"] = "|".join(selected_channels) if selected_channels else "AUTO:first-n"

        all_results.append(results_df)
        all_summary.append(summary_df)

    if not all_results:
        raise RuntimeError("No ablation run executed. Check channel counts and candidate channels.")

    final_results = pd.concat(all_results, ignore_index=True)
    final_summary = pd.concat(all_summary, ignore_index=True)

    # Determine output directory
    output_dir = Path(args.output_dir) if args.output_dir else Path(RESULTS_PATH)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_results = output_dir / get_timestamped_filename("sleep_channel_ablation_results", "csv")
    out_summary = output_dir / get_timestamped_filename("sleep_channel_ablation_summary", "csv")
    out_plot = output_dir / get_timestamped_filename("sleep_channel_ablation_curve", "png")

    final_results.to_csv(out_results, index=False)
    final_summary.to_csv(out_summary, index=False)
    _plot_channel_ablation(final_summary, out_plot)

    print("Done.")
    print(f"Ablation detailed results: {out_results}")
    print(f"Ablation summary: {out_summary}")
    print(f"Ablation curve: {out_plot}")


if __name__ == "__main__":
    main()
