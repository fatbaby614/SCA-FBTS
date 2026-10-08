#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Generate statistical significance tests using SUBJECTS as independent units.

Addresses JBHI reviewer comment R3-3: fold-level statistics with n=5 cannot reach
p<0.05 (Wilcoxon) and Friedman statistic is bounded by 5*(k-1)=30. This script
instead uses per-subject metrics (one observation per subject per algorithm) as
the independent experimental units, which is statistically valid:
  - Friedman test: n_blocks = number of subjects
  - Wilcoxon signed-rank test: n_pairs = number of subjects

Input: sleep_evaluation_subjects_*.csv produced by evaluate_sleep.py
       (each row = one subject's metrics computed on its own held-out epochs).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from scipy.stats import friedmanchisquare, wilcoxon
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "scipy is required for significance tests. Install with: pip install scipy"
    ) from exc


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT.parent / "results" / "sleep_edf"
OUT_DIR = ROOT.parent / "results" / "common" / "generated_tables"


def _latest_file(directory: Path, pattern: str) -> Path:
    files = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No files matched: {pattern} in {directory}")
    return files[-1]


def _subject_matrix(subjects_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Pivot subject-level results into a subject x algorithm matrix."""
    pivot = subjects_df.pivot_table(index=["subject"], columns="algorithm", values=metric)
    pivot = pivot.dropna(axis=0, how="any")
    return pivot


def _friedman_test(pivot: pd.DataFrame) -> dict:
    n_algos = pivot.shape[1]
    n_blocks = pivot.shape[0]
    if n_algos < 3 or n_blocks < 2:
        return {
            "n_subjects": n_blocks,
            "n_algorithms": n_algos,
            "statistic": np.nan,
            "p_value": np.nan,
            "max_possible_statistic": n_blocks * (n_algos - 1),
            "note": "Need >=3 algorithms and >=2 subjects for Friedman test",
        }

    samples = [pivot[col].to_numpy() for col in pivot.columns]
    stat, p_value = friedmanchisquare(*samples)
    max_stat = n_blocks * (n_algos - 1)
    note = ""
    if stat > max_stat + 1e-9:
        note = "WARNING: statistic exceeds theoretical maximum - check input data!"
    return {
        "n_subjects": n_blocks,
        "n_algorithms": n_algos,
        "statistic": float(stat),
        "p_value": float(p_value),
        "max_possible_statistic": max_stat,
        "note": note,
    }


def _wilcoxon_vs_reference(pivot: pd.DataFrame, reference_algo: str):
    if reference_algo not in pivot.columns:
        return None

    ref = pivot[reference_algo].to_numpy()
    rows = []
    for algo in pivot.columns:
        if algo == reference_algo:
            continue
        vals = pivot[algo].to_numpy()
        # Wilcoxon signed-rank on n=20 subject pairs (valid: 20 > minimal n for p<0.05)
        stat, p_value = wilcoxon(ref, vals, zero_method="wilcox", correction=False)
        rows.append(
            {
                "reference": reference_algo,
                "algorithm": algo,
                "n_subjects": len(ref),
                "mean_reference": float(np.mean(ref)),
                "mean_algorithm": float(np.mean(vals)),
                "delta_mean": float(np.mean(ref - vals)),
                "wilcoxon_stat": float(stat),
                "p_value": float(p_value),
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        alpha_corr = 0.05 / len(out)
        out["significant_0.05_bonf"] = out["p_value"] < alpha_corr
        out = out.sort_values("p_value")
    return out


def _resolve_reference(pivot: pd.DataFrame, preferred: str) -> str | None:
    """Resolve the reference algorithm name, trying common aliases of the proposed method."""
    if preferred in pivot.columns:
        return preferred
    for alias in ["SCA-FBTS", "FilterBankTangentSpace+SVM"]:
        if alias in pivot.columns:
            return alias
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Subject-level statistical significance tests (Friedman + Wilcoxon)"
    )
    parser.add_argument('--results-dir', type=str, default=str(RESULTS_DIR),
                        help='Directory containing sleep_evaluation_subjects_*.csv')
    parser.add_argument('--output-dir', type=str, default=str(OUT_DIR),
                        help='Directory to save output tables')
    parser.add_argument('--reference', type=str, default="SCA-FBTS",
                        help='Reference algorithm for paired Wilcoxon tests '
                             '(aliases SCA-FBTS / FilterBankTangentSpace+SVM are auto-resolved)')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    subjects_csv = _latest_file(results_dir, "sleep_evaluation_subjects_*.csv")
    print(f"Using subject-level results file: {subjects_csv.name}")

    subjects_df = pd.read_csv(subjects_csv)
    n_subjects = subjects_df['subject'].nunique() if 'subject' in subjects_df.columns else 0
    print(f"Subjects found: {n_subjects}")

    metrics = [m for m in ["accuracy", "kappa", "macro_f1"] if m in subjects_df.columns]
    if not metrics:
        raise ValueError("No supported metrics found in subjects CSV")

    friedman_rows = []
    wilcoxon_tables = {}

    for metric in metrics:
        pivot = _subject_matrix(subjects_df, metric)
        fr = _friedman_test(pivot)
        fr["metric"] = metric
        friedman_rows.append(fr)
        print(f"\n[{metric}] Friedman: n_subjects={fr['n_subjects']}, "
              f"statistic={fr['statistic']:.3f}, p={fr['p_value']:.4f}")

        if pivot.shape[1] >= 2:
            ref = _resolve_reference(pivot, args.reference)
            if ref is None:
                print(f"[{metric}] WARNING: no reference algorithm (tried {args.reference}, SCA-FBTS) "
                      f"in columns {list(pivot.columns)}; skipping Wilcoxon")
            else:
                wdf = _wilcoxon_vs_reference(pivot, ref)
                if wdf is not None and not wdf.empty:
                    wilcoxon_tables[metric] = wdf
                    print(f"[{metric}] Wilcoxon vs {ref}: n_subjects={len(pivot)}")

    friedman_df = pd.DataFrame(friedman_rows)[
        ["metric", "n_subjects", "n_algorithms", "statistic",
         "p_value", "max_possible_statistic", "note"]
    ]
    friedman_csv = out_dir / "table_stat_friedman_subject_level.csv"
    friedman_md = out_dir / "table_stat_friedman_subject_level.md"
    friedman_df.to_csv(friedman_csv, index=False)
    friedman_df.to_markdown(friedman_md, index=False)

    for metric, wdf in wilcoxon_tables.items():
        csv_path = out_dir / f"table_stat_wilcoxon_subject_level_{metric}.csv"
        md_path = out_dir / f"table_stat_wilcoxon_subject_level_{metric}.md"
        wdf.to_csv(csv_path, index=False)
        wdf.to_markdown(md_path, index=False)

    print("\nDone. Generated subject-level significance tables in:")
    print(out_dir)


if __name__ == "__main__":
    main()
