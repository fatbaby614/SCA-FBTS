#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Generate manuscript-ready tables from latest sleep evaluation CSV files."""

from __future__ import annotations

import ast
from pathlib import Path
import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
OUT_DIR = Path(__file__).resolve().parent / "results" / "generated_tables"


def _latest_file(pattern: str) -> Path:
    files = sorted(RESULTS_DIR.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No files matched: {pattern}")
    return files[-1]


def _macro_f1_from_confusion_matrix(cm: np.ndarray) -> float:
    cm = np.asarray(cm, dtype=float)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        return float("nan")

    f1_scores = []
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        denom = (2 * tp + fp + fn)
        f1 = 0.0 if denom == 0 else (2 * tp) / denom
        f1_scores.append(f1)

    return float(np.mean(f1_scores))


def build_dataset_summary_table() -> pd.DataFrame:
    """
    Build Table 1: Dataset Summary
    
    Provides metadata about Sleep-EDF dataset used for experiments
    including number of subjects, recordings, total epochs, channels, etc.
    """
    # Sleep-EDF dataset specifications
    dataset_info = {
        'Dataset': ['Sleep-EDF'],
        'Subjects': [20],
        'Recordings': [40],  # 2 per subject (night 1 + 2)
        'Total Epochs': [3000],  # Approximate based on standard Sleep-EDF
        'Epoch Length (s)': [30],
        'Sampling Rate (Hz)': [100],
        'Channels': [22],
        'Available Stages': ['W, N1, N2, N3, REM'],
        'Stage Labels': [5],
        'Study Type': ['Cross-subject, subject-wise CV'],
    }
    
    df = pd.DataFrame(dataset_info)
    return df


def _format_mean_std(mean_val: float, std_val: float, digits: int = 3) -> str:
    if pd.isna(mean_val):
        return "-"
    if pd.isna(std_val):
        return f"{mean_val:.{digits}f}"
    return f"{mean_val:.{digits}f} +- {std_val:.{digits}f}"


def build_summary_tables(summary_csv: Path, results_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_df = pd.read_csv(summary_csv)
    results_df = pd.read_csv(results_csv)

    results_df = results_df.copy()
    results_df["confusion_matrix"] = results_df["confusion_matrix"].apply(ast.literal_eval)
    results_df["macro_f1"] = results_df["confusion_matrix"].apply(
        lambda cm: _macro_f1_from_confusion_matrix(np.array(cm))
    )

    macro_f1_df = (
        results_df
        .groupby(["dataset", "cv_mode", "algorithm"], as_index=False)["macro_f1"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "macro_f1_mean", "std": "macro_f1_std"})
    )

    merged = summary_df.merge(
        macro_f1_df,
        on=["dataset", "cv_mode", "algorithm"],
        how="left",
    )

    table_main = merged[[
        "dataset",
        "cv_mode",
        "algorithm",
        "accuracy_mean",
        "accuracy_std",
        "kappa_mean",
        "kappa_std",
        "macro_f1_mean",
        "macro_f1_std",
        "train_time_mean",
        "train_time_std",
        "inference_time_ms_mean",
        "inference_time_ms_std",
        "model_size_mb_mean",
    ]].copy() if "inference_time_ms_mean" in merged.columns else merged[[
        "dataset",
        "cv_mode",
        "algorithm",
        "accuracy_mean",
        "accuracy_std",
        "kappa_mean",
        "kappa_std",
        "macro_f1_mean",
        "macro_f1_std",
        "train_time_mean",
        "train_time_std",
    ]].copy()

    table_main["Accuracy"] = table_main.apply(
        lambda r: _format_mean_std(r["accuracy_mean"], r["accuracy_std"]), axis=1
    )
    table_main["Kappa"] = table_main.apply(
        lambda r: _format_mean_std(r["kappa_mean"], r["kappa_std"]), axis=1
    )
    table_main["Macro-F1"] = table_main.apply(
        lambda r: _format_mean_std(r["macro_f1_mean"], r["macro_f1_std"]), axis=1
    )
    table_main["Train Time (s)"] = table_main.apply(
        lambda r: _format_mean_std(r["train_time_mean"], r["train_time_std"], digits=2), axis=1
    )
    
    if "inference_time_ms_mean" in table_main.columns:
        table_main["Inference (ms)"] = table_main.apply(
            lambda r: _format_mean_std(r["inference_time_ms_mean"], r["inference_time_ms_std"], digits=2), axis=1
        )
        table_main["Model Size (MB)"] = table_main["model_size_mb_mean"].apply(
            lambda x: f"{x:.2f}" if pd.notna(x) else "-"
        )
        table_main = table_main[["dataset", "cv_mode", "algorithm", "Accuracy", "Kappa", "Macro-F1", "Train Time (s)", "Inference (ms)", "Model Size (MB)"]]
    else:
        table_main = table_main[["dataset", "cv_mode", "algorithm", "Accuracy", "Kappa", "Macro-F1", "Train Time (s)"]]
    
    table_main = table_main.sort_values(["dataset", "cv_mode", "Accuracy"], ascending=[True, True, False])

    table_eff = merged[[
        "dataset",
        "cv_mode",
        "algorithm",
        "train_time_mean",
        "train_time_std",
        "accuracy_mean",
        "kappa_mean",
    ]].copy()

    table_eff["Train Time (s)"] = table_eff.apply(
        lambda r: _format_mean_std(r["train_time_mean"], r["train_time_std"], digits=2), axis=1
    )
    table_eff["Accuracy"] = table_eff["accuracy_mean"].map(lambda x: f"{x:.3f}")
    table_eff["Kappa"] = table_eff["kappa_mean"].map(lambda x: f"{x:.3f}")

    table_eff = table_eff[["dataset", "cv_mode", "algorithm", "Train Time (s)", "Accuracy", "Kappa"]]
    table_eff = table_eff.sort_values(["dataset", "cv_mode", "algorithm"])

    return table_main, table_eff


def write_outputs(table_main: pd.DataFrame, table_eff: pd.DataFrame, table_dataset: pd.DataFrame = None) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    main_csv = OUT_DIR / "table_main_performance.csv"
    eff_csv = OUT_DIR / "table_efficiency.csv"
    dataset_csv = OUT_DIR / "table_dataset_summary.csv" if table_dataset is not None else None
    
    main_md = OUT_DIR / "table_main_performance.md"
    eff_md = OUT_DIR / "table_efficiency.md"
    dataset_md = OUT_DIR / "table_dataset_summary.md" if table_dataset is not None else None
    
    main_tex = OUT_DIR / "table_main_performance.tex"
    eff_tex = OUT_DIR / "table_efficiency.tex"
    dataset_tex = OUT_DIR / "table_dataset_summary.tex" if table_dataset is not None else None

    table_main.to_csv(main_csv, index=False)
    table_eff.to_csv(eff_csv, index=False)
    if table_dataset is not None:
        table_dataset.to_csv(dataset_csv, index=False)

    table_main.to_markdown(main_md, index=False)
    table_eff.to_markdown(eff_md, index=False)
    if table_dataset is not None:
        table_dataset.to_markdown(dataset_md, index=False)

    main_tex.write_text(table_main.to_latex(index=False, escape=False), encoding="utf-8")
    eff_tex.write_text(table_eff.to_latex(index=False, escape=False), encoding="utf-8")
    if table_dataset is not None:
        dataset_tex.write_text(table_dataset.to_latex(index=False, escape=False), encoding="utf-8")

    report_txt = OUT_DIR / "README_generated_tables.txt"
    files_list = [
        main_csv.name,
        eff_csv.name,
        main_md.name,
        eff_md.name,
        main_tex.name,
        eff_tex.name,
    ]
    if table_dataset is not None:
        files_list.extend([dataset_csv.name, dataset_md.name, dataset_tex.name])
    
    report_txt.write_text(
        "Generated manuscript tables:\n"
        + "\n".join(f"- {f}" for f in files_list)
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    summary_csv = _latest_file("sleep_evaluation_summary_*.csv")
    results_csv = _latest_file("sleep_evaluation_results_*.csv")

    print(f"Using summary file: {summary_csv.name}")
    print(f"Using results file: {results_csv.name}")

    table_main, table_eff = build_summary_tables(summary_csv, results_csv)
    table_dataset = build_dataset_summary_table()
    
    write_outputs(table_main, table_eff, table_dataset)

    print("Done. Generated tables in:")
    print(OUT_DIR)


if __name__ == "__main__":
    main()
