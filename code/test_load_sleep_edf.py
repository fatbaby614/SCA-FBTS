#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Smoke-test the rewritten load_sleep_edf (Expanded direct-scan) and legacy path.
Usage:
    python test_load_sleep_edf.py > test_load_sleep_edf_log.txt
"""
import sys
from pathlib import Path

import data_loader_sleep as dl

ROOT = Path(r"E:\datasets\Sleep\sleep-edf-database-expanded-1.0.0")
LEGACY = Path(r"E:\datasets\Sleep\physionet-sleep-data")


def check(name, X, y, meta, expect_subjects):
    print(f"\n=== {name} ===")
    print(f"X shape: {X.shape}")
    print(f"classes: {sorted(set(y.tolist()))}")
    print(f"class dist: {dict(zip(*map(list, __import__('numpy').unique(y, return_counts=True))))}")
    print(f"subjects: {sorted(meta['subject'].unique())}")
    print(f"n_subjects: {meta['subject'].nunique()} (expected {expect_subjects})")
    print(f"nights per subject:\n{meta.groupby('subject')['night'].unique()}")
    ok = meta['subject'].nunique() == expect_subjects
    print(f"PASS={ok}")
    return ok


def main():
    results = []

    # 1. Expanded, first 3 subjects (indices 0,1,2 -> SC-00, SC-01, SC-02)
    X, y, meta = dl.load_sleep_edf(
        data_path=str(ROOT), subjects=[0, 1, 2], select_n_channels=2)
    results.append(check("Expanded subjects=[0,1,2] (2ch)", X, y, meta, 3))

    # 2. Expanded, string subject codes
    X, y, meta = dl.load_sleep_edf(
        data_path=str(ROOT), subjects=["SC-00", "SC4001"], select_n_channels=2)
    results.append(check("Expanded subjects=['SC-00','SC4001'] (2ch)", X, y, meta, 1))

    # 3. Expanded, all cassette subjects would be 78; check subset=telemetry first 2
    X, y, meta = dl.load_sleep_edf(
        data_path=str(ROOT), subjects=[0, 1], subset="telemetry", select_n_channels=2)
    results.append(check("Telemetry subjects=[0,1] (2ch)", X, y, meta, 2))

    # 4. Expanded, first 20 subjects (SC-20 subset for comparison with old paper)
    X, y, meta = dl.load_sleep_edf(
        data_path=str(ROOT), subjects=list(range(20)), select_n_channels=2)
    results.append(check("Expanded SC-20 subset subjects=0..19 (2ch)", X, y, meta, 20))

    # 5. Legacy flat directory
    try:
        X, y, meta = dl.load_sleep_edf(
            data_path=str(LEGACY), subjects=[0, 1], select_n_channels=2)
        results.append(check("Legacy flat dir subjects=[0,1] (2ch)", X, y, meta, 2))
    except Exception as exc:
        print(f"\n[!] Legacy path failed: {exc}")
        results.append(False)

    print("\n" + "=" * 60)
    print(f"ALL PASS: {all(results)} ({sum(results)}/{len(results)})")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
