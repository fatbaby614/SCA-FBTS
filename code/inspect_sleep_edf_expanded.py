#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Inspect the full Sleep-EDF Expanded dataset (v1.0.0).
Scans sleep-cassette and sleep-telemetry subsets, reads EDF headers and
hypnogram annotations, and writes a structured summary report.

Usage:
    python inspect_sleep_edf_expanded.py --data-path E:/datasets/Sleep/sleep-edf-database-expanded-1.0.0
                                         --output results/dataset_report.json
"""
import argparse
import json
import re
from pathlib import Path
from collections import Counter, OrderedDict

import mne
import numpy as np

# 30-s epoch as used in sleep staging
EPOCH_SECONDS = 30.0

STAGE_MAP = {
    "Sleep stage W": "W",
    "Sleep stage 1": "N1",
    "Sleep stage 2": "N2",
    "Sleep stage 3": "N3",
    "Sleep stage 4": "N3",  # R&K: stage 3+4 merged into N3
    "Sleep stage R": "REM",
}


def parse_record(psg_name):
    """Parse a PSG filename like SC4001E0-PSG.edf / ST7011J0-PSG.edf.

    Returns (subset, subject_code, subject_int, night, record_code) or None.
    """
    stem = psg_name.split("-PSG")[0]  # e.g. SC4001E0 / ST7011J0
    # Structure: SC|ST + fixed digit + 2-digit subject + 1-digit night +
    #            type letter + version digit  (e.g. SC4 00 1 E0)
    m = re.match(r"^(SC|ST)(\d)(\d{2})(\d)([A-Za-z])(\d)$", stem)
    if not m:
        return None
    subset, fixed, subj_digits, night, rec_letter, version = m.groups()
    return {
        "subset": "cassette" if subset == "SC" else "telemetry",
        "subject_code": f"{subset}-{subj_digits}",
        "subject_int": int(subj_digits),
        "night": int(night),
        "record_code": stem,
    }


def main():
    parser = argparse.ArgumentParser(description="Inspect Sleep-EDF Expanded dataset")
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to sleep-edf-database-expanded-1.0.0 directory")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON report path (default: printed to console)")
    args = parser.parse_args()

    root = Path(args.data_path)
    report = OrderedDict()
    report["data_root"] = str(root)
    report["subsets"] = OrderedDict()
    report["summary"] = {}

    all_stage_counts = Counter()
    all_record_info = []

    for subset_name, subset_dir in [("cassette", "sleep-cassette"),
                                    ("telemetry", "sleep-telemetry")]:
        sub_dir = root / subset_dir
        if not sub_dir.is_dir():
            print(f"[!] Subset directory not found: {sub_dir}")
            continue

        psg_files = sorted(sub_dir.glob("*-PSG.edf"))
        hyp_files = sorted(sub_dir.glob("*-Hypnogram.edf"))
        subset_info = OrderedDict()
        subset_info["n_psg"] = len(psg_files)
        subset_info["n_hypnograms"] = len(hyp_files)

        records = []
        subjects = OrderedDict()  # subject_code -> nights
        channel_counter = Counter()
        sfreq_counter = Counter()
        stage_counter = Counter()
        epoch_counts = []

        for psg_path in psg_files:
            parsed = parse_record(psg_path.name)
            if parsed is None:
                print(f"[!] Cannot parse filename: {psg_path.name}")
                continue

            # Find matching hypnogram: same prefix with last char dropped, e.g.
            # SC4261F0-PSG.edf -> SC4261FM-Hypnogram.edf
            prefix = parsed["record_code"]  # e.g. SC4261F0
            hyp_glob = f"{prefix[:-1]}*-Hypnogram.edf"
            hyp_matches = list(sub_dir.glob(hyp_glob))
            if len(hyp_matches) == 0:
                print(f"[!] No hypnogram for {psg_path.name} (glob {hyp_glob})")
                continue
            hyp_path = hyp_matches[0]
            if len(hyp_matches) > 1:
                print(f"[!] Multiple hypnograms for {psg_path.name}: {hyp_matches}")

            rec = dict(parsed)
            rec["psg"] = psg_path.name
            rec["hypnogram"] = hyp_path.name

            try:
                raw = mne.io.read_raw_edf(psg_path, preload=False, verbose="error")
                rec["sfreq"] = float(raw.info["sfreq"])
                rec["duration_h"] = round(raw.n_times / raw.info["sfreq"] / 3600.0, 3)
                rec["channels"] = list(raw.ch_names)
                rec["n_channels"] = len(raw.ch_names)
                channel_counter[tuple(raw.ch_names)] += 1
                sfreq_counter[float(raw.info["sfreq"])] += 1
            except Exception as exc:
                print(f"[!] EDF read error {psg_path.name}: {exc}")
                rec["error"] = str(exc)

            try:
                annot = mne.read_annotations(hyp_path)
                onset_first = annot.onset[0] if len(annot) else np.nan
                onset_last = annot.onset[-1] if len(annot) else np.nan
                rec["n_annotations"] = len(annot)
                rec["lights_off_h"] = round(onset_first / 3600.0, 3) if len(annot) else None
                rec["recorded_h"] = round((onset_last - onset_first) / 3600.0, 3) if len(annot) else None
                # Map hypnogram descriptions to AASM stages
                rec_stages = Counter()
                for desc in annot.description:
                    stage = STAGE_MAP.get(desc, "?")
                    if stage == "?":
                        print(f"[!] Unknown annotation '{desc}' in {hyp_path.name}")
                    rec_stages[stage] += 1
                rec["stages"] = dict(rec_stages)
                stage_counter.update(rec_stages)
                epoch_counts.append(sum(rec_stages.values()))
            except Exception as exc:
                print(f"[!] Hypnogram read error {hyp_path.name}: {exc}")
                rec["error"] = str(exc)

            # Aggregate per subject (both nights)
            if parsed["subject_code"] in subjects:
                subjects[parsed["subject_code"]].append(parsed["night"])
            else:
                subjects[parsed["subject_code"]] = [parsed["night"]]

            records.append(rec)
            all_record_info.append(rec)

        subset_info["n_subjects"] = len(subjects)
        subset_info["subjects"] = OrderedDict(sorted(subjects.items()))
        subset_info["channel_layouts"] = {str(k): v for k, v in channel_counter.items()}
        subset_info["sfreqs"] = dict(sfreq_counter)
        subset_info["stage_counts"] = dict(stage_counter)
        subset_info["total_epochs"] = int(sum(epoch_counts))
        subset_info["records"] = records
        report["subsets"][subset_name] = subset_info
        all_stage_counts.update(stage_counter)

    n_total_records = len(all_record_info)
    n_total_subjects = (len(report["subsets"].get("cassette", {}).get("subjects", {}))
                        + len(report["subsets"].get("telemetry", {}).get("subjects", {})))
    report["summary"] = {
        "n_recordings": n_total_records,
        "n_subjects": n_total_subjects,
        "total_epochs_30s": int(sum(all_stage_counts.values())),
        "total_stage_counts": dict(all_stage_counts),
        "total_hours": round(sum(
            (r.get("duration_h") or 0.0) for r in all_record_info), 2),
    }

    out_text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(out_text, encoding="utf-8")
        print(f"Report saved to {args.output}")

    # Console summary
    print("=" * 70)
    print("Sleep-EDF Expanded dataset summary")
    print("=" * 70)
    for subset_name in ["cassette", "telemetry"]:
        info = report["subsets"].get(subset_name)
        if not info:
            continue
        print(f"\n[{subset_name}] recordings={info['n_psg']} subjects={info['n_subjects']} "
              f"epochs={info['total_epochs']}")
        print(f"  channel layouts: {info['channel_layouts']}")
        print(f"  sfreqs: {info['sfreqs']}")
        print(f"  stage counts: {info['stage_counts']}")
        print(f"  subjects: {list(info['subjects'].keys())}")
    print("\n" + "=" * 70)
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
