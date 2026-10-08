"""
Sleep EEG Data Loader
Supports multiple sleep datasets including Sleep-EDF, ISRUC-Sleep, etc.
"""
import numpy as np
import pandas as pd
import mne
import re  # 新增导入
from pathlib import Path
try:
    import pyedflib  # 使用pyedflib读取.rec文件
    PYEDFLIB_AVAILABLE = True
except ImportError:
    PYEDFLIB_AVAILABLE = False
    print("Warning: pyedflib not available, .rec files will be skipped")
from mne.datasets import sleep_physionet
from config.algorithms_config import RANDOM_STATE, LOW_FREQ, HIGH_FREQ
import warnings
warnings.filterwarnings('ignore')


def _parse_subject_night(record_id):
    """Parse canonical subject and night code from Sleep-EDF record id."""
    match = re.match(r"^(.*?)(E\d)$", str(record_id))
    if match:
        return match.group(1), match.group(2)
    return str(record_id), "NA"


def _normalize_sleep_physionet_root(data_path):
    """Normalize fetch root to avoid duplicated physionet-sleep-data nesting."""
    if data_path is None:
        return None
    path = Path(data_path)
    if path.name.lower() == "physionet-sleep-data":
        return path.parent
    return path


# ===== Sleep-EDF Expanded (v1.0.0) direct-file loading =====
# Directory layout: <root>/sleep-cassette/*-PSG.edf, <root>/sleep-telemetry/*-PSG.edf
# Filename: SC|ST + fixed digit + 2-digit subject + 1-digit night + type letter + version,
# e.g. SC4001E0-PSG.edf (SC cassette, subject 00, night 1), ST7011J0-PSG.edf (ST telemetry).
SLEEP_EDF_SUBSETS = {
    "cassette": "sleep-cassette",
    "telemetry": "sleep-telemetry",
}
SLEEP_EDF_SUBSET_LETTER = {"cassette": "SC", "telemetry": "ST"}


def _parse_sleep_edf_record(psg_name):
    """Parse a Sleep-EDF PSG filename into structured record info.

    Returns dict(subset, subject, subject_int, night, record_code) or None.
    """
    stem = psg_name.split("-PSG")[0]  # e.g. SC4001E0 / ST7011J0
    m = re.match(r"^(SC|ST)(\d)(\d{2})(\d)([A-Za-z])(\d)$", stem)
    if not m:
        return None
    subset, _fixed, subj_digits, night, _rec_letter, _version = m.groups()
    return {
        "subset": "cassette" if subset == "SC" else "telemetry",
        "subject": f"{subset}-{subj_digits}",  # canonical subject id (both nights merged)
        "subject_int": int(subj_digits),
        "night": int(night),
        "record_code": stem,
    }


def _discover_sleep_edf_records(data_path, subsets=None):
    """Scan Expanded root or a flat directory for (psg_path, hyp_path, info) tuples."""
    data_path = Path(data_path)
    records = []

    # Expanded layout: <root>/sleep-cassette, <root>/sleep-telemetry
    expanded_dirs = {name: data_path / d for name, d in SLEEP_EDF_SUBSETS.items()
                     if (data_path / d).is_dir()}
    if expanded_dirs:
        for sub_name, sub_dir in expanded_dirs.items():
            if subsets is not None and sub_name not in subsets:
                continue
            for psg_path in sorted(sub_dir.glob("*-PSG.edf")):
                info = _parse_sleep_edf_record(psg_path.name)
                if info is None:
                    print(f"  Warning: cannot parse filename {psg_path.name}")
                    continue
                # Hypnogram shares the same prefix but the trailing char varies,
                # e.g. SC4001E0-PSG.edf <-> SC4001EC-Hypnogram.edf
                hyp_glob = f"{info['record_code'][:-1]}*-Hypnogram.edf"
                hyp_matches = list(sub_dir.glob(hyp_glob))
                if not hyp_matches:
                    print(f"  Warning: no hypnogram found for {psg_path.name}")
                    continue
                records.append((psg_path, hyp_matches[0], info))
        return records

    # Flat layout (e.g. a partial copy of the dataset in one directory)
    for psg_path in sorted(data_path.glob("*-PSG.edf")):
        info = _parse_sleep_edf_record(psg_path.name)
        if info is None:
            continue
        hyp_glob = f"{info['record_code'][:-1]}*-Hypnogram.edf"
        hyp_matches = list(data_path.glob(hyp_glob))
        if not hyp_matches:
            continue
        records.append((psg_path, hyp_matches[0], info))
    return records


def _subject_sort_key(subject_code):
    """Sort key: cassette before telemetry, then numeric subject id."""
    sub, num = subject_code.split("-")
    return (0 if sub == "SC" else 1, int(num))


def _resolve_sleep_edf_subjects(subjects, all_subject_codes):
    """Resolve the `subjects` argument into a set of canonical subject codes.

    Accepts:
      - None                    -> all subjects
      - list of ints            -> indices into the sorted subject-code list
      - list of str             -> 'SC-00' or 'SC4001'-style codes
    """
    ordered = sorted(all_subject_codes, key=_subject_sort_key)
    if subjects is None:
        return set(ordered)

    selected = set()
    for s in subjects:
        if isinstance(s, str):
            s_up = s.strip().upper()
            if s_up in all_subject_codes:
                selected.add(s_up)
                continue
            # 'SC4001'-style: SC|ST + fixed digit + 2-digit subject (+ night/type)
            m = re.match(r"^(SC|ST)\d(\d{2})", s_up)
            if m:
                selected.add(f"{m.group(1)}-{m.group(2)}")
                continue
            raise ValueError(f"Invalid Sleep-EDF subject code: {s}")
        elif isinstance(s, (int, np.integer)):
            if 0 <= int(s) < len(ordered):
                selected.add(ordered[int(s)])
            else:
                raise ValueError(
                    f"Sleep-EDF subject index {s} out of range [0, {len(ordered)}). "
                    f"Available subjects: {ordered}"
                )
        else:
            raise ValueError(f"Invalid Sleep-EDF subject spec: {s!r}")
    return selected


def load_sleep_edf(data_path=None, subjects=None, select_channels=None, select_n_channels=None,
                   recording=(1, 2), subset="cassette"):
    """
    Load Sleep-EDF data, preferring direct file scanning of the Sleep-EDF
    Expanded (v1.0.0) dataset directory.

    Args:
        data_path: Root of the Sleep-EDF Expanded dataset
            (e.g. .../sleep-edf-database-expanded-1.0.0) or a flat directory
            containing *-PSG.edf / *-Hypnogram.edf files. If the directory is
            empty/unavailable, falls back to mne.datasets.sleep_physionet
            (original 20-subject Sleep-EDF) for backward compatibility.
        subjects: None -> all subjects; list of ints -> indices into the sorted
            subject list (0 = SC-00, ...); list of str -> 'SC-00' or 'SC4001'.
        select_channels: List of channel names to use (default: auto EEG selection)
        select_n_channels: Keep first N EEG channels after channel selection
        recording: Nights to include, e.g. (1, 2) for both nights
        subset: Which Expanded subset to load: 'cassette' (default),
            'telemetry', or None/'both' for both.

    Returns:
        X: EEG data of shape (n_epochs, n_channels, n_times)
        y: Sleep stage labels (0=W, 1=N1, 2=N2, 3=N3, 4=REM)
        meta: Metadata dataframe with a canonical 'subject' column
    """
    all_X = []
    all_y = []
    all_meta = []

    annotation_desc_2_event_id = {
        "Sleep stage W": 1,
        "Sleep stage 1": 2,
        "Sleep stage 2": 3,
        "Sleep stage 3": 4,
        "Sleep stage 4": 4,
        "Sleep stage R": 5,
        "W": 1,
        "N1": 2,
        "N2": 3,
        "N3": 4,
        "N4": 4,
        "R": 5,
        "0": 1,
        "1": 2,
        "2": 3,
        "3": 4,
        "4": 4,
        "5": 5,
    }
    event_id = {
        "Sleep stage W": 1,
        "Sleep stage 1": 2,
        "Sleep stage 2": 3,
        "Sleep stage 3": 4,
        "Sleep stage 4": 4,
        "Sleep stage 3/4": 4,
        "Sleep stage R": 5,
        "W": 1,
        "N1": 2,
        "N2": 3,
        "N3": 4,
        "N4": 4,
        "R": 5,
        "0": 1,
        "1": 2,
        "2": 3,
        "3": 4,
        "4": 4,
        "5": 5,
    }
    # Note: 'Sleep stage ?' and 'Movement time' annotations are intentionally
    # absent from the mapping, so unscored/movement epochs are dropped.

    # --- Direct file scanning (Expanded or flat directory) ---
    scan_paths = []
    if data_path is not None:
        scan_paths = [data_path]
        # If data_path points at a legacy physionet-sleep-data root, also try the
        # nested sleep-edf-database-expanded layout under the same parent.
        parent = Path(data_path).parent
        if (parent / "sleep-edf-database-expanded-1.0.0").is_dir():
            scan_paths.append(str(parent / "sleep-edf-database-expanded-1.0.0"))

    records = []
    used_path = None
    for p in scan_paths:
        recs = _discover_sleep_edf_records(p, subsets={subset} if subset in SLEEP_EDF_SUBSETS else None)
        if recs:
            records = recs
            used_path = p
            break

    if records:
        print(f"\n[data_loader] Loading Sleep-EDF (Expanded direct scan) from: {used_path}")
        all_subject_codes = {info["subject"] for _, _, info in records}
        selected_subjects = _resolve_sleep_edf_subjects(subjects, all_subject_codes)
        nights = set(recording) if recording is not None else None

        for psg_path, hyp_path, info in records:
            if info["subject"] not in selected_subjects:
                continue
            if nights is not None and info["night"] not in nights:
                continue
            print(f"Processing {psg_path.name}...")

            try:
                # stim_channel differs across subsets ('Event marker' vs 'Marker');
                # infer_types auto-detects EEG/EOG/EMG and stimulus channels.
                try:
                    raw = mne.io.read_raw_edf(
                        psg_path, stim_channel="Event marker", infer_types=True,
                        preload=True, verbose="error")
                except Exception:
                    raw = mne.io.read_raw_edf(
                        psg_path, stim_channel="Marker", infer_types=True,
                        preload=True, verbose="error")

                annot = mne.read_annotations(hyp_path)

                if len(annot) >= 3:
                    # Trim long wake periods to reduce class imbalance.
                    annot.crop(annot[1]["onset"] - 30 * 60, annot[-2]["onset"] + 30 * 60)
                raw.set_annotations(annot, emit_warning=False)

                if select_channels is None:
                    preferred_channels = ["EEG Fpz-Cz", "EEG Pz-Oz", "EEG F3-M2", "EEG F4-M1",
                                          "EEG C3-M2", "EEG C4-M1", "EEG O1-M2", "EEG O2-M2"]
                    available_channels = raw.ch_names
                    channels = [ch for ch in preferred_channels if ch in available_channels]
                    if not channels:
                        eeg_picks = mne.pick_types(raw.info, eeg=True)
                        channels = [raw.ch_names[idx] for idx in eeg_picks[:min(8, len(eeg_picks))]]
                else:
                    channels = [ch for ch in select_channels if ch in raw.ch_names]
                    if not channels:
                        print(f"  Warning: requested channels not found in {psg_path.name}, fallback to first EEG channels")
                        eeg_picks = mne.pick_types(raw.info, eeg=True)
                        channels = [raw.ch_names[idx] for idx in eeg_picks]

                if select_n_channels is not None and select_n_channels > 0 and channels:
                    channels = channels[:min(select_n_channels, len(channels))]

                if channels:
                    raw.pick(channels)

                if raw.info['sfreq'] != 100:
                    raw.resample(100, verbose=False)

                raw.filter(LOW_FREQ, HIGH_FREQ, verbose=False)

                events, _ = mne.events_from_annotations(
                    raw, event_id=annotation_desc_2_event_id,
                    chunk_duration=30.0, verbose=False)

                if len(events) == 0:
                    print(f"  Warning: No valid sleep events for {info['subject']} ({psg_path.name})")
                    continue

                tmax = 30.0 - 1.0 / raw.info["sfreq"]
                epochs = mne.Epochs(
                    raw=raw, events=events, event_id=event_id,
                    tmin=0.0, tmax=tmax, baseline=None, preload=True, verbose=False)

                X_subj = epochs.get_data(copy=True)
                y_subj = epochs.events[:, 2] - 1

                all_X.append(X_subj)
                all_y.append(y_subj)
                all_meta.extend([
                    {
                        "subject": info["subject"],
                        "night": info["night"],
                        "record_id": info["record_code"],
                        "epoch": i,
                    }
                    for i in range(len(X_subj))
                ])

            except Exception as e:
                print(f"  Error processing {psg_path.name}: {e}")
    else:
        # --- Fallback: MNE sleep_physionet (original 20-subject Sleep-EDF) ---
        print("\n[data_loader] Direct scan found no Sleep-EDF files, falling back to "
              "mne.datasets.sleep_physionet (original Sleep-EDF).")
        fetch_root = _normalize_sleep_physionet_root(data_path)
        if subjects is None:
            subjects = list(range(20))
        subject_files = sleep_physionet.age.fetch_data(
            subjects=subjects, recording=recording, path=fetch_root, on_missing="warn")

        for psg_path, hyp_path in subject_files:
            record_id = Path(psg_path).stem.split("-")[0]
            subject_id, night_id = _parse_subject_night(record_id)
            print(f"Processing {Path(psg_path).name}...")

            try:
                raw = mne.io.read_raw_edf(
                    psg_path, stim_channel="Event marker", infer_types=True,
                    preload=True, verbose="error")
                annot = mne.read_annotations(hyp_path)

                if len(annot) >= 3:
                    annot.crop(annot[1]["onset"] - 30 * 60, annot[-2]["onset"] + 30 * 60)
                raw.set_annotations(annot, emit_warning=False)

                if select_channels is None:
                    preferred_channels = ["EEG Fpz-Cz", "EEG Pz-Oz", "EEG F3-M2", "EEG F4-M1",
                                          "EEG C3-M2", "EEG C4-M1", "EEG O1-M2", "EEG O2-M2"]
                    available_channels = raw.ch_names
                    channels = [ch for ch in preferred_channels if ch in available_channels]
                    if not channels:
                        eeg_picks = mne.pick_types(raw.info, eeg=True)
                        channels = [raw.ch_names[idx] for idx in eeg_picks[:min(8, len(eeg_picks))]]
                else:
                    channels = [ch for ch in select_channels if ch in raw.ch_names]
                    if not channels:
                        print(f"  Warning: requested channels not found in {Path(psg_path).name}, fallback to first EEG channels")
                        eeg_picks = mne.pick_types(raw.info, eeg=True)
                        channels = [raw.ch_names[idx] for idx in eeg_picks]

                if select_n_channels is not None and select_n_channels > 0 and channels:
                    channels = channels[:min(select_n_channels, len(channels))]

                if channels:
                    raw.pick(channels)

                if raw.info['sfreq'] != 100:
                    raw.resample(100, verbose=False)

                raw.filter(LOW_FREQ, HIGH_FREQ, verbose=False)

                events, _ = mne.events_from_annotations(
                    raw, event_id=annotation_desc_2_event_id,
                    chunk_duration=30.0, verbose=False)

                if len(events) == 0:
                    print(f"  Warning: No valid sleep events for {subject_id}")
                    continue

                tmax = 30.0 - 1.0 / raw.info["sfreq"]
                epochs = mne.Epochs(
                    raw=raw, events=events, event_id=event_id,
                    tmin=0.0, tmax=tmax, baseline=None, preload=True, verbose=False)

                X_subj = epochs.get_data(copy=True)
                y_subj = epochs.events[:, 2] - 1

                all_X.append(X_subj)
                all_y.append(y_subj)
                all_meta.extend([
                    {
                        "subject": subject_id,
                        "night": night_id,
                        "record_id": record_id,
                        "epoch": i,
                    }
                    for i in range(len(X_subj))
                ])

            except Exception as e:
                print(f"  Error processing {Path(psg_path).name}: {e}")

    if len(all_X) == 0:
        raise ValueError("No valid data loaded")

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    meta = pd.DataFrame(all_meta)

    print(f"\nSleep-EDF data loaded:")
    print(f"  Total epochs: {len(X)}")
    print(f"  Channels: {X.shape[1]}")
    print(f"  Time points: {X.shape[2]}")
    print(f"  Subjects: {meta['subject'].nunique()}")
    print(f"  Classes: {len(np.unique(y))}")
    print(f"  Class distribution: {np.bincount(y)}")

    return X, y, meta


def load_isruc_sleep(data_path, subjects=None, select_channels=None, select_n_channels=None, subgroup=1):
    """
    Load ISRUC-Sleep dataset
    
    Args:
        data_path: Path to ISRUC-Sleep dataset directory (containing subgroup1/2/3 folders)
        subjects: List of subject IDs to load (1-indexed)
        select_channels: List of channel names to use (default: F3, F4, C3, C4, O1, O2)
        select_n_channels: Keep first N EEG channels after channel selection
        subgroup: ISRUC subgroup (1, 2, or 3)
    
    Returns:
        X: EEG data of shape (n_epochs, n_channels, n_times)
        y: Sleep stage labels (0=W, 1=N1, 2=N2, 3=N3, 4=REM)
        meta: Metadata dataframe
    """
    all_X = []
    all_y = []
    all_meta = []
    
    data_path = Path(data_path)
    subgroup_dir = data_path / f"subgroup{subgroup}"
    
    if not subgroup_dir.exists():
        raise FileNotFoundError(f"ISRUC-Sleep subgroup directory not found: {subgroup_dir}")
    
    if subjects is None:
        if subgroup == 1:
            subjects = list(range(1, 11))
        elif subgroup == 2:
            subjects = list(range(1, 9))
        elif subgroup == 3:
            subjects = list(range(1, 11))
        else:
            subjects = list(range(1, 11))
    
    preferred_eeg_channels = ["F3", "F4", "C3", "C4", "O1", "O2"]
    
    stage_map = {
        "W": 0,
        "WAKE": 0,
        "S1": 1,
        "N1": 1,
        "S2": 2,
        "N2": 2,
        "S3": 3,
        "N3": 3,
        "S4": 3,
        "N4": 3,
        "R": 4,
        "REM": 4,
    }
    
    for subject_id in subjects:
        # Try both formats: 02d and plain
        subject_strs = [f"{subject_id:02d}", str(subject_id)]
        subject_dir = None
        
        for s_str in subject_strs:
            s_dir = subgroup_dir / s_str
            if s_dir.exists():
                subject_dir = s_dir
                break
        
        if not subject_dir:
            print(f"Warning: Subject {subject_id} directory not found: {subgroup_dir}/{subject_strs[0]} or {subject_strs[1]}")
            continue
        
        rec_files = list(subject_dir.glob("*.Rec")) + list(subject_dir.glob("*.rec"))
        edf_files = list(subject_dir.glob("*.edf")) + list(subject_dir.glob("*.EDF"))
        hypnogram_files = list(subject_dir.glob("*hypno*")) + list(subject_dir.glob("*Hypno*"))
        hypnogram_files += list(subject_dir.glob("*.txt"))
        
        if not rec_files and not edf_files:
            print(f"Warning: No .Rec or .edf file found for subject {subject_id}")
            continue
        
        # Try EDF files first, then Rec files
        if edf_files:
            psg_file = edf_files[0]
            try:
                raw = mne.io.read_raw_edf(str(psg_file), preload=True, verbose=False)
                sfreq = raw.info["sfreq"]
                ch_names = raw.ch_names
                
                if select_channels:
                    channels = [ch for ch in select_channels if ch in ch_names]
                else:
                    channels = [ch for ch in preferred_eeg_channels if ch in ch_names]
                    if not channels:
                        eeg_picks = mne.pick_types(raw.info, eeg=True)
                        channels = [ch_names[idx] for idx in eeg_picks[:min(6, len(eeg_picks))]]
                
                if select_n_channels is not None and select_n_channels > 0:
                    channels = channels[:min(select_n_channels, len(channels))]
                
                if not channels:
                    print(f"Warning: No EEG channels found for subject {subject_id}")
                    continue
                
                raw.pick_channels(channels)
                raw.filter(LOW_FREQ, HIGH_FREQ, verbose=False)
                
                # Get data
                data = raw.get_data()
            except Exception as e:
                print(f"Warning: Failed to load {psg_file}: {e}")
                continue
        elif rec_files:
            if not PYEDFLIB_AVAILABLE:
                print(f"Warning: pyedflib not available, skipping .rec file: {rec_files[0]}")
                continue
            
            psg_file = rec_files[0]
            try:
                # Use pyedflib to load .rec file
                print(f"Loading .rec file with pyedflib: {psg_file}")
                f = pyedflib.EdfReader(str(psg_file))
                
                # Get all channel names
                ch_names = f.getSignalLabels()
                sfreq = f.getSampleFrequency(0)
                
                # Select channels
                if select_channels:
                    channels = [ch for ch in select_channels if ch in ch_names]
                else:
                    channels = [ch for ch in preferred_eeg_channels if ch in ch_names]
                    if not channels:
                        # Use first available channels
                        channels = ch_names[:min(6, len(ch_names))]
                
                if select_n_channels is not None and select_n_channels > 0:
                    channels = channels[:min(select_n_channels, len(channels))]
                
                if not channels:
                    print(f"Warning: No EEG channels found for subject {subject_id}")
                    f.close()
                    continue
                
                # Read signals
                signals = []
                for ch in channels:
                    idx = ch_names.index(ch)
                    sig = f.readSignal(idx)
                    signals.append(sig)
                
                data = np.array(signals)
                f.close()
                
                # Apply filter
                from scipy.signal import butter, filtfilt
                def butter_bandpass(lowcut, highcut, fs, order=4):
                    nyq = 0.5 * fs
                    low = lowcut / nyq
                    high = highcut / nyq
                    b, a = butter(order, [low, high], btype='band')
                    return b, a
                
                def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
                    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
                    y = filtfilt(b, a, data, axis=-1)
                    return y
                
                data = butter_bandpass_filter(data, LOW_FREQ, HIGH_FREQ, sfreq)
            except Exception as e:
                print(f"Warning: Failed to load {psg_file}: {e}")
                continue
        else:
            continue
        
        if not hypnogram_files:
            print(f"Warning: No hypnogram file found for subject {subject_id}")
            continue
        
        hypnogram_file = hypnogram_files[0]
        
        try:
            hypnogram = _load_isruc_hypnogram(hypnogram_file)
        except Exception as e:
            print(f"Warning: Failed to load hypnogram {hypnogram_file}: {e}")
            continue
        
        epoch_duration = 30.0
        n_samples_per_epoch = int(epoch_duration * sfreq)
        
        n_epochs = min(len(hypnogram), int(data.shape[1] / n_samples_per_epoch))
        
        for epoch_idx in range(n_epochs):
            start_sample = epoch_idx * n_samples_per_epoch
            end_sample = start_sample + n_samples_per_epoch
            
            if end_sample > data.shape[1]:
                break
            
            epoch_data = data[:, start_sample:end_sample]
            stage_label = hypnogram[epoch_idx]
            
            if stage_label < 0:
                continue
            
            all_X.append(epoch_data)
            all_y.append(stage_label)
            all_meta.append({
                "subject": subject_id,
                "epoch": epoch_idx,
                "stage": ["W", "N1", "N2", "N3", "REM"][stage_label],
                "recording": 1,
            })
    
    if not all_X:
        raise RuntimeError("No valid epochs loaded from ISRUC-Sleep dataset")
    
    X = np.array(all_X)
    y = np.array(all_y)
    meta = pd.DataFrame(all_meta)
    
    print(f"\nLoaded ISRUC-Sleep (subgroup {subgroup}):")
    print(f"  Subjects: {sorted(meta['subject'].unique())}")
    print(f"  Total epochs: {len(X)}")
    print(f"  Channels: {X.shape[1]}")
    print(f"  Samples per epoch: {X.shape[2]}")
    print(f"  Classes: {len(np.unique(y))}")
    print(f"  Class distribution: {np.bincount(y)}")
    
    return X, y, meta


def _normalize_sleep_stage_desc(desc):
    """Normalize annotation text to a compact token for stage mapping."""
    s = str(desc).strip().lower()
    s = s.replace("sleep stage", "").replace("stage", "")
    s = s.replace("-", " ").replace("_", " ")
    s = " ".join(s.split())
    return s


def _build_stage_mapping_from_annotations(annotations):
    """Build robust mapping from annotation descriptions to 5-class sleep staging IDs.

    Returns event_id mapping compatible with MNE events_from_annotations:
    1=W, 2=N1, 3=N2, 4=N3, 5=REM
    """
    stage_aliases = {
        1: {"w", "wake", "sleep stage w", "0"},
        2: {"1", "n1", "s1", "sleep stage 1", "stage 1"},
        3: {"2", "n2", "s2", "sleep stage 2", "stage 2"},
        4: {"3", "4", "n3", "n4", "s3", "s4", "sleep stage 3", "sleep stage 4", "stage 3", "stage 4"},
        5: {"r", "rem", "sleep stage r", "stage r", "5"},
    }

    event_id = {}
    for d in sorted(set(annotations.description)):
        norm = _normalize_sleep_stage_desc(d)
        target = None
        for stage_id, aliases in stage_aliases.items():
            if norm in aliases:
                target = stage_id
                break
        if target is None:
            if norm.startswith("n1"):
                target = 2
            elif norm.startswith("n2"):
                target = 3
            elif norm.startswith("n3") or norm.startswith("n4"):
                target = 4
            elif norm.startswith("rem") or norm == "r":
                target = 5
            elif norm.startswith("w") or norm.startswith("wake"):
                target = 1

        if target is not None:
            event_id[str(d)] = target

    return event_id


def _find_dreams_annotation_file(psg_path):
    """Best-effort search for an annotation file corresponding to a DREAMS PSG file."""
    p = Path(psg_path)
    stem = p.stem.lower()
    parent = p.parent

    exact_num_match = re.search(r"(\d+)$", stem)
    patient_id = exact_num_match.group(1) if exact_num_match else None

    if patient_id is not None:
        for candidate_name in (
            f"HypnogramAASM_patient{patient_id}.txt",
            f"HypnogramR&K_patient{patient_id}.txt",
            f"patient{patient_id}.txt",
        ):
            candidate = parent / candidate_name
            if candidate.exists():
                return candidate

    candidates = []
    for ext in ("*.xml", "*.txt", "*.csv"):
        candidates.extend(parent.glob(ext))

    ann_keywords = ("hyp", "hypnogram", "stage", "scor", "annot")

    ranked = []
    for c in candidates:
        name = c.name.lower()
        if c.resolve() == p.resolve():
            continue
        if not any(k in name for k in ann_keywords):
            continue

        score = 0
        if patient_id is not None and f"patient{patient_id}" in name:
            score += 2
        if "hypnogram" in name:
            score += 2
        if "annot" in name or "scor" in name:
            score += 1
        ranked.append((score, c))

    if not ranked:
        return None
    ranked.sort(key=lambda x: x[0], reverse=True)
    return ranked[0][1]


def _parse_dreams_hypnogram_file(hypnogram_file):
    """Parse DREAMS hypnogram text file into numeric stage labels.

    The DREAMS Patients database typically uses integer stage codes 1-5:
    1=Wake, 2=N1, 3=N2, 4=N3, 5=REM.
    We convert them into generic 0-4 labels used in this codebase:
    1->0(W), 2->1(N1), 3->2(N2), 4->3(N3), 5->4(REM).
    """
    labels = []

    with open(hypnogram_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("["):
                continue
            
            token = line.split()[0].upper()
            
            # Text fallbacks
            if token in ["W", "WAKE", "AWAKE", "A"]:
                labels.append(0)
                continue
            elif token in ["REM", "R"]:
                labels.append(4)
                continue
            elif token in ["N1", "S1"]:
                labels.append(1)
                continue
            elif token in ["N2", "S2"]:
                labels.append(2)
                continue
            elif token in ["N3", "S3", "N4", "S4"]:
                labels.append(3)
                continue
            
            # Numeric stage codes (1-5 format)
            try:
                stage = int(token)
            except ValueError:
                continue

            if stage == 1:     # Wake
                labels.append(0)
            elif stage == 2:   # N1
                labels.append(1)
            elif stage == 3:   # N2
                labels.append(2)
            elif stage == 4:   # N3
                labels.append(3)
            elif stage == 5:   # REM
                labels.append(4)
            # stage 0 is ignored if the files genuinely use 1-5 format

    return np.asarray(labels, dtype=int)


def _extract_dreams_patient_sort_key(path):
    """Extract numeric patient id for stable DREAMS file ordering."""
    match = re.search(r"patient(\d+)", Path(path).stem.lower())
    if match:
        return int(match.group(1))
    return Path(path).stem.lower()


def load_dreams_sleep(data_path, subjects=None, select_channels=None, select_n_channels=None, database="patients"):
    """Load DREAMS dataset with best-effort EDF + annotation pairing.

    Args:
        data_path: Path to DREAMS root or directly to DatabasePatients folder
        subjects: Optional 0-based subject indices/ranges
        select_channels: Optional explicit channel names
        select_n_channels: Optional number of EEG channels to keep
        database: DREAMS subset, currently supports "patients"

    Returns:
        X, y, meta
    """
    if data_path is None:
        raise ValueError("DREAMS data_path is required")

    root = Path(data_path)
    if database.lower() == "patients" and root.name.lower() != "databasepatients":
        root = root / "DatabasePatients"

    if not root.exists():
        raise FileNotFoundError(f"DREAMS directory not found: {root}")

    psg_files = []
    for ext in ("*.edf", "*.rec"):
        for f in sorted(root.rglob(ext)):
            lname = f.name.lower()
            if any(k in lname for k in ("hyp", "hypnogram", "annot", "scor", "stage")):
                continue
            psg_files.append(f)

    psg_files = sorted(psg_files, key=_extract_dreams_patient_sort_key)

    if not psg_files:
        raise ValueError(f"No DREAMS PSG files found under: {root}")

    if subjects is not None:
        valid_idx = [i for i in subjects if isinstance(i, int) and i >= 0 and i < len(psg_files)]
        if not valid_idx:
            raise ValueError(
                f"No valid DREAMS subjects in requested list. Requested={subjects}, available=[0..{len(psg_files)-1}]"
            )
        psg_files = [psg_files[i] for i in valid_idx]

    all_X = []
    all_y = []
    all_meta = []

    for idx, psg_path in enumerate(psg_files):
        print(f"Processing DREAMS PSG: {psg_path.name}")
        try:
            raw = mne.io.read_raw_edf(str(psg_path), preload=True, verbose="error")

            ann = None
            ann_file = _find_dreams_annotation_file(psg_path)
            if ann_file is None:
                print(f"  Warning: no hypnogram file found for {psg_path.name}, skipped")
                continue

            y_subj = _parse_dreams_hypnogram_file(ann_file)
            if len(y_subj) == 0:
                print(f"  Warning: empty hypnogram for {psg_path.name}, skipped")
                continue

            if select_channels is None:
                eeg_picks = mne.pick_types(raw.info, eeg=True)
                channels = [raw.ch_names[i] for i in eeg_picks]
            else:
                channels = [ch for ch in select_channels if ch in raw.ch_names]

            if select_n_channels is not None and select_n_channels > 0 and channels:
                channels = channels[:min(select_n_channels, len(channels))]
            else:
                channels = channels[:2]

            if channels:
                raw.pick(channels)

            if raw.info["sfreq"] != 100:
                raw.resample(100, verbose=False)

            raw.filter(LOW_FREQ, HIGH_FREQ, verbose=False)

            data = raw.get_data()
            sfreq = raw.info["sfreq"]
            epoch_samples = int(30.0 * sfreq)
            # DREAMS AASM hypnograms are often 1 label per 5 seconds (6 labels per 30s)
            # Check if y_subj length is roughly 6x the EEG 30s epochs
            eeg_epochs = data.shape[1] // epoch_samples
            if len(y_subj) > eeg_epochs * 4: # heuristics: probably 6x
                factor = len(y_subj) // eeg_epochs
                if factor not in [5, 6] and abs(len(y_subj) - eeg_epochs) > 5:
                    print(f"\n[!] CRITICAL WARNING: DREAMS hypnogram length ({len(y_subj)}) does not tightly align with EEG epochs ({eeg_epochs}). Factor is {factor}. Data for {psg_path.name} may be misaligned or invalid.")
                y_subj_new = []
                for i in range(eeg_epochs):
                    chunk = y_subj[i*factor : (i+1)*factor]
                    if len(chunk) > 0:
                        vals, counts = np.unique(chunk, return_counts=True)
                        y_subj_new.append(vals[np.argmax(counts)])
                y_subj = np.array(y_subj_new)
            
            n_epochs = min(len(y_subj), eeg_epochs)

            if n_epochs == 0:
                print(f"  Warning: no valid 30s epochs for {psg_path.name}, skipped")
                continue

            data = data[:, : n_epochs * epoch_samples]
            X_subj = data.reshape(data.shape[0], n_epochs, epoch_samples).transpose(1, 0, 2)
            y_subj = y_subj[:n_epochs]

            if len(X_subj) == 0:
                print(f"  Warning: no valid W/N1/N2/N3/REM epochs for {psg_path.name}, skipped")
                continue

            subj_id = psg_path.stem
            all_X.append(X_subj)
            all_y.append(y_subj)
            all_meta.extend(
                {
                    "subject": subj_id,
                    "night": "NA",
                    "record_id": subj_id,
                    "epoch": i,
                }
                for i in range(len(X_subj))
            )

        except Exception as e:
            print(f"  Error processing {psg_path.name}: {e}")

    if len(all_X) == 0:
        raise ValueError(f"No valid DREAMS data loaded from: {root}")

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    meta = pd.DataFrame(all_meta)

    print("\nDREAMS data loaded:")
    print(f"  Total epochs: {len(X)}")
    print(f"  Channels: {X.shape[1]}")
    print(f"  Time points: {X.shape[2]}")
    print(f"  Classes: {len(np.unique(y))}")
    print(f"  Class distribution: {np.bincount(y)}")

    return X, y, meta


def _load_isruc_hypnogram(hypnogram_file):
    """Load ISRUC-Sleep hypnogram file and convert to numeric labels."""
    stage_map = {
        "W": 0, "WAKE": 0, "0": 0,
        "S1": 1, "N1": 1, "1": 1,
        "S2": 2, "N2": 2, "2": 2,
        "S3": 3, "N3": 3, "3": 3,
        "S4": 3, "N4": 3, "4": 3,
        "R": 4, "REM": 4, "5": 4,
    }
    
    hypnogram = []
    
    with open(hypnogram_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            stage_str = line.upper().split()[0] if " " in line else line.upper()
            
            if stage_str in stage_map:
                hypnogram.append(stage_map[stage_str])
            else:
                try:
                    stage_num = int(stage_str)
                    if stage_num in [0, 1, 2, 3, 4, 5]:
                        hypnogram.append(stage_num if stage_num <= 4 else 4)
                    else:
                        hypnogram.append(-1)
                except ValueError:
                    hypnogram.append(-1)
    
    return np.array(hypnogram)


def load_dummy_sleep_data(n_epochs=100, n_channels=8, n_times=3000):
    """
    Generate dummy sleep data for testing
    
    Args:
        n_epochs: Number of epochs
        n_channels: Number of channels
        n_times: Number of time points per epoch (3000 = 30s @ 100Hz)
    
    Returns:
        X: Dummy EEG data
        y: Dummy sleep stage labels
        meta: Dummy metadata
    """
    np.random.seed(RANDOM_STATE)
    
    # Generate random EEG-like data
    X = np.random.randn(n_epochs, n_channels, n_times) * 10
    
    # Add some oscillatory activity in different frequency bands
    t = np.linspace(0, 30, n_times)
    for i in range(n_epochs):
        for j in range(n_channels):
            # Add delta (0.5-4 Hz) for deep sleep
            X[i, j] += 5 * np.sin(2 * np.pi * 2 * t)
            # Add alpha (8-12 Hz) for wake
            X[i, j] += 3 * np.sin(2 * np.pi * 10 * t)
    
    # Generate sleep stage labels (imbalanced distribution)
    y = np.random.choice([0, 1, 2, 3, 4], size=n_epochs, 
                          p=[0.15, 0.10, 0.45, 0.15, 0.15])
    
    meta = pd.DataFrame({
        'subject': [f'S{i%10+1:02d}' for i in range(n_epochs)],
        'night': ['N0' for _ in range(n_epochs)],
        'record_id': [f'S{i%10+1:02d}_N0' for i in range(n_epochs)],
        'epoch': list(range(n_epochs))
    })
    
    print(f"Dummy sleep data generated:")
    print(f"  Total epochs: {len(X)}")
    print(f"  Channels: {X.shape[1]}")
    print(f"  Time points: {X.shape[2]}")
    print(f"  Classes: {len(np.unique(y))}")
    print(f"  Class distribution: {np.bincount(y)}")
    
    return X, y, meta


def load_sleep_dataset(dataset_name='dummy', **kwargs):
    """
    Unified interface to load sleep datasets
    
    Args:
        dataset_name: Name of dataset ('sleep_edf', 'isruc', 'combined', 'dummy')
        **kwargs: Additional arguments for specific loaders
    
    Returns:
        X, y, meta
    """
    def validate_data(X, y, meta, dataset):
        if len(X) == 0:
            raise ValueError(f"CRITICAL: No valid data loaded for {dataset}.")
        # Define minimum expected subjects
        min_expected = {
            'sleep_edf': 10,
            'isruc': 5,
            'dreams': 5,
            'combined': 10
        }.get(dataset.lower(), 1)
        
        n_subjects = len(meta['subject'].unique()) if 'subject' in meta else 0
        if n_subjects < min_expected:
            print(f"\n[!] WARNING: Only {n_subjects}/{min_expected} min expected subjects loaded for {dataset}.")
            print("If this is a quick test, ignore this. Otherwise, data might be missing/corrupt.")
            
        return X, y, meta

    if dataset_name.lower() == 'sleep_edf':
        return validate_data(*load_sleep_edf(**kwargs), dataset_name)
    elif dataset_name.lower() == 'isruc':
        return validate_data(*load_isruc_sleep(**kwargs), dataset_name)
    elif dataset_name.lower() == 'dreams':
        return validate_data(*load_dreams_sleep(**kwargs), dataset_name)
    elif dataset_name.lower() == 'combined':
        return validate_data(*load_combined_datasets(**kwargs), dataset_name)
    elif dataset_name.lower() == 'dummy':
        return load_dummy_sleep_data(**kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def load_combined_datasets(
    sleep_edf_path=None,
    isruc_path=None,
    sleep_edf_subjects=None,
    isruc_subjects=None,
    isruc_subgroups=None,
    select_n_channels=2,
):
    """
    Load combined Sleep-EDF and ISRUC-Sleep datasets.
    
    Uses only the first N EEG channels to ensure consistency across datasets.
    Sleep-EDF has 2 channels (Fpz-Cz, Pz-Oz), ISRUC has 6 channels.
    
    Args:
        sleep_edf_path: Path to Sleep-EDF data
        isruc_path: Path to ISRUC-Sleep data (required)
        sleep_edf_subjects: Subjects for Sleep-EDF (default: all 20)
        isruc_subjects: Subjects for ISRUC (default: all for each subgroup)
        isruc_subgroups: List of ISRUC subgroups to load (default: [1, 2, 3] for all)
        select_n_channels: Number of channels to use (default: 2)
    
    Returns:
        X, y, meta with combined data
    """
    if isruc_subgroups is None:
        isruc_subgroups = [1, 2, 3]
    
    all_X = []
    all_y = []
    all_meta = []
    
    print("=" * 60)
    print("Loading Combined Datasets")
    print("=" * 60)
    
    print("\n[1/2] Loading Sleep-EDF...")
    try:
        X_edf, y_edf, meta_edf = load_sleep_edf(
            data_path=sleep_edf_path,
            subjects=sleep_edf_subjects,
            select_n_channels=select_n_channels,
        )
        meta_edf['dataset'] = 'Sleep-EDF'
        all_X.append(X_edf)
        all_y.append(y_edf)
        all_meta.append(meta_edf)
        print(f"  Sleep-EDF loaded: {len(X_edf)} epochs")
    except Exception as e:
        print(f"  Warning: Failed to load Sleep-EDF: {e}")
    
    print(f"\n[2/2] Loading ISRUC-Sleep (subgroups: {isruc_subgroups})...")
    if isruc_path is None:
        raise ValueError("isruc_path is required for combined dataset")
    
    for subgroup in isruc_subgroups:
        print(f"  Loading ISRUC Subgroup {subgroup}...")
        try:
            X_isruc, y_isruc, meta_isruc = load_isruc_sleep(
                data_path=isruc_path,
                subjects=isruc_subjects,
                select_n_channels=select_n_channels,
                subgroup=subgroup,
            )
            meta_isruc['dataset'] = f'ISRUC-Sleep-SG{subgroup}'
            all_X.append(X_isruc)
            all_y.append(y_isruc)
            all_meta.append(meta_isruc)
            print(f"    ISRUC Subgroup {subgroup} loaded: {len(X_isruc)} epochs")
        except Exception as e:
            print(f"    Warning: Failed to load ISRUC Subgroup {subgroup}: {e}")
    
    if not all_X:
        raise ValueError("No data loaded from any dataset")
    
    # Resample all datasets to match the first dataset's time dimension
    # This ensures consistent time points across datasets
    target_samples = all_X[0].shape[2]
    print(f"\nResampling datasets to match target samples: {target_samples}")
    
    # Ensure all datasets have the same number of channels
    # Use the minimum number of channels across all datasets
    min_channels = min(X_data.shape[1] for X_data in all_X)
    print(f"Ensuring consistent channel count: using {min_channels} channels")
    
    resampled_X = []
    for i, X_data in enumerate(all_X):
        # Resample time dimension if needed
        if X_data.shape[2] != target_samples:
            from scipy.signal import resample
            print(f"  Resampling dataset {i+1} from {X_data.shape[2]} to {target_samples} samples")
            X_resampled = resample(X_data, target_samples, axis=2)
        else:
            X_resampled = X_data
        
        # Ensure consistent channel count
        if X_resampled.shape[1] != min_channels:
            print(f"  Truncating dataset {i+1} from {X_resampled.shape[1]} to {min_channels} channels")
            X_resampled = X_resampled[:, :min_channels, :]
        
        resampled_X.append(X_resampled)
    
    X = np.concatenate(resampled_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    meta = pd.concat(all_meta, ignore_index=True)
    
    print("\n" + "=" * 60)
    print("Combined Dataset Summary:")
    print("=" * 60)
    print(f"  Total epochs: {len(X)}")
    print(f"  Channels: {X.shape[1]}")
    print(f"  Time points: {X.shape[2]}")
    print(f"  Classes: {len(np.unique(y))}")
    print(f"  Class distribution: {np.bincount(y)}")
    print(f"  Datasets: {meta['dataset'].value_counts().to_dict()}")
    print(f"  Subjects: {meta['subject'].nunique()}")
    
    return X, y, meta
