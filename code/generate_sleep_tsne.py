#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Generate t-SNE visualization for sleep EEG features."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import data_loader_sleep as data_loader
from algorithms_collection import get_algorithm
from visualization import plot_tsne_visualization


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate t-SNE for sleep EEG algorithms")
    parser.add_argument("--algorithm", type=str, default="SCA-FBTS",
                        help="Algorithm name for feature extraction")
    parser.add_argument("--dataset", type=str, default="sleep_edf",
                        choices=["dummy", "sleep_edf", "isruc"],
                        help="Sleep dataset name")
    parser.add_argument("--data-path", type=str, default=None,
                        help="Dataset root path for sleep_edf/isruc")
    parser.add_argument("--subjects", type=int, nargs="+", default=None,
                        help="Sleep PhysioNet subject IDs")
    parser.add_argument("--recording", type=int, nargs="+", default=[1, 2],
                        help="Sleep PhysioNet recording indices (default: both nights)")
    parser.add_argument("--n-epochs-dummy", type=int, default=300,
                        help="Number of epochs when dataset=dummy")
    parser.add_argument("--max-samples", type=int, default=1200,
                        help="Max samples used for t-SNE (for speed)")
    parser.add_argument("--use-pca", action="store_true",
                        help="Apply PCA before t-SNE")
    parser.add_argument("--save-dir", type=str, default="results",
                        help="Directory for output figure")
    return parser.parse_args()


def _extract_features(model, X: np.ndarray, algo_name: str) -> np.ndarray:
    """Extract 2D-plot-friendly features from a fitted model."""
    try:
        if algo_name in ["CSP+LDA", "CSP+SVM"]:
            return model.pipeline.named_steps["csp"].transform(X)

        if algo_name in ["HandcraftedFeatures+RF", "HandcraftedRF", "Handcrafted+RF"]:
            feats = model._extract_features(X)
            return model.scaler.transform(feats)

        if algo_name in [
            "FilterBankTangentSpace+SVM",
            "FilterBankTangentSpace+LDA",
            "FilterBankTangentSpace+RF",
            "SCA-FBTS",
            "SCA-FBTS+SVM",
        ]:
            features_list = []
            for i, (low, high) in enumerate(model.freq_bands):
                from algorithms_collection import apply_bandpass_filter
                X_band = np.array([apply_bandpass_filter(trial, low, high, model.fs) for trial in X])
                cov = model.cov_estimators[i].transform(X_band)
                ts = model.ts_transformers[i].transform(cov)
                features_list.append(ts)
            X_combined = np.hstack(features_list)
            if model.feature_selector is not None:
                return model.feature_selector.transform(X_combined)
            return X_combined

        if algo_name in ["RiemannTangentSpace", "RiemannTangentSpace+SVM", "RiemannTangentSpace+RF", "RiemannTangentSpace+PCA"]:
            cov = model.pipeline.named_steps["cov"].transform(X)
            ts = model.pipeline.named_steps["ts"].transform(cov)
            if "pca" in model.pipeline.named_steps:
                ts = model.pipeline.named_steps["pca"].transform(ts)
            return ts

        if algo_name == "MDM":
            cov = model.cov_estimator.transform(X)
            return cov.reshape(cov.shape[0], -1)

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            if isinstance(proba, np.ndarray):
                return proba

    except Exception as exc:
        print(f"Feature extraction fallback for {algo_name}: {exc}")

    return X.reshape(X.shape[0], -1)


def _load_sleep_data(args: argparse.Namespace):
    if args.dataset == "dummy":
        return data_loader.load_dummy_sleep_data(n_epochs=args.n_epochs_dummy)

    if args.dataset == "isruc" and args.data_path is None:
        raise ValueError("--data-path is required for ISRUC")

    return data_loader.load_sleep_dataset(
        args.dataset,
        data_path=args.data_path,
        subjects=args.subjects,
        recording=args.recording,
    )


def main() -> None:
    args = parse_args()

    print(f"Loading dataset: {args.dataset}")
    X, y, meta = _load_sleep_data(args)

    if args.max_samples is not None and X.shape[0] > args.max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(X.shape[0], size=args.max_samples, replace=False)
        X = X[idx]
        y = y[idx]

    n_samples, n_channels, n_times = X.shape
    n_classes = len(np.unique(y))
    fs_est = max(1, int(round(n_times / 30.0)))

    print(f"Data: samples={n_samples}, channels={n_channels}, times={n_times}, classes={n_classes}")
    print(f"Fitting model: {args.algorithm}")

    model = get_algorithm(
        algo_name=args.algorithm,
        n_channels=n_channels,
        n_times=n_times,
        n_classes=n_classes,
        fs=fs_est,
    )

    import inspect
    fit_signature = inspect.signature(model.fit)
    if "epochs" in fit_signature.parameters:
        model.fit(X, y, epochs=80)
    else:
        model.fit(X, y)

    print("Extracting features for t-SNE...")
    features = _extract_features(model, X, args.algorithm)
    print(f"Feature matrix shape: {features.shape}")

    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"tsne_sleep_{args.algorithm.replace('+', 'plus').replace(' ', '_')}.png"

    plot_tsne_visualization(
        features=features,
        labels=y,
        algorithm_name=args.algorithm,
        subject_id=0,
        dataset_name=args.dataset,
        save_path=str(out_path),
        use_pca=args.use_pca,
    )

    print(f"Saved t-SNE plot to: {out_path}")


if __name__ == "__main__":
    main()
