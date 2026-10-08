import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader_sleep import load_sleep_dataset
from algorithms_collection import get_algorithm

def calculate_feature_importance(data_path, dataset_name="sleep_edf", subjects="0"):
    print("Loading data for feature importance analysis...")
    X, y, meta = load_sleep_dataset(dataset_name, data_path=data_path, subjects=subjects)
    groups = meta['subject']
    
    print("\nInitializing SCA-FBTS model to extract Spatial-Spectral feature importance...")
    # Use FBTS to extract features and get ANOVA F-values
    model = get_algorithm("SCA-FBTS", n_channels=X.shape[1], n_times=X.shape[2])
    
    # We just need it to fit to get the feature_selector scores
    print("Fitting model to compute spatial-spectral covariance features...")
    model.fit(X, y)
    
    if model.feature_selector is None:
        print("Error: Feature selector was not used.")
        return
        
    scores = model.feature_selector.scores_
    
    # Feature dimension per band in tangent space: C*(C+1)/2. For 2 channels = 3 features per band
    spatial_feats_per_band = X.shape[1] * (X.shape[1] + 1) // 2
    
    bands = [f"{low}-{high}Hz" for low, high in model.freq_bands]
    band_importance = []
    
    # Group importance scores by frequency band
    for i, band in enumerate(bands):
        start_idx = i * spatial_feats_per_band
        end_idx = start_idx + spatial_feats_per_band
        # Average importance of the spatial patterns in this frequency band
        avg_score = np.mean(scores[start_idx:end_idx])
        band_importance.append(avg_score)
        
    # Normalize
    band_importance = np.array(band_importance)
    band_importance = 100 * band_importance / np.sum(band_importance)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    colors = sns.color_palette("viridis", len(bands))
    bars = plt.bar(bands, band_importance, color=colors, edgecolor='black')
    
    plt.title("Spatial-Spectral Feature Importance in SCA-FBTS", fontsize=15, pad=15)
    plt.xlabel("Frequency Bands", fontsize=12)
    plt.ylabel("Relative Importance (ANOVA F-value %)", fontsize=12)
    plt.xticks(rotation=45)
    
    # Add values on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f'{yval:.1f}%', ha='center', va='bottom')
        
    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    out_path = "results/figures/feature_importance_bands.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    
    print(f"\nFeature importance analysis complete! Plot saved to: {out_path}")
    print("\nImportance Ranking:")
    for b, i in sorted(zip(bands, band_importance), key=lambda x: x[1], reverse=True):
        print(f"  {b}: {i:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="E:/datasets/Sleep/sleep-edf-database-expanded-1.0.0")
    parser.add_argument("--dataset", type=str, default="sleep_edf")
    args = parser.parse_args()
    
    from evaluate_sleep import _parse_subjects_arg
    subjects_list = _parse_subjects_arg(["0~1"])
    calculate_feature_importance(args.data_path, args.dataset, subjects=subjects_list)
