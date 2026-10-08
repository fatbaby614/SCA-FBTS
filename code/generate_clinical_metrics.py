import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader_sleep import load_sleep_dataset

def calculate_clinical_metrics(y_true, y_pred, epoch_duration=30):
    """
    Calculate clinical metrics: TST (mins), WASO (mins), Sleep Efficiency (%)
    from a sequence of epochs. 0=W, 1=N1, 2=N2, 3=N3, 4=REM
    """
    metrics = []
    
    # Each sequence is a recording for a subject
    for true_seq, pred_seq in zip(y_true, y_pred):
        true_seq = np.array(true_seq)
        pred_seq = np.array(pred_seq)
        
        # Calculate TST (Total Sleep Time): sum of all sleep stages (1, 2, 3, 4) * 30 / 60
        tst_true = np.sum(true_seq > 0) * (epoch_duration / 60)
        tst_pred = np.sum(pred_seq > 0) * (epoch_duration / 60)
        
        # Time in bed (total length)
        tib = len(true_seq) * (epoch_duration / 60)
        
        se_true = (tst_true / tib) * 100 if tib > 0 else 0
        se_pred = (tst_pred / tib) * 100 if tib > 0 else 0
        
        # WASO: Wake after sleep onset. Wake time after first sleep epoch
        def calc_waso(seq):
            sleep_idx = np.where(seq > 0)[0]
            if len(sleep_idx) == 0: return 0
            first_sleep = sleep_idx[0]
            last_sleep = sleep_idx[-1]
            return np.sum(seq[first_sleep:last_sleep] == 0) * (epoch_duration / 60)
            
        waso_true = calc_waso(true_seq)
        waso_pred = calc_waso(pred_seq)
        
        metrics.append({
            "TST": (tst_true, tst_pred),
            "SE": (se_true, se_pred),
            "WASO": (waso_true, waso_pred)
        })
        
    return metrics

def plot_bland_altman(true_vals, pred_vals, label, unit, out_path):
    true_vals = np.array(true_vals)
    pred_vals = np.array(pred_vals)
    
    diffs = pred_vals - true_vals
    means = (pred_vals + true_vals) / 2
    
    md = np.mean(diffs)
    sd = np.std(diffs, axis=0)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(means, diffs, alpha=0.7, color='teal', edgecolors='black')
    
    plt.axhline(md, color='red', linestyle='-', label=f'Mean Diff ({md:.2f})')
    plt.axhline(md + 1.96*sd, color='gray', linestyle='--', label=f'+1.96 SD ({md + 1.96*sd:.2f})')
    plt.axhline(md - 1.96*sd, color='gray', linestyle='--', label=f'-1.96 SD ({md - 1.96*sd:.2f})')
    
    # 0 line
    plt.axhline(0, color='black', linewidth=1)
    
    plt.title(f'Bland-Altman Plot: {label}')
    plt.xlabel(f'Mean of True and Predicted {label} {unit}')
    plt.ylabel(f'Difference (Predicted - True) {unit}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def generate_clinical_analysis(data_path, dataset_name="sleep_edf", subjects="0~3"):
    print("Running evaluation to collect predictions for clinical metrics...")
    
    # We load data, map true to predicted using SCA-FBTS
    X, y, meta = load_sleep_dataset(dataset_name, data_path=data_path, subjects=subjects)
    groups = meta['subject']
    
    from sklearn.model_selection import LeaveOneGroupOut
    logo = LeaveOneGroupOut()
    
    true_seq_all = []
    pred_seq_all = []
    
    print("\nStarting subject-wise cross-validation...")
    fold_idx = 1
    total_folds = len(np.unique(groups))
    
    from algorithms_collection import get_algorithm
    
    for train_idx, test_idx in logo.split(X, y, groups):
        print(f"  Fold {fold_idx}/{total_folds}")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model = get_algorithm("SCA-FBTS", n_channels=X.shape[1], n_times=X.shape[2])
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        true_seq_all.append(y_test)
        pred_seq_all.append(y_pred)
        fold_idx += 1
        
    metrics = calculate_clinical_metrics(true_seq_all, pred_seq_all)
    
    tst_t, tst_p = zip(*[m["TST"] for m in metrics])
    se_t, se_p = zip(*[m["SE"] for m in metrics])
    waso_t, waso_p = zip(*[m["WASO"] for m in metrics])
    
    # Calculate MAE for metrics
    tst_mae = np.mean(np.abs(np.array(tst_t) - np.array(tst_p)))
    se_mae = np.mean(np.abs(np.array(se_t) - np.array(se_p)))
    waso_mae = np.mean(np.abs(np.array(waso_t) - np.array(waso_p)))
    
    print("\nClinical Metrics Evaluation Complete!")
    print(f"  TST MAE: {tst_mae:.2f} mins")
    print(f"  SE MAE: {se_mae:.2f} %")
    print(f"  WASO MAE: {waso_mae:.2f} mins")
    
    os.makedirs("results/figures", exist_ok=True)
    
    print("\nGenerating Bland-Altman Plots...")
    plot_bland_altman(tst_t, tst_p, "TST", "(mins)", "results/figures/BA_plot_TST.png")
    plot_bland_altman(se_t, se_p, "SE", "(%)", "results/figures/BA_plot_SE.png")
    plot_bland_altman(waso_t, waso_p, "WASO", "(mins)", "results/figures/BA_plot_WASO.png")
    
    print(f"Done! Plots saved to results/figures/BA_plot_*.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="E:/datasets/Sleep/sleep-edf-database-expanded-1.0.0")
    parser.add_argument("--dataset", type=str, default="sleep_edf")
    args = parser.parse_args()
    
    # Testing quickly on 4 subjects for the clinical plots
    from evaluate_sleep import _parse_subjects_arg
    subjects_list = _parse_subjects_arg(["0~3"])
    generate_clinical_analysis(args.data_path, args.dataset, subjects=subjects_list)