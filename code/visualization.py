import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from config.algorithms_config import RESULTS_PATH, get_timestamped_filename
import os
import platform


system = platform.system()
if system == 'Windows':
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
elif system == 'Darwin':
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'SimHei']
else:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12


def plot_accuracy_comparison(results, save_path=None):
    df = pd.DataFrame(results)
    
    plt.figure(figsize=(14, 8))
    ax = sns.boxplot(x='algorithm', y='accuracy', data=df, palette='Set2')
    ax = sns.stripplot(x='algorithm', y='accuracy', data=df, color='black', alpha=0.5, size=6)
    
    plt.title('Algorithm Accuracy Comparison', fontsize=16, fontweight='bold')
    plt.xlabel('Algorithm', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.ylim([0, 1])
    plt.xticks(rotation=15)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(os.path.join(RESULTS_PATH, save_path), dpi=300, bbox_inches='tight')
    
    plt.close()


def plot_kappa_comparison(results, save_path=None):
    df = pd.DataFrame(results)
    
    plt.figure(figsize=(14, 8))
    ax = sns.boxplot(x='algorithm', y='kappa', data=df, palette='Set3')
    ax = sns.stripplot(x='algorithm', y='kappa', data=df, color='black', alpha=0.5, size=6)
    
    plt.title("Cohen's Kappa Comparison", fontsize=16, fontweight='bold')
    plt.xlabel('Algorithm', fontsize=14)
    plt.ylabel('Kappa', fontsize=14)
    plt.ylim([0, 1])
    plt.xticks(rotation=15)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(os.path.join(RESULTS_PATH, save_path), dpi=300, bbox_inches='tight')
    
    plt.close()


def plot_training_time(results, save_path=None):
    df = pd.DataFrame(results)
    
    plt.figure(figsize=(14, 8))
    ax = sns.barplot(x='algorithm', y='train_time', data=df, palette='viridis', errorbar='sd')
    
    plt.title('Training Time Comparison', fontsize=16, fontweight='bold')
    plt.xlabel('Algorithm', fontsize=14)
    plt.ylabel('Training Time (s)', fontsize=14)
    plt.xticks(rotation=15)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(os.path.join(RESULTS_PATH, save_path), dpi=300, bbox_inches='tight')
    
    plt.close()


def plot_subject_comparison(results, save_path=None):
    df = pd.DataFrame(results)
    
    plt.figure(figsize=(16, 10))
    ax = sns.boxplot(x='subject', y='accuracy', hue='algorithm', data=df, palette='Set2')
    
    plt.title('Accuracy Comparison by Subject', fontsize=16, fontweight='bold')
    plt.xlabel('Subject', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.ylim([0, 1])
    plt.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(os.path.join(RESULTS_PATH, save_path), dpi=300, bbox_inches='tight')
    
    plt.close()


def plot_confusion_matrix(results, algorithm_name, save_path=None):
    df = pd.DataFrame(results)
    algorithm_results = df[df['algorithm'] == algorithm_name]
    
    all_cms = []
    for cm_str in algorithm_results['confusion_matrix']:
        cm = np.array(cm_str)
        all_cms.append(cm)
    
    mean_cm = np.mean(all_cms, axis=0)
    
    # 根据混淆矩阵的形状动态确定类别数量
    n_classes = mean_cm.shape[0]
    
    # 根据数据集类型和类别数量动态生成类别标签
    dataset_name = algorithm_results['dataset'].iloc[0] if 'dataset' in algorithm_results.columns else 'Unknown'
    
    if n_classes == 2:
        classes = ['Hand', 'Foot']
    elif n_classes == 4:
        if 'PhysionetMI' in dataset_name:
            classes = ['Left Hand', 'Right Hand', 'Both Hands', 'Both Feet']
        else:
            classes = ['Left', 'Right', 'Foot', 'Tongue']
    else:
        classes = [f'Class {i}' for i in range(n_classes)]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(mean_cm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    
    plt.title(f'{algorithm_name} Confusion Matrix (Average)', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(os.path.join(RESULTS_PATH, save_path), dpi=300, bbox_inches='tight')
    
    plt.close()


def plot_performance_summary(summary_df, save_path=None):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    axes[0, 0].bar(summary_df['algorithm'], summary_df['accuracy_mean'], 
                   yerr=summary_df['accuracy_std'], capsize=5, color='steelblue')
    axes[0, 0].set_title('Mean Accuracy', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Accuracy', fontsize=12)
    axes[0, 0].tick_params(axis='x', rotation=15)
    axes[0, 0].set_ylim([0, 1])
    
    axes[0, 1].bar(summary_df['algorithm'], summary_df['kappa_mean'], 
                   yerr=summary_df['kappa_std'], capsize=5, color='coral')
    axes[0, 1].set_title('Mean Kappa', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Kappa', fontsize=12)
    axes[0, 1].tick_params(axis='x', rotation=15)
    axes[0, 1].set_ylim([0, 1])
    
    axes[1, 0].bar(summary_df['algorithm'], summary_df['train_time_mean'], 
                   yerr=summary_df['train_time_std'], capsize=5, color='forestgreen')
    axes[1, 0].set_title('Mean Training Time', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('Time (s)', fontsize=12)
    axes[1, 0].tick_params(axis='x', rotation=15)
    
    axes[1, 1].bar(summary_df['algorithm'], summary_df['accuracy_max'], 
                   alpha=0.7, color='purple', label='Max')
    axes[1, 1].bar(summary_df['algorithm'], summary_df['accuracy_min'], 
                   alpha=0.7, color='orange', label='Min')
    axes[1, 1].set_title('Accuracy Range', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('Accuracy', fontsize=12)
    axes[1, 1].legend()
    axes[1, 1].tick_params(axis='x', rotation=15)
    axes[1, 1].set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(os.path.join(RESULTS_PATH, save_path), dpi=300, bbox_inches='tight')
    
    plt.close()


def _get_algorithm_display_name(algorithm_name):
    """Get display name for algorithm to ensure consistency."""
    name_map = {
        'FilterBankTangentSpace+SVM': 'FBTS-SVM',
        'FilterBankTangentSpace+LDA': 'FBTS-LDA',
        'FilterBankTangentSpace+RF': 'FBTS-RF',
        'SCA-FBTS': 'SCA-FBTS',
        'HandcraftedFeatures+RF': 'Handcrafted+RF',
        'RiemannTangentSpace': 'Riemann-TS',
        'RiemannTangentSpace+SVM': 'Riemann-TS+SVM',
        'CSP+SVM': 'CSP+SVM',
        'CSP+LDA': 'CSP+LDA',
    }
    return name_map.get(algorithm_name, algorithm_name)


def plot_tsne_visualization(features, labels, algorithm_name, subject_id, dataset_name, save_path=None, use_pca=True):
    """
    Generate t-SNE visualization for features extracted by an algorithm
    
    Parameters:
    -----------
    features : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    labels : np.ndarray
        True labels of shape (n_samples,)
    algorithm_name : str
        Name of the algorithm
    subject_id : int
        Subject ID
    dataset_name : str
        Name of the dataset
    save_path : str, optional
        Path to save the figure
    use_pca : bool, optional
        Whether to use PCA preprocessing before t-SNE (default: True)
    """
    if features.shape[0] < 3:
        print(f"Warning: Not enough samples ({features.shape[0]}) for t-SNE visualization")
        return
    
    n_classes = len(np.unique(labels))
    
    # Generate class labels based on dataset and number of classes
    if n_classes == 2:
        class_names = ['Hand', 'Foot']
    elif n_classes == 4:
        if 'PhysionetMI' in dataset_name:
            class_names = ['Left Hand', 'Right Hand', 'Both Hands', 'Both Feet']
        else:
            class_names = ['Left', 'Right', 'Foot', 'Tongue']
    elif n_classes == 5 and ('sleep' in dataset_name.lower() or 'edf' in dataset_name.lower() or 'isruc' in dataset_name.lower()):
        # Sleep staging: 5 classes (W, N1, N2, N3, REM)
        class_names = ['W', 'N1', 'N2', 'N3', 'REM']
    else:
        class_names = [f'Class {i}' for i in range(n_classes)]
    
    print(f"  Applying t-SNE on {features.shape} features...")
    
    # Method 2: PCA preprocessing for better t-SNE results
    if use_pca and features.shape[1] > 50:
        print(f"  Applying PCA preprocessing...")
        from sklearn.decomposition import PCA
        # Reduce to 50 dimensions first
        n_pca_components = min(50, features.shape[0] - 1, features.shape[1])
        pca = PCA(n_components=n_pca_components, random_state=42)
        features = pca.fit_transform(features)
        print(f"  PCA reduced dimensions to {features.shape[1]} (explained variance: {pca.explained_variance_ratio_.sum():.3f})")
    
    # Method 3: Optimized t-SNE parameters
    n_samples = features.shape[0]
    
    # Better perplexity calculation
    perplexity = min(50, max(5, n_samples // 4))
    
    # Adjust learning rate based on perplexity
    learning_rate = max(100, perplexity * 4)
    
    print(f"  t-SNE parameters: perplexity={perplexity}, learning_rate={learning_rate}")
    
    # Apply t-SNE with optimized parameters
    tsne = TSNE(
        n_components=2, 
        random_state=42, 
        perplexity=perplexity, 
        learning_rate=learning_rate,
        max_iter=2000,
        early_exaggeration=12.0,
        init='pca',
        method='barnes_hut' if n_samples > 1000 else 'exact'
    )
    features_2d = tsne.fit_transform(features)
    
    # Create color palette
    colors = plt.cm.Set1(np.linspace(0, 1, n_classes))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot each class separately for better legend
    for i, class_name in enumerate(class_names):
        mask = labels == i
        ax.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                  c=[colors[i]], label=class_name, alpha=0.7, s=60, edgecolors='k', linewidth=0.5)
    
    pca_text = " (with PCA)" if use_pca and features.shape[1] > 50 else ""
    display_name = _get_algorithm_display_name(algorithm_name)
    ax.set_title(f't-SNE Visualization{pca_text}: {display_name}\n{dataset_name} - Subject {subject_id} (n={n_samples})', 
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('t-SNE Component 1', fontsize=14)
    ax.set_ylabel('t-SNE Component 2', fontsize=14)
    ax.legend(title='Sleep Stage', fontsize=12, title_fontsize=13, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Add interpretation text
    interpretation_text = (
        f"Interpretation: t-SNE reveals feature clustering patterns.\n"
        f"Well-separated clusters indicate discriminative features.\n"
        f"Overlapping regions suggest transitional stages (e.g., N1↔W, N1↔N2)."
    )
    ax.text(0.02, 0.98, interpretation_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  t-SNE plot saved to {save_path}")
    
    plt.close()


def plot_tsne_comparison(features_dict, labels, algorithm_names, subject_id, dataset_name, save_path=None, use_pca=True):
    """
    Generate t-SNE visualization comparing multiple algorithms
    
    Parameters:
    -----------
    features_dict : dict
        Dictionary mapping algorithm names to feature matrices
    labels : np.ndarray
        True labels of shape (n_samples,)
    algorithm_names : list
        List of algorithm names to visualize
    subject_id : int
        Subject ID
    dataset_name : str
        Name of the dataset
    save_path : str, optional
        Path to save the figure
    use_pca : bool, optional
        Whether to use PCA preprocessing before t-SNE (default: True)
    """
    n_algorithms = len(algorithm_names)
    n_cols = min(3, n_algorithms)
    n_rows = (n_algorithms + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_algorithms == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    n_classes = len(np.unique(labels))
    
    # Generate class labels based on dataset and number of classes
    if n_classes == 2:
        class_names = ['Hand', 'Foot']
    elif n_classes == 4:
        if 'PhysionetMI' in dataset_name:
            class_names = ['Left Hand', 'Right Hand', 'Both Hands', 'Both Feet']
        else:
            class_names = ['Left', 'Right', 'Foot', 'Tongue']
    elif n_classes == 5 and ('sleep' in dataset_name.lower() or 'edf' in dataset_name.lower() or 'isruc' in dataset_name.lower()):
        # Sleep staging: 5 classes (W, N1, N2, N3, REM)
        class_names = ['W', 'N1', 'N2', 'N3', 'REM']
    else:
        class_names = [f'Class {i}' for i in range(n_classes)]
    
    colors = plt.cm.Set1(np.linspace(0, 1, n_classes))
    
    for idx, algo_name in enumerate(algorithm_names):
        ax = axes[idx]
        
        if algo_name not in features_dict:
            ax.text(0.5, 0.5, 'No features available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{algo_name}', fontsize=12, fontweight='bold')
            continue
        
        features = features_dict[algo_name]
        
        if features.shape[0] < 3:
            ax.text(0.5, 0.5, f'Insufficient samples\n({features.shape[0]})', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{algo_name}', fontsize=12, fontweight='bold')
            continue
        
        # Apply PCA preprocessing if enabled
        if use_pca and features.shape[1] > 50:
            from sklearn.decomposition import PCA
            n_pca_components = min(50, features.shape[0] - 1, features.shape[1])
            pca = PCA(n_components=n_pca_components, random_state=42)
            features = pca.fit_transform(features)
        
        # Apply t-SNE with optimized parameters
        n_samples = features.shape[0]
        perplexity = min(50, max(5, n_samples // 4))
        learning_rate = max(100, perplexity * 4)
        
        tsne = TSNE(
            n_components=2, 
            random_state=42, 
            perplexity=perplexity, 
            learning_rate=learning_rate,
            max_iter=2000,
            early_exaggeration=12.0,
            init='pca',
            method='barnes_hut' if n_samples > 1000 else 'exact'
        )
        features_2d = tsne.fit_transform(features)
        
        # Plot each class separately
        for i, class_name in enumerate(class_names):
            mask = labels == i
            ax.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                      c=[colors[i]], label=class_name, alpha=0.7, s=50, edgecolors='k', linewidth=0.3)
        
        ax.set_title(f'{algo_name}', fontsize=13, fontweight='bold')
        ax.set_xlabel('t-SNE 1', fontsize=11)
        ax.set_ylabel('t-SNE 2', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        if idx == 0:
            ax.legend(title='Class', fontsize=9, title_fontsize=10, loc='best')
    
    # Hide unused subplots
    for idx in range(n_algorithms, len(axes)):
        axes[idx].axis('off')
    
    pca_text = " (with PCA)" if use_pca else ""
    plt.suptitle(f't-SNE Feature Comparison{pca_text}\n{dataset_name} - Subject {subject_id}', 
                 fontsize=16, fontweight='bold', y=1.0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  t-SNE comparison plot saved to {save_path}")
    
    plt.close()


def _compute_tsne_embedding(features, use_pca=True):
    """Compute 2D t-SNE embedding with optional PCA preprocessing."""
    X = np.asarray(features)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D feature matrix, got shape {X.shape}")

    if use_pca and X.shape[1] > 50:
        from sklearn.decomposition import PCA
        n_pca_components = min(50, X.shape[0] - 1, X.shape[1])
        if n_pca_components >= 2:
            pca = PCA(n_components=n_pca_components, random_state=42)
            X = pca.fit_transform(X)

    n_samples = X.shape[0]
    perplexity = min(50, max(5, n_samples // 4))
    learning_rate = max(100, perplexity * 4)

    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=perplexity,
        learning_rate=learning_rate,
        max_iter=2000,
        early_exaggeration=12.0,
        init='pca',
        method='barnes_hut' if n_samples > 1000 else 'exact'
    )
    return tsne.fit_transform(X)


def plot_sleep_trajectory_tsne(features, labels, meta, algorithm_name, dataset_name, save_path=None, use_pca=True):
    """Plot overnight sleep trajectory in t-SNE space with time-gradient lines."""
    if features.shape[0] < 3:
        print(f"Warning: Not enough samples ({features.shape[0]}) for trajectory t-SNE")
        return

    if meta is None or len(meta) != len(features):
        print("Warning: trajectory plot skipped due to missing/inconsistent metadata")
        return

    coords = _compute_tsne_embedding(features, use_pca=use_pca)
    meta_df = pd.DataFrame(meta).reset_index(drop=True).copy()
    meta_df['x'] = coords[:, 0]
    meta_df['y'] = coords[:, 1]
    meta_df['label'] = np.asarray(labels)

    if 'night' in meta_df.columns:
        group_key = meta_df['subject'].astype(str) + '_' + meta_df['night'].astype(str)
    else:
        group_key = meta_df['subject'].astype(str)
    meta_df['group_key'] = group_key

    fig, ax = plt.subplots(figsize=(12, 10))
    cmap = plt.cm.coolwarm

    for _, g in meta_df.groupby('group_key'):
        if 'epoch' in g.columns:
            g = g.sort_values('epoch')
        else:
            g = g.sort_index()

        pts = g[['x', 'y']].to_numpy()
        if len(pts) < 2:
            continue

        segments = np.stack([pts[:-1], pts[1:]], axis=1)
        t = np.linspace(0, 1, len(segments))
        lc = LineCollection(segments, cmap=cmap, norm=plt.Normalize(0, 1))
        lc.set_array(t)
        lc.set_linewidth(1.6)
        lc.set_alpha(0.85)
        ax.add_collection(lc)

        ax.scatter(pts[0, 0], pts[0, 1], c=[cmap(0.05)], s=18, marker='o', alpha=0.9)
        ax.scatter(pts[-1, 0], pts[-1, 1], c=[cmap(0.95)], s=18, marker='s', alpha=0.9)

    sc = ax.scatter(meta_df['x'], meta_df['y'], c=np.linspace(0, 1, len(meta_df)),
                    cmap=cmap, s=8, alpha=0.12)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Time progression (night -> morning)')

    display_name = _get_algorithm_display_name(algorithm_name)
    ax.set_title(f'Overnight Sleep Trajectory t-SNE: {display_name}\n{dataset_name}',
                 fontsize=15, fontweight='bold')
    ax.set_xlabel('t-SNE Component 1', fontsize=13)
    ax.set_ylabel('t-SNE Component 2', fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Sleep trajectory t-SNE saved to {save_path}")
    plt.close()


def plot_sleep_stability_tsne(features, labels, meta, algorithm_name, dataset_name, save_path=None, use_pca=True):
    """Plot cross-subject/cross-night t-SNE stability map.

    Same stage -> same color; different subjects -> different marker/alpha.
    """
    if features.shape[0] < 3:
        print(f"Warning: Not enough samples ({features.shape[0]}) for stability t-SNE")
        return

    if meta is None or len(meta) != len(features):
        print("Warning: stability plot skipped due to missing/inconsistent metadata")
        return

    coords = _compute_tsne_embedding(features, use_pca=use_pca)
    meta_df = pd.DataFrame(meta).reset_index(drop=True).copy()
    meta_df['x'] = coords[:, 0]
    meta_df['y'] = coords[:, 1]
    meta_df['label'] = np.asarray(labels)

    unique_labels = sorted(np.unique(labels))
    stage_colors = plt.cm.Set1(np.linspace(0, 1, max(3, len(unique_labels))))
    label_to_color = {lab: stage_colors[i % len(stage_colors)] for i, lab in enumerate(unique_labels)}

    subject_series = meta_df['subject'].astype(str) if 'subject' in meta_df.columns else pd.Series(['S0'] * len(meta_df))
    unique_subjects = sorted(subject_series.unique().tolist())
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', '<', '>']

    fig, ax = plt.subplots(figsize=(12, 10))
    for s_idx, subject in enumerate(unique_subjects):
        sub_mask = subject_series == subject
        marker = markers[s_idx % len(markers)]
        alpha = 0.35 + 0.45 * (s_idx / max(1, len(unique_subjects) - 1))
        for lab in unique_labels:
            mask = sub_mask & (meta_df['label'] == lab)
            if not np.any(mask):
                continue
            ax.scatter(
                meta_df.loc[mask, 'x'],
                meta_df.loc[mask, 'y'],
                c=[label_to_color[lab]],
                marker=marker,
                s=26,
                alpha=alpha,
                edgecolors='k',
                linewidths=0.2,
            )

    stage_names = {0: 'W', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM'}
    stage_handles = [
        Line2D([0], [0], marker='o', color='w', label=stage_names.get(lab, f'Class {lab}'),
               markerfacecolor=label_to_color[lab], markeredgecolor='k', markersize=8)
        for lab in unique_labels
    ]
    legend_stage = ax.legend(handles=stage_handles, title='Sleep Stage', loc='upper right')
    ax.add_artist(legend_stage)

    shown_subjects = unique_subjects[:10]
    subject_handles = [
        Line2D([0], [0], marker=markers[i % len(markers)], color='gray', label=s,
               linestyle='None', markersize=7)
        for i, s in enumerate(shown_subjects)
    ]
    ax.legend(handles=subject_handles, title='Subject (subset)', loc='lower right')

    display_name = _get_algorithm_display_name(algorithm_name)
    ax.set_title(f'Cross-Subject/Cross-Night Stability t-SNE: {display_name}\n{dataset_name}',
                 fontsize=15, fontweight='bold')
    ax.set_xlabel('t-SNE Component 1', fontsize=13)
    ax.set_ylabel('t-SNE Component 2', fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Sleep stability t-SNE saved to {save_path}")
    plt.close()


def plot_sleep_tsne(features, labels, meta, algorithm_name, dataset_name, save_path=None, use_pca=True):
    """Create a publication-ready 3-panel t-SNE figure for sleep staging.

    Panel A: sleep stage distribution.
    Panel B: overnight trajectory with time-gradient lines.
    Panel C: cross-subject/cross-night stability (same stage color, subject marker/alpha).
    """
    if features.shape[0] < 3:
        print(f"Warning: Not enough samples ({features.shape[0]}) for combined t-SNE figure")
        return
    if meta is None or len(meta) != len(features):
        print("Warning: combined t-SNE figure skipped due to missing/inconsistent metadata")
        return

    coords = _compute_tsne_embedding(features, use_pca=use_pca)
    meta_df = pd.DataFrame(meta).reset_index(drop=True).copy()
    meta_df['x'] = coords[:, 0]
    meta_df['y'] = coords[:, 1]
    meta_df['label'] = np.asarray(labels)

    unique_labels = sorted(np.unique(labels))
    stage_names = {0: 'W', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM'}
    stage_colors = plt.cm.Set1(np.linspace(0, 1, max(3, len(unique_labels))))
    label_to_color = {lab: stage_colors[i % len(stage_colors)] for i, lab in enumerate(unique_labels)}

    subject_series = meta_df['subject'].astype(str) if 'subject' in meta_df.columns else pd.Series(['S0'] * len(meta_df))
    if 'night' in meta_df.columns:
        group_key = subject_series + '_' + meta_df['night'].astype(str)
    else:
        group_key = subject_series
    meta_df['group_key'] = group_key

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    # Panel A: Stage distribution
    ax = axes[0]
    for lab in unique_labels:
        mask = meta_df['label'] == lab
        ax.scatter(
            meta_df.loc[mask, 'x'],
            meta_df.loc[mask, 'y'],
            c=[label_to_color[lab]],
            label=stage_names.get(lab, f'Class {lab}'),
            s=20,
            alpha=0.7,
            edgecolors='k',
            linewidths=0.2,
        )
    ax.set_title('A. Sleep Stage Distribution', fontsize=13, fontweight='bold')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.grid(True, alpha=0.3)
    ax.legend(title='Stage', fontsize=9, title_fontsize=10, loc='best')

    # Panel B: Overnight trajectory (time gradient with sleep stage markers)
    ax = axes[1]
    cmap = plt.cm.coolwarm
    for _, g in meta_df.groupby('group_key'):
        g = g.sort_values('epoch') if 'epoch' in g.columns else g.sort_index()
        pts = g[['x', 'y']].to_numpy()
        labels = g['label'].to_numpy()
        if len(pts) < 2:
            continue
        segments = np.stack([pts[:-1], pts[1:]], axis=1)
        t = np.linspace(0, 1, len(segments))
        lc = LineCollection(segments, cmap=cmap, norm=plt.Normalize(0, 1))
        lc.set_array(t)
        lc.set_linewidth(1.5)
        lc.set_alpha(0.85)
        ax.add_collection(lc)
        # Plot points with sleep stage colors
        for i, (pt, lab) in enumerate(zip(pts, labels)):
            ax.scatter(pt[0], pt[1], c=[label_to_color[lab]], 
                       s=12, marker='o', alpha=0.85, edgecolors='k', linewidths=0.5)
        # Mark start (circle) and end (square) with time color
        ax.scatter(pts[0, 0], pts[0, 1], c=[cmap(0.05)], s=20, marker='o', alpha=0.9, edgecolors='k', linewidths=1)
        ax.scatter(pts[-1, 0], pts[-1, 1], c=[cmap(0.95)], s=20, marker='s', alpha=0.9, edgecolors='k', linewidths=1)
    sm = plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap=cmap)
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Time (night -> morning)')
    # Add sleep stage legend
    stage_handles = [
        Line2D([0], [0], marker='o', color='w', label=stage_names.get(lab, f'Class {lab}'),
               markerfacecolor=label_to_color[lab], markeredgecolor='k', markersize=6)
        for lab in unique_labels
    ]
    ax.legend(handles=stage_handles, title='Sleep Stage', loc='lower right', fontsize=8, title_fontsize=9)
    ax.set_title('B. Overnight Sleep Trajectory', fontsize=13, fontweight='bold')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.grid(True, alpha=0.3)

    # Panel C: Cross-subject/cross-night stability
    ax = axes[2]
    unique_subjects = sorted(subject_series.unique().tolist())
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', '<', '>']
    for s_idx, subject in enumerate(unique_subjects):
        sub_mask = subject_series == subject
        marker = markers[s_idx % len(markers)]
        alpha = 0.35 + 0.45 * (s_idx / max(1, len(unique_subjects) - 1))
        for lab in unique_labels:
            mask = sub_mask & (meta_df['label'] == lab)
            if not np.any(mask):
                continue
            ax.scatter(
                meta_df.loc[mask, 'x'],
                meta_df.loc[mask, 'y'],
                c=[label_to_color[lab]],
                marker=marker,
                s=24,
                alpha=alpha,
                edgecolors='k',
                linewidths=0.2,
            )
    stage_handles = [
        Line2D([0], [0], marker='o', color='w', label=stage_names.get(lab, f'Class {lab}'),
               markerfacecolor=label_to_color[lab], markeredgecolor='k', markersize=7)
        for lab in unique_labels
    ]
    legend_stage = ax.legend(handles=stage_handles, title='Stage', loc='upper right', fontsize=8, title_fontsize=9)
    ax.add_artist(legend_stage)
    shown_subjects = unique_subjects[:10]
    subject_handles = [
        Line2D([0], [0], marker=markers[i % len(markers)], color='gray', label=s,
               linestyle='None', markersize=6)
        for i, s in enumerate(shown_subjects)
    ]
    ax.legend(handles=subject_handles, title='Subject', loc='lower right', fontsize=8, title_fontsize=9)
    ax.set_title('C. Cross-Subject/Cross-Night Stability', fontsize=13, fontweight='bold')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.grid(True, alpha=0.3)

    plt.suptitle(f't-SNE Analysis: {algorithm_name} on {dataset_name}',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Combined t-SNE saved to {save_path}")
    plt.close()


def generate_all_plots(results, summary_df):
    print("\nGenerating visualizations...")
    
    # Generate timestamped filenames for all plots
    plot_accuracy_comparison(results, save_path=get_timestamped_filename('accuracy_comparison', 'png'))
    plot_kappa_comparison(results, save_path=get_timestamped_filename('kappa_comparison', 'png'))
    plot_training_time(results, save_path=get_timestamped_filename('training_time', 'png'))
    plot_performance_summary(summary_df, save_path=get_timestamped_filename('performance_summary', 'png'))
    
    algorithms = pd.DataFrame(results)['algorithm'].unique()
    for algo in algorithms:
        safe_algo_name = algo.replace('+', '_').replace('-', '_')
        plot_confusion_matrix(results, algo, save_path=get_timestamped_filename(f'confusion_matrix_{safe_algo_name}', 'png'))
    
    print("All plots saved to results/ directory")


def plot_algorithm_comparison_bars(summary_df, save_path=None, figsize=(14, 8)):
    """
    Publication-ready grouped bar chart for algorithm comparison
    
    Shows Accuracy, Kappa, and Macro-F1 with error bars for all algorithms
    """
    if 'macro_f1_mean' not in summary_df.columns:
        # Keep compatibility with existing summary schema that may not include Macro-F1.
        summary_df = summary_df.copy()
        summary_df['macro_f1_mean'] = np.nan
        summary_df['macro_f1_std'] = np.nan
    
    df = summary_df.copy()
    df = df.sort_values('accuracy_mean', ascending=False)
    
    x = np.arange(len(df))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=figsize)
    
    bars1 = ax.bar(x - width, df['accuracy_mean'], width, 
                   yerr=df['accuracy_std'] if 'accuracy_std' in df.columns else None, label='Accuracy', 
                   color='steelblue', alpha=0.8, capsize=5)
    
    if 'kappa_mean' in df.columns:
        bars2 = ax.bar(x, df['kappa_mean'], width, 
                       yerr=df['kappa_std'] if 'kappa_std' in df.columns else None, label='Cohen\'s Kappa', 
                       color='coral', alpha=0.8, capsize=5)
    
    if 'macro_f1_mean' in df.columns and not df['macro_f1_mean'].isna().all():
        bars3 = ax.bar(x + width, df['macro_f1_mean'], width, 
                       yerr=df['macro_f1_std'] if 'macro_f1_std' in df.columns else None, label='Macro-F1', 
                       color='forestgreen', alpha=0.8, capsize=5)
    
    ax.set_xlabel('Algorithm', fontsize=13, fontweight='bold')
    ax.set_ylabel('Score', fontsize=13, fontweight='bold')
    ax.set_title('Algorithm Performance Comparison', fontsize=15, fontweight='bold')
    # Convert algorithm names for display
    display_names = []
    for algo in df['algorithm']:
        if algo == 'FilterBankTangentSpace+SVM':
            display_names.append('FBTS-SVM')
        else:
            display_names.append(algo)
    
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, rotation=45, ha='right')
    ax.set_ylim([0, 1.05])
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Algorithm comparison saved to {save_path}")
    plt.close()


def plot_confusion_matrices_grid(results_df, algorithm_names=None, save_path=None, figsize=(14, 6)):
    """
    Publication-ready confusion matrix heatmaps
    
    Displays confusion matrices for FBTS and best baseline side-by-side
    """
    import ast
    
    if algorithm_names is None:
        # Main evaluation records the proposed method as 'SCA-FBTS'
        algorithm_names = ['SCA-FBTS']
        if 'SCA-FBTS' not in results_df['algorithm'].values:
            algorithm_names = ['FilterBankTangentSpace+SVM']
        if 'CSP+LDA' in results_df['algorithm'].values:
            algorithm_names.append('CSP+LDA')
        else:
            score_col = 'accuracy_mean' if 'accuracy_mean' in results_df.columns else 'accuracy'
            if score_col in results_df.columns:
                best = results_df.groupby('algorithm')[score_col].mean().idxmax()
                if best not in algorithm_names:
                    algorithm_names.append(best)
    else:
        algorithm_names = [a for a in algorithm_names if a in results_df['algorithm'].values]
    
    if len(algorithm_names) == 0:
        print("Warning: No matching algorithms found. Skipping confusion matrix plot.")
        return
    
    n_algos = min(len(algorithm_names), 2)
    fig, axes = plt.subplots(1, n_algos, figsize=figsize)
    if n_algos == 1:
        axes = [axes]
    
    stage_names = {0: 'W', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM'}
    
    for idx, algo_name in enumerate(algorithm_names[:n_algos]):
        ax = axes[idx]
        
        algo_results = results_df[results_df['algorithm'] == algo_name]
        if len(algo_results) == 0:
            ax.text(0.5, 0.5, f'No data for {algo_name}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(algo_name, fontsize=12, fontweight='bold')
            continue
        
        # Parse confusion matrices and average
        cms = []
        for cm_str in algo_results['confusion_matrix']:
            try:
                if isinstance(cm_str, str):
                    cm = np.array(ast.literal_eval(cm_str))
                else:
                    cm = np.array(cm_str)
                cms.append(cm)
            except:
                pass
        
        if len(cms) == 0:
            ax.text(0.5, 0.5, 'Invalid confusion matrix data', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(algo_name, fontsize=12, fontweight='bold')
            continue
        
        mean_cm = np.mean(cms, axis=0).astype(int)
        
        # Normalize for better visualization
        cm_normalized = mean_cm.astype('float') / (mean_cm.sum(axis=1, keepdims=True) + 1e-10)
        
        # Get class labels
        n_classes = mean_cm.shape[0]
        classes = [stage_names.get(i, f'C{i}') for i in range(n_classes)]
        
        sns.heatmap(cm_normalized, annot=mean_cm, fmt='d', cmap='Blues', 
                   xticklabels=classes, yticklabels=classes, ax=ax,
                   cbar_kws={'label': 'Normalized'}, vmin=0, vmax=1)
        
        ax.set_title(f'{algo_name}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted Stage', fontsize=11)
        ax.set_ylabel('True Stage', fontsize=11)
    
    plt.suptitle('Confusion Matrices (averaged over folds)', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrices saved to {save_path}")
    plt.close()


def plot_hypnogram(true_labels, pred_labels, epoch_duration=30, subject_id=None, 
                   algorithm_name='FBTS-SVM', save_path=None, figsize=(12, 4)):
    """
    Plot sleep hypnogram comparing true labels vs predicted labels.
    
    Args:
        true_labels: Array of true sleep stage labels (0=W, 1=N1, 2=N2, 3=N3, 4=REM)
        pred_labels: Array of predicted sleep stage labels
        epoch_duration: Duration of each epoch in seconds (default: 30s)
        subject_id: Subject identifier for title
        algorithm_name: Algorithm name for legend
        save_path: Path to save the figure
        figsize: Figure size (width, height)
    """
    stage_names = {0: 'W', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM'}
    stage_colors = {
        0: '#FF6B6B',
        1: '#4ECDC4',
        2: '#45B7D1',
        3: '#2C3E50',
        4: '#9B59B6',
    }
    
    n_epochs = len(true_labels)
    time_hours = np.arange(n_epochs) * epoch_duration / 3600.0
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    for stage in range(5):
        mask_true = true_labels == stage
        mask_pred = pred_labels == stage
        
        if np.any(mask_true):
            ax1.fill_between(time_hours, stage - 0.4, stage + 0.4, 
                           where=mask_true, color=stage_colors[stage], alpha=0.8, 
                           label=stage_names[stage], step='mid')
        
        if np.any(mask_pred):
            ax2.fill_between(time_hours, stage - 0.4, stage + 0.4,
                           where=mask_pred, color=stage_colors[stage], alpha=0.8,
                           label=stage_names[stage], step='mid')
    
    ax1.set_ylabel('True Stage', fontsize=11)
    ax1.set_yticks(range(5))
    ax1.set_yticklabels([stage_names[i] for i in range(5)])
    ax1.set_ylim(-0.5, 4.5)
    ax1.invert_yaxis()
    ax1.legend(loc='upper left', bbox_to_anchor=(1.02, 1), ncol=1, fontsize=9)
    ax1.grid(True, alpha=0.3, axis='x')
    
    ax2.set_ylabel(f'Predicted\n({algorithm_name})', fontsize=11)
    ax2.set_yticks(range(5))
    ax2.set_yticklabels([stage_names[i] for i in range(5)])
    ax2.set_ylim(-0.5, 4.5)
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3, axis='x')
    
    ax2.set_xlabel('Time (hours)', fontsize=11)
    
    accuracy = np.mean(true_labels == pred_labels)
    
    if subject_id is not None:
        title = f'Sleep Hypnogram - Subject {subject_id} (Accuracy: {accuracy:.1%})'
    else:
        title = f'Sleep Hypnogram (Accuracy: {accuracy:.1%})'
    fig.suptitle(title, fontsize=13, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Hypnogram saved to {save_path}")
    plt.close()
    
    return accuracy


def plot_hypnogram_comparison(true_labels, pred_labels_dict, epoch_duration=30, 
                              subject_id=None, save_path=None, figsize=(14, 6)):
    """
    Plot hypnogram comparing multiple algorithms' predictions.
    
    Args:
        true_labels: Array of true sleep stage labels
        pred_labels_dict: Dictionary of {algorithm_name: pred_labels}
        epoch_duration: Duration of each epoch in seconds
        subject_id: Subject identifier for title
        save_path: Path to save the figure
        figsize: Figure size (width, height)
    """
    stage_names = {0: 'W', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM'}
    stage_colors = {
        0: '#FF6B6B',
        1: '#4ECDC4',
        2: '#45B7D1',
        3: '#2C3E50',
        4: '#9B59B6',
    }
    
    n_algos = len(pred_labels_dict) + 1
    n_epochs = len(true_labels)
    time_hours = np.arange(n_epochs) * epoch_duration / 3600.0
    
    fig, axes = plt.subplots(n_algos, 1, figsize=figsize, sharex=True)
    
    if n_algos == 1:
        axes = [axes]
    
    ax_idx = 0
    ax = axes[ax_idx]
    for stage in range(5):
        mask = true_labels == stage
        if np.any(mask):
            ax.fill_between(time_hours, stage - 0.4, stage + 0.4,
                           where=mask, color=stage_colors[stage], alpha=0.8,
                           label=stage_names[stage], step='mid')
    ax.set_ylabel('True', fontsize=10)
    ax.set_yticks(range(5))
    ax.set_yticklabels([stage_names[i] for i in range(5)])
    ax.set_ylim(-0.5, 4.5)
    ax.invert_yaxis()
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), ncol=1, fontsize=8)
    ax.grid(True, alpha=0.3, axis='x')
    
    for algo_name, pred_labels in pred_labels_dict.items():
        ax_idx += 1
        ax = axes[ax_idx]
        
        accuracy = np.mean(true_labels == pred_labels)
        
        for stage in range(5):
            mask = pred_labels == stage
            if np.any(mask):
                ax.fill_between(time_hours, stage - 0.4, stage + 0.4,
                               where=mask, color=stage_colors[stage], alpha=0.8,
                               label=stage_names[stage], step='mid')
        
        # Replace FilterBankTangentSpace+SVM with FBTS-SVM in algorithm name
        display_name = algo_name.replace('FilterBankTangentSpace+SVM', 'FBTS-SVM')
        ax.set_ylabel(f'{display_name}\n({accuracy:.1%})', fontsize=10)
        ax.set_yticks(range(5))
        ax.set_yticklabels([stage_names[i] for i in range(5)])
        ax.set_ylim(-0.5, 4.5)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
    
    axes[-1].set_xlabel('Time (hours)', fontsize=11)
    
    if subject_id is not None:
        title = f'Sleep Hypnogram Comparison - Subject {subject_id}'
    else:
        title = 'Sleep Hypnogram Comparison'
    fig.suptitle(title, fontsize=13, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Hypnogram comparison saved to {save_path}")
    plt.close()


def plot_per_stage_f1_scores(results_df, algorithm_names=None, save_path=None, figsize=(12, 6)):
    """
    Plot per-stage F1 scores for multiple algorithms.
    
    Args:
        results_df: DataFrame with columns ['algorithm', 'fold', 'confusion_matrix'] or
                   pre-computed per-stage F1 scores
        algorithm_names: List of algorithms to plot (default: all)
        save_path: Path to save the figure
        figsize: Figure size (width, height)
    """
    import ast
    from sklearn.metrics import f1_score
    
    stage_names = ['W', 'N1', 'N2', 'N3', 'REM']
    stage_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#2C3E50', '#9B59B6']
    
    if algorithm_names is None:
        algorithm_names = results_df['algorithm'].unique().tolist()
    else:
        algorithm_names = [a for a in algorithm_names if a in results_df['algorithm'].values]
    
    if len(algorithm_names) == 0:
        print("Warning: No matching algorithms found for per-stage F1 plot.")
        return
    
    per_stage_f1 = {algo: {stage: [] for stage in range(5)} for algo in algorithm_names}
    
    for algo_name in algorithm_names:
        algo_results = results_df[results_df['algorithm'] == algo_name]
        
        for _, row in algo_results.iterrows():
            cm_str = row['confusion_matrix']
            try:
                if isinstance(cm_str, str):
                    cm = np.array(ast.literal_eval(cm_str))
                else:
                    cm = np.array(cm_str)
                
                y_true = []
                y_pred = []
                for true_class in range(cm.shape[0]):
                    for pred_class in range(cm.shape[1]):
                        count = int(cm[true_class, pred_class])
                        y_true.extend([true_class] * count)
                        y_pred.extend([pred_class] * count)
                
                for stage in range(5):
                    y_true_binary = (np.array(y_true) == stage).astype(int)
                    y_pred_binary = (np.array(y_pred) == stage).astype(int)
                    
                    if np.sum(y_true_binary) > 0 and np.sum(y_pred_binary) > 0:
                        f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
                        per_stage_f1[algo_name][stage].append(f1)
            except Exception as e:
                print(f"Warning: Could not process confusion matrix for {algo_name}: {e}")
    
    mean_f1 = {}
    std_f1 = {}
    for algo_name in algorithm_names:
        mean_f1[algo_name] = []
        std_f1[algo_name] = []
        for stage in range(5):
            scores = per_stage_f1[algo_name][stage]
            if len(scores) > 0:
                mean_f1[algo_name].append(np.mean(scores))
                std_f1[algo_name].append(np.std(scores))
            else:
                mean_f1[algo_name].append(0)
                std_f1[algo_name].append(0)
    
    x = np.arange(len(stage_names))
    width = 0.8 / len(algorithm_names)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for i, algo_name in enumerate(algorithm_names):
        display_name = algo_name.replace('FilterBankTangentSpace+SVM', 'FBTS-SVM')
        display_name = display_name.replace('HandcraftedFeatures+RF', 'Handcrafted+RF')
        display_name = display_name.replace('SleepTransformer-Light', 'SleepTrans.')
        display_name = display_name.replace('EEGNet-Light', 'EEGNet')
        
        bars = ax.bar(x + i * width, mean_f1[algo_name], width, 
                     yerr=std_f1[algo_name], capsize=3,
                     label=display_name, alpha=0.8)
    
    ax.set_xlabel('Sleep Stage', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('Per-Stage F1 Scores Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * (len(algorithm_names) - 1) / 2)
    ax.set_xticklabels(stage_names, fontsize=11)
    ax.set_ylim([0, 1.0])
    ax.legend(loc='lower right', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Per-stage F1 scores saved to {save_path}")
        
        # Save data to CSV for later replotting
        import pandas as pd
        csv_path = save_path.replace('.png', '.csv')
        
        # Create DataFrame with mean and std F1 scores
        data = []
        for algo_name in algorithm_names:
            display_name = algo_name.replace('FilterBankTangentSpace+SVM', 'FBTS-SVM')
            display_name = display_name.replace('HandcraftedFeatures+RF', 'Handcrafted+RF')
            display_name = display_name.replace('SleepTransformer-Light', 'SleepTrans.')
            display_name = display_name.replace('EEGNet-Light', 'EEGNet')
            
            for stage_idx, stage_name in enumerate(stage_names):
                data.append({
                    'algorithm': display_name,
                    'stage': stage_name,
                    'mean_f1': mean_f1[algo_name][stage_idx],
                    'std_f1': std_f1[algo_name][stage_idx]
                })
        
        df_f1 = pd.DataFrame(data)
        df_f1.to_csv(csv_path, index=False)
        print(f"Per-stage F1 data saved to {csv_path}")
    
    plt.close()
    
    return mean_f1


def generate_per_stage_f1_table(results_df, algorithm_names=None):
    """
    Generate a LaTeX table for per-stage F1 scores.
    
    Args:
        results_df: DataFrame with confusion matrices
        algorithm_names: List of algorithms to include
        
    Returns:
        LaTeX table string
    """
    import ast
    from sklearn.metrics import f1_score
    
    stage_names = ['W', 'N1', 'N2', 'N3', 'REM']
    
    if algorithm_names is None:
        algorithm_names = results_df['algorithm'].unique().tolist()
    
    per_stage_f1 = {algo: {stage: [] for stage in range(5)} for algo in algorithm_names}
    
    for algo_name in algorithm_names:
        algo_results = results_df[results_df['algorithm'] == algo_name]
        
        for _, row in algo_results.iterrows():
            cm_str = row['confusion_matrix']
            try:
                if isinstance(cm_str, str):
                    cm = np.array(ast.literal_eval(cm_str))
                else:
                    cm = np.array(cm_str)
                
                y_true = []
                y_pred = []
                for true_class in range(cm.shape[0]):
                    for pred_class in range(cm.shape[1]):
                        count = int(cm[true_class, pred_class])
                        y_true.extend([true_class] * count)
                        y_pred.extend([pred_class] * count)
                
                for stage in range(5):
                    y_true_binary = (np.array(y_true) == stage).astype(int)
                    y_pred_binary = (np.array(y_pred) == stage).astype(int)
                    
                    if np.sum(y_true_binary) > 0 and np.sum(y_pred_binary) > 0:
                        f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
                        per_stage_f1[algo_name][stage].append(f1)
            except Exception:
                pass
    
    latex = "\\begin{table}[htbp]\n"
    latex += "\\centering\n"
    latex += "\\caption{Per-Stage F1 Scores (Mean $\\pm$ Std)}\n"
    latex += "\\label{tab:per_stage_f1}\n"
    latex += "\\begin{tabular}{l" + "c" * 5 + "}\n"
    latex += "\\toprule\n"
    latex += "Algorithm & W & N1 & N2 & N3 & REM \\\\\n"
    latex += "\\midrule\n"
    
    for algo_name in algorithm_names:
        display_name = algo_name.replace('FilterBankTangentSpace+SVM', 'FBTS-SVM')
        display_name = display_name.replace('HandcraftedFeatures+RF', 'Handcrafted+RF')
        display_name = display_name.replace('SleepTransformer-Light', 'SleepTrans.')
        
        scores = []
        for stage in range(5):
            stage_scores = per_stage_f1[algo_name][stage]
            if len(stage_scores) > 0:
                mean_val = np.mean(stage_scores)
                std_val = np.std(stage_scores)
                scores.append(f"{mean_val:.3f}$\\pm${std_val:.3f}")
            else:
                scores.append("-")
        
        latex += f"{display_name} & " + " & ".join(scores) + " \\\\\n"
    
    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}"
    
    return latex


def replot_per_stage_f1(csv_path, output_path=None, figsize=(12, 6)):
    """
    Replot per-stage F1 scores from CSV data.
    
    Args:
        csv_path: Path to the CSV file containing F1 scores
        output_path: Path to save the figure (default: same name as CSV but .png)
        figsize: Figure size (width, height)
    """
    # Read CSV data
    df = pd.read_csv(csv_path)
    
    # Get unique algorithms and stages
    algorithms = df['algorithm'].unique()
    stage_names = ['W', 'N1', 'N2', 'N3', 'REM']
    stage_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#2C3E50', '#9B59B6']
    
    x = np.arange(len(stage_names))
    width = 0.8 / len(algorithms)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for i, algo_name in enumerate(algorithms):
        algo_data = df[df['algorithm'] == algo_name]
        # Ensure correct stage order
        algo_data = algo_data.set_index('stage').reindex(stage_names).reset_index()
        
        mean_f1 = algo_data['mean_f1'].values
        std_f1 = algo_data['std_f1'].values
        
        bars = ax.bar(x + i * width, mean_f1, width, 
                     yerr=std_f1, capsize=3,
                     label=algo_name, alpha=0.8)
    
    ax.set_xlabel('Sleep Stage', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('Per-Stage F1 Scores Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * (len(algorithms) - 1) / 2)
    ax.set_xticklabels(stage_names, fontsize=11)
    ax.set_ylim([0, 1.0])
    ax.legend(loc='lower right', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Determine output path
    if output_path is None:
        output_path = str(csv_path).replace('.csv', '_replotted.png')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {output_path}")
    
    plt.close()


def replot_noise_robustness(csv_path, output_dir=None):
    """
    Replot noise robustness charts from CSV data.
    
    Args:
        csv_path: Path to the CSV file containing noise robustness results
        output_dir: Directory to save the figures (default: same directory as CSV)
    """
    # Read CSV data
    results_df = pd.read_csv(csv_path)
    
    # Determine output directory
    if output_dir is None:
        output_dir = Path(csv_path).parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract timestamp from CSV filename
    csv_filename = Path(csv_path).stem
    timestamp = csv_filename.replace('noise_robustness_results_', '')
    
    # Set plot style
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.linewidth'] = 1.2
    
    noise_levels = ['clean', 'low', 'medium', 'high']
    noise_labels = ['Clean', 'Low', 'Medium', 'High']
    
    algorithms = results_df['algorithm'].unique()
    
    # Plot 1: Line plots with error bars
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(algorithms)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
    
    for idx, (metric, title) in enumerate([
        ('accuracy_mean', 'Accuracy'),
        ('kappa_mean', 'Kappa'),
        ('macro_f1_mean', 'Macro-F1')
    ]):
        ax = axes[idx]
        
        for i, algo in enumerate(algorithms):
            algo_data = results_df[results_df['algorithm'] == algo]
            
            x_pos = [noise_levels.index(nl) for nl in algo_data['noise_level']]
            y_vals = [algo_data[algo_data['noise_level'] == nl][metric].values[0] 
                     if nl in algo_data['noise_level'].values else np.nan 
                     for nl in noise_levels]
            y_std = [algo_data[algo_data['noise_level'] == nl]['accuracy_std'].values[0] 
                    if nl in algo_data['noise_level'].values else 0 
                    for nl in noise_levels]
            
            ax.errorbar(range(4), y_vals, yerr=y_std, 
                       label=algo, color=colors[i], marker=markers[i % len(markers)],
                       linewidth=2, markersize=8, capsize=3)
        
        ax.set_xlabel('Noise Level', fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        ax.set_xticks(range(4))
        ax.set_xticklabels(noise_labels)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.3, 3.3)
    
    # Remove individual legends and add a single legend at the bottom
    legend_handles = []
    legend_labels = []
    
    for i, algo in enumerate(algorithms):
        handle = plt.Line2D([0], [0], marker=markers[i % len(markers)], color=colors[i], 
                           linestyle='-', linewidth=2, markersize=8)
        legend_handles.append(handle)
        legend_labels.append(algo)
    
    # Add single legend at the bottom, below the subplots
    fig.legend(handles=legend_handles, labels=legend_labels, 
               loc='lower center', fontsize=9, ncol=len(algorithms), 
               bbox_to_anchor=(0.5, -0.15))
    plt.subplots_adjust(bottom=0.25)
    
    plt.tight_layout()
    
    curve_path = output_dir / f'noise_robustness_curve_{timestamp}_replotted.png'
    plt.savefig(curve_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Curve plot saved to: {curve_path}")
    
    # Plot 2: Bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    pivot_df = results_df.pivot(index='algorithm', columns='noise_level', values='accuracy_mean')
    pivot_df = pivot_df[['clean', 'low', 'medium', 'high']]
    
    x = np.arange(len(pivot_df.index))
    width = 0.18
    
    bar_colors = ['#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']
    
    for i, noise_level in enumerate(['clean', 'low', 'medium', 'high']):
        bars = ax.bar(x + i * width, pivot_df[noise_level], width, 
                     label=noise_level.capitalize(), alpha=0.8, color=bar_colors[i])
    
    ax.set_xlabel('Algorithm', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(pivot_df.index, rotation=45, ha='right')
    # Legend moved to bottom center, horizontally arranged
    ax.legend(title='Noise Level', loc='lower center', fontsize=9, ncol=4, bbox_to_anchor=(0.5, -0.1))
    plt.subplots_adjust(bottom=0.2)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    bar_path = output_dir / f'noise_robustness_bars_{timestamp}_replotted.png'
    plt.savefig(bar_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Bar chart saved to: {bar_path}")
    
    print(f"\nAll figures replotted successfully!")


def main():
    """
    Command-line interface for visualization functions.
    """
    parser = argparse.ArgumentParser(
        description='Visualization tools for sleep EEG classification results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Replot per-stage F1 scores
    python visualization.py replot-per-stage-f1 --csv results/per_stage_f1_20260403_120000.csv
    
    # Replot noise robustness
    python visualization.py replot-noise-robustness --csv results/noise_robustness_results_20260403_120000.csv
    
    # Custom output for per-stage F1
    python visualization.py replot-per-stage-f1 --csv results/per_stage_f1_20260403_120000.csv --output my_figure.png --width 14 --height 8
        """
    )
    
    subparsers = parser.add_subparsers(title='Commands', dest='command')
    
    # Replot per-stage F1 scores
    parser_per_stage = subparsers.add_parser('replot-per-stage-f1', 
                                             help='Replot per-stage F1 scores from CSV')
    parser_per_stage.add_argument('--csv', type=str, required=True,
                                 help='Path to the CSV file containing F1 scores')
    parser_per_stage.add_argument('--output', type=str, default=None,
                                 help='Output path for the figure (default: same name as CSV but .png)')
    parser_per_stage.add_argument('--width', type=float, default=12,
                                 help='Figure width (default: 12)')
    parser_per_stage.add_argument('--height', type=float, default=6,
                                 help='Figure height (default: 6)')
    
    # Replot noise robustness
    parser_noise = subparsers.add_parser('replot-noise-robustness',
                                         help='Replot noise robustness from CSV')
    parser_noise.add_argument('--csv', type=str, required=True,
                             help='Path to the CSV file containing noise robustness results')
    parser_noise.add_argument('--output-dir', type=str, default=None,
                             help='Output directory for the figures (default: same directory as CSV)')
    
    args = parser.parse_args()
    
    if args.command == 'replot-per-stage-f1':
        replot_per_stage_f1(
            csv_path=args.csv,
            output_path=args.output,
            figsize=(args.width, args.height)
        )
    elif args.command == 'replot-noise-robustness':
        replot_noise_robustness(
            csv_path=args.csv,
            output_dir=args.output_dir
        )
    else:
        parser.print_help()


if __name__ == '__main__':
    import argparse
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path
    main()
