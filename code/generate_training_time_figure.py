#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate training time comparison figure from experiment results.

Usage:
    python generate_training_time_figure.py --csv results/sleep_edf/sleep_evaluation_summary_*.csv
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def generate_training_time_figure(csv_path, output_dir=None):
    """Generate training time comparison figure from CSV data."""
    df = pd.read_csv(csv_path)
    
    if output_dir is None:
        output_dir = Path(csv_path).parent
    else:
        output_dir = Path(output_dir)
    
    specified_order = ['MDM', 'RiemannTangentSpace', 'HandcraftedFeatures+RF', 
                       'DeepSleepNet', 'SSC-SleepNet', 'TinySleepNet', 'SCA-FBTS']
    
    df_filtered = df[df['algorithm'].isin(specified_order)].copy()
    df_filtered['algorithm_order'] = df_filtered['algorithm'].apply(
        lambda x: specified_order.index(x) if x in specified_order else len(specified_order)
    )
    df_filtered = df_filtered.sort_values('algorithm_order').drop('algorithm_order', axis=1)
    
    algorithms = df_filtered['algorithm'].values
    train_times = df_filtered['train_time_mean'].values
    
    display_names = {
        'SCA-FBTS': 'SCA-FBTS\n(Ours)',
        'SSC-SleepNet': 'SSC-SleepNet',
        'HandcraftedFeatures+RF': 'Handcrafted\n+RF',
        'DeepSleepNet': 'DeepSleepNet',
        'TinySleepNet': 'TinySleepNet',
        'RiemannTangentSpace': 'Riemann\nTangentSpace',
        'MDM': 'MDM'
    }
    
    colors = {
        'SCA-FBTS': '#FF0000',
        'SSC-SleepNet': '#5F27CD',
        'HandcraftedFeatures+RF': '#96CEB4',
        'DeepSleepNet': '#AA6B6B',
        'TinySleepNet': '#54A0FF',
        'RiemannTangentSpace': '#FF9FF3',
        'MDM': '#FECA57'
    }
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x_pos = np.arange(len(algorithms))
    bars = ax.bar(x_pos, train_times, color=[colors[a] for a in algorithms], 
                  edgecolor='black', linewidth=1.2)
    
    for i, (bar, time) in enumerate(zip(bars, train_times)):
        height = bar.get_height()
        if time < 10:
            label = f'{time:.1f}s'
        else:
            label = f'{time:.0f}s'
        if algorithms[i] == 'SCA-FBTS':
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    label, ha='center', va='bottom', fontsize=10, fontweight='bold', color='red')
        else:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    label, ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Algorithm', fontsize=12, fontweight='bold')
    ax.set_ylabel('Training Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Training Time Comparison on Sleep-EDF Dataset', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([display_names.get(a, a) for a in algorithms], fontsize=10)
    ax.set_yscale('log')
    ax.set_ylim(1, 15000)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#FF0000', edgecolor='black', label='SCA-FBTS (Ours)'),
        Patch(facecolor='#54A0FF', edgecolor='black', label='Deep Learning'),
        Patch(facecolor='#96CEB4', edgecolor='black', label='Traditional ML'),
        Patch(facecolor='#FECA57', edgecolor='black', label='Riemannian')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    plt.tight_layout()
    
    output_path = output_dir / 'training_time_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Training time comparison figure saved to: {output_path}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Generate training time comparison figure from experiment results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
        python generate_training_time_figure.py --csv results/sleep_edf/sleep_evaluation_summary_sleepedf_20260429_161629.csv
        python generate_training_time_figure.py --csv results/sleep_edf/sleep_evaluation_summary_sleepedf_20260429_161629.csv --output figures/
        """
    )
    
    parser.add_argument('--csv', type=str, required=True,
                        help='Path to the CSV file containing algorithm performance data')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for the figure (default: same as CSV directory)')
    
    args = parser.parse_args()
    
    generate_training_time_figure(args.csv, args.output)


if __name__ == '__main__':
    main()
