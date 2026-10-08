#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate extended TABLE I with training time and model size information.
"""

import pandas as pd
import sys
import argparse
from pathlib import Path

# Add the project root to path
sys.path.insert(0, str(Path(__file__).parent))

def generate_extended_table(csv_path, output_dir):
    # Read the data
    df = pd.read_csv(csv_path)
    
    # Select relevant columns
    extended_table = df[[
        'algorithm',
        'accuracy_mean',
        'kappa_mean',
        'macro_f1_mean',
        'train_time_mean',
        'model_size_mb_mean'
    ]].copy()
    
    # Rename columns for better readability
    extended_table.columns = [
        'Algorithm',
        'Acc.(%)',
        'Kappa',
        'F1',
        'Training Time (s)',
        'Model Size (MB)'
    ]
    
    # Convert accuracy to percentage
    extended_table['Acc.(%)'] = extended_table['Acc.(%)'] * 100
    
    # Sort by accuracy
    extended_table = extended_table.sort_values('Acc.(%)', ascending=False)
    
    # Move SCA-FBTS to the end and mark as (ours)
    # (main evaluation records the method under either alias)
    fbts_names = ['FilterBankTangentSpace+SVM', 'SCA-FBTS']
    fbts_mask = extended_table['Algorithm'].isin(fbts_names)
    fbts_row = extended_table[fbts_mask]
    if not fbts_row.empty:
        extended_table = extended_table[~fbts_mask]
        fbts_row = fbts_row.copy()
        fbts_row['Algorithm'] = 'SCA-FBTS(ours)'
        extended_table = pd.concat([extended_table, fbts_row], ignore_index=True)
    
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the extended table
    output_path = output_dir / 'extended_table_i.csv'
    extended_table.to_csv(output_path, index=False, float_format='%.3f')
    print(f"Extended TABLE I saved to: {output_path}")
    
    # Generate LaTeX table
    latex_table = generate_latex_table(extended_table)
    latex_path = output_dir / 'extended_table_i.tex'
    with open(latex_path, 'w') as f:
        f.write(latex_table)
    print(f"LaTeX TABLE I saved to: {latex_path}")
    
    return extended_table

def generate_latex_table(df):
    """Generate LaTeX table from DataFrame"""
    latex = r"""\begin{table}[!t]
\caption{Cross-subject results on Sleep-EDF dataset (40 recordings, 5-fold CV). Best values in bold.}
\label{tab:main}
\centering
\footnotesize
\begin{tabular}{lcccccc}
\toprule
Algorithm & Acc.(\%) & Kappa & F1 & Training Time (s) & Model Size (MB) \\
\midrule
"""
    
    # Find best values for bold formatting
    best_acc = df['Acc.(%)'].max()
    best_kappa = df['Kappa'].max()
    best_f1 = df['F1'].max()
    
    for _, row in df.iterrows():
        algo = row['Algorithm']
        acc = row['Acc.(%)']
        kappa = row['Kappa']
        f1 = row['F1']
        train_time = row['Training Time (s)']
        model_size = row['Model Size (MB)']
        
        # Apply bold formatting to best values
        acc_str = f"\\textbf{{{acc:.2f}}}" if abs(acc - best_acc) < 0.01 else f"{acc:.2f}"
        kappa_str = f"\\textbf{{{kappa:.3f}}}" if abs(kappa - best_kappa) < 0.001 else f"{kappa:.3f}"
        f1_str = f"\\textbf{{{f1:.3f}}}" if abs(f1 - best_f1) < 0.001 else f"{f1:.3f}"
        
        # Add row to LaTeX table
        row_str = f"{algo} & {acc_str} & {kappa_str} & {f1_str} & {train_time:.1f} & {model_size:.2f} \\\n"
        latex += row_str
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}"""
    
    return latex

def main():
    parser = argparse.ArgumentParser(description='Generate extended TABLE I')
    parser.add_argument('--csv', type=str, required=True,
                        help='Path to the CSV file containing algorithm performance data')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory for the table')
    
    args = parser.parse_args()
    
    generate_extended_table(args.csv, args.output_dir)

if __name__ == '__main__':
    main()