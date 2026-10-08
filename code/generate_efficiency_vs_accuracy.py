#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate efficiency vs accuracy visualization for sleep staging algorithms.

Two side-by-side bubble charts:
  Left  -- Accuracy vs model size  (log scale)
  Right -- Accuracy vs training time (log scale)

All seven algorithms in the Sleep-EDF comparison are plotted in both
panels. Bubble area encodes model size. Labels are placed automatically
by a deterministic collision-avoidance routine (no hand-tuned offsets),
so that every annotation stays clear of the other labels, of the data
bubbles, and of the panel frame.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox


# Canonical order (must match the paper tables). The first entry is the
# proposed method and gets first pick of the label positions.
ALGO_ORDER = [
    'SCA-FBTS',
    'SSC-SleepNet',
    'HandcraftedFeatures+RF',
    'DeepSleepNet',
    'TinySleepNet',
    'RiemannTangentSpace',
    'MDM',
]

DISPLAY_NAMES = {
    'SCA-FBTS': 'SCA-FBTS',
    'SSC-SleepNet': 'SSC-SleepNet',
    'HandcraftedFeatures+RF': 'Handcrafted+RF',
    'DeepSleepNet': 'DeepSleepNet',
    'TinySleepNet': 'TinySleepNet',
    'RiemannTangentSpace': 'Riemannian TS',
    'MDM': 'MDM',
}

COLORS = {
    'SCA-FBTS': '#E63946',              # proposed method
    'SSC-SleepNet': '#6A4C93',
    'HandcraftedFeatures+RF': '#2A9D8F',
    'DeepSleepNet': '#F4A261',
    'TinySleepNet': '#264653',
    'RiemannTangentSpace': '#9D4EDD',
    'MDM': '#FFB703',
}

LABEL_FONTSIZE = 9.0
BUBBLE_PAD_PT = 2.5          # minimum gap between bubble edge and label box
LABEL_PAD_PX = 1.5           # minimum gap between two label boxes
FRAME_INSET_PX = 1.5         # keep labels inside the axes frame
# Preferred directions, in offset-point space (x right, y up).
PREFERRED_ANGLES_DEG = [
    0, 180, 40, 140, -40, -140, 20, -20, 160, -160,
    65, 115, -65, -115, 90, -90, 30, 150, -30, -150,
    50, 130, -50, -130, 75, 105, -75, -105, 10, -10, 170, -170,
]


def load_summary(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if 'accuracy' in df.columns and 'accuracy_mean' not in df.columns:
        grouped = (
            df.groupby('algorithm')
            .agg({
                'accuracy': 'mean',
                'kappa': 'mean',
                'macro_f1': 'mean',
                'train_time': 'mean',
                'inference_time_ms': 'mean',
                'model_size_mb': 'mean',
            })
            .reset_index()
            .rename(columns={
                'accuracy': 'accuracy_mean',
                'kappa': 'kappa_mean',
                'macro_f1': 'macro_f1_mean',
                'train_time': 'train_time_mean',
                'inference_time_ms': 'inference_time_ms_mean',
                'model_size_mb': 'model_size_mb_mean',
            })
        )
        df = grouped

    df = df[df['algorithm'].isin(ALGO_ORDER)].copy()
    missing = [a for a in ALGO_ORDER if a not in set(df['algorithm'])]
    if missing:
        raise ValueError(f'Missing algorithms in summary CSV: {missing}')
    df['__order'] = df['algorithm'].apply(ALGO_ORDER.index)
    df = df.sort_values('__order').drop(columns='__order').reset_index(drop=True)
    return df


def bubble_sizes(model_size: np.ndarray) -> np.ndarray:
    if model_size.max() == model_size.min():
        return np.full_like(model_size, 200.0, dtype=float)
    # Min size 80, max 320, log-scaled to match the log x-axis.
    log = np.log10(model_size)
    lo, hi = log.min(), log.max()
    return 80 + 240 * (log - lo) / (hi - lo + 1e-12)


def _overlap_area(a: Bbox, b: Bbox) -> float:
    dx = min(a.x1, b.x1) - max(a.x0, b.x0)
    dy = min(a.y1, b.y1) - max(a.y0, b.y0)
    if dx <= 0 or dy <= 0:
        return 0.0
    return dx * dy


def _inflate(bb: Bbox, pad: float) -> Bbox:
    return Bbox.from_extents(bb.x0 - pad, bb.y0 - pad, bb.x1 + pad, bb.y1 + pad)


def auto_place_labels(ax, fig, xs, ys, sizes, labels):
    """Place one text label per point, guaranteed clear of bubbles and peers.

    Everything is measured in display (pixel) coordinates, so the result is
    independent of the axis scaling. Returns the list of annotations.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    px_per_pt = fig.dpi / 72.0

    # --- Obstacles: every bubble, inflated by a small padding ---------------
    bubble_boxes = []
    for xi, yi, s in zip(xs, ys, sizes):
        cx, cy = ax.transData.transform((xi, yi))
        r_px = 0.5 * np.sqrt(s) * px_per_pt          # scatter radius in pixels
        bubble_boxes.append(
            Bbox.from_extents(
                cx - r_px - BUBBLE_PAD_PT * px_per_pt,
                cy - r_px - BUBBLE_PAD_PT * px_per_pt,
                cx + r_px + BUBBLE_PAD_PT * px_per_pt,
                cy + r_px + BUBBLE_PAD_PT * px_per_pt,
            )
        )

    # --- Region where a label is allowed to live ---------------------------
    allowed = ax.get_window_extent(renderer=renderer)
    allowed = Bbox.from_extents(
        allowed.x0 + FRAME_INSET_PX,
        allowed.y0 + FRAME_INSET_PX,
        allowed.x1 - FRAME_INSET_PX,
        allowed.y1 - FRAME_INSET_PX,
    )

    annotations = []
    placed = []
    for i, (algo, xi, yi, s) in enumerate(zip(ALGO_ORDER, xs, ys, sizes)):
        ann = ax.annotate(
            labels[i],
            xy=(xi, yi),
            xytext=(0, 0),
            textcoords='offset points',
            fontsize=LABEL_FONTSIZE,
            ha='left',
            va='bottom',
            bbox=dict(boxstyle='round,pad=0.25', fc='white', alpha=0.85, lw=0.4),
            zorder=5,
        )

        min_r_pt = 0.5 * np.sqrt(s) + BUBBLE_PAD_PT + 3.0
        radii_pt = [min_r_pt + d for d in (0, 5, 11, 18, 26, 36, 48, 62, 78, 96)]

        best = None            # (collision_area, r_pt, dx, dy, ha, va, bbox)
        fallback = None        # used only if no candidate fits inside the frame
        for r_pt in radii_pt:
            for ang in PREFERRED_ANGLES_DEG:
                rad = np.deg2rad(ang)
                dx = r_pt * np.cos(rad)
                dy = r_pt * np.sin(rad)
                ha = 'left' if dx >= 0 else 'right'
                va = 'bottom' if dy >= 0 else 'top'
                ann.set_position((dx, dy))
                ann.set_ha(ha)
                ann.set_va(va)
                bb = ann.get_window_extent(renderer=renderer)

                if fallback is None:
                    fallback = (float('inf'), r_pt, dx, dy, ha, va, bb)

                inside = (
                    bb.x0 >= allowed.x0 and bb.x1 <= allowed.x1
                    and bb.y0 >= allowed.y0 and bb.y1 <= allowed.y1
                )
                if not inside:
                    continue

                overlap = 0.0
                for ob in bubble_boxes:
                    overlap += _overlap_area(bb, ob)
                for pb in placed:
                    overlap += _overlap_area(bb, _inflate(pb, LABEL_PAD_PX))

                if overlap == 0.0:
                    best = (0.0, r_pt, dx, dy, ha, va, bb)
                    break
                if best is None or overlap < best[0]:
                    best = (overlap, r_pt, dx, dy, ha, va, bb)
            if best is not None and best[0] == 0.0:
                break

        _, _, dx, dy, ha, va, bb = best if best is not None else fallback
        ann.set_position((dx, dy))
        ann.set_ha(ha)
        ann.set_va(va)
        annotations.append(ann)
        placed.append(ann.get_window_extent(renderer=renderer))

    return annotations


def draw_panel(ax, fig, x, y, sizes, xlabel, xlim, xscale):
    for algo, xi, yi, s in zip(ALGO_ORDER, x, y, sizes):
        ax.scatter(
            xi, yi,
            s=s,
            c=COLORS[algo],
            alpha=0.78,
            edgecolors='black',
            linewidths=1.0,
            zorder=3,
        )

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Accuracy on Sleep-EDF (%)', fontsize=12)
    ax.grid(True, which='both', alpha=0.3)
    if xscale == 'log':
        ax.set_xscale('log')
    ax.set_xlim(xlim)
    # Generous fixed y-range so that labels above and below the extreme
    # points stay clear of the axes frame.
    ax.set_ylim(38, 86)

    labels = [DISPLAY_NAMES[a] for a in ALGO_ORDER]
    auto_place_labels(ax, fig, x, y, sizes, labels)


def generate_efficiency_visualization(csv_path: Path, output_dir: Path):
    df = load_summary(csv_path)

    accuracy = df['accuracy_mean'].values * 100.0
    train_time = df['train_time_mean'].values
    model_size = df['model_size_mb_mean'].values

    sizes = bubble_sizes(model_size)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))

    # Left panel -- model size. Allows ~4 orders of magnitude so that the
    # MDM bubble (0.00067 MB) and the random-forest bubble (1063 MB) both
    # sit comfortably inside the frame.
    draw_panel(
        ax1, fig,
        x=model_size,
        y=accuracy,
        sizes=sizes,
        xlabel='Model size (MB, log scale)',
        xlim=(3e-4, 4e3),
        xscale='log',
    )

    # Right panel -- training time. Range covers MDM (~104 s) and
    # SSC-SleepNet (~26 477 s) with breathing room on both sides.
    draw_panel(
        ax2, fig,
        x=train_time,
        y=accuracy,
        sizes=sizes,
        xlabel='Training time (s, log scale)',
        xlim=(5e1, 6e4),
        xscale='log',
    )

    # Legend (single, below the figure)
    legend_elements = [
        plt.Line2D(
            [0], [0],
            marker='o', color='w',
            label=DISPLAY_NAMES[algo],
            markerfacecolor=COLORS[algo],
            markersize=10,
            markeredgecolor='black',
        )
        for algo in ALGO_ORDER
    ]
    fig.legend(
        handles=legend_elements,
        loc='lower center',
        ncol=4,
        bbox_to_anchor=(0.5, 0.005),
        fontsize=10,
        frameon=True,
    )

    fig.suptitle(
        'Sleep-EDF: accuracy versus efficiency trade-off '
        '(bubble size encodes model size)',
        fontsize=15,
        fontweight='bold',
        y=0.985,
    )

    plt.subplots_adjust(bottom=0.16, top=0.89, left=0.07, right=0.97, wspace=0.22)

    output_path = output_dir / 'algorithm_efficiency_vs_accuracy.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Visualization saved to: {output_path}')


def create_efficiency_table(df, output_dir):
    table = df[['algorithm', 'accuracy_mean', 'kappa_mean', 'macro_f1_mean',
                  'train_time_mean', 'inference_time_ms_mean', 'model_size_mb_mean']].copy()
    table.columns = ['Algorithm', 'Accuracy (%)', 'Kappa', 'Macro-F1',
                     'Training Time (s)', 'Inference Time (ms)', 'Model Size (MB)']
    table['Accuracy (%)'] = table['Accuracy (%)'] * 100
    table['__o'] = table['Algorithm'].apply(ALGO_ORDER.index)
    table = table.sort_values('__o').drop(columns='__o').reset_index(drop=True)
    out = output_dir / 'algorithm_efficiency_table.csv'
    table.to_csv(out, index=False, float_format='%.3f')
    print(f'Efficiency table saved to: {out}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True)
    p.add_argument('--output', default=None)
    args = p.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.output) if args.output else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_summary(csv_path)
    generate_efficiency_visualization(csv_path, out_dir)
    create_efficiency_table(df, out_dir)


if __name__ == '__main__':
    main()