import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

COLUMNS = ['right_shoulder', 'right_trap', 'left_shoulder', 'left_trap']
FS = 1000  # Hz


def parse_csv(file_path):
    with open(file_path) as f:
        lines = f.read().strip().split('\n')

    data_start = None
    for i, line in enumerate(lines):
        parts = line.split(',')
        try:
            [float(p) for p in parts if p.strip()]
            if len([p for p in parts if p.strip()]) >= 6:
                data_start = i
                break
        except ValueError:
            continue

    if data_start is None:
        raise ValueError("Could not find data rows in file.")

    rows = []
    for line in lines[data_start:]:
        parts = line.split(',')
        numeric = [float(p) for p in parts if p.strip()]
        if len(numeric) >= 6:
            rows.append(numeric[2:6])  # ch1-4: right_shoulder, right_trap, left_shoulder, left_trap

    return pd.DataFrame(rows, columns=COLUMNS)


def convert_volt_to_ohms(df: pd.DataFrame) -> pd.DataFrame:
    """Voltage divider → resistance: R = (V * 1000) / (1 - V/5)"""
    result = df.copy()
    for col in COLUMNS:
        if col in result.columns:
            v = result[col]
            result[col] = (v * 1000) / (1 - v / 5)
    return result


def apply_lowpass_filter(df: pd.DataFrame, cutoff_hz: float = 5.0, order: int = 4) -> pd.DataFrame:
    """Zero-phase Butterworth low-pass filter."""
    result = df.copy()
    nyq = FS / 2
    b, a = butter(order, cutoff_hz / nyq, btype='low')
    for col in COLUMNS:
        if col in result.columns:
            result[col] = filtfilt(b, a, result[col].values)
    return result


def load_trial(file_path, filter_data: bool = True) -> pd.DataFrame:
    """Load a trial CSV and return a resistance (Ω) DataFrame at 1000 Hz."""
    df = parse_csv(file_path)
    df = convert_volt_to_ohms(df)
    if filter_data:
        df = apply_lowpass_filter(df)
    df.index = pd.RangeIndex(len(df))
    return df


def plot(df, output_path=None, rep_boundaries=None, title=None):
    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    axes = axes.flatten()
    time = df.index / FS

    for i, col in enumerate(COLUMNS):
        ax = axes[i]
        ax.plot(time, df[col], linewidth=1.2, color=colors[i])
        ax.fill_between(time, df[col], alpha=0.08, color=colors[i])
        if rep_boundaries:
            for start, peak, end in rep_boundaries:
                ax.axvline(start / FS, color='gray', linewidth=0.6, alpha=0.5)
        ax.set_title(col.replace('_', ' ').title(), fontsize=11, fontweight='bold', color=colors[i])
        ax.set_ylabel('Resistance (Ω)', fontsize=9, color='#444')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(colors='#555', labelsize=9)
        ax.set_facecolor('#f9f9f9')
        if i >= 2:
            ax.set_xlabel('Time (s)', fontsize=9, color='#444')

    if title:
        fig.suptitle(title, fontsize=13, fontweight='bold', y=1.01)
    fig.patch.set_facecolor('white')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {output_path}")
    return fig


def main():
    parser = argparse.ArgumentParser(description='Parse and plot motion tape CSV.')
    parser.add_argument('file', help='Path to the CSV file')
    parser.add_argument('-o', '--output', help='Output image path (default: <file>.png)')
    parser.add_argument('--no-filter', action='store_true', help='Skip low-pass filter')
    args = parser.parse_args()

    input_path = Path(args.file)
    output_path = args.output or input_path.with_suffix('.png')

    df = load_trial(input_path, filter_data=not args.no_filter)
    print(df.describe().round(1))
    plot(df, output_path, title=input_path.stem)


if __name__ == '__main__':
    main()
