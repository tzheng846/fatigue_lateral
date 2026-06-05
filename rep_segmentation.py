import numpy as np
import pandas as pd
from scipy.signal import find_peaks, butter, filtfilt
from parse_and_plot import FS

VARIANCE_MULTIPLIER = 3 # active window threshold: N× baseline rolling variance
ENVELOPE_WINDOW_MS = 200  # ms for rolling mean envelope smoothing
MIN_REP_SECS = 1.5       # minimum seconds between rep peaks
MIN_REP_HEIGHT_MULT = 1.5 # peak height must exceed N× mean active envelope


def _envelope(signal: np.ndarray, window_ms: int = ENVELOPE_WINDOW_MS) -> np.ndarray:
    """Rectified, smoothed envelope centred on the signal's resting level.

    Uses the 5th percentile as the resting-level estimate — robust to the
    initial settling drift and to peak outliers.  During a lateral raise the
    arm is at rest (tape slack) most of the time, so p5 ≈ the true trough.
    """
    rest_level = float(np.percentile(signal, 5))
    centered = signal - rest_level
    window = int(window_ms * FS / 1000)
    return pd.Series(np.abs(centered)).rolling(window, center=True, min_periods=1).mean().values


def detect_active_period(df: pd.DataFrame) -> tuple[int, int]:
    """
    Find the sample indices where the subject starts and stops doing reps.
    Uses rolling variance of the shoulder envelope vs. quiet baseline.
    Returns (start_sample, end_sample).

    Baseline is estimated from the quietest 1s window within the first 8s.
    This is robust to an initial settling drift in the first few seconds
    (person standing still while sensor resistance stabilises).
    """
    shoulder_avg = (df['right_shoulder'].values + df['left_shoulder'].values) / 2

    env = _envelope(shoulder_avg)
    rolling_var = pd.Series(env).rolling(int(0.25 * FS), min_periods=1).var().fillna(0).values

    # Find the quietest 1s window within the first 8s; record where it is
    scan_n = max(1, min(8 * FS, len(rolling_var) - FS))
    win = FS  # 1s window
    stride = win // 2
    best_mean = float('inf')
    baseline_window_start = 0
    for i in range(0, scan_n, stride):
        w_mean = float(np.mean(rolling_var[i:i + win]))
        if w_mean < best_mean:
            best_mean = w_mean
            baseline_window_start = i
    baseline_var = best_mean + 1e-9

    active_mask = rolling_var > VARIANCE_MULTIPLIER * baseline_var

    all_starts = np.where(np.diff(active_mask.astype(int)) == 1)[0]
    ends = np.where(np.diff(active_mask.astype(int)) == -1)[0]

    if len(all_starts) == 0:
        return 0, len(df) - 1

    # Only look for the active start AFTER the baseline window ends.
    # This skips the settling drift (which also looks high-variance) and finds
    # the genuine transition from quiet→exercise.
    baseline_window_end = baseline_window_start + win
    valid_starts = all_starts[all_starts >= baseline_window_end]
    if len(valid_starts) == 0:
        valid_starts = all_starts  # fallback if exercise starts within first 8s

    start = max(0, int(valid_starts[0]) - int(0.5 * FS))  # 0.5s buffer before

    # active_end: use the last inactive transition only if the signal genuinely stays
    # quiet afterwards (= end of exercise).  A brief inter-rep pause creates a
    # transition at near-zero threshold but the signal immediately goes active again;
    # in that case fall back to the full signal so rep segmentation isn't truncated.
    valid_ends = ends[ends > start]
    end = len(df) - 1
    if len(valid_ends) > 0:
        last_end = int(valid_ends[-1])
        post_window = active_mask[last_end:min(last_end + 2 * FS, len(active_mask))]
        if len(post_window) > 0 and float(np.mean(post_window)) < 0.3:
            end = min(last_end + int(0.5 * FS), len(df) - 1)

    return int(start), int(end)


def segment_reps(df: pd.DataFrame, active_start: int, active_end: int,
                 min_rep_secs: float = MIN_REP_SECS) -> list[tuple[int, int, int]]:
    """
    Detect individual rep cycles within the active window.
    Returns list of (start_sample, peak_sample, end_sample) tuples.
    Uses the average of both shoulder channels for detection.
    Auto-detects polarity (contractions = peaks or troughs).
    """
    active_df = df.iloc[active_start:active_end]
    shoulder_avg = (active_df['right_shoulder'].values + active_df['left_shoulder'].values) / 2

    env = _envelope(shoulder_avg)
    min_distance = int(min_rep_secs * FS)

    # Try detecting peaks (contraction = resistance increase)
    mean_env = np.mean(env)
    peaks_up, _ = find_peaks(env, distance=min_distance,
                              height=MIN_REP_HEIGHT_MULT * mean_env)
    # Try detecting troughs (contraction = resistance decrease → envelope of inverted signal)
    inv_env = _envelope(-shoulder_avg + np.mean(shoulder_avg))
    peaks_down, _ = find_peaks(inv_env, distance=min_distance,
                                height=MIN_REP_HEIGHT_MULT * np.mean(inv_env))

    peak_indices = peaks_up if len(peaks_up) >= len(peaks_down) else peaks_down
    global_peaks = peak_indices + active_start

    if len(global_peaks) == 0:
        return []

    reps = []
    for k, peak in enumerate(global_peaks):
        half_gap = min_distance // 2
        start = int(global_peaks[k - 1] + half_gap) if k > 0 else max(active_start, peak - half_gap)
        end = int(global_peaks[k + 1] - half_gap) if k < len(global_peaks) - 1 else min(active_end, peak + half_gap)
        reps.append((start, peak, end))

    return reps


def get_reps(df: pd.DataFrame) -> tuple[list[tuple[int, int, int]], int, int]:
    """
    Full pipeline: detect active period, then segment reps.
    Returns (reps, active_start, active_end).
    """
    active_start, active_end = detect_active_period(df)
    reps = segment_reps(df, active_start, active_end)
    return reps, active_start, active_end
