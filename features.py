import numpy as np
import pandas as pd
from parse_and_plot import FS

BASELINE_SKIP = 1   # skip first rep (truncated trough at active window boundary)
BASELINE_REPS = 3   # number of reps used for normalization baseline
TEMPLATE_LEN = 100  # samples for waveform resampling


# ---------------------------------------------------------------------------
# Low-level feature helpers
# ---------------------------------------------------------------------------

def _rep_amplitude(signal: np.ndarray) -> float:
    """Peak-to-trough amplitude within a rep segment."""
    return float(np.max(signal) - np.min(signal))


def _rise_time(signal: np.ndarray) -> float:
    """Concentric phase duration (seconds): time from 10% to 90% of range.

    Finds the trough within the window first so the measurement always starts
    from the actual bottom of the rep, not from wherever the window edge falls.
    """
    trough_idx = int(np.argmin(signal))
    post = signal[trough_idx:]
    lo, hi = float(post.min()), float(post.max())
    span = hi - lo
    if span < 1e-6:
        return np.nan
    above_10 = np.where(post >= lo + 0.1 * span)[0]
    above_90 = np.where(post >= lo + 0.9 * span)[0]
    if not len(above_10) or not len(above_90):
        return np.nan
    return float((above_90[0] - above_10[0]) / FS)


def _trough_duration(signal: np.ndarray) -> float:
    """Total time (seconds) the signal spends in the bottom 20% of its range.

    With midpoint-split rep windows, the trough is split between the start and
    end of the window — summing all below-threshold samples captures the full
    rest period regardless of where it falls within the window.
    """
    lo, hi = float(signal.min()), float(signal.max())
    threshold = lo + 0.2 * (hi - lo)
    return float(np.sum(signal <= threshold) / FS)


def _trough_level(signal: np.ndarray) -> float:
    """Mean resistance during the bottom 20% of the rep window.

    If trap compensation keeps the shoulder slightly elevated between reps,
    this level drifts upward across the set.
    """
    lo, hi = float(signal.min()), float(signal.max())
    threshold = lo + 0.2 * (hi - lo)
    at_bottom = signal[signal <= threshold]
    return float(np.mean(at_bottom)) if len(at_bottom) > 0 else lo


def _resample(signal: np.ndarray, n: int = TEMPLATE_LEN) -> np.ndarray:
    """Resample signal to a fixed length for template comparison."""
    return np.interp(
        np.linspace(0, 1, n),
        np.linspace(0, 1, len(signal)),
        signal,
    )


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------

def extract_per_rep_features(df: pd.DataFrame,
                              reps: list[tuple[int, int, int]],
                              active_start: int = 0) -> pd.DataFrame:
    """
    Compute per-rep features for shoulder and trap channels.
    Returns a DataFrame with one row per rep.

    Parameters
    ----------
    df            : full trial DataFrame (all channels, all samples)
    reps          : list of (start_sample, peak_sample, end_sample) per rep
    active_start  : first sample of the active window (used for rep-1 interval)

    Features
    --------
    Shoulder (expected flat amplitude; fatigue shows in timing/form):
        shoulder_amplitude        – peak-to-trough per rep (expect ~constant)
        rs_amplitude / ls_amplitude – per-side amplitudes (for asymmetry)
        shoulder_rise_time        – concentric phase duration (arm going up)
        shoulder_trough_duration  – time at rest within rep window
        shoulder_trough_level     – resting resistance level
        inter_rep_interval_s      – time between consecutive peaks (full cycle)
        waveform_corr             – Pearson r vs first-3-rep mean template (0–1)
        waveform_dissimilarity    – 1 − waveform_corr (0 = identical, 1 = different)

    Trap (expected rising amplitude; the direct fatigue/compensation signal):
        trap_amplitude            – peak-to-trough per rep (expect rising)
        rt_amplitude / lt_amplitude – per-side amplitudes
        trap_shoulder_ratio       – trap_amplitude / shoulder_amplitude

    Normalisations (suffix _norm = divided by first-BASELINE_REPS-rep mean):
        shoulder_amplitude_norm, trap_amplitude_norm,
        inter_rep_interval_s_norm, shoulder_rise_time_norm,
        shoulder_trough_duration_norm, shoulder_trough_level_norm,
        trap_ratio_norm (normalised to p10 of the ratio distribution)
    """
    if len(reps) == 0:
        return pd.DataFrame()

    # Build waveform template from BASELINE_REPS reps, skipping the first
    # (rep 0 often has a truncated trough at the active window boundary)
    shoulder_raw = (df['right_shoulder'].values + df['left_shoulder'].values) / 2
    template_segs = []
    for start, _, end in reps[BASELINE_SKIP:BASELINE_SKIP + BASELINE_REPS]:
        seg = shoulder_raw[start:end]
        if len(seg) > 1:
            template_segs.append(_resample(seg))
    template = np.mean(template_segs, axis=0) if template_segs else None

    records = []
    prev_peak = active_start  # for first rep's inter-rep interval

    for rep_idx, (start, peak, end) in enumerate(reps):
        seg = df.iloc[start:end]

        rs = seg['right_shoulder'].values
        ls = seg['left_shoulder'].values
        rt = seg['right_trap'].values
        lt = seg['left_trap'].values
        shoulder_avg = (rs + ls) / 2
        trap_avg = (rt + lt) / 2

        # Waveform correlation vs early-set template
        if template is not None and len(shoulder_avg) > 1:
            r_seg = _resample(shoulder_avg)
            if np.std(r_seg) > 1e-6 and np.std(template) > 1e-6:
                waveform_corr = float(np.corrcoef(r_seg, template)[0, 1])
            else:
                waveform_corr = 1.0
        else:
            waveform_corr = np.nan

        records.append({
            'rep': rep_idx + 1,
            'start_sample': start,
            'peak_sample': peak,
            'end_sample': end,
            'duration_s': (end - start) / FS,

            # Timing features
            'inter_rep_interval_s': (peak - prev_peak) / FS,

            # Shoulder features
            'rs_amplitude': _rep_amplitude(rs),
            'ls_amplitude': _rep_amplitude(ls),
            'shoulder_amplitude': _rep_amplitude(shoulder_avg),
            'shoulder_rise_time': _rise_time(shoulder_avg),
            'shoulder_trough_duration': _trough_duration(shoulder_avg),
            'shoulder_trough_level': _trough_level(shoulder_avg),
            'waveform_corr': waveform_corr,
            'waveform_dissimilarity': max(0.0, 1.0 - waveform_corr) if not np.isnan(waveform_corr) else np.nan,

            # Trap features
            'rt_amplitude': _rep_amplitude(rt),
            'lt_amplitude': _rep_amplitude(lt),
            'trap_amplitude': _rep_amplitude(trap_avg),

            # Form quality
            'trap_shoulder_ratio': _rep_amplitude(trap_avg) / (_rep_amplitude(shoulder_avg) + 1e-6),
        })
        prev_peak = peak

    feat = pd.DataFrame(records)
    n_base_start = min(BASELINE_SKIP, len(feat))
    n_base_end = min(BASELINE_SKIP + BASELINE_REPS, len(feat))

    # --- trap:shoulder ratio → p10 normalisation (robust to 1-2 outlier reps) ---
    p10 = np.percentile(feat['trap_shoulder_ratio'], 10)
    feat['trap_ratio_norm'] = feat['trap_shoulder_ratio'] / p10 if p10 > 1e-6 else 1.0

    # --- Baseline normalisation: skip rep 0, use reps BASELINE_SKIP…BASELINE_SKIP+BASELINE_REPS ---
    for col in [
        'shoulder_amplitude', 'trap_amplitude',
        'inter_rep_interval_s',
        'shoulder_rise_time',
        'shoulder_trough_duration',
        'shoulder_trough_level',
    ]:
        base = feat[col].iloc[n_base_start:n_base_end].mean()
        feat[f'{col}_norm'] = feat[col] / base if base > 1e-6 else pd.Series(1.0, index=feat.index)

    return feat
