import numpy as np
import pandas as pd

SMOOTHING_WINDOW = 3

# Empirical thresholds — set on this dataset, not cross-validated.
# Use threshold_sweep() to inspect sensitivity before citing in a paper.
ONSET_TRAP_THRESHOLD    = 1.5   # trap:shoulder ratio N× best-form baseline
ONSET_SHOULDER_THRESHOLD = 0.3  # shoulder-derived score (0–1 scale)

ONSET_MIN_REP      = 3  # skip first N reps (sensor settling / warm-up)
ONSET_MIN_DURATION = 2  # threshold must hold for this many consecutive reps


def compute_fatigue_score(feat: pd.DataFrame) -> pd.DataFrame:
    """
    Add fatigue score columns to the per-rep features DataFrame.

    Trap ground truth (the compensation signal):
        trap_score        : per-rep trap:shoulder ratio normalised to p10 baseline
        trap_score_smooth : 3-rep trailing mean (for visualisation)

    Shoulder predictors (independent; no weighting):
        shoulder_trough_score   : normalised rest-time increase (0–1).
                                  >1× baseline trough duration → score rises.
                                  Most directly tied to unconscious rep-pacing.
        shoulder_waveform_score : waveform dissimilarity vs early-set template (0–1).
                                  Rises when movement pattern changes (form breakdown).

    Onset columns:
        trap_onset_rep             : first sustained trap_score >= ONSET_TRAP_THRESHOLD
        shoulder_trough_onset_rep  : first sustained trough_score >= ONSET_SHOULDER_THRESHOLD
        shoulder_waveform_onset_rep: first sustained waveform_score >= ONSET_SHOULDER_THRESHOLD
    """
    out = feat.copy()

    # --- Trap score ---
    out['trap_score'] = out['trap_ratio_norm']
    out['trap_score_smooth'] = (
        out['trap_ratio_norm']
        .rolling(SMOOTHING_WINDOW, min_periods=1, center=False)
        .mean()
    )

    # --- Shoulder trough score ---
    # How much longer is the rest period compared to the early-set baseline?
    # Normalised to 0–1 over the trial.
    trough_above = (out['shoulder_trough_duration_norm'] - 1).clip(lower=0)
    t_max = trough_above.max()
    out['shoulder_trough_score'] = trough_above / t_max if t_max > 1e-6 else trough_above

    # --- Shoulder waveform score ---
    # How different is this rep's movement shape from the early template?
    # waveform_dissimilarity is already 0–1; normalise to trial max.
    if 'waveform_dissimilarity' in out.columns:
        w_max = out['waveform_dissimilarity'].max()
        out['shoulder_waveform_score'] = (
            out['waveform_dissimilarity'] / w_max if w_max > 1e-6
            else out['waveform_dissimilarity']
        )
    else:
        out['shoulder_waveform_score'] = np.nan

    # --- Onset detection ---
    out['trap_onset_rep'] = _first_onset(out['trap_score'], ONSET_TRAP_THRESHOLD)
    out['shoulder_trough_onset_rep'] = _first_onset(
        out['shoulder_trough_score'], ONSET_SHOULDER_THRESHOLD)
    out['shoulder_waveform_onset_rep'] = _first_onset(
        out['shoulder_waveform_score'], ONSET_SHOULDER_THRESHOLD)

    # Sample index for raw-signal overlay (trap only)
    onset = out['trap_onset_rep'].iloc[0]
    out['trap_onset_sample'] = int(out.iloc[onset - 1]['start_sample']) if onset else None

    return out


def _first_onset(score_series: pd.Series, threshold: float) -> int | None:
    """First sustained threshold crossing (ONSET_MIN_DURATION consecutive reps),
    skipping the first ONSET_MIN_REP reps. Returns 1-indexed rep number or None."""
    search = score_series.iloc[ONSET_MIN_REP:].dropna()
    vals = search.values
    for i in range(len(vals) - ONSET_MIN_DURATION + 1):
        if all(vals[i:i + ONSET_MIN_DURATION] >= threshold):
            return int(search.index[i] + 1)
    return None


def threshold_sweep(all_features: dict,
                    score_col: str = 'shoulder_trough_score',
                    trap_thresholds: np.ndarray | None = None,
                    component_thresholds: np.ndarray | None = None) -> pd.DataFrame:
    """
    Sweep trap and component score thresholds across all trials.

    For every (trap_thresh, comp_thresh) pair, re-runs onset detection and records:
        n_valid    : trials where BOTH onsets were detected
        mean_lead  : mean(trap_onset − component_onset); positive = component earlier
        frac_early : fraction of valid trials where component onset <= trap onset
        leads      : list of per-trial leads (for distribution inspection)

    Parameters
    ----------
    all_features        : dict mapping (participant, set_num) → scored features DataFrame
    score_col           : column to sweep (default: shoulder_trough_score)
    trap_thresholds     : 1-D array of trap threshold values to test
    component_thresholds: 1-D array of component threshold values to test
    """
    if trap_thresholds is None:
        trap_thresholds = np.round(np.arange(1.1, 2.55, 0.1), 2)
    if component_thresholds is None:
        component_thresholds = np.round(np.arange(0.05, 0.76, 0.05), 2)

    rows = []
    for t_thresh in trap_thresholds:
        for c_thresh in component_thresholds:
            leads = []
            for feat in all_features.values():
                t_onset = _first_onset(feat['trap_score'], t_thresh)
                s_onset = _first_onset(feat[score_col], c_thresh)
                if t_onset is not None and s_onset is not None:
                    leads.append(t_onset - s_onset)

            rows.append({
                'trap_threshold': t_thresh,
                'component_threshold': c_thresh,
                'n_valid': len(leads),
                'mean_lead': float(np.mean(leads)) if leads else np.nan,
                'frac_early': float(np.mean([l >= 0 for l in leads])) if leads else np.nan,
                'leads': leads,
            })

    return pd.DataFrame(rows)


def trial_summary(feat: pd.DataFrame, participant: str, trial: str) -> dict:
    """Compact per-trial summary dict."""
    n = len(feat)
    t_onset  = feat['trap_onset_rep'].iloc[0]
    tr_onset = feat['shoulder_trough_onset_rep'].iloc[0]
    wf_onset = feat['shoulder_waveform_onset_rep'].iloc[0]
    return {
        'participant': participant,
        'trial': trial,
        'n_reps': n,
        'trap_onset_rep': t_onset,
        'trough_onset_rep': tr_onset,
        'waveform_onset_rep': wf_onset,
        'trough_lead':   (t_onset - tr_onset) if (t_onset and tr_onset) else None,
        'waveform_lead': (t_onset - wf_onset) if (t_onset and wf_onset) else None,
        'final_trap_score':     round(float(feat['trap_score'].iloc[-1]), 3),
        'final_trough_score':   round(float(feat['shoulder_trough_score'].iloc[-1]), 3),
        'final_waveform_score': round(float(feat['shoulder_waveform_score'].iloc[-1])
                                      if not pd.isna(feat['shoulder_waveform_score'].iloc[-1])
                                      else 0.0, 3),
    }
