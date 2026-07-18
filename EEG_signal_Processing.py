"""
 eeg pipline analysis
IMPORTANT — READ BEFORE USING:
──────────────────────────────
1. SATURATION CHECK: ADS1115 values at ±32767/±32768 indicate ADC saturation.
   If your raw CSV shows most samples at these limits, your hardware gain is
   wrong (too low a gain range for the signal amplitude).  Fix on the hardware
   side first.  This pipeline will flag and discard saturated epochs.

2. SAMPLING RATE: Your firmware runs two ADS conversion phases per loop, each
   at 860 SPS single-shot.  Measured real-world Fs is ~215–250 Hz depending on
   I2C overhead.  Measure it empirically using the 'counter' column and set
   FS_HZ below accordingly.  Wrong Fs breaks every frequency-domain result.

3. ADS1115 GAIN: Set ADS_GAIN_RANGE_V to match your firmware's ADS_GAIN
   register value so raw ADC counts are converted to µV correctly.


"""

# ── Standard library ──────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # safe for all OSes; switch to TkAgg if interactive
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from scipy.signal import (
    butter, filtfilt, iirnotch, welch, coherence,
    sosfiltfilt, butter as butter_sos
)
from scipy.stats import zscore
from scipy.interpolate import Rbf
from matplotlib.patches import Polygon
from matplotlib.path import Path

import os
import sys
from datetime import datetime
from pathlib import Path as FilePath

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION — edit these before running
# ═══════════════════════════════════════════════════════════════════════════════

CSV_FILE = "/home/jonah/online reciving/ثبت جدید سالم/awake/awake60min.csv"          # path to your recorded CSV

# ── Sampling rate ─────────────────────────────────────────────────────────────
# Measure empirically: count rows in 1 second of known-duration recording.
# Your firmware targets ~250 Hz but I2C overhead typically gives 215–250 Hz.
FS_HZ = 257.1                           # Hz  ← measure and set this precisely
# ⚠ UNRESOLVED: this value is currently a firmware *target*, not a per-recording
# empirical measurement. Your packet stream carries only an 8-bit counter, no
# wall-clock timestamp, so the true Fs cannot be recovered from this CSV alone.
# validate_sampling_rate() below will warn every run until either (a) the CSV
# gains a host-side receive-timestamp column so Fs can be computed from
# (n_samples-1)/(t_last-t_first), or (b) you supply a per-file empirical Fs
# measured independently (e.g. a timed test recording of known duration).
# Do not report frequency-domain results as final until this is closed.

# ── ADS1115 hardware gain ─────────────────────────────────────────────────────
# CONFIRMED against firmware: ADS_GAIN = 0x0800, BASE_CONFIG_0_1 = 0x81F3,
# BASE_CONFIG_2_3 = 0xB1F3 decode to PGA=+/-0.512V, MUX=AIN0-AIN1 / AIN2-AIN3,
# single-shot mode, DR=860 SPS -- all consistent with the manuscript's stated
# channel wiring and conversion rate. Maps to the PGA setting in your firmware
# (ADS_GAIN register):
#   0x0000 = ±6.144 V  →  187.5 µV/LSB
#   0x0200 = ±4.096 V  →  125.0 µV/LSB   
#   0x0400 = ±2.048 V  →   62.5 µV/LSB
#   0x0600 = ±1.024 V  →   31.25 µV/LSB
#   0x0800 = ±0.512 V  →   15.625 µV/LSB (current firmware default, CONFIRMED)
#   0x0A00 = ±0.256 V  →    7.8125 µV/LSB
# NOTE: the manuscript text currently states 7.8125 µV for this setting, computed
# as FSR/2^16. That divisor is wrong for the ADS1115's signed 16-bit code
# (range -32768..+32767, i.e. 2^15 steps per side) -- the correct value for
# ±0.512V is 15.625 µV/LSB, as used below and as this firmware is actually
# configured. Fix the manuscript number, not this one.
TOTAL_PREAMP_GAIN = 902.0          # INA333 (x2) x OPA2349 stage A (x11) x stage B (x41)

ADS_GAIN_RANGE_V = 0.512
ADS_BITS         = 16
UV_PER_LSB_ADC   = (ADS_GAIN_RANGE_V / (2 ** (ADS_BITS - 1))) * 1e6   # 15.625 µV/count (ADC input)
UV_PER_LSB       = UV_PER_LSB_ADC / TOTAL_PREAMP_GAIN                 # 0.01953 µV/count (electrode)

# ── ADXL345 IMU ───────────────────────────────────────────────────────────────
# Your firmware sets DATA_FORMAT = 0x08 (full-res, ±16g).
# In full-resolution mode, scale factor is always 3.9 mg/LSB regardless of range.
ADXL345_MG_PER_LSB  = 3.9              # mg per raw LSB (full-res mode, ADXL345 datasheet p.4)
ADXL345_G_PER_LSB   = ADXL345_MG_PER_LSB / 1000.0

# Motion artifact rejection: discard any epoch where peak resultant acceleration > threshold.
# See MOTION_MAD_K / MOTION_G_THRESHOLD in the epoching/artifact-rejection section
# below for the actual (robust, data-driven) rejection rule used.

# ── EEG channel names and approximate electrode positions ─────────────────────
# Positions are in normalized head coordinates (AP axis = Y, ML axis = X).
# Adjust to match your actual electrode placement on the rat skull.
EEG_CHANNELS = ['EEG1', 'EEG2', 'EEG3', 'EEG4']
ELECTRODE_POS = np.array([
    [-0.32,  0.42],    # EEG1  — e.g. left frontal
    [ 0.32,  0.42],    # EEG2  — e.g. right frontal
    [-0.28, -0.18],    # EEG3  — e.g. left parietal
    [ 0.28, -0.18],    # EEG4  — e.g. right parietal
])

# ── Frequency bands (standard rodent EEG literature) ─────────────────────────
# References: Buzsaki (2006) "Rhythms of the Brain"; Volk et al. (2016)
FREQ_BANDS = {
    'Delta': (1,   4),
    'Theta': (4,   12),   # rodent theta extends to ~12 Hz (Buzsaki 2002)
    'Beta':  (12,  30),
    'Gamma': (30,  48),
}
# Note: rodent theta = 4–12 Hz, unlike human 4–8 Hz. Adjust to your paradigm.

# ── Preprocessing parameters ──────────────────────────────────────────────────
NOTCH_FREQ_HZ     = 50.0     # power-line frequency (50 Hz for EU/Iran; use 60 for USA)
NOTCH_Q           = 30.0     # quality factor — higher = narrower notch (30–40 typical)
BANDPASS_LOW_HZ   = 1.0      # high-pass cutoff: removes DC drift and slow movement artefacts
# 1-70 Hz is the standard convention for rodent cortical EEG/ECoG (cortical activity
# is broadly accepted to live in ~1-70/80 Hz; below 1 Hz is electrode drift, above
# ~70-80 Hz increasingly overlaps EMG/muscle artifact in skull-surface recordings).
# This also clears the Gamma band's 48 Hz upper edge with margin (was 48 Hz exactly,
# which put the filter's own -3dB point inside the band being measured), and stays
# well under Nyquist (FS/2 = 125 Hz at FS_HZ=250).
BANDPASS_HIGH_HZ  = 50.0     # low-pass cutoff: must be < Nyquist (FS/2)
FILTER_ORDER      = 4        # Butterworth order; 4 is standard for EEG (zero-phase via filtfilt = effective order 8)

# ── Epoching and artifact rejection ──────────────────────────────────────────
EPOCH_SEC         = 10.0      # epoch length (seconds)
EPOCH_OVERLAP     = 0        # fractional overlap between epochs (0 = no overlap)

# Amplitude-based artifact rejection.
# A single fixed µV cutoff picked in advance doesn't generalize across animals,
# electrode impedance, or sessions -- standard EEG/MEG artifact-rejection practice
# (e.g. good-practice guidance in Gross et al. 2013) instead flags epochs whose
# peak amplitude is a robust outlier *relative to that recording's own
# distribution*: median + AMPLITUDE_MAD_K * MAD (median absolute deviation).
# This is the PRIMARY criterion below. AMPLITUDE_THRESH_UV is kept only as an
# absolute sanity ceiling that should never be exceeded regardless of a
# recording's own statistics (e.g. a fully disconnected/floating electrode).
AMPLITUDE_MAD_K     = 5.0    # robust z-score-equivalent multiplier (5 MAD ~ 3.5 SD for Gaussian data)
AMPLITUDE_THRESH_UV = 6000  # µV — absolute ceiling regardless of MAD-based threshold

# Motion artifact rejection: same robust-statistics rationale as amplitude above.
# MOTION_G_THRESHOLD is kept as an absolute ceiling; the data-driven MAD-based
# criterion is the primary rejection rule and adapts to how much a given animal
# actually moves during the session (a 2 g fixed cutoff is closer to an impact/
# fall than typical grooming or ambulatory head acceleration, so on its own it
# would under-reject ordinary movement artifact for most sessions).
MOTION_MAD_K        = 5.0
MOTION_G_THRESHOLD  = 2.0    # g — absolute ceiling

# Saturation rejection: ADS1115 saturates at ±32767 raw counts
SATURATION_THRESH_COUNTS = 32700 # raw counts — slightly below rail to catch near-saturation

# ── Output ────────────────────────────────────────────────────────────────────
OUTPUT_DIR = "EEG_out"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SAVE_FIGURES = True
FIGURE_DPI   = 300          # publication quality
FIGURE_FMT   = "pdf"        # "pdf" for submission, "png" for preview


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 0 — Load and validate data
# ═══════════════════════════════════════════════════════════════════════════════

def load_and_validate(csv_path: str) -> pd.DataFrame:
    """
    Load CSV, validate structure, detect and report saturation.
    Returns the raw DataFrame with a microsecond timestamp column added.
    """
    print("\n" + "═"*60)
    print("  STEP 0 — Load & Validate")
    print("═"*60)

    df = pd.read_csv(csv_path)

    required = ['counter'] + EEG_CHANNELS + ['IMU_X', 'IMU_Y', 'IMU_Z']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    # Reconstruct time axis from counter column.
    # The 'counter' wraps at 256 (uint8 on firmware side).
    # Unwrap it to get a monotonic sample index.
    c = df['counter'].values.astype(np.int32)
    unwrapped = np.zeros(len(c), dtype=np.int64)
    unwrapped[0] = c[0]
    for i in range(1, len(c)):
        diff = (c[i] - c[i-1]) % 256
        unwrapped[i] = unwrapped[i-1] + diff
    df['sample_idx'] = unwrapped
    df['time_sec']   = unwrapped / FS_HZ

    # Detect missing packets (gaps in unwrapped counter)
    gaps = np.where(np.diff(unwrapped) > 1)[0]
    total_missing = int(np.sum(np.diff(unwrapped)[np.diff(unwrapped) > 1] - 1))
    print(f"  Loaded       : {len(df)} samples")
    print(f"  Duration     : {df['time_sec'].iloc[-1]:.1f} s  ({df['time_sec'].iloc[-1]/60:.2f} min)")
    print(f"  Packet gaps  : {len(gaps)} events, {total_missing} missing samples "
          f"({100*total_missing/max(len(df),1):.2f}%)")

    # Saturation check — critical quality gate
    for ch in EEG_CHANNELS:
        sat = np.abs(df[ch].values) >= SATURATION_THRESH_COUNTS
        pct = 100 * sat.mean()
        flag = " ← ⚠ SATURATED — CHECK HARDWARE GAIN" if pct > 5 else ""
        print(f"  {ch} saturated: {pct:.1f}%{flag}")

    # Convert raw counts to µV immediately
    for ch in EEG_CHANNELS:
        df[ch + '_uV'] = df[ch].values * UV_PER_LSB

    # Convert IMU raw counts to g
    for ax in ['X', 'Y', 'Z']:
        df[f'IMU_{ax}_g'] = df[f'IMU_{ax}'].values * ADXL345_G_PER_LSB

    # Resultant acceleration vector magnitude
    df['accel_g'] = np.sqrt(
        df['IMU_X_g']**2 + df['IMU_Y_g']**2 + df['IMU_Z_g']**2
    )
    # Subtract 1g static component (gravity) to get dynamic acceleration
    df['accel_dynamic_g'] = np.abs(df['accel_g'] - 1.0)

    print(f"  Scale factor : {UV_PER_LSB:.4f} µV/LSB  (gain range ±{ADS_GAIN_RANGE_V} V)")
    print(f"  Peak accel   : {df['accel_dynamic_g'].max():.3f} g (dynamic, gravity-subtracted)")

    validate_sampling_rate(df)
    return df


def validate_sampling_rate(df: pd.DataFrame) -> None:
    """
    Cross-check the hardcoded FS_HZ against any host-side receive timestamp
    column, if present. Since the BLE packet itself carries only an 8-bit
    counter (no timestamp, by firmware design), this is the only place Fs can
    be verified from the file. If no timestamp column exists, this prints a
    hard warning rather than silently trusting FS_HZ.
    """
    ts_candidates = [c for c in df.columns
                      if c.lower() in ('timestamp', 'recv_time', 'host_time', 'time')]
    if not ts_candidates:
        print("\n  ⚠  FS_HZ NOT VERIFIED: no host-side timestamp column found in CSV.")
        print(f"     Proceeding with FS_HZ = {FS_HZ} Hz as an ASSUMPTION, not a measurement.")
        print("     Add a receive-timestamp column in the Python receiver, or supply an")
        print("     independently measured Fs, before treating spectral results as final.")
        return

    ts = df[ts_candidates[0]].values.astype(np.float64)
    duration = ts[-1] - ts[0]
    if duration <= 0:
        print("\n  ⚠  Timestamp column present but non-monotonic/zero duration; cannot verify Fs.")
        return
    measured_fs = (len(ts) - 1) / duration
    pct_err = 100 * abs(measured_fs - FS_HZ) / FS_HZ
    flag = " ← ⚠ MISMATCH >2%: update FS_HZ" if pct_err > 2 else " (OK)"
    print(f"\n  Empirical Fs : {measured_fs:.2f} Hz measured from '{ts_candidates[0]}' "
          f"vs FS_HZ={FS_HZ:.1f} Hz ({pct_err:.1f}% diff){flag}")


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — Preprocessing
# ═══════════════════════════════════════════════════════════════════════════════

def build_filters(fs: float):
    """
    Build and return filter coefficients.
    Uses second-order sections (SOS) for numerical stability with high-order
    Butterworth filters — important for double-precision EEG filtering.
    """
    nyq = 0.5 * fs

    # Notch filter: IIR notch (narrow, zero-phase via filtfilt)
    b_notch, a_notch = iirnotch(NOTCH_FREQ_HZ, NOTCH_Q, fs)

    # Bandpass: SOS form for stability
    sos_bp = butter(FILTER_ORDER,
                    [BANDPASS_LOW_HZ / nyq, BANDPASS_HIGH_HZ / nyq],
                    btype='band', output='sos')

    return b_notch, a_notch, sos_bp


def preprocess_channel(raw_uv: np.ndarray, b_notch, a_notch, sos_bp) -> np.ndarray:
    """
    Standard EEG preprocessing chain for a single channel:
      1. DC removal (mean subtraction)
      2. Linear detrend (removes slow drift without distorting spectral baseline)
      3. 50 Hz notch filter (zero-phase)
      4. 1–90 Hz bandpass Butterworth (zero-phase, SOS)

    All filters applied with zero-phase (forward–backward) to avoid group delay.
    """
    x = raw_uv.copy().astype(np.float64)

    # 1. DC offset removal
    x -= np.mean(x)

    # 2. Linear detrend (removes electrode drift)
    x -= np.polyval(np.polyfit(np.arange(len(x)), x, 1), np.arange(len(x)))

    # 3. Notch
    x = filtfilt(b_notch, a_notch, x)

    # 4. Bandpass (SOS form for numerical stability)
    x = sosfiltfilt(sos_bp, x)

    return x


def preprocess(df: pd.DataFrame) -> np.ndarray:
    """
    Preprocess all EEG channels.
    Returns array shape (n_samples, n_channels) in µV.
    """
    print("\n" + "═"*60)
    print("  STEP 1 — Preprocessing")
    print("═"*60)
    print(f"  Notch filter : {NOTCH_FREQ_HZ} Hz  (Q = {NOTCH_Q})")
    print(f"  Bandpass     : {BANDPASS_LOW_HZ}–{BANDPASS_HIGH_HZ} Hz  "
          f"(Butterworth order {FILTER_ORDER}, zero-phase)")

    b_notch, a_notch, sos_bp = build_filters(FS_HZ)

    clean = np.zeros((len(df), len(EEG_CHANNELS)), dtype=np.float64)
    for i, ch in enumerate(EEG_CHANNELS):
        clean[:, i] = preprocess_channel(df[ch + '_uV'].values, b_notch, a_notch, sos_bp)
        rms = np.sqrt(np.mean(clean[:, i]**2))
        print(f"  {ch}: RMS = {rms:.2f} µV  |  peak = {np.max(np.abs(clean[:,i])):.2f} µV")

    return clean


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — Epoching + Artifact Rejection
# ═══════════════════════════════════════════════════════════════════════════════

def epoch_signal(data: np.ndarray, df: pd.DataFrame, epoch_sec: float = None, epoch_overlap: float = None):
    """
    Segment continuous signal into epochs.
    Args:
        epoch_sec, epoch_overlap: override EPOCH_SEC/EPOCH_OVERLAP for this call
            (used to build the 2-s, non-overlapping epochs for classical
            coherence, separately from the default 8-s/50%-overlap epochs
            used for artifact-rejection bookkeeping and band-power-over-time).
    Returns:
        epochs       : (n_epochs, n_samples_per_epoch, n_channels) — filtered, µV
        epoch_times  : (n_epochs,) — epoch centre time in seconds
        accel_per_epoch : (n_epochs,) — max dynamic acceleration per epoch
        raw_epochs   : (n_epochs, n_samples_per_epoch, n_channels) — RAW ADC counts,
                       same window boundaries as `epochs`. Needed because saturation
                       must be checked on raw counts: bandpass filtering removes the
                       DC offset and spreads clipping distortion, so a genuinely
                       rail-saturated epoch will NOT show a large peak in `epochs`.
        starts, win  : window start indices and window length in samples (needed
                       by callers that build their own derived epoch sets)
    """
    epoch_sec     = EPOCH_SEC if epoch_sec is None else epoch_sec
    epoch_overlap = EPOCH_OVERLAP if epoch_overlap is None else epoch_overlap

    win  = int(epoch_sec * FS_HZ)
    step = int(win * (1 - epoch_overlap))
    n    = len(data)

    starts = np.arange(0, n - win + 1, step)
    epochs = np.stack([data[s:s+win, :] for s in starts])
    epoch_times = (starts + win / 2) / FS_HZ

    raw_counts = df[EEG_CHANNELS].values  # raw ADC counts, pre-filter, pre-scaling
    raw_epochs = np.stack([raw_counts[s:s+win, :] for s in starts])

    # Per-epoch max dynamic acceleration
    accel = df['accel_dynamic_g'].values
    accel_per_epoch = np.array([accel[s:s+win].max() for s in starts])

    return epochs, epoch_times, accel_per_epoch, raw_epochs, starts, win


def reject_artifacts(epochs, epoch_times, accel_per_epoch, raw_epochs):
    """
    Three-stage artifact rejection.

    Stage 1 — Motion rejection:
        Primary criterion is robust and data-driven: an epoch's peak dynamic
        acceleration is flagged if it exceeds median + MOTION_MAD_K * MAD of
        that session's own per-epoch acceleration distribution (robust
        z-score equivalent). This adapts to how much a given animal actually
        moves in a given session, rather than assuming one fixed cutoff fits
        every animal/session. MOTION_G_THRESHOLD is applied as well, as an
        absolute ceiling that should never be exceeded regardless of the
        session's own statistics.
        NOTE: the g-conversion assumes the ADXL345 DATA_FORMAT register is set
        to full-resolution mode (3.9 mg/LSB constant) — confirm this against
        firmware before trusting the absolute g ceiling.

    Stage 2 — Amplitude rejection:
        Same robust-statistics approach: median + AMPLITUDE_MAD_K * MAD of the
        session's own per-epoch peak amplitude distribution, plus
        AMPLITUDE_THRESH_UV as an absolute ceiling. Catches electrode
        artifacts, chewing, and saturation remnants.

    Stage 3 — Saturation rejection:
        Checked on RAW ADC counts (raw_epochs), not the filtered signal.
        Bandpass filtering removes DC offset and spreads clipping distortion,
        so a genuinely rail-saturated epoch would NOT reliably show up as a
        large peak after filtering — checking filtered data here (as the
        original version did) made this stage effectively non-functional.

    Returns good_mask (bool array), detailed reason array.
    """
    print("\n" + "═"*60)
    print("  STEP 2 — Artifact Rejection")
    print("═"*60)

    n = len(epochs)
    mask   = np.ones(n, dtype=bool)
    reason = np.full(n, '', dtype=object)

    # Stage 1: motion — robust MAD-based threshold, capped by absolute ceiling
    accel_median = np.median(accel_per_epoch)
    accel_mad    = np.median(np.abs(accel_per_epoch - accel_median)) * 1.4826  # normal-consistent MAD
    motion_adaptive_thresh = accel_median + MOTION_MAD_K * accel_mad if accel_mad > 0 else MOTION_G_THRESHOLD
    motion_thresh = min(motion_adaptive_thresh, MOTION_G_THRESHOLD)
    motion_bad = accel_per_epoch > motion_thresh
    mask[motion_bad]   = False
    reason[motion_bad] = 'motion'
    print(f"  Motion thresh  : adaptive={motion_adaptive_thresh:.3f}g, ceiling={MOTION_G_THRESHOLD}g "
          f"-> using {motion_thresh:.3f}g")
    print(f"  Motion         : {motion_bad.sum():4d} / {n} epochs rejected "
          f"({100*motion_bad.mean():.1f}%)")

    # Stage 2: amplitude — robust MAD-based threshold, capped by absolute ceiling
    peak_amp = np.max(np.abs(epochs), axis=(1, 2))
    amp_median = np.median(peak_amp)
    amp_mad    = np.median(np.abs(peak_amp - amp_median)) * 1.4826
    amp_adaptive_thresh = amp_median + AMPLITUDE_MAD_K * amp_mad if amp_mad > 0 else AMPLITUDE_THRESH_UV
    amp_thresh = min(amp_adaptive_thresh, AMPLITUDE_THRESH_UV)
    amp_bad  = peak_amp > amp_thresh
    mask[amp_bad] = False
    reason = np.where(amp_bad, np.where(reason != '', reason + '+amp', 'amp'), reason)
    print(f"  Amplitude thresh: adaptive={amp_adaptive_thresh:.1f}µV, ceiling={AMPLITUDE_THRESH_UV}µV "
          f"-> using {amp_thresh:.1f}µV")
    print(f"  Amplitude      : {amp_bad.sum():4d} / {n} epochs rejected "
          f"({100*amp_bad.mean():.1f}%)")

    # Stage 3: saturation — checked on RAW counts, not filtered epochs
    peak_raw_counts = np.max(np.abs(raw_epochs), axis=(1, 2))
    sat_bad = peak_raw_counts >= SATURATION_THRESH_COUNTS
    mask[sat_bad] = False
    reason = np.where(sat_bad, np.where(reason != '', reason + '+sat', 'sat'), reason)
    print(f"  Saturation     : {sat_bad.sum():4d} / {n} epochs rejected "
          f"({100*sat_bad.mean():.1f}%)  [checked on raw ADC counts, thresh={SATURATION_THRESH_COUNTS}]")

    total_bad = (~mask).sum()
    print(f"  ─────────────────────────────────────────")
    print(f"  Total rejected : {total_bad:4d} / {n} ({100*total_bad/n:.1f}%)")
    print(f"  Retained       : {mask.sum():4d} / {n} ({100*mask.mean():.1f}%)")

    if mask.sum() < 10:
        print("\n  ⚠  WARNING: Fewer than 10 clean epochs remain.")
        print("     Check hardware gain, cable connections, and animal movement.")

    return mask, reason


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — Power Spectral Density
# ═══════════════════════════════════════════════════════════════════════════════

def compute_psd(clean_epochs: np.ndarray):
    """
    Compute PSD using Welch's method, averaged PER EPOCH rather than on a
    concatenation of all clean epochs.
    """
    print("\n" + "═"*60)
    print("  STEP 3 — Power Spectral Density (Welch, averaged per-epoch)")
    print("═"*60)

    n_epochs, n_samp, n_ch = clean_epochs.shape
    nperseg  = min(n_samp, int(4 * FS_HZ))   # 4-second Welch window → 0.25 Hz resolution
    noverlap = nperseg // 2                   # 50% overlap (within an epoch only)

    freqs = None
    psd_all = np.zeros((n_ch, 0))
    for ch in range(n_ch):
        per_epoch_psd = []
        for w in range(n_epochs):
            f, psd = welch(
                clean_epochs[w, :, ch],
                fs       = FS_HZ,
                window   = 'hann',
                nperseg  = nperseg,
                noverlap = noverlap,
                scaling  = 'density'       # µV²/Hz
            )
            per_epoch_psd.append(psd)
        freqs = f
        if ch == 0:
            psd_all = np.zeros((n_ch, len(f)))
        psd_all[ch] = np.mean(per_epoch_psd, axis=0)

    # Band power summary
    print(f"\n  {'Channel':<8}", end="")
    for band, (lo, hi) in FREQ_BANDS.items():
        print(f"  {band:<8}", end="")
    print()
    print("  " + "─"*55)

    for ch, name in enumerate(EEG_CHANNELS):
        print(f"  {name:<8}", end="")
        for band, (lo, hi) in FREQ_BANDS.items():
            idx = (freqs >= lo) & (freqs <= hi)
            bp = np.trapezoid(psd_all[ch, idx], freqs[idx])
            print(f"  {bp:8.2f}", end="")
        print()
    print("  (units: µV²)")

    return freqs, psd_all


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 5 — Time-frequency: band power over time
# ═══════════════════════════════════════════════════════════════════════════════

def compute_band_power_timeseries(clean_epochs: np.ndarray, epoch_times: np.ndarray):
    """
    For each epoch and each frequency band, compute band power via Welch.
    Returns dict mapping band_name → (n_epochs, n_channels) array of power in µV².
    """
    print("\n" + "═"*60)
    print("  STEP 5 — Band Power Time-Series")
    print("═"*60)

    n_epochs, n_samp, n_ch = clean_epochs.shape
    nperseg = min(n_samp, int(2 * FS_HZ))   # 2-s sub-window within epoch

    band_power = {band: np.full((n_epochs, n_ch), np.nan) for band in FREQ_BANDS}

    for w in range(n_epochs):
        for ch in range(n_ch):
            x = clean_epochs[w, :, ch]
            if np.nanstd(x) < 1e-15:
                continue
            freqs, psd = welch(x, fs=FS_HZ, nperseg=nperseg, window='hann')
            for band, (lo, hi) in FREQ_BANDS.items():
                idx = (freqs >= lo) & (freqs <= hi)
                if idx.sum() >= 2:
                    band_power[band][w, ch] = np.trapezoid(psd[idx], freqs[idx])

    print(f"  Computed band power for {n_epochs} clean epochs × {n_ch} channels")
    return band_power



# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — Coherence (classical, matching a standard BrainVision-Analyzer-style
#  workflow: fixed-length artifact-free epochs -> random subsample -> Hanning-
#  windowed FFT per epoch -> average cross-/auto-spectra across epochs ->
#  Cxy(f) = |mean(Pxy)|^2 / (mean(Pxx) * mean(Pyy)). No bias correction, no
#  imaginary coherence, no re-referencing -- this is the plain, standard
#  magnitude-squared coherence estimator, computed the way scipy.signal.coherence
#  (and BrainVision Analyzer, and most Welch-based coherence tools) define it.
#
#  The only thing enforced beyond "the classical formula" is that segments used
#  in the average are (a) all exactly COHERENCE_EPOCH_SEC long and (b) each
#  drawn from a single contiguous artifact-free stretch, so no FFT segment
#  straddles a discontinuity between two unrelated pieces of the recording and
#  no sample is counted twice. That is a correctness requirement of the
#  classical formula itself (Welch's method assumes segments are drawn cleanly
#  from the signal), not an added method.
# ═══════════════════════════════════════════════════════════════════════════════

COHERENCE_EPOCH_SEC = 2.0     # match target paper's 2-second artifact-free epochs
COHERENCE_N_EPOCHS  = 200     # match target paper's ~200 randomly selected clean epochs
COHERENCE_SEED      = 42      # fixed seed so "random selection" is reproducible run-to-run


def select_coherence_epochs(clean: np.ndarray, df: pd.DataFrame,
                             epoch_sec: float = COHERENCE_EPOCH_SEC,
                             n_epochs: int = COHERENCE_N_EPOCHS,
                             seed: int = COHERENCE_SEED):
    """
    Build epoch_sec-length, non-overlapping epochs, run them through the same
    artifact rejection used elsewhere in this pipeline (motion / amplitude /
    saturation -- an automated stand-in for the paper's manual rejection step),
    then randomly select up to n_epochs of the clean ones.

    Returns:
        selected_epochs : (n_selected, epoch_len, n_channels) -- the randomly
                           chosen clean epochs, in ORIGINAL time order removed
                           (order doesn't matter for the averaged-periodogram
                           formula, each epoch is scored independently)
        n_selected       : number of epochs actually used (<= n_epochs; fewer
                           if the recording doesn't have that many clean
                           epochs at this epoch length)
        n_available       : total clean epochs available before subsampling
    """
    epochs, epoch_times, accel_per_epoch, raw_epochs, starts, win = epoch_signal(
        clean, df, epoch_sec=epoch_sec, epoch_overlap=0.0)
    good_mask, reason = reject_artifacts(epochs, epoch_times, accel_per_epoch, raw_epochs)

    clean_epochs = epochs[good_mask]
    n_available = clean_epochs.shape[0]

    rng = np.random.default_rng(seed)
    if n_available <= n_epochs:
        idx = np.arange(n_available)
        if n_available < n_epochs:
            print(f"  ⚠  Only {n_available} clean {epoch_sec:.0f}-s epochs available "
                  f"(< requested {n_epochs}); using all of them.")
    else:
        idx = rng.choice(n_available, size=n_epochs, replace=False)

    selected_epochs = clean_epochs[idx]
    return selected_epochs, selected_epochs.shape[0], n_available


def compute_coherence(selected_epochs: np.ndarray):
    """
    Classical magnitude-squared coherence, averaged across the randomly
    selected clean epochs from select_coherence_epochs():

        Cxy(f) = |mean_epochs(Pxy(f))|^2 / (mean_epochs(Pxx(f)) * mean_epochs(Pyy(f)))

    Implemented via scipy.signal.coherence on the epochs concatenated
    end-to-end with nperseg = epoch length and noverlap = 0, which makes
    scipy's internal Welch segmentation land exactly on epoch boundaries --
    mathematically identical to averaging one Hanning-windowed periodogram
    per epoch by hand, with no segment ever spanning two different epochs.

    Returns:
        freqs   : frequency array (Hz)
        coh_dict: {(i,j): Cxy array (n_freqs,)}
        n_used  : number of epochs actually averaged (the classical method's L)
    """
    print("\n" + "═"*60)
    print("  STEP 4 — Coherence (classical, BrainVision-style)")
    print("═"*60)

    n_epochs, seg_len, n_ch = selected_epochs.shape
    pooled = selected_epochs.reshape(-1, n_ch)   # (n_epochs*seg_len, n_ch), epoch-aligned

    print(f"  Epochs averaged     : {n_epochs}  (target: {COHERENCE_N_EPOCHS})")
    print(f"  Epoch length        : {seg_len/FS_HZ:.1f} s ({seg_len} samples)")
    print(f"  Window              : Hann, no sub-segment overlap (one FFT per epoch)")

    pairs = [(i, j) for i in range(n_ch) for j in range(i + 1, n_ch)]
    coh_dict = {}
    f = None

    print(f"\n  {'Pair':<15}", end="")
    for band in FREQ_BANDS:
        print(f"  {band:<8}", end="")
    print()
    print("  " + "─"*55)

    for i, j in pairs:
        f, Cxy = coherence(pooled[:, i], pooled[:, j], fs=FS_HZ, window='hann',
                            nperseg=seg_len, noverlap=0)
        coh_dict[(i, j)] = Cxy
        pair_name = f"{EEG_CHANNELS[i]}-{EEG_CHANNELS[j]}"
        print(f"  {pair_name:<15}", end="")
        for band, (lo, hi) in FREQ_BANDS.items():
            idx = (f >= lo) & (f <= hi)
            print(f"  {Cxy[idx].mean():.3f}  ", end="")
        print()

    return f, coh_dict, n_epochs


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURES
# ═══════════════════════════════════════════════════════════════════════════════

def _savefig(fig, name: str):
    if SAVE_FIGURES:
        path = os.path.join(OUTPUT_DIR, f"{name}.{FIGURE_FMT}")
        fig.savefig(path, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"  [saved] {path}")
    plt.close(fig)


def fig_raw_vs_clean(df: pd.DataFrame, clean: np.ndarray):

    # ---------- Select middle 5-minute window ----------
    five_min = min(int(5 * 60 * FS_HZ), len(df))
    start = (len(df) - five_min) // 2
    end = start + five_min

    # ---------- 10-second zoom from center of 5-minute window ----------
    zoom_len = int(10 * FS_HZ)
    zoom_start = start + (five_min - zoom_len) // 2
    zoom_end = zoom_start + zoom_len

    t5 = df["time_sec"].values[start:end]
    tz = df["time_sec"].values[zoom_start:zoom_end]

    fig = plt.figure(figsize=(14, len(EEG_CHANNELS) * 5.2))
    gs = fig.add_gridspec(
        nrows=len(EEG_CHANNELS) * 2,
        ncols=2,
        hspace=0.45,
        wspace=0.25
    )

    fig.suptitle(
        "Raw vs Preprocessed EEG",
        fontsize=14,
        fontweight="bold"
    )

    for i, ch in enumerate(EEG_CHANNELS):

        # ---------- RAW ----------
        ax_raw_full = fig.add_subplot(gs[2 * i, 0])
        ax_raw_zoom = fig.add_subplot(gs[2 * i + 1, 0])

        raw5 = df[ch + "_uV"].values[start:end]
        rawz = df[ch + "_uV"].values[zoom_start:zoom_end]

        ax_raw_full.plot(t5, raw5, lw=0.3, color="gray")

        # Highlight zoom window
        ax_raw_full.axvspan(
            tz[0], tz[-1],
            color="tomato",
            alpha=0.25
        )

        ax_raw_zoom.plot(tz, rawz, lw=0.7, color="gray")

        # ---------- CLEAN ----------
        ax_cln_full = fig.add_subplot(gs[2 * i, 1])
        ax_cln_zoom = fig.add_subplot(gs[2 * i + 1, 1])

        clean5 = clean[start:end, i]
        cleanz = clean[zoom_start:zoom_end, i]

        ax_cln_full.plot(t5, clean5, lw=0.3, color="#000000")
        ax_cln_full.axvspan(
            tz[0], tz[-1],
            color="tomato",
            alpha=0.25
        )

        ax_cln_zoom.plot(tz, cleanz, lw=0.8, color="#000000")

        # ---------- Labels ----------
        ax_raw_full.set_ylabel(f"{ch}\n(µV)")
        ax_raw_zoom.set_ylabel("(µV)")

        if i == 0:
            ax_raw_full.set_title("Raw")
            ax_cln_full.set_title(
                "Preprocessed\n(detrend + notch + bandpass)"
            )

        for ax in [ax_raw_full, ax_raw_zoom,
                   ax_cln_full, ax_cln_zoom]:
            ax.grid(False)

        if i == len(EEG_CHANNELS) - 1:
            ax_raw_zoom.set_xlabel("Time (s)")
            ax_cln_zoom.set_xlabel("Time (s)")

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    _savefig(fig, "fig1_raw_vs_clean")


def fig_psd(freqs, psd_all):
    fig, ax = plt.subplots(figsize=(9, 5))

    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']

    for i, (ch, c) in enumerate(zip(EEG_CHANNELS, colors)):
        ax.semilogy(freqs, psd_all[i], color=c, lw=1.5, label=ch)

    # Slightly darker band colors
    band_colors = {
        'Delta': '#6F95B7',
        'Theta': '#7FA67F',
        'Beta' : '#D2BE72',
        'Gamma': '#C97C86'
    }

    for band, (lo, hi) in FREQ_BANDS.items():
        ax.axvspan(lo, hi,
                   color=band_colors[band],
                   alpha=0.30,
                   label=f'_{band}')

        # Band label at a fixed position (5% above bottom)
        ax.text(
            (lo + hi) / 2,
            0.05,
            band,
            transform=ax.get_xaxis_transform(),
            ha='center',
            va='bottom',
            fontsize=9,
            fontweight='bold',
            color='#333333'
        )

    ax.set_xlim(0, min(BANDPASS_HIGH_HZ, FS_HZ / 2))
    ax.set_ylim(1e0, 1e4)

    ax.set_xlabel("Frequency (Hz)", fontsize=11)
    ax.set_ylabel("PSD (µV²/Hz)", fontsize=11)

    ax.set_title(
        "Power Spectral Density — Welch's Method\n"
        f"(4-s window, Hann, 50% overlap, {FS_HZ:.0f} Hz)",
        fontsize=11
    )

    ax.legend(fontsize=9)
    ax.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    _savefig(fig, "fig2_psd")


def fig_coherence(freqs, coh_dict, n_used):
    """Classic coherence spectra, one panel per channel pair, plus a per-band
    summary table printed alongside (no correction/adjustment applied)."""
    pairs = list(coh_dict.keys())
    n = len(pairs)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    # رنگ‌های باند (همانند نمودار PSD)
    band_colors = {
        'Delta': '#6F95B7',
        'Theta': '#7FA67F',
        'Beta' : '#D2BE72',
        'Gamma': '#C97C86'
    }

    for ax, (i, j) in zip(axes, pairs):
        Cxy = coh_dict[(i, j)]
        ax.plot(freqs, Cxy, lw=1.5, color='#6c3483')
        ax.fill_between(freqs, 0, Cxy, alpha=0.2, color='#6c3483')

        for band, (lo, hi) in FREQ_BANDS.items():
            ax.axvspan(lo, hi,
                       color=band_colors[band],
                       alpha=0.25,
                       label=f'_{band}') 

            ax.text(
                (lo + hi) / 2,
                0.975,                       
                band,
                transform=ax.get_xaxis_transform(),
                ha='center',          
                va='center',                 
                rotation=0,              
                fontsize=5,              
                fontweight='bold',
                color='#333333'
            )

        ax.set_xlim(0, min(BANDPASS_HIGH_HZ, FS_HZ/2))
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{EEG_CHANNELS[i]} – {EEG_CHANNELS[j]}", fontsize=10)
        ax.set_xlabel("Frequency (Hz)")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Magnitude-Squared Coherence", fontsize=10)
    fig.suptitle(f"Coherence Spectra (classical, {COHERENCE_EPOCH_SEC:.0f}-s Hann epochs, "
                 f"n={n_used})", fontsize=12, fontweight='bold')
    plt.tight_layout()
    _savefig(fig, "fig3_coherence_spectra")

def fig_artifact_diagnostic(df, all_epoch_times, accel_per_epoch, good_mask, reason):
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    t_min = df['time_sec'].values / 60
    raw_uv = df[[ch + '_uV' for ch in EEG_CHANNELS]].values
    axes[0].fill_between(t_min, raw_uv.min(axis=1), raw_uv.max(axis=1),
                         alpha=0.4, color='#1a6faf', label='EEG amplitude envelope')
    axes[0].set_ylabel("µV", fontsize=10); axes[0].set_title("Raw EEG Amplitude Envelope", fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(t_min, df['accel_dynamic_g'].values, lw=0.5, color='#555')
    axes[1].axhline(MOTION_G_THRESHOLD, color='red', lw=1.5, ls='--',
                    label=f'Ceiling ({MOTION_G_THRESHOLD} g)')
    axes[1].set_ylabel("g", fontsize=10); axes[1].set_title("Dynamic Acceleration", fontsize=10)
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)
    ep_t = all_epoch_times / 60
    axes[2].scatter(ep_t[good_mask], np.ones(good_mask.sum()), marker='|', s=40,
                    color='green', label='Accepted', alpha=0.6)
    axes[2].scatter(ep_t[~good_mask], np.ones((~good_mask).sum()), marker='|', s=40,
                    color='red', label='Rejected', alpha=0.6)
    axes[2].set_yticks([]); axes[2].set_title("Epoch Acceptance / Rejection", fontsize=10)
    axes[2].set_xlabel("Time (min)", fontsize=11); axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)
    plt.suptitle("Artifact Rejection Diagnostic (8-s epochs, band-power/artifact bookkeeping)",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    _savefig(fig, "fig4_artifact_diagnostic")


# ═══════════════════════════════════════════════════════════════════════════════
#  SUMMARY (for Methods section)
# ═══════════════════════════════════════════════════════════════════════════════

def print_methods_summary(df, n_clean, n_coh_used, n_coh_available):
    print("\n" + "═"*60)
    print("  METHODS SUMMARY  (suitable for paper Methods section)")
    print("═"*60)
    print(f"""
  Recording
  ─────────
  Sampling rate    : {FS_HZ:.0f} Hz  [NOT YET EMPIRICALLY VERIFIED -- see validate_sampling_rate()]
  ADC              : ADS1115 16-bit, gain ±{ADS_GAIN_RANGE_V} V ({UV_PER_LSB_ADC:.3f} µV/LSB at ADC input,
                     {UV_PER_LSB:.5f} µV/LSB referred to electrode after /{TOTAL_PREAMP_GAIN:.0f} gain)
  IMU              : ADXL345, full-res, ±2 g (firmware-confirmed), {ADXL345_MG_PER_LSB} mg/LSB
  Total samples    : {len(df)}  ({df['time_sec'].iloc[-1]:.1f} s)

  Preprocessing
  ─────────────
  1. DC mean subtraction + linear detrend
  2. {NOTCH_FREQ_HZ:.0f} Hz IIR notch filter (Q = {NOTCH_Q})  [zero-phase]
  3. {BANDPASS_LOW_HZ}-{BANDPASS_HIGH_HZ} Hz Butterworth bandpass, order {FILTER_ORDER}  [zero-phase, SOS]

  Artifact Rejection
  ──────────────────
  Epoch length     : {EPOCH_SEC:.1f} s, {int(EPOCH_OVERLAP*100)}% overlap (band-power-over-time / general bookkeeping)
  Motion threshold : robust (median + {MOTION_MAD_K:.0f}xMAD of session), ceiling {MOTION_G_THRESHOLD} g
  Amplitude thresh : robust (median + {AMPLITUDE_MAD_K:.0f}xMAD of session), ceiling {AMPLITUDE_THRESH_UV} µV
  Saturation thresh: >{SATURATION_THRESH_COUNTS} raw ADC counts (checked on raw counts, not filtered signal)
  Clean epochs     : {n_clean}

  Coherence Analysis (classical)
  ───────────────────────────────
  Epoch length     : {COHERENCE_EPOCH_SEC:.0f} s, non-overlapping, artifact-free
  Epoch selection  : random subsample, seed={COHERENCE_SEED} ({n_coh_used} of {n_coh_available} clean
                     {COHERENCE_EPOCH_SEC:.0f}-s epochs used; target {COHERENCE_N_EPOCHS})
  Window           : Hann, one FFT per epoch, no sub-segment overlap
  Formula          : Cxy(f) = |mean_epochs(Pxy(f))|^2 / (mean_epochs(Pxx(f)) * mean_epochs(Pyy(f)))
                     (standard Welch-based magnitude-squared coherence)

  Spectral Analysis (PSD)
  ─────────────────────────
  Method      : Welch, averaged per-epoch (Hann, 4-s sub-windows, 50% overlap)

  Frequency Bands (rodent LFP)
  ──────────────────────────────""")
    for band, (lo, hi) in FREQ_BANDS.items():
        print(f"  {band:<8}: {lo}-{hi} Hz")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "█"*60)
    print("  Rat EEG Analysis Pipeline")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("█"*60)

    df = load_and_validate(CSV_FILE)
    clean = preprocess(df)

    epochs, epoch_times, accel_per_epoch, raw_epochs, starts, win = epoch_signal(clean, df)
    good_mask, reason = reject_artifacts(epochs, epoch_times, accel_per_epoch, raw_epochs)

    clean_epochs = epochs[good_mask]
    clean_times  = epoch_times[good_mask]
    n_clean      = good_mask.sum()

    if n_clean == 0:
        print("\n[FATAL] No clean epochs after artifact rejection.")
        sys.exit(1)

    freqs, psd_all = compute_psd(clean_epochs)

    coh_epochs, n_coh_used, n_coh_available = select_coherence_epochs(clean, df)
    f_coh, coh_dict, n_used = compute_coherence(coh_epochs)

    band_power = compute_band_power_timeseries(clean_epochs, clean_times)

    print("\n" + "═"*60)
    print("  Generating Figures")
    print("═"*60)
    fig_raw_vs_clean(df, clean)
    fig_psd(freqs, psd_all)
    fig_coherence(f_coh, coh_dict, n_used)
    fig_artifact_diagnostic(df, epoch_times, accel_per_epoch, good_mask, reason)

    print_methods_summary(df, n_clean, n_coh_used, n_coh_available)

    print(f"\n  All outputs saved to: {os.path.abspath(OUTPUT_DIR)}/")
    print("  Done.\n")


def run_session(csv_path: str, state_label: str) -> dict:
    """Run load->preprocess->epoch->reject->PSD->coherence->band power on one
    session's CSV and return the results instead of only printing."""
    df = load_and_validate(csv_path)
    clean = preprocess(df)
    epochs, epoch_times, accel_per_epoch, raw_epochs, starts, win = epoch_signal(clean, df)
    good_mask, reason = reject_artifacts(epochs, epoch_times, accel_per_epoch, raw_epochs)

    clean_epochs = epochs[good_mask]
    clean_times  = epoch_times[good_mask]

    freqs, psd_all = compute_psd(clean_epochs)

    coh_epochs, n_coh_used, n_coh_available = select_coherence_epochs(clean, df)
    f_coh, coh_dict, n_used = compute_coherence(coh_epochs)

    band_power = compute_band_power_timeseries(clean_epochs, clean_times)

    return {
        "state": state_label,
        "csv": csv_path,
        "n_clean": int(good_mask.sum()),
        "n_total": int(len(good_mask)),
        "freqs": freqs,
        "psd_all": psd_all,
        "f_coh": f_coh,
        "coh_dict": coh_dict,        # {(i,j): Cxy array}
        "n_coh_used": n_used,
        "band_power": band_power,
    }


def compare_states(sessions: list) -> list:
    """sessions: list of {'csv': path, 'state': 'awake'|'anesthetized'|'sleep'}."""
    results = [run_session(s["csv"], s["state"]) for s in sessions]

    print("\n" + "="*70)
    print("  CROSS-STATE BAND POWER SUMMARY (µV², mean over clean epochs)")
    print("="*70)
    for band in FREQ_BANDS:
        print(f"\n  {band}")
        print(f"    {'State':<14}", end="")
        for ch in EEG_CHANNELS:
            print(f"  {ch:>10}", end="")
        print()
        for r in results:
            print(f"    {r['state']:<14}", end="")
            for ch_i in range(len(EEG_CHANNELS)):
                vals = r["band_power"][band][:, ch_i]
                print(f"  {np.nanmean(vals):10.2f}", end="")
            print(f"   (n={r['n_clean']}/{r['n_total']} clean epochs)")

    print("\n" + "="*70)
    print("  CROSS-STATE COHERENCE SUMMARY (classical, mean per band)")
    print("="*70)
    for band, (lo, hi) in FREQ_BANDS.items():
        print(f"\n  {band}")
        for r in results:
            idx = (r["f_coh"] >= lo) & (r["f_coh"] <= hi)
            mean_coh = np.mean([Cxy[idx].mean() for Cxy in r["coh_dict"].values()])
            print(f"    {r['state']:<14}: mean coherence = {mean_coh:.3f}  (n_epochs={r['n_coh_used']})")

    fig, axes = plt.subplots(1, len(EEG_CHANNELS), figsize=(5*len(EEG_CHANNELS), 4.5), sharey=True)
    colors = {"awake": "#2ecc71", "anesthetized": "#3498db", "sleep": "#9b59b6", "rest": "#9b59b6"}
    for ch_i, ch in enumerate(EEG_CHANNELS):
        ax = axes[ch_i]
        for r in results:
            c = colors.get(r["state"], "#333")
            ax.semilogy(r["freqs"], r["psd_all"][ch_i], color=c, lw=1.5, label=r["state"])
        ax.set_xlim(0, min(BANDPASS_HIGH_HZ, FS_HZ/2))
        ax.set_title(ch, fontsize=10); ax.set_xlabel("Frequency (Hz)"); ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("PSD (µV²/Hz)"); axes[0].legend(fontsize=8)
    fig.suptitle("PSD by Vigilance/Anesthesia State", fontsize=12, fontweight="bold")
    plt.tight_layout()
    _savefig(fig, "fig5_psd_by_state")

    return results


if __name__ == "__main__":
    # compare_states([
    #     {"csv": "/home/jonah/Downloads/online reciving/ctrl_A.csv",        "state": "awake"},
    #     {"csv": "/home/jonah/Downloads/online reciving/ctl_baghjari.csv", "state": "anesthetized"},
    #     {"csv": "/home/jonah/Downloads/online reciving/آلزایمری - 203.csv","state": "sleep"},
    # ])
    main()
