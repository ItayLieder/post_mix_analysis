

# DSP Primitives Layer — Notebook Implementation
# Reusable, low-level DSP blocks for building post‑mix features.
# Dependencies: numpy, scipy.signal
#
# All functions accept/return numpy arrays:
# - audio: shape (N,) mono or (N,2) stereo
# - sr: sample rate (int)
#
# Notes:
# - Uses numerically-stable SOS filters where applicable
# - Sanitizes NaN/Inf and clamps frequencies to safe ranges
# - Stereo‑aware (processes both channels consistently)
#
# ---------------------------------------------------------------------

from __future__ import annotations
import numpy as np
from scipy import signal
from typing import Tuple

# --------- Core helpers ---------

def _sanitize(x: np.ndarray) -> np.ndarray:
    """Replace NaN/Inf and clamp extreme outliers to avoid IIR blowups."""
    return np.clip(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), -4.0, 4.0).astype(np.float32)

def _ensure_stereo(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return np.stack([x, x], axis=-1)
    if x.shape[1] == 1:
        return np.repeat(x, 2, axis=1)
    return x

def _mono(x: np.ndarray) -> np.ndarray:
    return x if x.ndim == 1 else np.mean(x, axis=1)

def _safe_freq(f: float, sr: int, lo: float = 10.0, hi_ratio: float = 0.49) -> float:
    """Clamp frequency to a safe absolute Hz range based on sample rate."""
    return float(max(lo, min(f, hi_ratio * sr)))

def _db_to_lin(db: float) -> float:
    return 10.0 ** (db / 20.0)

def _lin_to_db(lin: float, eps: float = 1e-12) -> float:
    return 20.0 * np.log10(max(eps, abs(lin)))

# --------- Gain & leveling ---------

def apply_gain_db(audio: np.ndarray, db: float) -> np.ndarray:
    """Apply linear gain in dB (stereo‑safe)."""
    g = _db_to_lin(db)
    return (_sanitize(audio) * g).astype(np.float32)

def measure_peak(audio: np.ndarray) -> float:
    """Return peak linear amplitude (mono‑collapsed)."""
    x = _sanitize(audio)
    x = _mono(x)
    return float(np.max(np.abs(x)))

def measure_rms(audio: np.ndarray) -> float:
    """Return RMS (linear) on mono‑collapsed signal."""
    x = _sanitize(audio)
    x = _mono(x)
    return float(np.sqrt(np.mean(x**2)))

def normalize_peak(audio: np.ndarray, target_dbfs: float = -1.0, eps: float = 1e-9) -> np.ndarray:
    """Scale so that the absolute peak ≈ target dBFS."""
    x = _sanitize(audio)
    peak = float(np.max(np.abs(x)))
    if peak < eps:
        return np.zeros_like(x, dtype=np.float32)
    g = _db_to_lin(target_dbfs) / peak
    return (x * g).astype(np.float32)

# --------- K‑weighting & LUFS (approx) ---------

def _k_weighting_sos(sr: int) -> np.ndarray:
    """SOS for ITU‑R BS.1770 K‑weighting: 2nd‑order HPF (~38 Hz) + 2nd‑order high‑shelf (~+4 dB @ 1 kHz)."""
    f_hp = 38.0
    sos_hp = signal.butter(2, _safe_freq(f_hp, sr)/(sr*0.5), btype='highpass', output='sos')
    # High‑shelf (RBJ)
    f_shelf = 1681.974450955533
    Q_shelf = 0.7071752369554196
    gain_db = 3.99984385397
    A = 10**(gain_db/40.0)
    w0 = 2*np.pi*f_shelf/sr
    alpha = np.sin(w0)/(2*Q_shelf)
    cosw0 = np.cos(w0)
    b0 =    A*((A+1) + (A-1)*cosw0 + 2*np.sqrt(A)*alpha)
    b1 = -2*A*((A-1) + (A+1)*cosw0)
    b2 =    A*((A+1) + (A-1)*cosw0 - 2*np.sqrt(A)*alpha)
    a0 =        (A+1) - (A-1)*cosw0 + 2*np.sqrt(A)*alpha
    a1 =    2*((A-1) - (A+1)*cosw0)
    a2 =        (A+1) - (A-1)*cosw0 - 2*np.sqrt(A)*alpha
    sos_shelf = signal.tf2sos([b0/a0, b1/a0, b2/a0], [1.0, a1/a0, a2/a0])
    return np.vstack([sos_hp, sos_shelf])

def k_weight(audio: np.ndarray, sr: int) -> np.ndarray:
    """Apply K‑weighting to mono signal (array)."""
    x = _sanitize(audio)
    x = _mono(x)
    sos = _k_weighting_sos(sr)
    return signal.sosfilt(sos, x).astype(np.float32)

def lufs_integrated_approx(audio: np.ndarray, sr: int) -> float:
    """Lightweight integrated LUFS (BS.1770 K‑weighting, no gating)."""
    xk = k_weight(audio, sr)
    ms = float(np.mean(xk**2))
    return -0.691 + 10.0 * np.log10(max(1e-12, ms))

def normalize_lufs(audio: np.ndarray, sr: int, target_lufs: float = -14.0) -> np.ndarray:
    """Normalize integrated loudness to target LUFS (approximate, no gating)."""
    x = _sanitize(audio)
    current = lufs_integrated_approx(x, sr)
    delta = target_lufs - current  # dB to add
    return apply_gain_db(x, delta)

# --------- Filtering (SOS) ---------

def highpass_filter(audio: np.ndarray, sr: int, cutoff_hz: float, order: int = 4) -> np.ndarray:
    x = _sanitize(audio)
    sos = signal.butter(order, _safe_freq(cutoff_hz, sr)/(sr*0.5), btype='highpass', output='sos')
    return signal.sosfilt(sos, x, axis=0 if x.ndim > 1 else 0).astype(np.float32)

def lowpass_filter(audio: np.ndarray, sr: int, cutoff_hz: float, order: int = 4) -> np.ndarray:
    x = _sanitize(audio)
    sos = signal.butter(order, _safe_freq(cutoff_hz, sr)/(sr*0.5), btype='lowpass', output='sos')
    return signal.sosfilt(sos, x, axis=0 if x.ndim > 1 else 0).astype(np.float32)

def bandpass_filter(audio: np.ndarray, sr: int, f_lo: float, f_hi: float, order: int = 4) -> np.ndarray:
    x = _sanitize(audio)
    lo = _safe_freq(f_lo, sr)
    hi = _safe_freq(f_hi, sr)
    if hi <= lo:  # enforce valid band
        hi = min(max(lo * 1.2, lo + 5.0), 0.49 * sr)
    sos = signal.butter(order, [lo/(sr*0.5), hi/(sr*0.5)], btype='bandpass', output='sos')
    return signal.sosfilt(sos, x, axis=0 if x.ndim > 1 else 0).astype(np.float32)

# --------- Shelving & Parametric EQ (RBJ biquads -> SOS) ---------

def _biquad_peaking_sos(sr: int, f0: float, gain_db: float, Q: float = 0.707) -> np.ndarray:
    A = 10**(gain_db/40.0)
    w0 = 2*np.pi*_safe_freq(f0, sr)/sr
    alpha = np.sin(w0)/(2*Q)
    cosw0 = np.cos(w0)
    b0 = 1 + alpha*A
    b1 = -2*cosw0
    b2 = 1 - alpha*A
    a0 = 1 + alpha/A
    a1 = -2*cosw0
    a2 = 1 - alpha/A
    b = np.array([b0, b1, b2])/a0
    a = np.array([1.0, a1/a0, a2/a0])
    return signal.tf2sos(b, a)

def _biquad_lowshelf_sos(sr: int, f0: float, gain_db: float, S: float = 0.5) -> np.ndarray:
    A = 10**(gain_db/40.0)
    w0 = 2*np.pi*_safe_freq(f0, sr)/sr
    cosw0 = np.cos(w0); sinw0 = np.sin(w0)
    alpha = sinw0/2 * np.sqrt((A + 1/A)*(1/S - 1) + 2)
    b0 =    A*((A+1) - (A-1)*cosw0 + 2*np.sqrt(A)*alpha)
    b1 =  2*A*((A-1) - (A+1)*cosw0)
    b2 =    A*((A+1) - (A-1)*cosw0 - 2*np.sqrt(A)*alpha)
    a0 =        (A+1) + (A-1)*cosw0 + 2*np.sqrt(A)*alpha
    a1 =   -2*((A-1) + (A+1)*cosw0)
    a2 =        (A+1) + (A-1)*cosw0 - 2*np.sqrt(A)*alpha
    b = np.array([b0, b1, b2])/a0
    a = np.array([1.0, a1/a0, a2/a0])
    return signal.tf2sos(b, a)

def _biquad_highshelf_sos(sr: int, f0: float, gain_db: float, S: float = 0.5) -> np.ndarray:
    A = 10**(gain_db/40.0)
    w0 = 2*np.pi*_safe_freq(f0, sr)/sr
    cosw0 = np.cos(w0); sinw0 = np.sin(w0)
    alpha = sinw0/2 * np.sqrt((A + 1/A)*(1/S - 1) + 2)
    b0 =    A*((A+1) + (A-1)*cosw0 + 2*np.sqrt(A)*alpha)
    b1 = -2*A*((A-1) + (A+1)*cosw0)
    b2 =    A*((A+1) + (A-1)*cosw0 - 2*np.sqrt(A)*alpha)
    a0 =        (A+1) - (A-1)*cosw0 + 2*np.sqrt(A)*alpha
    a1 =    2*((A-1) - (A+1)*cosw0)
    a2 =        (A+1) - (A-1)*cosw0 - 2*np.sqrt(A)*alpha
    b = np.array([b0, b1, b2])/a0
    a = np.array([1.0, a1/a0, a2/a0])
    return signal.tf2sos(b, a)

def peaking_eq(audio: np.ndarray, sr: int, f0: float, gain_db: float, Q: float = 0.707) -> np.ndarray:
    """Parametric peaking EQ at f0 with gain_db and quality Q."""
    x = _sanitize(audio)
    sos = _biquad_peaking_sos(sr, f0, gain_db, Q)
    return signal.sosfilt(sos, x, axis=0 if x.ndim > 1 else 0).astype(np.float32)

def shelf_filter(audio: np.ndarray, sr: int, cutoff_hz: float, gain_db: float, kind: str = "low", S: float = 0.5) -> np.ndarray:
    """Low/High shelf EQ using RBJ biquad. kind ∈ {'low','high'}"""
    x = _sanitize(audio)
    if kind == "low":
        sos = _biquad_lowshelf_sos(sr, cutoff_hz, gain_db, S)
    elif kind == "high":
        sos = _biquad_highshelf_sos(sr, cutoff_hz, gain_db, S)
    else:
        raise ValueError("kind must be 'low' or 'high'")
    return signal.sosfilt(sos, x, axis=0 if x.ndim > 1 else 0).astype(np.float32)

def notch_filter(audio: np.ndarray, sr: int, f0: float, Q: float = 10.0) -> np.ndarray:
    """Narrow notch (peaking with large negative gain)."""
    # Implement via iirnotch for convenience
    w0 = _safe_freq(f0, sr)/(sr*0.5)
    b, a = signal.iirnotch(w0, Q)
    sos = signal.tf2sos(b, a)
    return signal.sosfilt(sos, _sanitize(audio), axis=0 if audio.ndim > 1 else 0).astype(np.float32)

def tilt_eq(audio: np.ndarray, sr: int, pivot_hz: float = 1000.0, gain_db: float = 1.5) -> np.ndarray:
    """Simple 'tilt' EQ via two wide peaks (approximate): low cut + high lift around pivot."""
    x = peaking_eq(audio, sr, f0=max(80.0, pivot_hz/5), gain_db=-gain_db/2, Q=0.7)
    x = peaking_eq(x, sr, f0=min(sr*0.45, pivot_hz*5), gain_db=+gain_db/2, Q=0.7)
    return x.astype(np.float32)

# --------- Mid/Side & Stereo ---------

def mid_side_encode(audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (M, S) mid/side from stereo input."""
    x = _ensure_stereo(_sanitize(audio))
    L, R = x[:,0], x[:,1]
    M = 0.5 * (L + R)
    S = 0.5 * (L - R)
    return M.astype(np.float32), S.astype(np.float32)

def mid_side_decode(M: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Return stereo from (M, S)."""
    L = M + S
    R = M - S
    return np.column_stack([L, R]).astype(np.float32)

def stereo_widener(audio: np.ndarray, width: float = 1.0) -> np.ndarray:
    """
    Adjust stereo width by scaling the side channel.
    width = 1.0 unchanged; >1 wider; <1 narrower.
    """
    x = _ensure_stereo(_sanitize(audio))
    M, S = mid_side_encode(x)
    y = mid_side_decode(M, S * float(width))
    return y.astype(np.float32)

# --------- Dynamics ---------

def _envelope_detector(mono: np.ndarray, sr: int, attack_ms: float = 10.0, release_ms: float = 100.0) -> np.ndarray:
    a_a = np.exp(-1.0 / ((attack_ms/1000.0) * sr))
    a_r = np.exp(-1.0 / ((release_ms/1000.0) * sr))
    env = np.zeros_like(mono, dtype=np.float32)
    prev = 0.0
    for i, v in enumerate(np.abs(mono)):
        if v > prev:
            prev = a_a*prev + (1 - a_a)*v
        else:
            prev = a_r*prev + (1 - a_r)*v
        env[i] = prev
    return env

def compressor(audio: np.ndarray, sr: int,
               threshold_db: float = -18.0, ratio: float = 2.0,
               attack_ms: float = 15.0, release_ms: float = 120.0,
               makeup_db: float = 0.0, knee_db: float = 3.0,
               link_stereo: bool = True) -> np.ndarray:
    """
    Feed‑forward compressor with soft knee. If link_stereo=True, uses shared gain for both channels.
    """
    x = _ensure_stereo(_sanitize(audio))
    M = _mono(x) if link_stereo else None
    if link_stereo:
        env = _envelope_detector(M, sr, attack_ms, release_ms)
    else:
        envL = _envelope_detector(x[:,0], sr, attack_ms, release_ms)
        envR = _envelope_detector(x[:,1], sr, attack_ms, release_ms)

    thr = _db_to_lin(threshold_db)
    knee = _db_to_lin(max(0.0, knee_db))

    def gain_curve(env_val: float) -> float:
        e = env_val
        if e <= thr / knee:
            g = 1.0
        elif e <= thr * knee:
            edb = _lin_to_db(e)
            over = max(0.0, edb - threshold_db)
            comp_db = over - (over / ratio)
            g = _db_to_lin(-comp_db)
        else:
            edb = _lin_to_db(e)
            over = edb - threshold_db
            comp_db = over - (over / ratio)
            g = _db_to_lin(-comp_db)
        return g

    if link_stereo:
        gains = np.array([gain_curve(v) for v in env], dtype=np.float32)
        y = np.column_stack([x[:,0]*gains, x[:,1]*gains]).astype(np.float32)
    else:
        gainsL = np.array([gain_curve(v) for v in envL], dtype=np.float32)
        gainsR = np.array([gain_curve(v) for v in envR], dtype=np.float32)
        y = np.column_stack([x[:,0]*gainsL, x[:,1]*gainsR]).astype(np.float32)

    if makeup_db != 0.0:
        y = apply_gain_db(y, makeup_db)
    return y.astype(np.float32)

def transient_shaper(audio: np.ndarray, sr: int,
                     attack_gain_db: float = 0.0, sustain_gain_db: float = 0.0,
                     split_hz: float = 4000.0) -> np.ndarray:
    """
    Simple transient shaper: high‑band emphasizes transient (attack), low‑band controls sustain.
    Not a full envelope‑splitter, but useful as a primitive.
    """
    x = _ensure_stereo(_sanitize(audio))
    # Split into low/high
    low = lowpass_filter(x, sr, cutoff_hz=split_hz, order=2)
    high = x - low
    # Envelope of high band ~ transients proxy
    env_high = _envelope_detector(_mono(high), sr, attack_ms=2.0, release_ms=50.0)
    att = apply_gain_db(high, attack_gain_db)
    sus = apply_gain_db(low, sustain_gain_db)
    y = att + sus
    return y.astype(np.float32)

# --------- Fades ---------

def fade_in(audio: np.ndarray, sr: int, dur_s: float = 0.01) -> np.ndarray:
    x = _sanitize(audio)
    n = int(max(1, dur_s * sr))
    env = np.linspace(0.0, 1.0, n, dtype=np.float32)
    y = x.copy()
    if x.ndim == 1:
        y[:n] *= env
    else:
        y[:n, :] *= env[:, None]
    return y.astype(np.float32)

def fade_out(audio: np.ndarray, sr: int, dur_s: float = 0.01) -> np.ndarray:
    x = _sanitize(audio)
    n = int(max(1, dur_s * sr))
    env = np.linspace(1.0, 0.0, n, dtype=np.float32)
    y = x.copy()
    if x.ndim == 1:
        y[-n:] *= env
    else:
        y[-n:, :] *= env[:, None]
    return y.astype(np.float32)

print("DSP Primitives Layer loaded:")
print("- Gain/level: apply_gain_db, normalize_peak, normalize_lufs, measure_peak, measure_rms")
print("- Filters: highpass_filter, lowpass_filter, bandpass_filter, shelf_filter, peaking_eq, notch_filter, tilt_eq")
print("- Stereo: mid_side_encode, mid_side_decode, stereo_widener")
print("- Dynamics: compressor (soft‑knee), transient_shaper")
print("- Fades: fade_in, fade_out")
print("- K‑weighting/LUFS approx: k_weight, lufs_integrated_approx")
