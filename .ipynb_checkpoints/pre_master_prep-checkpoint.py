

# --- Pre-Master Prep Layer ---

import numpy as np
import soundfile as sf

def sanitize_audio(x: np.ndarray) -> np.ndarray:
    """Replace NaN/Inf and clamp extreme outliers."""
    return np.clip(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0),
                   -4.0, 4.0).astype(np.float32)

def highpass_filter(audio: np.ndarray, sr: int, cutoff_hz: float = 20.0, order: int = 2) -> np.ndarray:
    """Gentle high-pass to clear subsonics."""
    from scipy.signal import butter, filtfilt
    b, a = butter(order, cutoff_hz / (sr/2.0), btype='highpass')
    return filtfilt(b, a, audio, axis=0)

def normalize_peak(audio: np.ndarray, target_dbfs: float = -6.0, eps: float = 1e-9) -> tuple[np.ndarray, float]:
    """Scale so peak hits target_dbfs. Returns (scaled_audio, applied_gain_db)."""
    x = sanitize_audio(audio)
    peak = float(np.max(np.abs(x)))
    if peak < eps:
        return x, 0.0
    current_db = 20 * np.log10(peak + eps)
    gain_db = target_dbfs - current_db
    gain_lin = 10 ** (gain_db/20)
    return (x * gain_lin).astype(np.float32), gain_db

def premaster_prep(audio: np.ndarray, sr: int,
                   target_peak_dbfs: float = -6.0,
                   hpf_hz: float = 20.0) -> tuple[np.ndarray, dict]:
    """Do full pre-master prep: sanitize, HPF, normalize to headroom."""
    y = sanitize_audio(audio)
    if hpf_hz:
        y = highpass_filter(y, sr, cutoff_hz=hpf_hz)
    y, gain_db = normalize_peak(y, target_dbfs=target_peak_dbfs)

    meta = {
        "target_peak_dbfs": target_peak_dbfs,
        "applied_gain_db": round(gain_db, 2),
        "hpf_hz": hpf_hz,
        "sr": sr,
        "peak_after": round(float(np.max(np.abs(y))), 4)
    }
    return y, meta
