
# Analysis Layer — Notebook Version
# Robust audio analysis utilities for post-mix:
# - Health checks (DC, peak/RMS, true-peak approx, headroom, NaN/Inf)
# - Loudness (K-weighted momentary/short-term + approx integrated LUFS)
# - Dynamics proxies (crest, short-term distribution, DR proxy)
# - Spectrum & band energy (bass/air %), spectral flatness
# - Stereo metrics (phase correlation, width proxy, mid/side peaks)
# - Plots: spectrum, short-term loudness, waveform excerpt (matplotlib; no seaborn)
#
# Designed to integrate with the earlier I/O layer (AudioBuffer, load_wav).

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt

try:
    from caas_jupyter_tools import display_dataframe_to_user
except Exception:
    display_dataframe_to_user = None


# ----------------------------- Utilities -----------------------------

def _lin_to_db(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(eps, np.abs(x)))

def _db_to_lin(db: float) -> float:
    return 10.0 ** (db / 20.0)

def _ensure_stereo(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return np.stack([x, x], axis=-1)
    if x.shape[1] == 1:
        return np.repeat(x, 2, axis=1)
    return x

def _mono(x: np.ndarray) -> np.ndarray:
    return x if x.ndim == 1 else np.mean(x, axis=1)

def _sanitize(x: np.ndarray) -> np.ndarray:
    return np.clip(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)


# ----------------------------- True Peak (approx) -----------------------------

def true_peak_dbfs(x: np.ndarray, sr: int, oversample: int = 4) -> float:
    """
    Approximate true peak by oversampling with polyphase resampling and taking the max.
    """
    x = _sanitize(x)
    x_os = signal.resample_poly(x, oversample, 1, axis=0 if x.ndim > 1 else 0)
    tp = float(np.max(np.abs(x_os)))
    return float(_lin_to_db(np.array([tp]))[0])


# ----------------------------- K-Weighting (BS.1770-style) -----------------------------

def _k_weighting_sos(sr: int):
    """
    Return SOS for the K-weighting pre-filter + high-frequency shelf per ITU-R BS.1770.
    Using standard bilinear transforms for the defined z-plane filters.
    """
    # High-pass (2nd order) at 38 Hz (pre-filter)
    f_hp = 38.0
    # High-shelf (2nd order) with +4 dB above ~1 kHz
    f_shelf = 1681.974450955533
    Q_shelf = 0.7071752369554196
    gain_db = 3.99984385397  # ~+4 dB

    # HPF
    sos_hp = signal.butter(2, f_hp/(sr*0.5), btype='highpass', output='sos')

    # High-shelf (RBJ biquad in SOS form)
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

def k_weight(x: np.ndarray, sr: int) -> np.ndarray:
    """Apply K-weighting to mono signal."""
    sos = _k_weighting_sos(sr)
    return signal.sosfilt(sos, x)

def lufs_momentary(x_mono: np.ndarray, sr: int, window_s: float = 0.4) -> Tuple[np.ndarray, np.ndarray]:
    """
    400 ms momentary LUFS approximation (BS.1770 weighting, no gating).
    Returns (time_axis, lufs_array).
    """
    xk = k_weight(x_mono, sr)
    win = int(max(1, round(window_s * sr)))
    # mean square via moving average
    kernel = np.ones(win, dtype=np.float32) / float(win)
    ms = np.convolve(xk**2, kernel, mode='same')
    lufs = -0.691 + 10.0 * np.log10(np.maximum(1e-12, ms))  # -0.691 is the BS.1770 absolute scale offset
    t = np.arange(len(lufs)) / sr
    return t, lufs

def lufs_integrated_approx(x_mono: np.ndarray, sr: int) -> float:
    """
    Very lightweight integrated LUFS approximation (K-weighted, no gating).
    For streaming normalization preview, this is often sufficient.
    """
    xk = k_weight(x_mono, sr)
    ms = np.mean(xk**2)
    lufs = -0.691 + 10.0 * np.log10(np.maximum(1e-12, ms))
    return float(lufs)


# ----------------------------- Spectrum & Bands -----------------------------

def spectrum_db(mono: np.ndarray, sr: int, n_fft: int = 1<<16) -> Tuple[np.ndarray, np.ndarray]:
    seg = mono[:min(len(mono), n_fft)]
    if len(seg) < 2048:
        pad = np.zeros(2048, dtype=seg.dtype)
        pad[:len(seg)] = seg
        seg = pad
    win = np.hanning(len(seg))
    sp = np.fft.rfft(seg * win)
    freqs = np.fft.rfftfreq(len(seg), 1/sr)
    mag_db = _lin_to_db(np.abs(sp))
    return freqs, mag_db

def band_energy_percent(mono: np.ndarray, sr: int, f_lo: float, f_hi: float) -> float:
    n = 1<<16
    seg = mono[:min(len(mono), n)]
    win = np.hanning(len(seg))
    sp = np.fft.rfft(seg * win)
    freqs = np.fft.rfftfreq(len(seg), 1/sr)
    power = (np.abs(sp)**2)
    total = np.sum(power) + 1e-20
    mask = (freqs >= f_lo) & (freqs < f_hi)
    band = np.sum(power[mask])
    return float(100.0 * band / total)

def spectral_flatness(mono: np.ndarray, sr: int) -> float:
    n = 1<<14
    seg = mono[:min(len(mono), n)]
    win = np.hanning(len(seg))
    sp = np.abs(np.fft.rfft(seg * win)) + 1e-12
    geo = np.exp(np.mean(np.log(sp)))
    ari = np.mean(sp)
    return float(np.clip(geo / ari, 0.0, 1.0))


# ----------------------------- Stereo Metrics -----------------------------

def stereo_metrics(x: np.ndarray) -> Dict[str, float]:
    x = _ensure_stereo(x)
    L = x[:, 0]; R = x[:, 1]
    # Phase correlation: mean of normalized instantaneous product
    denom = np.maximum(1e-12, np.sqrt(L**2) * np.sqrt(R**2))
    corr = float(np.mean((L * R) / denom))
    # Width proxy using mid/side
    M = 0.5 * (L + R); S = 0.5 * (L - R)
    width = float(np.mean(np.abs(S)) / (np.mean(np.abs(M)) + 1e-12))
    return {
        "phase_correlation": float(np.clip(corr, -1.0, 1.0)),
        "stereo_width": width,
        "mid_peak_db": float(_lin_to_db(np.array([np.max(np.abs(M))]))[0]),
        "side_peak_db": float(_lin_to_db(np.array([np.max(np.abs(S))]))[0]),
    }


# ----------------------------- Health & Dynamics -----------------------------

def health_metrics(x: np.ndarray, sr: int) -> Dict[str, float]:
    mono = _mono(x)
    dc = float(np.mean(mono))
    dc_db = float(_lin_to_db(np.array([abs(dc) if abs(dc) > 0 else 1e-12]))[0])
    peak = float(np.max(np.abs(mono)))
    peak_db = float(_lin_to_db(np.array([peak]))[0])
    rms = float(np.sqrt(np.mean(mono**2)))
    rms_db = float(_lin_to_db(np.array([rms]))[0])
    crest = peak_db - rms_db
    sub_pct = band_energy_percent(mono, sr, 0.0, 30.0)
    return {
        "peak_dbfs": peak_db,
        "rms_dbfs": rms_db,
        "crest_db": crest,
        "dc_offset": dc,
        "dc_dbfs": dc_db,
        "sub_30Hz_%": sub_pct,
    }

def short_term_loudness(mono: np.ndarray, sr: int, win_s: float = 3.0, hop_s: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    win = int(max(2, round(win_s * sr)))
    hop = int(max(1, round(hop_s * sr)))
    kernel = np.ones(win, dtype=np.float32) / float(win)
    pow_sig = mono**2
    rms = np.sqrt(np.maximum(1e-20, np.convolve(pow_sig, kernel, mode="same")))
    idx = np.arange(0, len(mono), hop)
    return idx / sr, _lin_to_db(rms[idx])

def dr_proxy(mono: np.ndarray, sr: int) -> float:
    t, st = short_term_loudness(mono, sr, win_s=3.0, hop_s=0.5)
    return float(np.percentile(st, 95) - np.percentile(st, 10))


# ----------------------------- Analyzer Facade -----------------------------

@dataclass
class AnalysisReport:
    sr: int
    duration_s: float
    basic: Dict[str, float]
    stereo: Dict[str, float]
    lufs_integrated: float
    true_peak_dbfs: float
    bass_energy_pct: float
    air_energy_pct: float
    spectral_flatness: float

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        rows.append({"metric": "sr", "value": self.sr})
        rows.append({"metric": "duration_s", "value": self.duration_s})
        for k, v in self.basic.items():
            rows.append({"metric": k, "value": v})
        for k, v in self.stereo.items():
            rows.append({"metric": k, "value": v})
        rows.append({"metric": "lufs_integrated", "value": self.lufs_integrated})
        rows.append({"metric": "true_peak_dbfs", "value": self.true_peak_dbfs})
        rows.append({"metric": "bass_energy_%", "value": self.bass_energy_pct})
        rows.append({"metric": "air_energy_%", "value": self.air_energy_pct})
        rows.append({"metric": "spectral_flatness", "value": self.spectral_flatness})
        return pd.DataFrame(rows)

def analyze_audio_array(x: np.ndarray, sr: int) -> AnalysisReport:
    x = _sanitize(x)
    x = _ensure_stereo(x)
    mono = _mono(x)
    dur_s = len(mono) / sr

    basic = health_metrics(x, sr)
    stereo = stereo_metrics(x)
    tp = true_peak_dbfs(x, sr, oversample=4)
    bass_pct = band_energy_percent(mono, sr, 20.0, 120.0)
    air_pct = band_energy_percent(mono, sr, 8000.0, sr/2)
    flat = spectral_flatness(mono, sr)
    lufs_i = lufs_integrated_approx(mono, sr)

    return AnalysisReport(
        sr=sr,
        duration_s=dur_s,
        basic=basic,
        stereo=stereo,
        lufs_integrated=lufs_i,
        true_peak_dbfs=tp,
        bass_energy_pct=bass_pct,
        air_energy_pct=air_pct,
        spectral_flatness=flat,
    )

def analyze_wav(path: str, target_sr: Optional[int] = None) -> AnalysisReport:
    sr, data = wavfile.read(path)
    x = data.astype(np.float32)
    if data.dtype == np.int16:
        x = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        x = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        x = (data.astype(np.float32) - 128.0) / 128.0
    # resample if requested
    if target_sr and target_sr != sr:
        gcd = np.gcd(sr, target_sr)
        x = signal.resample_poly(x, target_sr//gcd, sr//gcd, axis=0 if x.ndim > 1 else 0)
        sr = target_sr
    return analyze_audio_array(x, sr)


# ----------------------------- Plot Helpers -----------------------------

def plot_spectrum(path_or_array, sr: Optional[int] = None, fmax: float = 20000.0):
    if isinstance(path_or_array, str):
        rep = analyze_wav(path_or_array)
        sr0, data = wavfile.read(path_or_array)
        x = data.astype(np.float32)
        if data.dtype == np.int16:
            x = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            x = data.astype(np.float32) / 2147483648.0
        elif data.dtype == np.uint8:
            x = (data.astype(np.float32) - 128.0) / 128.0
        mono = _mono(x)
        freqs, mag_db = spectrum_db(mono, sr0)
    else:
        x = path_or_array
        assert sr is not None, "When passing an array, provide sr."
        mono = _mono(x)
        freqs, mag_db = spectrum_db(mono, sr)
        rep = None

    mask = freqs <= fmax
    plt.figure()
    plt.plot(freqs[mask], mag_db[mask])
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title("Magnitude Spectrum")
    plt.show()
    return rep

def plot_short_term_loudness(path_or_array, sr: Optional[int] = None, win_s: float = 3.0, hop_s: float = 0.5):
    if isinstance(path_or_array, str):
        sr0, data = wavfile.read(path_or_array)
        x = data.astype(np.float32)
        if data.dtype == np.int16:
            x = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            x = data.astype(np.float32) / 2147483648.0
        elif data.dtype == np.uint8:
            x = (data.astype(np.float32) - 128.0) / 128.0
        mono = _mono(x)
        t, st = short_term_loudness(mono, sr0, win_s=win_s, hop_s=hop_s)
    else:
        x = path_or_array
        assert sr is not None, "When passing an array, provide sr."
        mono = _mono(x)
        t, st = short_term_loudness(mono, sr, win_s=win_s, hop_s=hop_s)

    plt.figure()
    plt.plot(t, st)
    plt.xlabel("Time (s)")
    plt.ylabel("Short-term RMS (dBFS)")
    plt.title("Short-term Loudness")
    plt.show()

def plot_waveform_excerpt(path_or_array, sr: Optional[int] = None, start_s: float = 0.0, dur_s: float = 10.0):
    if isinstance(path_or_array, str):
        sr0, data = wavfile.read(path_or_array)
        x = data.astype(np.float32)
        if data.dtype == np.int16:
            x = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            x = data.astype(np.float32) / 2147483648.0
        elif data.dtype == np.uint8:
            x = (data.astype(np.float32) - 128.0) / 128.0
        sr = sr0
    else:
        x = path_or_array
        assert sr is not None, "When passing an array, provide sr."

    x = _ensure_stereo(x)
    n0 = int(start_s * sr); n1 = int((start_s + dur_s) * sr)
    n1 = min(n1, x.shape[0])
    t = np.arange(n0, n1) / sr
    mono = _mono(x[n0:n1, :]) if x.ndim > 1 else _mono(x[n0:n1])

    plt.figure()
    plt.plot(t, mono)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (mono)")
    plt.title(f"Waveform Excerpt ({start_s:.1f}s–{start_s+dur_s:.1f}s)")
    plt.show()


# ----------------------------- Tabular Summary -----------------------------

def analysis_table(report: AnalysisReport, name: str = "Track Analysis") -> pd.DataFrame:
    df = report.to_dataframe()
    if display_dataframe_to_user:
        display_dataframe_to_user(name, df)
    else:
        print(df.to_string(index=False))
    return df

print("Analysis layer loaded: analyze_wav/analyze_audio_array, analysis_table, plot_spectrum, plot_short_term_loudness, plot_waveform_excerpt, LUFS approx, true-peak approx, stereo & health metrics.")
