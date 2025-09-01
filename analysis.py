
# Analysis Layer — Cleaned Up Version
# Audio analysis utilities for post-mix processing:
# - Health checks (DC, peak/RMS, true-peak, headroom)
# - Loudness (K-weighted LUFS estimation)
# - Dynamics analysis (crest factor, dynamic range)
# - Spectrum & band energy analysis
# - Stereo metrics (phase correlation, width)
# - Visualization tools

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
from config import CONFIG
from utils import (
    to_float32, sanitize_audio, ensure_stereo, to_mono,
    linear_to_db, db_to_linear, true_peak_db, rms_db, crest_factor_db
)

try:
    from caas_jupyter_tools import display_dataframe_to_user
except Exception:
    display_dataframe_to_user = None


# ----------------------------- Audio Processing -----------------------------


# ----------------------------- K-Weighting (BS.1770-style) -----------------------------

def _k_weighting_sos(sr: int):
    """
    Return SOS for the K-weighting pre-filter + high-frequency shelf per ITU-R BS.1770.
    Using standard bilinear transforms for the defined z-plane filters.
    """
    # Use configuration values for K-weighting filter
    f_hp = CONFIG.analysis.k_weight_hp_freq
    f_shelf = CONFIG.analysis.k_weight_shelf_freq
    Q_shelf = CONFIG.analysis.k_weight_shelf_q
    gain_db = CONFIG.analysis.k_weight_shelf_gain_db

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

def lufs_momentary(x_mono: np.ndarray, sr: int, window_s: float = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Momentary LUFS approximation (BS.1770 weighting, no gating).
    Returns (time_axis, lufs_array).
    """
    if window_s is None:
        window_s = CONFIG.analysis.momentary_window_s
    
    xk = k_weight(x_mono, sr)
    win = int(max(1, round(window_s * sr)))
    # mean square via moving average
    kernel = np.ones(win, dtype=np.float32) / float(win)
    ms = np.convolve(xk**2, kernel, mode='same')
    lufs = CONFIG.audio.lufs_bs1770_offset + 10.0 * np.log10(np.maximum(1e-12, ms))
    t = np.arange(len(lufs)) / sr
    return t, lufs

def lufs_integrated_approx(x_mono: np.ndarray, sr: int) -> float:
    """
    Very lightweight integrated LUFS approximation (K-weighted, no gating).
    For streaming normalization preview, this is often sufficient.
    """
    xk = k_weight(x_mono, sr)
    ms = np.mean(xk**2)
    lufs = CONFIG.audio.lufs_bs1770_offset + 10.0 * np.log10(np.maximum(1e-12, ms))
    return float(lufs)


# ----------------------------- Spectrum & Bands -----------------------------

def spectrum_db(mono: np.ndarray, sr: int, n_fft: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate magnitude spectrum in dB."""
    if n_fft is None:
        n_fft = CONFIG.analysis.spectrum_nfft
    
    seg = mono[:min(len(mono), n_fft)]
    if len(seg) < 2048:
        pad = np.zeros(2048, dtype=seg.dtype)
        pad[:len(seg)] = seg
        seg = pad
    
    # Use configured window type
    if CONFIG.analysis.window_type == "hanning":
        win = np.hanning(len(seg))
    else:
        win = np.hanning(len(seg))  # Default fallback
    
    sp = np.fft.rfft(seg * win)
    freqs = np.fft.rfftfreq(len(seg), 1/sr)
    mag_db = linear_to_db(np.abs(sp))
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
    """Calculate spectral flatness (Wiener entropy)."""
    n = CONFIG.analysis.flatness_nfft
    seg = mono[:min(len(mono), n)]
    
    # Use configured window type
    if CONFIG.analysis.window_type == "hanning":
        win = np.hanning(len(seg))
    else:
        win = np.hanning(len(seg))  # Default fallback
    
    sp = np.abs(np.fft.rfft(seg * win)) + 1e-12
    geo = np.exp(np.mean(np.log(sp)))
    ari = np.mean(sp)
    return float(np.clip(geo / ari, 0.0, 1.0))


# ----------------------------- Stereo Metrics -----------------------------

def stereo_metrics(x: np.ndarray) -> Dict[str, float]:
    """Calculate stereo imaging metrics."""
    stereo = ensure_stereo(x)
    left, right = stereo[:, 0], stereo[:, 1]
    
    # Phase correlation
    denominator = np.maximum(1e-12, np.sqrt(left**2) * np.sqrt(right**2))
    correlation = float(np.mean((left * right) / denominator))
    correlation = np.clip(correlation, -1.0, 1.0)
    
    # Mid/Side analysis for width
    mid = 0.5 * (left + right)
    side = 0.5 * (left - right)
    width = float(np.mean(np.abs(side)) / (np.mean(np.abs(mid)) + 1e-12))
    
    return {
        "phase_correlation": float(correlation),
        "stereo_width": width,
        "mid_peak_db": linear_to_db(np.max(np.abs(mid))),
        "side_peak_db": linear_to_db(np.max(np.abs(side))),
    }


# ----------------------------- Health & Dynamics -----------------------------

def health_metrics(x: np.ndarray, sr: int) -> Dict[str, float]:
    """Calculate basic audio health metrics."""
    mono = to_mono(x)
    dc_offset = float(np.mean(mono))
    dc_db = linear_to_db(abs(dc_offset) if abs(dc_offset) > 0 else 1e-12)
    
    peak_db = linear_to_db(np.max(np.abs(mono)))
    rms_level_db = rms_db(mono)
    crest_db = crest_factor_db(mono)
    
    sub_30hz_pct = band_energy_percent(mono, sr, 0.0, CONFIG.audio.subsonic_cutoff)
    
    return {
        "peak_dbfs": peak_db,
        "rms_dbfs": rms_level_db,
        "crest_db": crest_db,
        "dc_offset": dc_offset,
        "dc_dbfs": dc_db,
        "sub_30Hz_%": sub_30hz_pct,
    }

def short_term_loudness(mono: np.ndarray, sr: int, win_s: float = None, hop_s: float = None) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate short-term RMS loudness over time."""
    if win_s is None:
        win_s = CONFIG.analysis.short_term_window_s
    if hop_s is None:
        hop_s = CONFIG.analysis.short_term_hop_s
        
    window_samples = int(max(2, round(win_s * sr)))
    hop_samples = int(max(1, round(hop_s * sr)))
    
    kernel = np.ones(window_samples, dtype=np.float32) / float(window_samples)
    power_signal = mono**2
    rms_values = np.sqrt(np.maximum(1e-20, np.convolve(power_signal, kernel, mode="same")))
    
    time_indices = np.arange(0, len(mono), hop_samples)
    time_axis = time_indices / sr
    loudness_db = linear_to_db(rms_values[time_indices])
    
    return time_axis, loudness_db

def dr_proxy(mono: np.ndarray, sr: int) -> float:
    """Calculate dynamic range proxy using configured percentiles."""
    t, st = short_term_loudness(mono, sr)
    return float(np.percentile(st, CONFIG.analysis.dr_high_percentile) - 
                np.percentile(st, CONFIG.analysis.dr_low_percentile))


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
    """Analyze audio array and return comprehensive metrics."""
    audio = sanitize_audio(x, clip_range=1.0)
    audio_stereo = ensure_stereo(audio)
    audio_mono = to_mono(audio_stereo)
    duration = len(audio_mono) / sr

    # Calculate all metrics
    basic_metrics = health_metrics(audio, sr)
    stereo_metrics_dict = stereo_metrics(audio_stereo)
    true_peak_dbfs = true_peak_db(audio, sr, oversample=4)
    bass_pct = band_energy_percent(audio_mono, sr, CONFIG.audio.bass_freq_low, CONFIG.audio.bass_freq_high)
    air_pct = band_energy_percent(audio_mono, sr, CONFIG.audio.air_freq_low, sr/2)
    spectral_flat = spectral_flatness(audio_mono, sr)
    lufs_integrated = lufs_integrated_approx(audio_mono, sr)

    return AnalysisReport(
        sr=sr,
        duration_s=duration,
        basic=basic_metrics,
        stereo=stereo_metrics_dict,
        lufs_integrated=lufs_integrated,
        true_peak_dbfs=true_peak_dbfs,
        bass_energy_pct=bass_pct,
        air_energy_pct=air_pct,
        spectral_flatness=spectral_flat,
    )

def analyze_wav(path: str, target_sr: Optional[int] = None) -> AnalysisReport:
    """Analyze WAV file and return comprehensive metrics."""
    sr, data = wavfile.read(path)
    audio = to_float32(data)
    
    # Resample if requested
    if target_sr and target_sr != sr:
        gcd = np.gcd(sr, target_sr)
        audio = signal.resample_poly(audio, target_sr//gcd, sr//gcd, axis=0 if audio.ndim > 1 else 0)
        sr = target_sr
    
    return analyze_audio_array(audio, sr)


# ----------------------------- Plot Helpers -----------------------------

def plot_spectrum(path_or_array, sr: Optional[int] = None, fmax: float = 20000.0):
    """Plot magnitude spectrum of audio."""
    if isinstance(path_or_array, str):
        # Load from file
        sr_file, data = wavfile.read(path_or_array)
        audio = to_float32(data)
        mono = to_mono(audio)
        freqs, mag_db = spectrum_db(mono, sr_file)
        analysis_report = analyze_wav(path_or_array)
    else:
        # Use provided array
        if sr is None:
            raise ValueError("Sample rate required when passing audio array")
        audio = path_or_array
        mono = to_mono(audio)
        freqs, mag_db = spectrum_db(mono, sr)
        analysis_report = None

    # Plot spectrum up to fmax
    freq_mask = freqs <= fmax
    plt.figure(figsize=(10, 6))
    plt.plot(freqs[freq_mask], mag_db[freq_mask])
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title("Magnitude Spectrum")
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return analysis_report

def plot_short_term_loudness(path_or_array, sr: Optional[int] = None, win_s: float = 3.0, hop_s: float = 0.5):
    """Plot short-term loudness over time."""
    if isinstance(path_or_array, str):
        # Load from file
        sr_file, data = wavfile.read(path_or_array)
        audio = to_float32(data)
        mono = to_mono(audio)
        time_axis, loudness = short_term_loudness(mono, sr_file, win_s=win_s, hop_s=hop_s)
    else:
        # Use provided array
        if sr is None:
            raise ValueError("Sample rate required when passing audio array")
        audio = path_or_array
        mono = to_mono(audio)
        time_axis, loudness = short_term_loudness(mono, sr, win_s=win_s, hop_s=hop_s)

    plt.figure(figsize=(12, 6))
    plt.plot(time_axis, loudness)
    plt.xlabel("Time (s)")
    plt.ylabel("Short-term RMS (dBFS)")
    plt.title("Short-term Loudness")
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_waveform_excerpt(path_or_array, sr: Optional[int] = None, start_s: float = 0.0, dur_s: float = 10.0):
    """Plot waveform excerpt."""
    if isinstance(path_or_array, str):
        # Load from file
        sr_file, data = wavfile.read(path_or_array)
        audio = to_float32(data)
        sr = sr_file
    else:
        # Use provided array
        if sr is None:
            raise ValueError("Sample rate required when passing audio array")
        audio = path_or_array

    # Extract excerpt
    audio_stereo = ensure_stereo(audio)
    start_sample = int(start_s * sr)
    end_sample = int((start_s + dur_s) * sr)
    end_sample = min(end_sample, audio_stereo.shape[0])
    
    time_axis = np.arange(start_sample, end_sample) / sr
    excerpt = audio_stereo[start_sample:end_sample]
    mono_excerpt = to_mono(excerpt)

    plt.figure(figsize=(12, 6))
    plt.plot(time_axis, mono_excerpt)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(f"Waveform Excerpt ({start_s:.1f}s–{start_s+dur_s:.1f}s)")
    plt.grid(True, alpha=0.3)
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
