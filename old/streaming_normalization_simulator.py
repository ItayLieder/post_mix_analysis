

# Streaming Normalization Simulator — Notebook Layer
# Requires: I/O + Analysis layer (for lufs_integrated_approx), and manifest helpers.
# Outputs "as-heard" WAVs for each platform and returns a summary table.

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Tuple
import os
import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal
from data_handler import *

# ---- Reuse analysis helpers if present; else provide fallbacks ----
def _sanitize(x: np.ndarray) -> np.ndarray:
    return np.clip(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), -4.0, 4.0).astype(np.float32)

def _db_to_lin(db: float) -> float:
    return 10.0**(db/20.0)

def _lin_to_db(l: float, eps: float = 1e-12) -> float:
    return 20.0*np.log10(max(eps, abs(l)))

def _k_weighting_sos(sr: int):
    f_hp = 38.0
    sos_hp = signal.butter(2, f_hp/(sr*0.5), btype="highpass", output="sos")
    # ~+4 dB shelf around 1 kHz
    f_shelf = 1681.974450955533; Q_shelf = 0.7071752369554196; gain_db = 3.99984385397
    A = 10**(gain_db/40.0); w0 = 2*np.pi*f_shelf/sr
    alpha = np.sin(w0)/(2*Q_shelf); cosw0 = np.cos(w0)
    b0 =    A*((A+1) + (A-1)*cosw0 + 2*np.sqrt(A)*alpha)
    b1 = -2*A*((A-1) + (A+1)*cosw0)
    b2 =    A*((A+1) + (A-1)*cosw0 - 2*np.sqrt(A)*alpha)
    a0 =        (A+1) - (A-1)*cosw0 + 2*np.sqrt(A)*alpha
    a1 =    2*((A-1) - (A+1)*cosw0)
    a2 =        (A+1) - (A-1)*cosw0 - 2*np.sqrt(A)*alpha
    sos_sh = signal.tf2sos([b0/a0, b1/a0, b2/a0], [1.0, a1/a0, a2/a0])
    return np.vstack([sos_hp, sos_sh])

def _lufs_integrated_approx(x: np.ndarray, sr: int) -> float:
    # Use existing function if present
    try:
        return lufs_integrated_approx(x, sr)  # defined in your Analysis layer
    except NameError:
        mono = x if x.ndim == 1 else np.mean(x, axis=1)
        sos = _k_weighting_sos(sr)
        xk = signal.sosfilt(sos, mono)
        ms = float(np.mean(xk**2))
        return -0.691 + 10.0*np.log10(max(1e-12, ms))

def _true_peak_dbtp(x: np.ndarray, sr: int, oversample: int = 4) -> float:
    try:
        # Prefer previously defined function name if present
        return true_peak_dbfs(x, sr, oversample=oversample)  # returns dBFS (≈ dBTP here)
    except NameError:
        x_os = signal.resample_poly(_sanitize(x), oversample, 1, axis=0 if x.ndim>1 else 0)
        tp = float(np.max(np.abs(x_os)))
        return 20.0*np.log10(max(1e-12, tp))

def _normalize_true_peak(x: np.ndarray, sr: int, target_dbtp: float = -1.0, oversample: int = 4) -> Tuple[np.ndarray, float]:
    tp = _true_peak_dbtp(x, sr, oversample=oversample)
    gain_db = target_dbtp - tp
    y = (_sanitize(x) * _db_to_lin(gain_db)).astype(np.float32)
    return y, gain_db

def _gentle_limiter(x: np.ndarray, ceiling_dbfs: float = -1.0, knee_db: float = 0.8) -> np.ndarray:
    # super-simple soft ceiling; not a brickwall TP limiter (good enough for preview)
    c = _db_to_lin(ceiling_dbfs)
    y = _sanitize(x).copy()
    mag = np.abs(y)
    over = np.maximum(0.0, mag - c)
    knee = _db_to_lin(-knee_db)
    over = over / (1.0 + (over / (knee*c))**2)
    y = np.sign(y) * np.minimum(mag, c)  # clamp
    # gentle blend to reduce clicks
    return (0.6*_sanitize(x) + 0.4*y).astype(np.float32)

# ---- Profiles ----
@dataclass
class StreamingProfile:
    name: str
    target_lufs: float       # platform loudness target (track mode)
    tp_ceiling_db: float     # approximate true-peak ceiling
    tp_strategy: str = "trim"  # "trim" (reduce gain) or "limit" (lightly limit)

def default_streaming_profiles() -> Dict[str, StreamingProfile]:
    # Typical/commonly-cited targets (approximate, for preview). Override as needed.
    return {
        "Spotify":     StreamingProfile("Spotify",     target_lufs=-14.0, tp_ceiling_db=-1.0, tp_strategy="trim"),
        "AppleMusic":  StreamingProfile("AppleMusic",  target_lufs=-16.0, tp_ceiling_db=-1.0, tp_strategy="trim"),
        "YouTube":     StreamingProfile("YouTube",     target_lufs=-14.0, tp_ceiling_db=-1.0, tp_strategy="trim"),
        "TIDAL":       StreamingProfile("TIDAL",       target_lufs=-14.0, tp_ceiling_db=-1.0, tp_strategy="trim"),
        "Amazon":      StreamingProfile("Amazon",      target_lufs=-14.0, tp_ceiling_db=-1.0, tp_strategy="trim"),
    }

# ---- Core simulation ----
def simulate_streaming_as_heard(x: np.ndarray, sr: int, profile: StreamingProfile,
                                oversample_tp: int = 4) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Return an 'as-heard' version after platform gain normalization and TP guard.
    """
    x = _sanitize(x)
    in_lufs = _lufs_integrated_approx(x, sr)
    in_tp = _true_peak_dbtp(x, sr, oversample=oversample_tp)

    # 1) loudness normalization (pure gain)
    delta_db = profile.target_lufs - in_lufs
    y = (x * _db_to_lin(delta_db)).astype(np.float32)

    # 2) true-peak guard
    after_tp = _true_peak_dbtp(y, sr, oversample=oversample_tp)
    tp_over = after_tp - profile.tp_ceiling_db
    tp_action = None
    trim_db = 0.0

    if tp_over > 0.0:
        if profile.tp_strategy == "limit":
            y = _gentle_limiter(y, ceiling_dbfs=profile.tp_ceiling_db, knee_db=0.8)
            tp_action = "limit"
        else:
            # trim enough to meet the ceiling
            y, trim_db = _normalize_true_peak(y, sr, target_dbtp=profile.tp_ceiling_db, oversample=oversample_tp)
            tp_action = "trim"

    out_lufs = _lufs_integrated_approx(y, sr)
    out_tp = _true_peak_dbtp(y, sr, oversample=oversample_tp)

    meta = {
        "profile": asdict(profile),
        "in_lufs": round(in_lufs, 2),
        "in_true_peak_dbTP": round(in_tp, 2),
        "gain_to_target_db": round(delta_db, 2),
        "tp_action": tp_action,
        "extra_trim_db": round(trim_db, 2) if tp_action == "trim" else 0.0,
        "out_lufs": round(out_lufs, 2),
        "out_true_peak_dbTP": round(out_tp, 2),
        "lufs_error_db": round(out_lufs - profile.target_lufs, 2)  # non-zero if we trimmed for TP
    }
    return y, meta

# ---- Batch runner + export ----
def simulate_and_export_for_platforms(
    input_path: str,
    out_dir: str,
    profiles: Optional[Dict[str, StreamingProfile]] = None,
    bit_depth: str = "PCM_24",
    register_to_manifest: Optional[tuple] = None,  # (manifest, kind_str)
) -> Tuple[List[str], pd.DataFrame]:
    """
    Generate 'as-heard' files for multiple platforms from a pre-master or master.
    Returns (list_of_paths, summary_dataframe).
    """
    profiles = profiles or default_streaming_profiles()

    # read
    x, sr = sf.read(input_path)
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    out_paths = []
    base = os.path.splitext(os.path.basename(input_path))[0]

    for name, prof in profiles.items():
        y, meta = simulate_streaming_as_heard(x, sr, prof)
        out_name = f"{base}__asheard_{name}.wav"
        out_path = os.path.join(out_dir, out_name)
        sf.write(out_path, y, sr, subtype=bit_depth)
        out_paths.append(out_path)

        row = {
            "platform": name,
            "target_lufs": prof.target_lufs,
            "tp_ceiling_db": prof.tp_ceiling_db,
            "tp_strategy": prof.tp_strategy,
            **{k: meta[k] for k in ["in_lufs","in_true_peak_dbTP","gain_to_target_db","tp_action","extra_trim_db","out_lufs","out_true_peak_dbTP","lufs_error_db"]},
            "asheard_path": out_path,
        }
        rows.append(row)

        # optional manifest registration
        if register_to_manifest is not None:
            man, kind = register_to_manifest
            register_artifact(man, out_path, kind=kind, params={"profile": name, **meta}, stage=f"asheard_{name}")

    df = pd.DataFrame(rows)
    return out_paths, df

def print_streaming_summary(df: pd.DataFrame):
    cols = [
        "platform","target_lufs","tp_ceiling_db",
        "in_lufs","gain_to_target_db","tp_action","extra_trim_db",
        "out_lufs","lufs_error_db","out_true_peak_dbTP"
    ]
    if set(cols).issubset(df.columns):
        print(df[cols].to_string(index=False))
    else:
        print(df.to_string(index=False))
