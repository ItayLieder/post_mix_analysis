
# Processors (Feature Macros) — Notebook Layer
# High-level, musical “dials” built on top of DSP primitives.
# - Bass (low-shelf + optional dynamic control)
# - Punch (kick/bass tightening via low-band ducking)
# - Clarity (mud dip around 160–250 Hz)
# - Air (HF shelf)
# - Width (M/S scaling with safety)
# - Pre‑master Prep (DC/sub cleanup + headroom target)
# - Dial mapping 0–100 → safe internal params
# - Fast preview via one-time preprocess cache
#
# Assumes the DSP primitives cell has been executed already.

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Tuple
import numpy as np
from dsp_premitives import *

# ---------- Dial mapping helpers ----------
# --------- Dynamics ---------

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


def _map(amount: float, a0: float, a1: float) -> float:
    """Linear map amount (0..100) to [a0..a1]."""
    amt = float(np.clip(amount, 0.0, 100.0))
    return a0 + (a1 - a0) * (amt / 100.0)

def _exp_map(amount: float, a0: float, a1: float) -> float:
    """Exponential-ish feel (more resolution at low values)."""
    amt = float(np.clip(amount, 0.0, 100.0)) / 100.0
    t = amt**1.6
    return a0 + (a1 - a0) * t

# ---------- Feature Macros (stateless) ----------

def make_bassier(x: np.ndarray, sr: int, amount: float,
                 base_hz: float = 80.0, max_db: float = 6.0,
                 dynamic: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Bass dial: low-shelf at ~80 Hz. amount=0..100 → 0..max_db dB.
    If dynamic=True, add mild low-band compression to control boom.
    """
    gain_db = _exp_map(amount, 0.0, max_db)  # gentle taper
    y = shelf_filter(x, sr, cutoff_hz=base_hz, gain_db=gain_db, kind="low", S=0.5)
    params = {"bass_gain_db": round(gain_db, 2), "bass_hz": base_hz}
    if dynamic and gain_db > 0.5:
        # compress lows below ~120 Hz slightly (ratio 1.5–2.0)
        low = lowpass_filter(y, sr, cutoff_hz=120.0, order=4)
        low_c = compressor(low, sr, threshold_db=-28.0, ratio=1.8, attack_ms=10, release_ms=120, makeup_db=0.0)
        y = y - low + low_c
        params["bass_dynamic"] = True
    else:
        params["bass_dynamic"] = False
    return y.astype(np.float32), params

def make_punchier(x: np.ndarray, sr: int, amount: float,
                  kick_lo: float = 40.0, kick_hi: float = 110.0,
                  low_cutoff: float = 120.0) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Punch dial: duck non-kick lows under the kick envelope. amount 0..100 → 0..~6 dB depth.
    """
    # Depth 0..6 dB, attack 3..6 ms, release 80..140 ms
    depth_db = _map(amount, 0.0, 6.0)
    atk_ms = _map(amount, 6.0, 3.0)
    rel_ms = _map(amount, 80.0, 140.0)  # a bit longer release for higher amounts

    # Build bands & envelope (single-pass per call; for fast preview use cache below)
    low = lowpass_filter(x, sr, cutoff_hz=low_cutoff, order=4)
    kick = bandpass_filter(x, sr, f_lo=kick_lo, f_hi=kick_hi, order=4)
    nonkick_low = low - kick

    env = _envelope_detector(_mono(kick), sr, attack_ms=4.0, release_ms=90.0)
    # Normalize envelope to 0..1 robustly
    p95 = np.percentile(env, 95) if env.size else 0.0
    if p95 <= 1e-9:
        gain_curve = np.ones_like(env, dtype=np.float32)
    else:
        env = np.clip(env / p95, 0.0, 1.0)
        # Map to gain curve
        floor_gain = 10**(-abs(depth_db)/20.0)
        gain_curve = (floor_gain + (1.0 - floor_gain) * (1.0 - env)).astype(np.float32)
    gain_curve = gain_curve[:, None]

    ducked_nonkick = nonkick_low * gain_curve
    lows_tight = ducked_nonkick + kick
    high = x - low
    y = lows_tight + high

    params = {"punch_depth_db": round(depth_db, 2), "kick_lo": kick_lo, "kick_hi": kick_hi, "low_cutoff": low_cutoff,
              "attack_ms": round(atk_ms, 1), "release_ms": round(rel_ms, 1)}
    return y.astype(np.float32), params

def reduce_mud(x: np.ndarray, sr: int, amount: float,
               mud_hz_center: float = 200.0) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Clarity dial: dip around 160–250 Hz. amount 0..100 → 0..3 dB cut.
    """
    cut_db = -_exp_map(amount, 0.0, 3.0)
    hz = _map(amount, 180.0, 230.0)  # shift center slightly with amount
    y = peaking_eq(x, sr, f0=hz, gain_db=cut_db, Q=1.0)
    params = {"mud_cut_db": round(cut_db, 2), "mud_hz": round(hz, 1)}
    return y.astype(np.float32), params

def add_air(x: np.ndarray, sr: int, amount: float) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Air dial: high-shelf at ~10 kHz. amount 0..100 → 0..4 dB.
    """
    db = _exp_map(amount, 0.0, 4.0)
    y = shelf_filter(x, sr, cutoff_hz=10000.0, gain_db=db, kind="high", S=0.5)
    return y.astype(np.float32), {"air_db": round(db, 2), "air_hz": 10000.0}

def widen_stereo(x: np.ndarray, sr: int, amount: float) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Width dial: scale side channel 1.0..1.4 while guarding mono-compat (soft limit).
    """
    width = _map(amount, 1.0, 1.4)
    # Soft limit if correlation is already low
    M, S = mid_side_encode(x)
    # Estimate correlation quickly
    denom = np.maximum(1e-9, np.sqrt(M**2) * np.sqrt(S**2))
    corr_est = float(np.mean((M * S) / denom))
    if corr_est < 0.15:
        width = min(width, 1.2)  # avoid over-wide if already decorrelated
    y = mid_side_decode(M, S * width)
    return y.astype(np.float32), {"width_factor": round(width, 3)}

def premaster_prep(x: np.ndarray, sr: int,
                   target_peak_dbfs: float = -6.0,
                   hpf_hz: float = 20.0) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Pre-master prep: gentle 20 Hz HPF + peak normalization to -6 dBFS.
    """
    y = highpass_filter(x, sr, cutoff_hz=hpf_hz, order=2)
    y = normalize_peak(y, target_dbfs=target_peak_dbfs)
    return y.astype(np.float32), {"hpf_hz": hpf_hz, "target_peak_dbfs": target_peak_dbfs}

# ---------- Fast preview via preprocess cache ----------

@dataclass
class PreviewCache:
    sr: int
    high: np.ndarray        # > low_cutoff
    low: np.ndarray         # < low_cutoff
    kick: np.ndarray        # kick band
    env01: np.ndarray       # normalized kick envelope 0..1
    low_cutoff: float
    kick_lo: float
    kick_hi: float

def build_preview_cache(x: np.ndarray, sr: int,
                        low_cutoff: float = 120.0,
                        kick_lo: float = 40.0,
                        kick_hi: float = 110.0) -> PreviewCache:
    low = lowpass_filter(x, sr, cutoff_hz=low_cutoff, order=4)
    high = x - low
    kick = bandpass_filter(x, sr, f_lo=kick_lo, f_hi=kick_hi, order=4)
    env = _envelope_detector(_mono(kick), sr, attack_ms=4.0, release_ms=90.0)
    p95 = np.percentile(env, 95) if env.size else 0.0
    env01 = np.zeros_like(env, dtype=np.float32) if p95 <= 1e-9 else np.clip(env / p95, 0.0, 1.0).astype(np.float32)
    return PreviewCache(sr=sr, high=high.astype(np.float32), low=low.astype(np.float32),
                        kick=kick.astype(np.float32), env01=env01,
                        low_cutoff=low_cutoff, kick_lo=kick_lo, kick_hi=kick_hi)

# Patch: fix render_from_cache width call (was returning a (array, params) tuple)
# Now we correctly unpack the tuple so Y stays a numpy array.

def render_from_cache(cache: PreviewCache,
                      bass_amount: float = 0.0,
                      punch_amount: float = 0.0,
                      clarity_amount: float = 0.0,
                      air_amount: float = 0.0,
                      width_amount: float = 0.0,
                      target_peak_dbfs: Optional[float] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Millisecond re-render from cached bands/envelope for interactive dials.
    """
    # Start from separate bands
    M = cache.low.copy()
    H = cache.high.copy()
    K = cache.kick.copy()
    nonkick = M - K

    # Bass: scale low band (acts like a shelf)
    bass_db = _exp_map(bass_amount, 0.0, 6.0)
    M = M * (10**(bass_db/20.0))

    # Punch: duck non-kick lows with precomputed envelope
    depth_db = _map(punch_amount, 0.0, 6.0)
    floor_gain = 10**(-abs(depth_db)/20.0)
    g_sc = (floor_gain + (1.0 - floor_gain) * (1.0 - cache.env01)).astype(np.float32)
    ducked_nonkick = nonkick * g_sc[:, None]
    lows_tight = ducked_nonkick + K

    Y = lows_tight + H

    # Clarity: light peaking dip ~200 Hz (approximate via single biquad now)
    if clarity_amount > 0.0:
        cut_db = -_exp_map(clarity_amount, 0.0, 3.0)
        Y = peaking_eq(Y, cache.sr, f0=_map(clarity_amount, 180.0, 230.0), gain_db=cut_db, Q=1.0)

    # Air: high shelf ~10 kHz
    if air_amount > 0.0:
        air_db = _exp_map(air_amount, 0.0, 4.0)
        Y = shelf_filter(Y, cache.sr, cutoff_hz=10000.0, gain_db=air_db, kind="high", S=0.5)

    # Width: side scaling (correctly unpack tuple)
    if width_amount > 0.0:
        Y, _params_w = widen_stereo(Y, cache.sr, amount=width_amount)

    params = {
        "bass_db": round(bass_db, 2),
        "punch_depth_db": round(depth_db, 2),
        "clarity_db": round(-_exp_map(clarity_amount, 0.0, 3.0), 2) if clarity_amount>0 else 0.0,
        "air_db": round(_exp_map(air_amount, 0.0, 4.0), 2) if air_amount>0 else 0.0,
        "width_amount": round(width_amount, 2),
    }

    if target_peak_dbfs is not None:
        Y = normalize_peak(Y, target_dbfs=target_peak_dbfs)

    return Y.astype(np.float32), params

print("Patched: render_from_cache now unpacks widen_stereo tuple correctly.")


print("Processors (Feature Macros) loaded: make_bassier, make_punchier, reduce_mud, add_air, widen_stereo, premaster_prep, build_preview_cache, render_from_cache.")
