"""
Centralized audio utilities to eliminate code duplication.
Handles format conversion, sanitization, and common audio operations.
"""

import numpy as np
from typing import Union
from scipy import signal


def to_float32(data: np.ndarray) -> np.ndarray:
    """
    Convert various integer audio formats to float32 in [-1, 1] range.
    
    Args:
        data: Input audio data in any supported format
        
    Returns:
        Audio data as float32 in [-1, 1] range
    """
    if data.dtype == np.int16:
        return data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        return data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        return (data.astype(np.float32) - 128.0) / 128.0
    else:
        return data.astype(np.float32)


def sanitize_audio(x: np.ndarray, clip_range: float = 4.0) -> np.ndarray:
    """
    Clean audio by removing NaN/Inf and clipping to safe range.
    
    Args:
        x: Input audio array
        clip_range: Maximum absolute value to clip to
        
    Returns:
        Cleaned audio array as float32
    """
    cleaned = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(cleaned, -clip_range, clip_range).astype(np.float32)


def ensure_stereo(x: np.ndarray) -> np.ndarray:
    """
    Convert mono audio to stereo by duplicating channels.
    
    Args:
        x: Audio array, mono or stereo
        
    Returns:
        Stereo audio array (N, 2)
    """
    if x.ndim == 1:
        return np.stack([x, x], axis=-1)
    elif x.shape[1] == 1:
        return np.repeat(x, 2, axis=1)
    return x


def to_mono(x: np.ndarray) -> np.ndarray:
    """
    Convert stereo audio to mono by averaging channels.
    
    Args:
        x: Audio array, mono or stereo
        
    Returns:
        Mono audio array (N,)
    """
    if x.ndim == 1:
        return x
    return np.mean(x, axis=1)


def db_to_linear(db: float) -> float:
    """Convert decibels to linear amplitude."""
    return 10.0 ** (db / 20.0)


def linear_to_db(amplitude: Union[float, np.ndarray], eps: float = 1e-12) -> Union[float, np.ndarray]:
    """Convert linear amplitude to decibels."""
    amplitude = np.maximum(eps, np.abs(amplitude))
    return 20.0 * np.log10(amplitude)


def true_peak_db(x: np.ndarray, sr: int, oversample: int = 4) -> float:
    """
    Estimate true peak in dB using oversampling.
    
    Args:
        x: Input audio
        sr: Sample rate
        oversample: Oversampling factor
        
    Returns:
        True peak estimate in dBFS
    """
    x_clean = sanitize_audio(x, clip_range=1.0)
    x_oversampled = signal.resample_poly(
        x_clean, oversample, 1, 
        axis=0 if x_clean.ndim > 1 else 0
    )
    peak = float(np.max(np.abs(x_oversampled)))
    return linear_to_db(peak)


def normalize_peak(x: np.ndarray, target_dbfs: float = -1.0) -> np.ndarray:
    """
    Normalize audio to target peak level.
    
    Args:
        x: Input audio
        target_dbfs: Target peak level in dBFS
        
    Returns:
        Normalized audio
    """
    current_peak = float(np.max(np.abs(x)))
    if current_peak == 0:
        return x
    
    current_db = linear_to_db(current_peak)
    gain_db = target_dbfs - current_db
    gain_linear = db_to_linear(gain_db)
    
    return (x * gain_linear).astype(np.float32)


def normalize_true_peak(x: np.ndarray, sr: int, target_dbtp: float = -1.0) -> np.ndarray:
    """
    Normalize audio to target true peak level.
    
    Args:
        x: Input audio
        sr: Sample rate
        target_dbtp: Target true peak in dBTP
        
    Returns:
        Normalized audio
    """
    current_tp = true_peak_db(x, sr)
    gain_db = target_dbtp - current_tp
    gain_linear = db_to_linear(gain_db)
    
    return (sanitize_audio(x) * gain_linear).astype(np.float32)


def validate_audio(x: np.ndarray, name: str = "audio") -> None:
    """
    Validate audio array for common issues.
    
    Args:
        x: Audio array to validate
        name: Name for error messages
        
    Raises:
        ValueError: If audio has issues
    """
    if not isinstance(x, np.ndarray):
        raise ValueError(f"{name}: expected numpy array, got {type(x)}")
    
    if x.ndim not in (1, 2):
        raise ValueError(f"{name}: expected 1D or 2D array, got shape {x.shape}")
    
    if x.size == 0:
        raise ValueError(f"{name}: empty array")
    
    if x.ndim == 2 and x.shape[1] not in (1, 2):
        raise ValueError(f"{name}: expected mono or stereo, got {x.shape[1]} channels")
    
    if not np.isfinite(x).all():
        raise ValueError(f"{name}: contains NaN or Inf values")


def rms_db(x: np.ndarray) -> float:
    """Calculate RMS level in dB."""
    rms = np.sqrt(np.mean(x**2))
    return linear_to_db(rms)


def crest_factor_db(x: np.ndarray) -> float:
    """Calculate crest factor (peak-to-RMS ratio) in dB."""
    peak = np.max(np.abs(x))
    rms = np.sqrt(np.mean(x**2))
    return linear_to_db(peak / max(rms, 1e-12))