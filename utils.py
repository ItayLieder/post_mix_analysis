"""
Consolidated utility functions for audio processing, exports, and reporting.
Combines audio_utils.py and original utils.py functionality.
"""

import os
import io
import html
import numpy as np
import soundfile as sf
from dataclasses import dataclass
from typing import Union, Optional, Dict, Any
from scipy import signal
import pandas as pd
from config import CONFIG

# ============================================================================
# AUDIO PROCESSING UTILITIES
# ============================================================================

def to_float32(data: np.ndarray) -> np.ndarray:
    """Convert various integer audio formats to float32 in [-1, 1] range."""
    if data.dtype == np.int16:
        return data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        return data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        return (data.astype(np.float32) - 128.0) / 128.0
    else:
        return data.astype(np.float32)


def sanitize_audio(x: np.ndarray, clip_range: float = 4.0) -> np.ndarray:
    """Clean audio by removing NaN/Inf and clipping to safe range."""
    cleaned = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(cleaned, -clip_range, clip_range).astype(np.float32)


def ensure_stereo(x: np.ndarray) -> np.ndarray:
    """Convert mono audio to stereo by duplicating channels."""
    if x.ndim == 1:
        return np.stack([x, x], axis=-1)
    elif x.shape[1] == 1:
        return np.repeat(x, 2, axis=1)
    return x


def to_mono(x: np.ndarray) -> np.ndarray:
    """Convert stereo audio to mono by averaging channels."""
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
    """Estimate true peak in dB using oversampling."""
    x_clean = sanitize_audio(x, clip_range=1.0)
    x_oversampled = signal.resample_poly(
        x_clean, oversample, 1, 
        axis=0 if x_clean.ndim > 1 else 0
    )
    peak = float(np.max(np.abs(x_oversampled)))
    return linear_to_db(peak)


def normalize_peak(x: np.ndarray, target_dbfs: float = -1.0) -> np.ndarray:
    """Normalize audio to target peak level."""
    current_peak = float(np.max(np.abs(x)))
    if current_peak == 0:
        return x
    
    current_db = linear_to_db(current_peak)
    gain_db = target_dbfs - current_db
    gain_linear = db_to_linear(gain_db)
    
    return (x * gain_linear).astype(np.float32)


def normalize_true_peak(x: np.ndarray, sr: int, target_dbtp: float = -1.0) -> np.ndarray:
    """Normalize audio to target true peak level."""
    current_tp = true_peak_db(x, sr)
    gain_db = target_dbtp - current_tp
    gain_linear = db_to_linear(gain_db)
    
    return (sanitize_audio(x) * gain_linear).astype(np.float32)


def rms_db(x: np.ndarray) -> float:
    """Calculate RMS level in dB."""
    rms = np.sqrt(np.mean(x**2))
    return linear_to_db(rms)


def crest_factor_db(x: np.ndarray) -> float:
    """Calculate crest factor (peak-to-RMS ratio) in dB."""
    peak = np.max(np.abs(x))
    rms = np.sqrt(np.mean(x**2))
    return linear_to_db(peak / max(rms, 1e-12))


def validate_audio(x: np.ndarray, name: str = "audio") -> None:
    """Validate audio array for common issues."""
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


# ============================================================================
# EXPORT UTILITIES
# ============================================================================

def save_wav_24bit(path: str, y: np.ndarray, sr: int, bit_depth: str = None):
    """Export audio with proper bit depth and directory creation."""
    if bit_depth is None:
        bit_depth = CONFIG.audio.default_bit_depth
    
    # Ensure output directory exists
    output_path = os.path.abspath(path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Sanitize and write audio
    clean_audio = sanitize_audio(y, clip_range=CONFIG.audio.safe_clip_range)
    sf.write(output_path, clean_audio, int(sr), subtype=bit_depth)
    
    return output_path


@dataclass
class TPGuardResult:
    out: np.ndarray
    gain_db: float
    in_dbtp: float
    out_dbtp: float
    ceiling_db: float


def safe_true_peak(y: np.ndarray, sr: int, ceiling_db: float = None) -> TPGuardResult:
    """Apply gain reduction to keep true peak below ceiling (no limiting)."""
    if ceiling_db is None:
        ceiling_db = CONFIG.audio.true_peak_ceiling_db
    
    # Clean input audio
    clean_audio = sanitize_audio(y)
    
    # Measure input true peak
    input_tp = true_peak_db(clean_audio, sr, oversample=CONFIG.audio.oversample_factor)
    
    # Calculate required gain
    gain_db = ceiling_db - input_tp
    
    # Apply gain
    output_audio = (clean_audio * db_to_linear(gain_db)).astype(np.float32)
    
    # Measure output true peak for verification
    output_tp = true_peak_db(output_audio, sr, oversample=CONFIG.audio.oversample_factor)
    
    return TPGuardResult(
        out=output_audio, 
        gain_db=gain_db, 
        in_dbtp=input_tp, 
        out_dbtp=output_tp, 
        ceiling_db=ceiling_db
    )


# ============================================================================
# VALIDATION UTILITIES
# ============================================================================

class InputError(Exception): 
    """Raised when input audio data is invalid."""
    pass


def ensure_audio_valid(x: np.ndarray, name: str = "audio"):
    """Validate audio array and raise InputError if invalid."""
    try:
        validate_audio(x, name)
    except ValueError as e:
        raise InputError(str(e))


# ============================================================================
# REPORTING UTILITIES
# ============================================================================

def df_with_links(df: pd.DataFrame) -> pd.DataFrame:
    """If 'path' column exists, add a clickable 'file' column for HTML report."""
    if "path" in df.columns:
        def mk_link(p):
            base = os.path.basename(str(p))
            return f"<a href='{html.escape(base)}' target='_blank'>{html.escape(base)}</a>"
        df2 = df.copy()
        df2["file"] = df2["path"].apply(mk_link)
        # put 'file' right after 'name' if present
        cols = list(df2.columns)
        if "name" in cols:
            cols.insert(cols.index("name")+1, cols.pop(cols.index("file")))
            df2 = df2[cols]
        return df2
    return df


def _safe_copy_to_dir(src_path: str, target_dir: str):
    """Copy src_path into target_dir unless it's already there."""
    if not os.path.exists(src_path):
        return None
    dst = os.path.join(target_dir, os.path.basename(src_path))
    try:
        # if already same path or same inode → skip
        if os.path.abspath(src_path) == os.path.abspath(dst) or (
            os.path.exists(dst) and os.path.samefile(src_path, dst)
        ):
            return dst
    except Exception:
        # os.path.samefile might fail on some platforms; fall through to copy
        pass
    import shutil
    shutil.copy2(src_path, dst)
    return dst


def write_report_html(summary_df, deltas_df, plots, out_path, title=None, extra_notes=None):
    """Write HTML comparison report with improved styling."""
    if title is None:
        title = CONFIG.reporting.report_title
    
    # Create output directory
    output_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Build HTML
    html_content = io.StringIO()
    html_content.write(f'<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(title)}</title>')
    
    # Enhanced CSS styling using config
    html_content.write(f'''<style>
        body {{ font-family: {CONFIG.reporting.font_family}; margin: 24px; line-height: 1.6; }}
        h1 {{ margin-top: 0; color: #333; }}
        h3 {{ color: #555; margin-top: 2rem; }}
        img {{ max-width: 100%; height: auto; border-radius: 4px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
        th, td {{ border: 1px solid {CONFIG.reporting.table_border_color}; padding: 8px 12px; text-align: left; }}
        th {{ background: #f8f9fa; font-weight: 600; }}
        tr:nth-child(even) {{ background: #f8f9fa; }}
        .notes {{ font-style: italic; color: #666; background: #f8f9fa; padding: 1rem; border-radius: 4px; margin: 1rem 0; }}
    </style>''')
    
    html_content.write('</head><body>')
    html_content.write(f'<h1>{html.escape(title)}</h1>')
    
    if extra_notes:
        html_content.write(f'<div class="notes">{html.escape(extra_notes)}</div>')
    
    # Tables
    html_content.write('<h3>Summary Metrics</h3>')
    html_content.write(summary_df.to_html(index=False, float_format=lambda v: f"{v:.6g}", escape=False))
    
    if deltas_df is not None and len(deltas_df):
        html_content.write('<h3>Changes vs Reference</h3>')
        html_content.write(deltas_df.to_html(index=False, float_format=lambda v: f"{v:.6g}"))
    
    # Plots
    for plot_type, plot_path in plots.items():
        if plot_path and os.path.exists(plot_path):
            plot_name = {
                'spectrum_png': 'Frequency Spectrum Comparison',
                'loudness_png': 'Short-Term Loudness Comparison'
            }.get(plot_type, plot_type.replace('_', ' ').title())
            
            html_content.write(f'<h3>{plot_name}</h3>')
            html_content.write(f'<img src="{os.path.basename(plot_path)}" alt="{plot_name}"/>')
    
    html_content.write('</body></html>')
    
    # Write HTML file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content.getvalue())
    
    # Copy plot assets to same directory
    target_dir = os.path.dirname(output_path)
    for plot_path in plots.values():
        if plot_path and os.path.exists(plot_path):
            _safe_copy_to_dir(plot_path, target_dir)
    
    return output_path