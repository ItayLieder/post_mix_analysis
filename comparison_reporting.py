

# ============================================
# Comparison & Reporting — Notebook Layer
# ============================================
# What this provides:
# - collect_metrics(paths): one-call metrics for many files (uses your Analysis layer)
# - build_comparison_tables(...): summary + deltas vs reference
# - plot_overlays(...): spectrum + short-term loudness overlays (saved to reports/)
# - make_blind_ab_pack(...): copies/renames files for unbiased listening
# - write_report_html(...): self-contained HTML report with tables + plots
# - write_report_bundle(...): one-shot wrapper that does all of the above
#
# Notes:
# - No code is executed on import; you’ll call these functions later.
# - Uses only matplotlib (no seaborn), and saves figures (no blocking .show()).
# - Registers generated artifacts in your manifest when you pass it in.

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
import os, io, shutil, warnings, json, uuid
import numpy as np
import pandas as pd
import matplotlib
from analysis import *
from data_handler import register_artifact

# Safe copy function to avoid SameFileError when source and dest are the same
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
    shutil.copy2(src_path, dst)
    return dst

matplotlib.rcParams["figure.dpi"] = 110
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.io import wavfile
try:
    from scipy.io.wavfile import WavFileWarning
except ImportError:
    class WavFileWarning(UserWarning): pass
warnings.filterwarnings("ignore", category=WavFileWarning)

# ---------- Config dataclasses ----------

@dataclass
class CompareConfig:
    preview_seconds: int = 60
    nfft: int = 1 << 16     # spectrum segment size
    win_s: float = 3.0      # short-term loudness window
    hop_s: float = 0.5
    level_match_mode: str = "none"   # "none" | "ref_lufs" | "target_lufs"
    target_lufs: float = -14.0       # used when level_match_mode == "target_lufs"
    reference_name: Optional[str] = None  # which row is the reference for Δ tables

# ---------- Internals: fast audio readers & helpers ----------

def _read_mono_preview(path: str, preview_seconds: Optional[int]) -> Tuple[int, np.ndarray]:
    sr0, data0 = wavfile.read(path)
    if data0.dtype == np.int16:
        x0 = data0.astype(np.float32) / 32768.0
    elif data0.dtype == np.int32:
        x0 = data0.astype(np.float32) / 2147483648.0
    elif data0.dtype == np.uint8:
        x0 = (data0.astype(np.float32) - 128.0) / 128.0
    else:
        x0 = data0.astype(np.float32)
    mono = x0 if x0.ndim == 1 else np.mean(x0, axis=1)
    if preview_seconds is not None:
        n = int(min(len(mono), preview_seconds * sr0))
        mono = mono[:n]
    return sr0, mono

def _spectrum_xy(mono: np.ndarray, sr: int, nfft: int) -> Tuple[np.ndarray, np.ndarray]:
    seg = mono[:min(len(mono), nfft)]
    if len(seg) < 2048:
        pad = np.zeros(2048, dtype=seg.dtype); pad[:len(seg)] = seg; seg = pad
    win = np.hanning(len(seg))
    sp = np.fft.rfft(seg * win)
    freqs = np.fft.rfftfreq(len(seg), 1/sr)
    mag_db = 20*np.log10(np.maximum(1e-12, np.abs(sp)))
    return freqs, mag_db

def _short_term_loudness(mono: np.ndarray, sr: int, win_s: float, hop_s: float) -> Tuple[np.ndarray, np.ndarray]:
    win = int(max(2, round(win_s * sr)))
    hop = int(max(1, round(hop_s * sr)))
    kernel = np.ones(win, dtype=np.float32) / float(win)
    pow_sig = mono**2
    rms = np.sqrt(np.maximum(1e-20, np.convolve(pow_sig, kernel, mode="same")))
    idx = np.arange(0, len(mono), hop)
    t = idx / sr
    return t, 20*np.log10(np.maximum(1e-12, rms[idx]))

def _lufs_approx(x: np.ndarray, sr: int) -> float:
    # Uses your Analysis layer function if available
    try:
        return lufs_integrated_approx(x, sr)
    except NameError:
        # tiny inline fallback (not gated)
        from scipy import signal
        sos_hp = signal.butter(2, 38.0/(sr*0.5), btype='highpass', output='sos')
        mono = x if x.ndim == 1 else np.mean(x, axis=1)
        y = signal.sosfilt(sos_hp, mono)
        ms = float(np.mean(y**2))
        return -0.691 + 10.0*np.log10(max(1e-12, ms))

def _tp_approx_db(x: np.ndarray, sr: int, oversample: int = 4) -> float:
    try:
        return true_peak_dbfs(x, sr, oversample=oversample)
    except NameError:
        from scipy import signal
        x_os = signal.resample_poly(x, oversample, 1, axis=0 if x.ndim>1 else 0)
        tp = float(np.max(np.abs(x_os)))
        return 20.0*np.log10(max(1e-12, tp))

# ---------- 1) Metrics collection ----------

def collect_metrics(file_paths: List[str]) -> pd.DataFrame:
    """
    For each path, run your Analysis layer and collect key metrics.
    """
    rows = []
    for p in file_paths:
        rep = analyze_wav(p)  # uses your Analysis layer
        rows.append({
            "name": os.path.splitext(os.path.basename(p))[0],
            "path": os.path.abspath(p),
            "sr": rep.sr,
            "duration_s": rep.duration_s,
            "peak_dbfs": rep.basic["peak_dbfs"],
            "true_peak_dbfs": rep.true_peak_dbfs,
            "rms_dbfs": rep.basic["rms_dbfs"],
            "lufs_int": rep.lufs_integrated,
            "crest_db": rep.basic["crest_db"],
            "bass_%": rep.bass_energy_pct,
            "air_%": rep.air_energy_pct,
            "phase_corr": rep.stereo["phase_correlation"],
            "stereo_width": rep.stereo["stereo_width"],
            "spectral_flatness": rep.spectral_flatness,
        })
    df = pd.DataFrame(rows)
    return df

# ---------- 2) Build comparison tables (summary + deltas) ----------

def build_comparison_tables(df_metrics: pd.DataFrame, cfg: CompareConfig) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Returns (summary_df, deltas_df_vs_reference|None).
    """
    summary = df_metrics.copy()

    deltas = None
    ref_name = cfg.reference_name or (summary.iloc[0]["name"] if len(summary) else None)
    if ref_name and ref_name in list(summary["name"]):
        ref_row = summary[summary["name"] == ref_name].iloc[0]
        # metrics to delta
        delta_cols = ["peak_dbfs","true_peak_dbfs","rms_dbfs","lufs_int","crest_db","bass_%","air_%","phase_corr","stereo_width","spectral_flatness"]
        rows = []
        for _, r in summary.iterrows():
            d = { "name": r["name"] }
            for c in delta_cols:
                d[f"Δ {c}"] = float(r[c] - ref_row[c])
            rows.append(d)
        deltas = pd.DataFrame(rows)
    return summary, deltas

# ---------- 3) Plots: spectrum & loudness overlays ----------
# Saves files under reports_dir and returns list of paths

def plot_overlays(file_paths: List[str], labels: Optional[List[str]], reports_dir: str, cfg: CompareConfig) -> Dict[str, str]:
    os.makedirs(reports_dir, exist_ok=True)
    labels = labels or [os.path.splitext(os.path.basename(p))[0] for p in file_paths]

    # Spectrum
    plt.figure()
    for p, lbl in zip(file_paths, labels):
        sr, mono = _read_mono_preview(p, cfg.preview_seconds)
        f, m = _spectrum_xy(mono, sr, cfg.nfft)
        plt.plot(f, m, label=lbl)
    plt.xscale('log'); plt.xlim(20, 20000)
    plt.xlabel("Frequency (Hz)"); plt.ylabel("Magnitude (dB)")
    plt.title(f"Spectrum Overlay (first {cfg.preview_seconds}s)")
    plt.legend()
    spec_png = os.path.join(reports_dir, "spectrum_overlay.png")
    plt.savefig(spec_png, bbox_inches="tight"); plt.close()

    # Short-term loudness
    plt.figure()
    series, min_len = [], None
    for p, lbl in zip(file_paths, labels):
        sr, mono = _read_mono_preview(p, cfg.preview_seconds)
        t, s = _short_term_loudness(mono, sr, cfg.win_s, cfg.hop_s)
        series.append((t, s, lbl))
        min_len = len(s) if min_len is None else min(min_len, len(s))
    for t, s, lbl in series:
        plt.plot(t[:min_len], s[:min_len], label=lbl)
    plt.xlabel("Time (s)"); plt.ylabel("Short-term RMS (dBFS)")
    plt.title(f"Short-term Loudness Overlay (first {cfg.preview_seconds}s)")
    plt.legend()
    loud_png = os.path.join(reports_dir, "loudness_overlay.png")
    plt.savefig(loud_png, bbox_inches="tight"); plt.close()

    return {"spectrum_png": spec_png, "loudness_png": loud_png}

# ---------- 4) Blind A/B pack (optional) ----------

def make_blind_ab_pack(file_paths: List[str], out_dir: str, bit_depth: str = "PCM_24") -> List[str]:
    """
    Copies files with randomized neutral labels (A/B/C...), returns the new paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    labels = [chr(ord('A') + i) for i in range(len(file_paths))]
    order = np.arange(len(file_paths))
    np.random.shuffle(order)

    out_paths = []
    for idx, new_lbl in zip(order, labels):
        src = file_paths[idx]
        x, sr = sf.read(src)
        dst = os.path.join(out_dir, f"Blind_{new_lbl}.wav")
        sf.write(dst, x, sr, subtype=bit_depth)
        out_paths.append(dst)
    return out_paths

# ---------- 5) HTML Report ----------

def _df_to_html_table(df: pd.DataFrame, caption: str) -> str:
    buf = io.StringIO()
    buf.write(f"<h3>{caption}</h3>\n")
    buf.write(df.to_html(index=False, float_format=lambda v: f"{v:.6g}"))
    return buf.getvalue()

def write_report_html(
    summary_df: pd.DataFrame,
    deltas_df: Optional[pd.DataFrame],
    plots: Dict[str, str],
    out_path: str,
    title: str = "Post-Mix Comparison Report",
    extra_notes: Optional[str] = None
) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    html = io.StringIO()
    html.write(f"<!doctype html><html><head><meta charset='utf-8'><title>{title}</title>")
    html.write("<style>body{font-family:system-ui,Arial,sans-serif;margin:24px} h1{margin-top:0} img{max-width:100%;height:auto} table{border-collapse:collapse} th,td{border:1px solid #ddd;padding:6px} caption{margin:6px 0}</style>")
    html.write("</head><body>")
    html.write(f"<h1>{title}</h1>")
    if extra_notes:
        html.write(f"<p><em>{extra_notes}</em></p>")
    # tables
    html.write(_df_to_html_table(summary_df, "Summary Metrics"))
    if deltas_df is not None and len(deltas_df):
        html.write(_df_to_html_table(deltas_df, "Δ vs Reference"))
    # plots
    if "spectrum_png" in plots and os.path.exists(plots["spectrum_png"]):
        html.write("<h3>Spectrum Overlay</h3>")
        html.write(f"<img src='{os.path.basename(plots['spectrum_png'])}' alt='Spectrum Overlay'/>")
    if "loudness_png" in plots and os.path.exists(plots["loudness_png"]):
        html.write("<h3>Short-Term Loudness Overlay</h3>")
        html.write(f"<img src='{os.path.basename(plots['loudness_png'])}' alt='Loudness Overlay'/>")
    html.write("</body></html>")

    # write HTML and copy plot assets next to it
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html.getvalue())
    # copy images to same folder for portability (safe copy to avoid SameFileError)
    target_dir = os.path.dirname(out_path)
    for k, p in plots.items():
        if p and os.path.exists(p):
            _safe_copy_to_dir(p, target_dir)
    return os.path.abspath(out_path)

# ---------- 6) One-shot bundle: metrics → plots → HTML (+ manifest) ----------

def write_report_bundle(
    file_paths: List[str],
    reports_dir: str,
    cfg: Optional[CompareConfig] = None,
    manifest: Optional[Any] = None,   # pass your Manifest object to auto-register
    report_name: str = "comparison_report.html",
    extra_notes: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience wrapper to:
      - compute metrics for all files
      - build summary + deltas (vs reference)
      - render plots (saved)
      - write HTML report
      - register artifacts in manifest (optional)
    Returns a dict with paths and DataFrames.
    """
    cfg = cfg or CompareConfig()
    os.makedirs(reports_dir, exist_ok=True)

    # 1) metrics
    df_metrics = collect_metrics(file_paths)
    # 2) tables
    summary_df, deltas_df = build_comparison_tables(df_metrics, cfg)
    # 3) plots
    labels = list(summary_df["name"])
    plot_paths = plot_overlays(file_paths, labels, reports_dir, cfg)
    # 4) HTML
    html_path = os.path.join(reports_dir, report_name)
    html_path = write_report_html(summary_df, deltas_df, plot_paths, html_path, extra_notes=extra_notes)

    # 5) optional manifest registration (HTML + images as a single "report" artifact)
    if manifest is not None:
        register_artifact(manifest, html_path, kind="report", params={
            "type": "comparison",
            "reference": cfg.reference_name,
            "preview_seconds": cfg.preview_seconds,
        }, stage="compare_html")
        for p in plot_paths.values():
            if os.path.exists(p):
                register_artifact(manifest, p, kind="report_asset", params={"linked_report": os.path.basename(html_path)})

    return {
        "html_path": html_path,
        "summary_df": summary_df,
        "deltas_df": deltas_df,
        "plots": plot_paths
    }

print("Comparison & Reporting layer loaded:")
print("- collect_metrics, build_comparison_tables")
print("- plot_overlays (saves to reports/)")
print("- make_blind_ab_pack (optional)")
print("- write_report_html (self-contained)")
print("- write_report_bundle (one-shot with manifest)")
