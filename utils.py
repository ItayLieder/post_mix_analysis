

# === Utility: exports + true-peak guard ===
import os, numpy as np, soundfile as sf
from dataclasses import dataclass

# Use your existing TP approx if available; else fallback
try:
    true_peak_dbfs
except NameError:
    from scipy import signal
    def true_peak_dbfs(x: np.ndarray, sr: int, oversample: int = 4) -> float:
        x_os = signal.resample_poly(np.asarray(x, dtype=np.float32), oversample, 1, axis=0 if np.asarray(x).ndim>1 else 0)
        tp = float(np.max(np.abs(x_os)))
        return 20.0*np.log10(max(1e-12, tp))

def _db_to_lin(db: float) -> float: return 10.0**(db/20.0)

def save_wav_24bit(path: str, y: np.ndarray, sr: int):
    """Always export 24-bit PCM with dirs created."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    sf.write(path, np.asarray(y, dtype=np.float32), int(sr), subtype="PCM_24")
    return os.path.abspath(path)

@dataclass
class TPGuardResult:
    out: np.ndarray
    gain_db: float
    in_dbtp: float
    out_dbtp: float
    ceiling_db: float

def safe_true_peak(y: np.ndarray, sr: int, ceiling_db: float = -1.0) -> TPGuardResult:
    """Trim overall gain so true-peak ≤ ceiling (no limiting)."""
    tp_in = true_peak_dbfs(y, sr, oversample=4)
    gain_db = ceiling_db - tp_in
    y2 = (np.asarray(y, dtype=np.float32) * _db_to_lin(gain_db)).astype(np.float32)
    tp_out = true_peak_dbfs(y2, sr, oversample=4)
    return TPGuardResult(out=y2, gain_db=gain_db, in_dbtp=tp_in, out_dbtp=tp_out, ceiling_db=ceiling_db)

# === Global config + validators ===
from dataclasses import dataclass, asdict
import numpy as np

@dataclass
class GlobalConfig:
    # I/O
    default_bit_depth: str = "PCM_24"
    # Prep
    prep_hpf_hz: float = 20.0
    prep_peak_target_dbfs: float = -6.0
    # Rendering
    render_peak_target_dbfs: float = -1.0
    # Streaming sim
    tp_ceiling_db: float = -1.0
    # Reporting
    preview_seconds: int = 60
    nfft: int = 1<<16

CFG = GlobalConfig()

class InputError(Exception): pass

def ensure_audio_valid(x: np.ndarray, name: str = "audio"):
    if not isinstance(x, np.ndarray):
        raise InputError(f"{name}: expected numpy array, got {type(x)}")
    if x.ndim not in (1,2):
        raise InputError(f"{name}: expected 1D (mono) or 2D (stereo), got shape {x.shape}")
    if not np.isfinite(x).all():
        raise InputError(f"{name}: contains NaN/Inf values; sanitize before processing")
    if x.size == 0:
        raise InputError(f"{name}: empty array")
    if x.ndim == 2 and x.shape[1] not in (1,2):
        raise InputError(f"{name}: expected (N,), (N,1) or (N,2); got {x.shape}")

# === Reporting helpers (enhanced HTML) ===
import os, io, html
from typing import Optional, Dict, Any
import pandas as pd

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

def html_header_block(title: str, code_versions: Dict[str,str], dials: Optional[Dict[str,Any]] = None, notes: Optional[str] = None) -> str:
    buf = io.StringIO()
    buf.write(f"<h1>{html.escape(title)}</h1>\n")
    if notes:
        buf.write(f"<p><em>{html.escape(notes)}</em></p>\n")
    # versions
    if code_versions:
        buf.write("<h3>Code Versions</h3><ul>")
        for k,v in code_versions.items():
            buf.write(f"<li><code>{html.escape(k)}</code>: {html.escape(v)}</li>")
        buf.write("</ul>")
    # dials snapshot
    if dials:
        buf.write("<h3>Dial Snapshot</h3><ul>")
        for k,v in dials.items():
            buf.write(f"<li>{html.escape(k)}: {html.escape(str(v))}</li>")
        buf.write("</ul>")
    return buf.getvalue()

# Patch your write_report_html to use these (drop-in):
def write_report_html_enhanced(
    summary_df: pd.DataFrame,
    deltas_df: Optional[pd.DataFrame],
    plots: Dict[str, str],
    out_path: str,
    title: str = "Post-Mix Comparison Report",
    extra_notes: Optional[str] = None,
    code_versions: Optional[Dict[str,str]] = None,
    dial_snapshot: Optional[Dict[str,Any]] = None
) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    # linkify
    summary_df2 = df_with_links(summary_df)
    # base HTML
    html_doc = io.StringIO()
    html_doc.write(f"<!doctype html><html><head><meta charset='utf-8'><title>{title}</title>")
    html_doc.write("<style>body{font-family:system-ui,Arial,sans-serif;margin:24px} h1{margin-top:0} img{max-width:100%;height:auto} table{border-collapse:collapse} th,td{border:1px solid #ddd;padding:6px} code{background:#f5f5f5;padding:2px 4px;border-radius:4px}</style>")
    html_doc.write("</head><body>")
    html_doc.write(html_header_block(title, code_versions or {}, dials=dial_snapshot, notes=extra_notes))
    # tables
    html_doc.write("<h3>Summary Metrics</h3>")
    html_doc.write(summary_df2.to_html(index=False, escape=False, float_format=lambda v: f"{v:.6g}"))
    if deltas_df is not None and len(deltas_df):
        html_doc.write("<h3>Δ vs Reference</h3>")
        html_doc.write(deltas_df.to_html(index=False, float_format=lambda v: f"{v:.6g}"))
    # plots
    if "spectrum_png" in plots and os.path.exists(plots["spectrum_png"]):
        html_doc.write("<h3>Spectrum Overlay</h3>")
        html_doc.write(f"<img src='{os.path.basename(plots['spectrum_png'])}' alt='Spectrum Overlay'/>")
    if "loudness_png" in plots and os.path.exists(plots["loudness_png"]):
        html_doc.write("<h3>Short-Term Loudness Overlay</h3>")
        html_doc.write(f"<img src='{os.path.basename(plots['loudness_png'])}' alt='Loudness Overlay'/>")
    html_doc.write("</body></html>")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_doc.getvalue())
    # copy assets next to HTML (same as before)
    import shutil
    for p in plots.values():
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(os.path.dirname(out_path), os.path.basename(p)))
    return os.path.abspath(out_path)


# ---- Patch: make report writers skip same-file copies ----
import os, shutil

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

# Rebind write_report_html to use safe copy
def write_report_html(summary_df, deltas_df, plots, out_path, title="Post-Mix Comparison Report", extra_notes=None):
    import io
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    html = io.StringIO()
    html.write(f"<!doctype html><html><head><meta charset='utf-8'><title>{title}</title>")
    html.write("<style>body{font-family:system-ui,Arial,sans-serif;margin:24px} h1{margin-top:0} img{max-width:100%;height:auto} table{border-collapse:collapse} th,td{border:1px solid #ddd;padding:6px} caption{margin:6px 0}</style>")
    html.write("</head><body>")
    html.write(f"<h1>{title}</h1>")
    if extra_notes:
        html.write(f"<p><em>{extra_notes}</em></p>")
    # tables
    html.write(summary_df.to_html(index=False, float_format=lambda v: f"{v:.6g}"))
    if deltas_df is not None and len(deltas_df):
        html.write(deltas_df.to_html(index=False, float_format=lambda v: f"{v:.6g}"))
    # plots
    if "spectrum_png" in plots and os.path.exists(plots["spectrum_png"]):
        html.write("<h3>Spectrum Overlay</h3>")
        html.write(f"<img src='{os.path.basename(plots['spectrum_png'])}' alt='Spectrum Overlay'/>")
    if "loudness_png" in plots and os.path.exists(plots["loudness_png"]):
        html.write("<h3>Short-Term Loudness Overlay</h3>")
        html.write(f"<img src='{os.path.basename(plots['loudness_png'])}' alt='Loudness Overlay'/>")
    html.write("</body></html>")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html.getvalue())

    # copy assets next to HTML, but skip if already there
    target_dir = os.path.dirname(out_path)
    for p in plots.values():
        if p:
            _safe_copy_to_dir(p, target_dir)
    return os.path.abspath(out_path)

# If you use the enhanced writer, patch its copy loop too
try:
    _old_enh = write_report_html_enhanced  # if defined earlier

    def write_report_html_enhanced(summary_df, deltas_df, plots, out_path,
                                   title="Post-Mix Comparison Report",
                                   extra_notes=None, code_versions=None, dial_snapshot=None):
        import io, html
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

        # (keep your previous enhanced content generation…)
        html_doc = io.StringIO()
        html_doc.write(f"<!doctype html><html><head><meta charset='utf-8'><title>{title}</title>")
        html_doc.write("<style>body{font-family:system-ui,Arial,sans-serif;margin:24px} h1{margin-top:0} img{max-width:100%;height:auto} table{border-collapse:collapse} th,td{border:1px solid #ddd;padding:6px} code{background:#f5f5f5;padding:2px 4px;border-radius:4px}</style>")
        html_doc.write("</head><body>")
        # simple header
        html_doc.write(f"<h1>{html.escape(title)}</h1>")
        if extra_notes:
            html_doc.write(f"<p><em>{html.escape(extra_notes)}</em></p>")
        if code_versions:
            html_doc.write("<h3>Code Versions</h3><ul>")
            for k,v in code_versions.items():
                html_doc.write(f"<li><code>{html.escape(k)}</code>: {html.escape(str(v))}</li>")
            html_doc.write("</ul>")
        if dial_snapshot:
            html_doc.write("<h3>Dial Snapshot</h3><ul>")
            for k,v in dial_snapshot.items():
                html_doc.write(f"<li>{html.escape(k)}: {html.escape(str(v))}</li>")
            html_doc.write("</ul>")

        html_doc.write("<h3>Summary Metrics</h3>")
        html_doc.write(summary_df.to_html(index=False, escape=False, float_format=lambda v: f"{v:.6g}"))
        if deltas_df is not None and len(deltas_df):
            html_doc.write("<h3>Δ vs Reference</h3>")
            html_doc.write(deltas_df.to_html(index=False, float_format=lambda v: f"{v:.6g}"))

        if "spectrum_png" in plots and os.path.exists(plots["spectrum_png"]):
            html_doc.write("<h3>Spectrum Overlay</h3>")
            html_doc.write(f"<img src='{os.path.basename(plots['spectrum_png'])}' alt='Spectrum Overlay'/>")
        if "loudness_png" in plots and os.path.exists(plots["loudness_png"]):
            html_doc.write("<h3>Short-Term Loudness Overlay</h3>")
            html_doc.write(f"<img src='{os.path.basename(plots['loudness_png'])}' alt='Loudness Overlay'/>")

        html_doc.write("</body></html>")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html_doc.getvalue())

        # safe-copy assets
        target_dir = os.path.dirname(out_path)
        for p in plots.values():
            if p:
                _safe_copy_to_dir(p, target_dir)
        return os.path.abspath(out_path)
except NameError:
    pass
