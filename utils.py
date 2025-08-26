# === Utility: exports + true-peak guard ===
import os
import numpy as np
import soundfile as sf
from dataclasses import dataclass
from config import CONFIG
from audio_utils import true_peak_db, db_to_linear, sanitize_audio

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

# === Audio validation ===
from audio_utils import validate_audio as _validate_audio

class InputError(Exception): 
    """Raised when input audio data is invalid."""
    pass

def ensure_audio_valid(x: np.ndarray, name: str = "audio"):
    """Validate audio array and raise InputError if invalid."""
    try:
        _validate_audio(x, name)
    except ValueError as e:
        raise InputError(str(e))

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
def write_report_html(summary_df, deltas_df, plots, out_path, title=None, extra_notes=None):
    """Write HTML comparison report with improved styling."""
    import io
    import html as html_module
    
    if title is None:
        title = CONFIG.reporting.report_title
    
    # Create output directory
    output_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Build HTML
    html_content = io.StringIO()
    html_content.write(f'<!doctype html><html><head><meta charset="utf-8"><title>{html_module.escape(title)}</title>')
    
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
    html_content.write(f'<h1>{html_module.escape(title)}</h1>')
    
    if extra_notes:
        html_content.write(f'<div class="notes">{html_module.escape(extra_notes)}</div>')
    
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
