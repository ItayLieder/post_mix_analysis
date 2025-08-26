

# ============================================
# Logging · Versioning · Reproducibility Layer
# ============================================
# What this provides:
# - RunLogger: creates a run_id and writes structured logs (JSONL) + summary JSON
# - Env capture: library versions, Python/OS, CPU info, pip freeze snapshot
# - Determinism: simple seed manager for NumPy & Python hash seed
# - Provenance: content hashes for audio/code/config; processing graph digest
# - Artifact registry helpers (alongside your Manifest)
# - Repro bundle: packs config, logs, environment, and outputs into a zip

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List, Iterable, Tuple
import os, sys, io, json, time, uuid, hashlib, platform, subprocess, zipfile, shutil, textwrap
from datetime import datetime
import numpy as np
from data_handler import *

# ---------- small utils ----------

def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def _make_json_serializable(obj):
    """Convert numpy types and other non-serializable objects to JSON-compatible types."""
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serializable(item) for item in obj]
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'item'):  # numpy scalar
        return obj.item()
    else:
        return obj

def _mkdirp(p: str) -> str:
    os.makedirs(p, exist_ok=True); return p

def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def file_sha256(path: str, bufsize: int = 1<<20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(bufsize), b""):
            h.update(chunk)
    return h.hexdigest()

def json_sha256(obj: Any) -> str:
    """Stable JSON hash (sorted keys, no whitespace)."""
    b = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(b)

# ---------- determinism ----------

class SeedScope:
    """
    Context manager to set deterministic seeds for NumPy and PYTHONHASHSEED.
    Use: with SeedScope(42): ...  (or call SeedScope.set_global(42))
    """
    def __init__(self, seed: int = 42):
        self.seed = int(seed)
        self._prev_hashseed = os.environ.get("PYTHONHASHSEED")

    def __enter__(self):
        np.random.seed(self.seed)
        os.environ["PYTHONHASHSEED"] = str(self.seed)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._prev_hashseed is None:
            os.environ.pop("PYTHONHASHSEED", None)
        else:
            os.environ["PYTHONHASHSEED"] = self._prev_hashseed

    @staticmethod
    def set_global(seed: int = 42):
        np.random.seed(int(seed))
        os.environ["PYTHONHASHSEED"] = str(int(seed))

# ---------- environment capture ----------

def capture_environment() -> Dict[str, Any]:
    info = {
        "timestamp": _now_iso(),
        "python": sys.version.replace("\n", " "),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "packages": {},
        "pip_freeze": None,
    }
    # library versions (best-effort)
    try:
        import scipy, soundfile, pandas, matplotlib
        info["packages"].update({
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "soundfile": soundfile.__version__,
            "pandas": pandas.__version__,
            "matplotlib": matplotlib.__version__,
        })
    except Exception:
        info["packages"]["numpy"] = np.__version__
    # pip freeze (optional, can be heavy)
    try:
        out = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], timeout=20)
        info["pip_freeze"] = out.decode("utf-8").splitlines()
    except Exception:
        info["pip_freeze"] = None
    return info

# ---------- processing graph digest ----------

def processing_digest(name: str, *, code_versions: Dict[str, str], params: Dict[str, Any]) -> str:
    """
    Create a short fingerprint for a processing step: hashes code + params.
    code_versions: e.g., {"dsp_primitives":"v1.2.0+sha...", "processors":"v0.9.3", ...}
    params: the dial values, thresholds, etc.
    """
    payload = {"step": name, "code": code_versions, "params": params}
    h = json_sha256(payload)
    return h[:16]  # short id

# ---------- RunLogger ----------

@dataclass
class RunLogger:
    root_dir: str
    run_id: str
    dir_logs: str
    dir_meta: str
    dir_bundle: str
    summary_path: str
    jsonl_path: str

    @staticmethod
    def start(workspace_root: str, tag: str = "session") -> "RunLogger":
        rid = f"{tag}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:6]}"
        logs = _mkdirp(os.path.join(workspace_root, "reports", "logs", rid))
        meta = _mkdirp(os.path.join(workspace_root, "reports", "meta", rid))
        bund = _mkdirp(os.path.join(workspace_root, "reports", "bundles", rid))
        return RunLogger(
            root_dir=workspace_root,
            run_id=rid,
            dir_logs=logs,
            dir_meta=meta,
            dir_bundle=bund,
            summary_path=os.path.join(meta, "summary.json"),
            jsonl_path=os.path.join(logs, "events.jsonl"),
        )

    # event logging (append JSON lines)
    def log_event(self, kind: str, payload: Dict[str, Any]):
        # Convert numpy types to JSON-serializable types
        safe_payload = _make_json_serializable(payload)
        evt = {"ts": _now_iso(), "run_id": self.run_id, "kind": kind, **safe_payload}
        
        # Ensure the log directory exists before writing
        log_dir = os.path.dirname(self.jsonl_path)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(evt, ensure_ascii=False) + "\n")

    # param/metric helpers
    def log_params(self, step: str, params: Dict[str, Any], code_versions: Optional[Dict[str,str]] = None):
        # Convert to JSON-safe types before processing
        safe_params = _make_json_serializable(params)
        digest = processing_digest(step, code_versions=code_versions or {}, params=safe_params)
        self.log_event("params", {"step": step, "digest": digest, "params": safe_params, "code_versions": code_versions})

    def log_metrics(self, step: str, metrics: Dict[str, Any]):
        # Convert numpy types to JSON-serializable types
        safe_metrics = _make_json_serializable(metrics)
        self.log_event("metrics", {"step": step, "metrics": safe_metrics})

    def log_artifact(self, kind: str, path: str, extra: Optional[Dict[str,Any]] = None):
        payload = {"kind": kind, "path": os.path.abspath(path)}
        if extra: payload.update(extra)
        self.log_event("artifact", payload)

    # summary (overwrites)
    def write_summary(self, summary: Dict[str, Any]):
        # Convert numpy types and ensure directory exists
        safe_summary = _make_json_serializable(summary)
        summary_dir = os.path.dirname(self.summary_path)
        if not os.path.exists(summary_dir):
            os.makedirs(summary_dir, exist_ok=True)
            
        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(safe_summary, f, indent=2, ensure_ascii=False)

# ---------- provenance for files & configs ----------

def capture_file_provenance(path: str, role: str = "input") -> Dict[str, Any]:
    return {
        "role": role,
        "path": os.path.abspath(path),
        "sha256": file_sha256(path),
        "bytes": os.path.getsize(path),
        "mtime": int(os.path.getmtime(path)),
    }

def capture_config_snapshot(config: Dict[str, Any], name: str = "config") -> Dict[str, Any]:
    snap = {"name": name, "sha256": json_sha256(config), "config": config}
    return snap

# ---------- reproducibility bundle ----------

def make_repro_zip(
    out_zip_path: str,
    *,
    workspace_root: str,
    run_logger: RunLogger,
    env_info: Dict[str, Any],
    inputs: List[str],
    outputs: List[str],
    extra_jsons: Optional[Dict[str, Any]] = None,
    readme_text: Optional[str] = None
) -> str:
    """
    Creates a zip with:
      - logs/events.jsonl, meta/summary.json
      - environment.json (+ pip_freeze)
      - file provenance for inputs/outputs
      - any extra JSON config you pass in
    Returns absolute path to the zip.
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_zip_path)), exist_ok=True)

    prov_inputs = [capture_file_provenance(p, role="input") for p in inputs]
    prov_outputs = [capture_file_provenance(p, role="output") for p in outputs]

    bundle_meta = {
        "created": _now_iso(),
        "run_id": run_logger.run_id,
        "workspace_root": os.path.abspath(workspace_root),
        "environment": env_info,
        "inputs": prov_inputs,
        "outputs": prov_outputs,
    }
    if extra_jsons:
        bundle_meta.update(extra_jsons)

    # write temp files in bundle dir for consistent names
    env_json = os.path.join(run_logger.dir_meta, "environment.json")
    with open(env_json, "w", encoding="utf-8") as f:
        json.dump(env_info, f, indent=2, ensure_ascii=False)

    prov_json = os.path.join(run_logger.dir_meta, "provenance.json")
    with open(prov_json, "w", encoding="utf-8") as f:
        json.dump({"inputs": prov_inputs, "outputs": prov_outputs}, f, indent=2, ensure_ascii=False)

    bundle_json = os.path.join(run_logger.dir_meta, "bundle_meta.json")
    with open(bundle_json, "w", encoding="utf-8") as f:
        json.dump(bundle_meta, f, indent=2, ensure_ascii=False)

    readme_md = os.path.join(run_logger.dir_meta, "README.txt")
    with open(readme_md, "w", encoding="utf-8") as f:
        f.write(textwrap.dedent(readme_text or f"""
        Post-Mix Reproducibility Bundle
        =================================
        Run ID: {run_logger.run_id}
        Created: {_now_iso()}

        Contents:
          - logs/events.jsonl      : structured event log
          - meta/summary.json      : high-level run summary
          - meta/environment.json  : Python/OS/pkg versions (+pip freeze when available)
          - meta/provenance.json   : input/output file hashes and sizes
          - meta/bundle_meta.json  : collected metadata for quick inspection

        To reproduce:
          1) Recreate Python env from pip_freeze (if present).
          2) Use the same inputs (verified by sha256) and configs.
          3) Run via the same code versions; dials/params are in events.jsonl and summary.json.
        """).strip() + "\n")

    # zip it up
    with zipfile.ZipFile(out_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        # logs + meta
        for root in [run_logger.dir_logs, run_logger.dir_meta]:
            for dirpath, _, filenames in os.walk(root):
                for fn in filenames:
                    ap = os.path.join(dirpath, fn)
                    arc = os.path.relpath(ap, os.path.dirname(run_logger.dir_logs))
                    z.write(ap, arcname=arc)

    return os.path.abspath(out_zip_path)

# ---------- convenience: wire into your Manifest ----------

def register_and_log_artifact(manifest, logger: RunLogger, path: str, kind: str, params: Optional[Dict[str,Any]] = None, stage: Optional[str] = None):
    register_artifact(manifest, path, kind=kind, params=params or {}, stage=stage)
    logger.log_artifact(kind, path, extra={"stage": stage, "params": params or {}})

# ---------- version stamps for your code layers (edit these in each layer once) ----------

CODE_VERSIONS = {
    "io_layer":           "v0.1.0",
    "analysis_layer":     "v0.3.0",
    "dsp_primitives":     "v0.4.1",
    "processors":         "v0.5.0",
    "render_engine":      "v0.2.0",
    "premaster_prep":     "v0.1.0",
    "orchestrator":       "v0.1.0",
    "stream_sim":         "v0.1.0",
    "compare_reporting":  "v0.1.0",
    "presets_recs":       "v0.1.0",
}

print("Logging · Versioning · Reproducibility layer loaded:")
print("- RunLogger.start(workspace_root, tag) → logger")
print("- logger.log_params/metrics/artifact(), logger.write_summary()")
print("- capture_environment(), file_sha256(), capture_config_snapshot()")
print("- processing_digest(name, code_versions, params)")
print("- make_repro_zip(out_zip, workspace_root, logger, env_info, inputs, outputs)")
print("- register_and_log_artifact(manifest, logger, path, kind, params, stage)")
print("- CODE_VERSIONS dict (update per layer when you change code)")
