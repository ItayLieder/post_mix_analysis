from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Optional, Tuple, Dict, Any, List, Iterable, Union

import os
import math
import hashlib
import numpy as np
from scipy.io import wavfile
from scipy import signal
from datetime import datetime
import os, glob, json, shutil, sys, platform
from datetime import datetime
import numpy as np
# Post-Mix I/O Layer — Notebook Version
# - WAV-focused (robust float32 pipeline; no external deps)
# - Safe loading/saving, resampling, preview slicing, hashing, path helpers
# - Designed to later lift into .py modules with minimal changes
#
# Note: This cell only defines functions/classes. No files are written.
# ----------------------------- Dataclasses -----------------------------

@dataclass
class AudioBuffer:
    """In-memory audio container standardized to float32 in [-1, 1]."""
    sr: int
    samples: np.ndarray  # shape: (N,) mono or (N, 2) stereo (float32)
    path: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None

    @property
    def n_samples(self) -> int:
        return int(self.samples.shape[0])

    @property
    def n_channels(self) -> int:
        return 1 if self.samples.ndim == 1 else int(self.samples.shape[1])

    @property
    def duration_s(self) -> float:
        return float(self.n_samples) / float(self.sr if self.sr else 1)

    @property
    def peak(self) -> float:
        return float(np.max(np.abs(self.samples))) if self.n_samples else 0.0

    @property
    def rms(self) -> float:
        return float(np.sqrt(np.mean(np.square(self.samples)))) if self.n_samples else 0.0

    def summary(self) -> Dict[str, Any]:
        return {
            "sr": self.sr,
            "channels": self.n_channels,
            "duration_s": round(self.duration_s, 3),
            "peak": round(self.peak, 6),
            "rms": round(self.rms, 6),
            "path": self.path,
            "meta": self.meta or {},
        }


# ----------------------------- Helpers -----------------------------

def _to_float32(x: np.ndarray) -> np.ndarray:
    """Convert common PCM/float types to float32 in [-1, 1]."""
    if x.dtype == np.int16:
        y = x.astype(np.float32) / 32768.0
    elif x.dtype == np.int32:
        y = x.astype(np.float32) / 2147483648.0
    elif x.dtype == np.uint8:
        # 8-bit WAV is offset binary [0..255]
        y = (x.astype(np.float32) - 128.0) / 128.0
    elif x.dtype in (np.float32, np.float64):
        y = x.astype(np.float32)
        # Assume already -1..1 but sanitize below
    else:
        raise ValueError(f"Unsupported WAV dtype: {x.dtype}")
    # Sanitize
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    # Clip extreme outliers for safety
    y = np.clip(y, -1.0, 1.0)
    return y


def _ensure_stereo(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return np.stack([x, x], axis=-1)
    if x.shape[1] == 1:
        return np.repeat(x, 2, axis=1)
    return x


def _downmix_mono(x: np.ndarray) -> np.ndarray:
    return x if x.ndim == 1 else np.mean(x, axis=1).astype(np.float32)


def tpdf_dither_16bit(x: np.ndarray) -> np.ndarray:
    """Triangular PDF dithering to 16-bit PCM."""
    x = np.clip(x, -1.0, 1.0)
    lsb = 1.0 / 32768.0
    noise = (np.random.rand(*x.shape).astype(np.float32) - np.random.rand(*x.shape).astype(np.float32)) * lsb
    y = x + noise
    y = np.clip(y, -1.0, 1.0)
    return np.int16(np.round(y * 32767.0))


def resample_poly(x: np.ndarray, sr_in: int, sr_out: int) -> Tuple[np.ndarray, int]:
    """High-quality rational resampling with polyphase filtering."""
    if sr_in == sr_out:
        return x, sr_in
    gcd = math.gcd(sr_in, sr_out)
    up = sr_out // gcd
    down = sr_in // gcd
    y = signal.resample_poly(x, up, down, axis=0 if x.ndim > 1 else 0)
    return y.astype(np.float32), sr_out


def ensure_dir(path: str) -> None:
    """Create parent dir for a file path if missing."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


def with_suffix(input_path: str, suffix: str, ext: Optional[str] = None) -> str:
    """Returns a new path by inserting a suffix before extension (or switching ext)."""
    d, base = os.path.split(input_path)
    stem, old_ext = os.path.splitext(base)
    new_ext = ext if ext else old_ext
    name = f"{stem}{suffix}{new_ext}"
    return os.path.join(d, name)


def sha256_file(path: str, chunk_size: int = 1 << 20) -> str:
    """Compute SHA-256 for reproducibility tracking."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def slice_preview(buf: AudioBuffer, start_s: float = 0.0, dur_s: float = 30.0) -> AudioBuffer:
    """Return a window for fast preview rendering."""
    n0 = max(0, int(start_s * buf.sr))
    n1 = min(buf.n_samples, n0 + int(dur_s * buf.sr))
    sl = buf.samples[n0:n1].copy()
    return AudioBuffer(sr=buf.sr, samples=sl, path=buf.path, meta={"slice": [start_s, dur_s], **(buf.meta or {})})


# ----------------------------- I/O API -----------------------------

def load_wav(path: str,
             target_sr: Optional[int] = None,
             mono: bool = False,
             sanitize: bool = True) -> AudioBuffer:
    """Load a WAV into float32[-1,1], optional resample & mono downmix."""
    sr, data = wavfile.read(path)
    y = _to_float32(data)
    if mono:
        y = _downmix_mono(y)
    # Resample if needed
    if target_sr is not None and target_sr != sr:
        y, sr = resample_poly(y, sr, target_sr)
    # Final sanitize
    if sanitize:
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.clip(y, -1.0, 1.0)
    # Pack
    meta = {
        "loaded_at": datetime.utcnow().isoformat() + "Z",
        "source_sha256": sha256_file(path),
        "dtype_in": str(data.dtype),
        "channels_in": 1 if data.ndim == 1 else int(data.shape[1]),
    }
    return AudioBuffer(sr=sr, samples=y, path=os.path.abspath(path), meta=meta)


def save_wav(path: str,
             buf: AudioBuffer | np.ndarray,
             sr: Optional[int] = None,
             bitdepth: str = "float32",
             dither_16bit: bool = True) -> str:
    """Save an AudioBuffer or ndarray to WAV. Supports float32 or int16 PCM output."""
    if isinstance(buf, AudioBuffer):
        y = buf.samples
        sr_out = buf.sr
    else:
        if sr is None:
            raise ValueError("When saving a raw ndarray, 'sr' must be provided.")
        y = np.asarray(buf, dtype=np.float32)
        sr_out = sr

    ensure_dir(path)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.clip(y, -1.0, 1.0).astype(np.float32)

    if bitdepth == "float32":
        wavfile.write(path, sr_out, y.astype(np.float32))
    elif bitdepth == "int16":
        pcm16 = tpdf_dither_16bit(y) if dither_16bit else np.int16(np.round(np.clip(y, -1, 1) * 32767.0))
        wavfile.write(path, sr_out, pcm16)
    else:
        raise ValueError("bitdepth must be 'float32' or 'int16'")

    return os.path.abspath(path)


# ----------------------------- Convenience Summaries -----------------------------

def print_audio_summary(buf: AudioBuffer, name: str = "Audio"):
    s = buf.summary()
    print(f"{name}: sr={s['sr']} | ch={s['channels']} | dur={s['duration_s']}s | peak={s['peak']} | rms={s['rms']}")
    if buf.path:
        print(f"  path: {buf.path}")
    if buf.meta:
        if "source_sha256" in buf.meta:
            print(f"  sha256: {buf.meta['source_sha256'][:16]}...")
        if "dtype_in" in buf.meta:
            print(f"  src dtype: {buf.meta['dtype_in']} | src ch: {buf.meta.get('channels_in')}")


def auto_out_path(input_path: str, stage: str, ext: str = ".wav") -> str:
    """Create a standardized output filename like 'song__stage.wav' in same folder."""
    d, base = os.path.split(input_path)
    stem, _ = os.path.splitext(base)
    name = f"{stem}__{stage}{ext}"
    return os.path.join(d, name)


print("Post-Mix I/O layer loaded: AudioBuffer, load_wav, save_wav, resample_poly, slice_preview, with_suffix, auto_out_path, sha256_file, print_audio_summary.")


# Post-Mix I/O — Convenience helpers for notebook workflows
# Adds:
# - batch_load_wavs (accepts glob or list)
# - run workspace scaffolding (timestamped work dir tree)
# - manifest creation/update (JSON)
# - artifact registration (copy + hash + metadata)
# - small utilities for copying, naming, and environment capture
#
# This cell *extends* the I/O layer previously loaded.




# --- Workspace paths model ---
@dataclass
class RunPaths:
    root: str
    inputs: str
    work: str
    outputs: str
    reports: str

def _timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d-%H%M%S")

def make_workspace(base_dir: str = "./postmix_runs", project: str = "default", slug: Optional[str] = None) -> RunPaths:
    ts = _timestamp()
    slug_part = f"_{slug}" if slug else ""
    root = os.path.abspath(os.path.join(base_dir, f"{project}_{ts}{slug_part}"))
    paths = RunPaths(
        root=root,
        inputs=os.path.join(root, "inputs"),
        work=os.path.join(root, "work"),
        outputs=os.path.join(root, "outputs"),
        reports=os.path.join(root, "reports"),
    )
    for p in asdict(paths).values():
        os.makedirs(p, exist_ok=True)
    print(f"Workspace created at: {paths.root}")
    return paths

# --- Batch load WAVs ---
def batch_load_wavs(paths: Union[str, Iterable[str]], target_sr: Optional[int] = None, mono: bool = False) -> Dict[str, AudioBuffer]:
    """
    Load multiple WAVs. 'paths' can be a glob pattern ('/path/*.wav') or an iterable of paths.
    Returns dict: {stem: AudioBuffer}
    """
    if isinstance(paths, str):
        file_list = sorted(glob.glob(paths))
    else:
        file_list = list(paths)
    if not file_list:
        print("No files matched.")
        return {}

    buffers: Dict[str, AudioBuffer] = {}
    for p in file_list:
        try:
            buf = load_wav(p, target_sr=target_sr, mono=mono)
            stem = os.path.splitext(os.path.basename(p))[0]
            buffers[stem] = buf
            print_audio_summary(buf, name=stem)
        except Exception as e:
            print(f"Failed to load {p}: {e}")
    return buffers

# --- Environment capture (for reproducibility) ---
def env_fingerprint() -> Dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        # scipy and others may be present from earlier cells
        "scipy": __import__("scipy").__version__ if "scipy" in sys.modules else None,
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
    }

# --- Manifest model ---
@dataclass
class Manifest:
    project: str
    workspace: RunPaths
    inputs: List[Dict[str, Any]] = field(default_factory=list)     # list of audio inputs and hashes
    params: Dict[str, Any] = field(default_factory=dict)           # top-level run params (optional)
    artifacts: List[Dict[str, Any]] = field(default_factory=list)  # produced files
    env: Dict[str, Any] = field(default_factory=env_fingerprint)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Expand dataclass paths as dict
        d["workspace"] = asdict(self.workspace)
        return d

def manifest_path(paths: RunPaths) -> str:
    return os.path.join(paths.root, "manifest.json")

def write_manifest(man: Manifest) -> str:
    path = manifest_path(man.workspace)
    with open(path, "w") as f:
        json.dump(man.to_dict(), f, indent=2)
    print(f"Manifest written: {path}")
    return path

def read_manifest(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)

# --- Artifact registration ---
def copy_into(dst_dir: str, src_path: str, new_name: Optional[str] = None) -> str:
    os.makedirs(dst_dir, exist_ok=True)
    base = new_name if new_name else os.path.basename(src_path)
    dst = os.path.join(dst_dir, base)
    shutil.copy2(src_path, dst)
    return os.path.abspath(dst)

def register_input(man: Manifest, path: str, alias: Optional[str] = None) -> Dict[str, Any]:
    info = {
        "alias": alias or os.path.splitext(os.path.basename(path))[0],
        "path": os.path.abspath(path),
        "sha256": sha256_file(path),
    }
    man.inputs.append(info)
    return info

def register_artifact(man: Manifest, file_path: str, kind: str, params: Optional[Dict[str, Any]] = None, stage: Optional[str] = None) -> Dict[str, Any]:
    rec = {
        "kind": kind,                       # e.g., "premaster", "master_landr", "stream_spotify"
        "stage": stage,                     # optional string label
        "path": os.path.abspath(file_path),
        "sha256": sha256_file(file_path),
        "params": params or {},
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    man.artifacts.append(rec)
    return rec

# --- Convenience: bring source mix into workspace/inputs ---
def import_mix(paths: RunPaths, source_path: str, alias: Optional[str] = None) -> str:
    dst_name = (alias or os.path.basename(source_path))
    dst = copy_into(paths.inputs, source_path, new_name=dst_name)
    print(f"Imported mix → {dst}")
    return dst

print("Convenience helpers loaded: make_workspace, batch_load_wavs, env_fingerprint, Manifest, write/read_manifest, register_input, register_artifact, import_mix.")
