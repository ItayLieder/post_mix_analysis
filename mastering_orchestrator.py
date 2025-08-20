

# Mastering Orchestrator — Notebook Layer
# Provider-agnostic runner that:
#  - accepts a pre-master WAV
#  - runs 1..N mastering providers (local + external)
#  - collects outputs, level-matches (optional), and registers artifacts
#
# Includes:
#  - LocalMasterProvider: simple "house master" with a few styles (neutral/warm/bright/loud)
#  - LandrProvider (stub): method contracts to implement when you wire real API calls

# ---- SAFE Mastering patch: limiter + toned-down styles ----
import numpy as np
from dataclasses import dataclass, asdict

from typing import Optional, Tuple, Dict, Any, List
import soundfile as sf
from scipy import signal
import os
import soundfile as sf
import numpy as np
from data_handler import *

# --- Mastering request/result contracts ---

@dataclass
class MasterRequest:
    input_path: str
    style: str = "neutral"   # e.g. "neutral" | "warm" | "bright" | "loud"
    strength: float = 0.5
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class MasterResult:
    provider: str
    style: str
    strength: float
    out_path: str
    sr: int
    bit_depth: str
    params: Dict[str, Any]



@dataclass
class MasteringResult:
    out_path: str
    params: Dict[str, Any]


class MasteringProvider:
    """
    Abstract mastering provider interface.
    Subclasses implement actual mastering logic
    (local DSP, API call to LANDR, etc.)
    """

    def __init__(self, out_dir: str, name: str = "local"):
        self.out_dir = out_dir
        self.name = name
        os.makedirs(out_dir, exist_ok=True)

    def process(self, in_path: str, preset: Optional[str] = None) -> MasteringResult:
        """
        Apply mastering to `in_path` and return MasteringResult.
        This default implementation is a transparent passthrough.
        Override in subclasses to apply real mastering.
        """
        y, sr = sf.read(in_path)
        # --- passthrough ---
        out_path = os.path.join(
            self.out_dir, f"{os.path.splitext(os.path.basename(in_path))[0]}_mastered.wav"
        )
        sf.write(out_path, y, sr, subtype="PCM_24")
        return MasteringResult(out_path=out_path, params={"preset": preset or "transparent"})



class LocalMasteringProvider(MasteringProvider):
    """
    Example local mastering chain: EQ + compression + limiter.
    Replace the DSP stubs with real processing functions.
    """

    def __init__(self, out_dir: str):
        super().__init__(out_dir, name="local")

    def process(self, in_path: str, preset: Optional[str] = None) -> MasteringResult:
        y, sr = sf.read(in_path)

        # --- TODO: implement DSP here ---
        # Example: mild normalization instead of real chain
        peak = np.max(np.abs(y))
        if peak > 0:
            y = y / peak * 0.98  # normalize to -0.2 dBFS

        out_path = os.path.join(
            self.out_dir, f"{os.path.splitext(os.path.basename(in_path))[0]}_mastered.wav"
        )
        sf.write(out_path, y, sr, subtype="PCM_24")

        return MasteringResult(out_path=out_path, params={"preset": preset or "basic_normalize"})


class LandrStubProvider(MasteringProvider):
    """
    Stub for LANDR or other API-based mastering services.
    Currently just copies the file and tags metadata.
    """

    def __init__(self, out_dir: str):
        super().__init__(out_dir, name="landr_stub")

    def process(self, in_path: str, preset: Optional[str] = None) -> MasteringResult:
        y, sr = sf.read(in_path)
        out_path = os.path.join(
            self.out_dir, f"{os.path.splitext(os.path.basename(in_path))[0]}_landr_stub.wav"
        )
        sf.write(out_path, y, sr, subtype="PCM_24")
        return MasteringResult(out_path=out_path, params={"preset": preset or "api_stub"})



def _sanitize(x): 
    return np.clip(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), -4.0, 4.0).astype(np.float32)

def _db_to_lin(db): return 10.0**(db/20.0)

def true_peak_db_approx(x: np.ndarray, sr: int, oversample: int = 4) -> float:
    x_os = signal.resample_poly(_sanitize(x), oversample, 1, axis=0 if x.ndim>1 else 0)
    tp = float(np.max(np.abs(x_os)))
    return 20*np.log10(max(1e-12, tp))

def normalize_true_peak(x: np.ndarray, sr: int, target_dbtp: float = -1.0) -> np.ndarray:
    tp = true_peak_db_approx(x, sr, oversample=4)
    gain_db = target_dbtp - tp
    return (_sanitize(x) * _db_to_lin(gain_db)).astype(np.float32)

# Gentle shelves/peaks (unchanged; just keep amounts conservative)
def lowshelf(x, sr, hz, db): 
    A = 10**(db/40.0)
    w0 = 2*np.pi*hz/sr; cosw0=np.cos(w0); sinw0=np.sin(w0); S=0.5
    alpha = sinw0/2*np.sqrt((A+1/A)*(1/S-1)+2)
    b0=A*((A+1)-(A-1)*cosw0+2*np.sqrt(A)*alpha)
    b1=2*A*((A-1)-(A+1)*cosw0)
    b2=A*((A+1)-(A-1)*cosw0-2*np.sqrt(A)*alpha)
    a0=(A+1)+(A-1)*cosw0+2*np.sqrt(A)*alpha
    a1=-2*((A-1)+(A+1)*cosw0)
    a2=(A+1)+(A-1)*cosw0-2*np.sqrt(A)*alpha
    sos = signal.tf2sos([b0/a0,b1/a0,b2/a0],[1.0,a1/a0,a2/a0])
    return signal.sosfilt(sos,_sanitize(x),axis=0 if x.ndim>1 else 0).astype(np.float32)

def highshelf(x, sr, hz, db):
    A = 10**(db/40.0)
    w0 = 2*np.pi*hz/sr; cosw0=np.cos(w0); sinw0=np.sin(w0); S=0.5
    alpha = sinw0/2*np.sqrt((A+1/A)*(1/S-1)+2)
    b0=A*((A+1)+(A-1)*cosw0+2*np.sqrt(A)*alpha)
    b1=-2*A*((A-1)+(A+1)*cosw0)
    b2=A*((A+1)+(A-1)*cosw0)
    a0=(A+1)-(A-1)*cosw0+2*np.sqrt(A)*alpha
    a1=2*((A-1)-(A+1)*cosw0)
    a2=(A+1)-(A-1)*cosw0-2*np.sqrt(A)*alpha
    sos = signal.tf2sos([b0/a0,b1/a0,b2/a0],[1.0,a1/a0,a2/a0])
    return signal.sosfilt(sos,_sanitize(x),axis=0 if x.ndim>1 else 0).astype(np.float32)

def broad_peak(x, sr, f0, db, Q=0.7):
    A = 10**(db/40.0)
    w0=2*np.pi*f0/sr; alpha=np.sin(w0)/(2*Q); cosw0=np.cos(w0)
    b0=1+alpha*A; b1=-2*cosw0; b2=1-alpha*A
    a0=1+alpha/A; a1=-2*cosw0; a2=1-alpha/A
    sos = signal.tf2sos([b0/a0,b1/a0,b2/a0],[1.0,a1/a0,a2/a0])
    return signal.sosfilt(sos,_sanitize(x),axis=0 if x.ndim>1 else 0).astype(np.float32)

# --- new: lookahead soft-knee limiter (no screechy clamp) ---
def lookahead_limiter(x: np.ndarray, sr: int,
                      ceiling_dbfs: float = -1.0,
                      lookahead_ms: float = 2.0,
                      attack_ms: float = 1.0,
                      release_ms: float = 50.0,
                      knee_db: float = 1.5) -> np.ndarray:
    """
    Feed-forward, soft-knee, lookahead limiter. Mono/stereo safe.
    """
    x = _sanitize(x)
    la = max(1, int(sr * lookahead_ms/1000.0))
    c = _db_to_lin(ceiling_dbfs)
    knee = _db_to_lin(-abs(knee_db))  # knee expressed as a soft blend near the ceiling

    # Lookahead via simple delay
    if x.ndim == 1:
        pad = np.zeros(la, dtype=x.dtype); x_del = np.concatenate([pad, x])
        x_for_det = np.concatenate([x, pad])
    else:
        pad = np.zeros((la, x.shape[1]), dtype=x.dtype); x_del = np.vstack([pad, x])
        x_for_det = np.vstack([x, pad])

    # Peak detector with attack/release
    atk = np.exp(-1.0 / max(1, int(sr*attack_ms/1000.0)))
    rel = np.exp(-1.0 / max(1, int(sr*release_ms/1000.0)))
    env = np.zeros_like(x_for_det, dtype=np.float32)
    mag = np.abs(x_for_det)
    if x.ndim == 1:
        e = 0.0
        for n in range(len(mag)):
            e = max(mag[n], e* (atk if mag[n] > e else rel))
            env[n] = e
    else:
        e = np.zeros(x.shape[1], dtype=np.float32)
        for n in range(len(mag)):
            cur = mag[n]
            e = np.maximum(cur, e*(atk if np.any(cur>e) else rel))
            env[n] = e

    # Gain computer
    # soft knee near the ceiling: reduce ratio smoothly as we approach c
    eps = 1e-12
    over = np.maximum(0.0, env - c)
    knee_mix = (env / (env + knee*c + eps))
    raw_gain = c / (env + eps)
    gain = 1.0 - knee_mix + knee_mix * np.minimum(1.0, raw_gain)

    # Apply gain to delayed signal, trim back to original length
    y = (x_del * gain[:len(x_del)]).astype(np.float32)
    y = y[la: la + len(x)]
    return y

# --- patched LocalMasterProvider using safe limiter and milder EQ ---
class LocalMasterProvider(MasteringProvider):
    name = "local"
    def __init__(self, bit_depth: str = "PCM_24"):
        self.bit_depth = bit_depth

    def _process(self, x: np.ndarray, sr: int, style: str, strength: float) -> Tuple[np.ndarray, Dict[str, Any]]:
        s = float(np.clip(strength, 0.0, 1.0))
        y = _sanitize(x)

        # tiny safety headroom before EQ
        y = normalize_true_peak(y, sr, target_dbtp=-2.5)

        # toned-down styles (keep boosts small; avoid big >10 kHz lifts)
        if style == "neutral":
            y = highshelf(y, sr, 9000, +0.8*s)
            y = lowshelf(y, sr, 90, +0.5*s)
            glue = 0.10 + 0.10*s
        elif style == "warm":
            y = lowshelf(y, sr, 120, +1.2*s)
            y = broad_peak(y, sr, 3500, -0.6*s, Q=1.0)
            glue = 0.12 + 0.12*s
        elif style == "bright":
            # cap bright lift and keep the shelf lower (8–10 kHz) to avoid fizz
            y = highshelf(y, sr, 8500, +1.4*s)
            y = broad_peak(y, sr, 220, -0.5*s, Q=0.9)
            glue = 0.10 + 0.12*s
        elif style == "loud":
            y = highshelf(y, sr, 9000, +1.0*s)
            y = lowshelf(y, sr, 90, +0.8*s)
            glue = 0.16 + 0.18*s
        else:
            style = "neutral"
            glue = 0.10 + 0.10*s

        # "glue" via parallel into limiter (safe, lookahead)
        limited = lookahead_limiter(y, sr, ceiling_dbfs=-1.2, lookahead_ms=2.0, attack_ms=1.0, release_ms=60.0, knee_db=1.5)
        y = (1.0 - glue)*y + glue*limited

        # Final true-peak trim to -1.0 dBTP (prevents inter-sample spikes)
        y = normalize_true_peak(y, sr, target_dbtp=-1.0)

        params = {
            "style": style,
            "strength": s,
            "glue": round(glue, 3),
            "true_peak_dbtp": round(true_peak_db_approx(y, sr), 3)
        }
        return y.astype(np.float32), params

    def submit(self, req: MasterRequest) -> str:
        return "local-sync"

    def run_sync(self, req: MasterRequest, out_path: str) -> MasterResult:
        y, sr = sf.read(req.input_path)
        y_proc, params = self._process(y, sr, req.style, req.strength)
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        sf.write(out_path, y_proc, sr, subtype=self.bit_depth)
        return MasterResult(provider=self.name, style=req.style, strength=req.strength,
                            out_path=os.path.abspath(out_path), sr=sr, bit_depth=self.bit_depth, params=params)

# --- LANDR provider (stub) ---
class LandrProvider(MasteringProvider):
    """
    Stubbed adapter. Fill in with real API calls when ready:
      - __init__(api_key: str)
      - submit(): upload pre-master, select style/strength, returns job_id
      - poll(job_id): query job status ("queued"|"processing"|"done"|"error")
      - download(job_id, out_path): fetch mastered WAV to out_path
    """
    name = "landr"
    def __init__(self, api_key: Optional[str] = None, bit_depth: str = "PCM_24"):
        self.api_key = api_key or os.environ.get("LANDR_API_KEY", None)
        self.bit_depth = bit_depth
        # self.endpoint = "https://api.landr.com/..."  # example placeholder

    def submit(self, req: MasterRequest) -> str:
        raise NotImplementedError("LANDR adapter not wired yet. Implement API call here.")

    def poll(self, job_id: str) -> str:
        raise NotImplementedError("LANDR adapter not wired yet. Implement job status polling.")

    def download(self, job_id: str, out_path: str) -> MasterResult:
        raise NotImplementedError("LANDR adapter not wired yet. Implement download to out_path.")

# --- Orchestrator ---
class MasteringOrchestrator:
    """
    Runs 1..N providers for a given pre-master and registers outputs.
    """
    def __init__(self, workspace_paths, manifest):
        self.paths = workspace_paths
        self.man = manifest

    def run(self,
            premaster_path: str,
            providers: List[MasteringProvider],
            styles: List[Tuple[str,float]],     # list of (style, strength 0..1)
            out_tag: str = "master",
            level_match_preview_lufs: Optional[float] = None  # if set, write *preview* copies level-matched for A/B
            ) -> List[MasterResult]:

        results: List[MasterResult] = []
        base_outdir = os.path.join(self.paths.outputs, out_tag)
        os.makedirs(base_outdir, exist_ok=True)

        for prov in providers:
            for style, strength in styles:
                name = f"{prov.name}_{style}_{int(round(strength*100))}"
                out_path = os.path.join(base_outdir, f"{name}.wav")

                if isinstance(prov, LocalMasterProvider):
                    res = prov.run_sync(MasterRequest(premaster_path, style=style, strength=strength), out_path)
                else:
                    # external provider flow (submit -> poll -> download)
                    job_id = prov.submit(MasterRequest(premaster_path, style=style, strength=strength))
                    status = prov.poll(job_id)
                    while status not in ("done", "error"):
                        time.sleep(2.0)
                        status = prov.poll(job_id)
                    if status == "error":
                        print(f"[{prov.name}] job failed for style={style} strength={strength}")
                        continue
                    res = prov.download(job_id, out_path)

                # register artifact
                register_artifact(self.man, res.out_path, kind=out_tag, params={
                    "provider": res.provider,
                    "style": res.style,
                    "strength": res.strength,
                    **res.params
                }, stage=name)

                results.append(res)

                # optional preview copies level-matched to a LUFS target (for A/B only)
                if level_match_preview_lufs is not None:
                    # lightweight LUFS approx + gain
                    from scipy import signal
                    def k_weight(mono, sr):
                        # simple K-weight from earlier; inline here for convenience
                        sos_hp = signal.butter(2, 38.0/(sr*0.5), btype='highpass', output='sos')
                        y = signal.sosfilt(sos_hp, mono)
                        return y
                    x, sr = sf.read(res.out_path)
                    mono = x if x.ndim==1 else np.mean(x, axis=1)
                    xk = k_weight(mono, sr)
                    ms = float(np.mean(xk**2)); cur_lufs = -0.691 + 10*np.log10(max(1e-12, ms))
                    delta = level_match_preview_lufs - cur_lufs
                    x_matched = (x * _db_to_lin(delta)).astype(np.float32)
                    # keep true peak safe
                    x_matched = normalize_true_peak(x_matched, sr, target_dbtp=-1.0)
                    prev_path = os.path.join(base_outdir, f"{name}__LM{int(level_match_preview_lufs)}LUFS.wav")
                    sf.write(prev_path, x_matched, sr, subtype=res.bit_depth)
                    register_artifact(self.man, prev_path, kind=f"{out_tag}_preview", params={
                        "provider": res.provider, "style": res.style, "strength": res.strength,
                        "level_matched_lufs": level_match_preview_lufs
                    }, stage=f"{name}__preview")

        return results
