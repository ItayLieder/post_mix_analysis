

# Render Engine — Notebook Layer
# Requires previous cells (I/O, Analysis, DSP Primitives, Processors) to be loaded.

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List, Tuple
import os
import numpy as np
import soundfile as sf
from processors import *
from dsp_premitives import *
# ---- Dial state (what the user controls) ----

@dataclass
class DialState:
    bass: float = 0.0     # 0..100
    punch: float = 0.0    # 0..100
    clarity: float = 0.0  # 0..100
    air: float = 0.0      # 0..100
    width: float = 0.0    # 0..100

@dataclass
class PreprocessConfig:
    low_cutoff: float = 120.0
    kick_lo: float = 40.0
    kick_hi: float = 110.0

@dataclass
class RenderOptions:
    target_peak_dbfs: Optional[float] = -1.0   # normalize peak at the end; set None to skip
    hpf_hz: Optional[float] = None             # optional extra HPF before normalize (None = skip)
    bit_depth: str = "PCM_24"                  # "PCM_24" (recommended), "FLOAT", "PCM_16"
    save_headroom_first: bool = False          # if True, do a premaster-style -6 dB pass before dials

# ---- Render Engine ----

class RenderEngine:
    def __init__(self, x: np.ndarray, sr: int, preprocess: Optional[PreprocessConfig] = None):
        """
        x: input stereo/mono numpy array
        sr: sample rate
        preprocess: parameters for cache building (bands + kick envelope)
        """
        self.x = x.astype(np.float32)
        self.sr = int(sr)
        self.pre_cfg = preprocess or PreprocessConfig()
        self.cache = None  # filled by self.preprocess()


    # ---------- BATCH VARIANTS ----------
    def commit_variants(self, out_dir, variants_plan, opts):
        """
        Render all variants in variants_plan into out_dir.
        Each variant is a dict with at least:
          - name: str
          - recommended: bool
          - params: dict of DSP params
        Recommended variants will get '_recommended' in their filename.
        """
    
        os.makedirs(out_dir, exist_ok=True)
        metas = []
    
        for variant in variants_plan:
            # Build output filename
            if variant.get("recommended", False):
                filename = f"{variant['name']}_recommended.wav"
            else:
                filename = f"{variant['name']}.wav"
    
            out_path = os.path.join(out_dir, filename)
    
            # Render the variant
            audio = self.render_variant(variant, opts)
    
            # Save at target bit depth (usually 24bit) and peak normalization
            save_wav_24bit(out_path, audio, opts.sr)
    
            # Collect metadata
            meta = {
                "name": variant["name"],
                "recommended": variant.get("recommended", False),
                "params": variant.get("params", {}),
                "out_path": out_path,
                "bit_depth": opts.bit_depth,
                "peak_target_dbfs": opts.target_peak_dbfs,
            }
            metas.append(meta)
    
        return metas


    
    def render_variants(self, out_dir, variants_plan, opts):
        """
        Render all premastered variants (without mastering) into out_dir.
        Each variant in variants_plan is (name, DialState).
        """
        os.makedirs(out_dir, exist_ok=True)
        metas = []
        for name, dials in variants_plan:
            out_path = os.path.join(out_dir, f"{name}.wav")
            
            self._ensure_cache()
            y, params = render_from_cache(
                self.cache,
                bass_amount=dials.bass,
                punch_amount=dials.punch,
                clarity_amount=dials.clarity,
                air_amount=dials.air,
                width_amount=dials.width,
                target_peak_dbfs=opts.target_peak_dbfs
            )
            sf.write(out_path, y, self.sr, subtype=opts.bit_depth)
            
            metas.append({
                "name": name,
                "out_path": out_path,
                "dials": dials.__dict__,
                "sr": self.sr,
                "bit_depth": opts.bit_depth,
            })
        return metas
    
    def preprocess(self) -> Dict[str, Any]:
        """Build fast preview cache (low/high split, kick band, envelope)."""
        self.cache = build_preview_cache(
            self.x, self.sr,
            low_cutoff=self.pre_cfg.low_cutoff,
            kick_lo=self.pre_cfg.kick_lo,
            kick_hi=self.pre_cfg.kick_hi
        )
        return {
            "sr": self.sr,
            "n_samples": int(self.x.shape[0]),
            "low_cutoff": self.pre_cfg.low_cutoff,
            "kick_lo": self.pre_cfg.kick_lo,
            "kick_hi": self.pre_cfg.kick_hi
        }
    
    def _ensure_cache(self):
        if self.cache is None:
            self.preprocess()

    # ---------- PREVIEW ----------
    def preview(self,
                dials: DialState,
                start_s: float = 0.0,
                dur_s: Optional[float] = 30.0,
                opts: Optional[RenderOptions] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Fast preview using cached bands/envelope. Optional time window.
        Returns (audio_preview, params).
        """
        self._ensure_cache()
        opts = opts or RenderOptions()
        
        # window the cached arrays (no re-filtering)
        n0 = int(max(0, start_s * self.sr))
        n1 = int(self.cache.high.shape[0]) if dur_s is None else int(min(self.cache.high.shape[0], n0 + dur_s * self.sr))
        
        # build a temporary mini-cache slice for fast render
        slice_cache = type(self.cache)(
            sr=self.cache.sr,
            high=self.cache.high[n0:n1],
            low=self.cache.low[n0:n1],
            kick=self.cache.kick[n0:n1],
            env01=self.cache.env01[n0:n1],
            low_cutoff=self.cache.low_cutoff,
            kick_lo=self.cache.kick_lo,
            kick_hi=self.cache.kick_hi
        )
        
        y, params = render_from_cache(
            slice_cache,
            bass_amount=dials.bass,
            punch_amount=dials.punch,
            clarity_amount=dials.clarity,
            air_amount=dials.air,
            width_amount=dials.width,
            target_peak_dbfs=opts.target_peak_dbfs
        )
        return y, params

    # ---------- COMMIT (FULL RENDER) ----------
    def commit(self,
               out_path: str,
               dials: DialState,
               opts: Optional[RenderOptions] = None) -> Dict[str, Any]:
        """
        Full-length render using the precomputed cache (fast) and export to disk.
        Returns a dict of render metadata.
        """
        self._ensure_cache()
        opts = opts or RenderOptions()
        
        # optional premaster headroom first (use the primitive so it HPFs and normalizes)
        x_work = self.x
        pre_meta = None
        if opts.save_headroom_first:
            self.preprocess()

        # dial render over the *full* cache
        y, params = render_from_cache(
            self.cache,
            bass_amount=dials.bass,
            punch_amount=dials.punch,
            clarity_amount=dials.clarity,
            air_amount=dials.air,
            width_amount=dials.width,
            target_peak_dbfs=opts.target_peak_dbfs
        )
        
        # optional extra HPF after dials
        if opts.hpf_hz is not None:
            y = highpass_filter(y, self.sr, cutoff_hz=opts.hpf_hz, order=2)

        # export
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        subtype = {
            "PCM_24": "PCM_24",
            "PCM_16": "PCM_16",
            "FLOAT": "FLOAT"
        }.get(opts.bit_depth.upper(), "PCM_24")
        sf.write(out_path, y, self.sr, subtype=subtype)

        meta = {
            "sr": self.sr,
            "samples": int(y.shape[0]),
            "dials": asdict(dials),
            "preprocess": asdict(self.pre_cfg),
            "params": params,
            "options": asdict(opts),
            "out_path": os.path.abspath(out_path),
            "bit_depth": subtype
        }
        if pre_meta:
            meta["premaster_first"] = pre_meta
        return meta


