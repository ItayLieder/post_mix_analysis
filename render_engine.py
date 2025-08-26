

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
            x_work, pre_meta = premaster_prep(x_work, self.sr, target_peak_dbfs=-6.0, hpf_hz=20.0)

            # If we premastered first, rebuild cache so dials work on the premastered signal
            self.x = x_work
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

        # Validate and sanitize audio before writing
        from audio_utils import sanitize_audio
        try:
            y = sanitize_audio(y, clip_range=2.0)  # Allow some headroom
        except Exception as e:
            print(f"⚠️ Audio sanitization failed: {e}")
            # Fallback: basic cleanup
            y = np.nan_to_num(y, nan=0.0, posinf=0.99, neginf=-0.99)
            y = np.clip(y, -1.0, 1.0)
        
        # Ensure proper dtype
        if y.dtype != np.float32:
            y = y.astype(np.float32)

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

    # ---------- BATCH VARIANTS ----------
    def commit_variants(self,
                        base_outdir: str,
                        variants: List[Tuple[str, DialState]],
                        opts: Optional[RenderOptions] = None) -> List[Dict[str, Any]]:
        """
        Render multiple named variants and return their metadata.
        variants: list of (name, DialState)
        """
        results = []
        for name, d in variants:
            out_path = os.path.join(base_outdir, f"{name}.wav")
            info = self.commit(out_path, dials=d, opts=opts)
            results.append(info)
        return results


# ---- Stem-Aware Render Engine ----

class StemRenderEngine:
    """
    Advanced render engine that processes multiple stems separately then combines intelligently.
    Works alongside the standard RenderEngine for single-file processing.
    """
    
    def __init__(self, stem_set, preprocess: Optional[PreprocessConfig] = None):
        """
        stem_set: StemSet object with drums, bass, vocals, music stems
        preprocess: parameters for cache building per stem
        """
        from stem_mastering import StemSet, validate_stem_set
        
        if not validate_stem_set(stem_set):
            raise ValueError("Invalid stem set provided")
            
        self.stem_set = stem_set
        self.pre_cfg = preprocess or PreprocessConfig()
        self.stem_engines = {}  # Individual render engines per stem
        
        # Create individual render engines for each active stem
        for stem_type in stem_set.get_active_stems():
            stem_audio = getattr(stem_set, stem_type)
            if stem_audio is not None:
                # Convert AudioBuffer to numpy array for RenderEngine
                audio_array = stem_audio.samples
                self.stem_engines[stem_type] = RenderEngine(audio_array, stem_audio.sr, preprocess)
    
    def preprocess_all_stems(self) -> Dict[str, Any]:
        """Build preview cache for all active stems."""
        metadata = {}
        for stem_type, engine in self.stem_engines.items():
            metadata[stem_type] = engine.preprocess()
        return metadata
    
    def commit_stem_variants(self,
                           base_outdir: str,
                           stem_combinations: List[Tuple[str, str]],  # [(variant_name, combination_key)]
                           opts: Optional[RenderOptions] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Process stems with different combination approaches and save results.
        
        Args:
            base_outdir: Base output directory
            stem_combinations: List of (variant_name, combination_key) tuples
            opts: Render options
            
        Returns:
            Dict mapping variant names to their processing metadata
        """
        from stem_mastering import get_stem_variant_for_combination, STEM_COMBINATIONS
        
        os.makedirs(base_outdir, exist_ok=True)
        all_results = {}
        
        for variant_name, combination_key in stem_combinations:
            print(f"🎛️ Processing stem combination: {variant_name}")
            
            variant_dir = os.path.join(base_outdir, variant_name)
            os.makedirs(variant_dir, exist_ok=True)
            
            # Process each stem with appropriate dial settings
            stem_results = {}
            processed_stems = {}
            
            for stem_type, engine in self.stem_engines.items():
                # Get dial settings for this stem in this combination
                dial_state = get_stem_variant_for_combination(stem_type, combination_key)
                
                # Process the stem
                stem_out_path = os.path.join(variant_dir, f"{stem_type}_processed.wav")
                stem_metadata = engine.commit(stem_out_path, dial_state, opts)
                stem_results[stem_type] = stem_metadata
                
                # Load processed stem for summing
                processed_audio, _ = sf.read(stem_out_path)
                processed_stems[stem_type] = processed_audio
                
                print(f"  ✅ Processed {stem_type} stem")
            
            # Intelligent stem summing
            final_mix = self._sum_stems_intelligently(processed_stems)
            
            # Save final mixed result
            final_out_path = os.path.join(variant_dir, f"{variant_name}.wav")
            sample_rate = list(self.stem_engines.values())[0].sr  # Get SR from first engine
            sf.write(final_out_path, final_mix, sample_rate, subtype=opts.bit_depth if opts else "PCM_24")
            
            # Create metadata for the complete variant
            variant_metadata = {
                "variant_name": variant_name,
                "combination_key": combination_key,
                "final_mix_path": os.path.abspath(final_out_path),
                "stem_results": stem_results,
                "active_stems": list(self.stem_engines.keys())
            }
            
            all_results[variant_name] = variant_metadata
            print(f"  🎉 Completed {variant_name} with {len(self.stem_engines)} stems")
        
        return all_results
    
    def _sum_stems_intelligently(self, processed_stems: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Combine processed stems with intelligent gain staging and bus processing.
        
        Args:
            processed_stems: Dict of stem_type -> processed audio arrays
            
        Returns:
            Final mixed stereo array
        """
        if not processed_stems:
            raise ValueError("No processed stems to sum")
        
        # Ensure all stems have the same length (pad shorter ones)
        max_length = max(len(audio) for audio in processed_stems.values())
        
        # Initialize final mix
        final_mix = np.zeros((max_length, 2), dtype=np.float32)
        
        # Calculate conservative gains based on number of active stems
        num_stems = len(processed_stems)
        base_headroom = 0.7  # Start with 70% headroom
        
        # Intelligent gain staging per stem type with headroom consideration
        stem_gains = {
            "drums": 0.7,    # Drums controlled to avoid overpowering
            "bass": 0.6,     # Bass needs the most control for headroom
            "vocals": 0.8,   # Vocals important but controlled
            "music": 0.65    # Music/instruments as foundation
        }
        
        # Apply additional scaling based on number of stems to prevent clipping
        stem_scale_factor = base_headroom / num_stems
        
        # Sum stems with conservative gains
        for stem_type, audio in processed_stems.items():
            # Pad if necessary
            if len(audio) < max_length:
                if audio.ndim == 1:
                    padded = np.zeros((max_length, 2), dtype=np.float32)
                    padded[:len(audio), :] = np.column_stack([audio, audio])
                else:
                    padded = np.zeros((max_length, 2), dtype=np.float32)
                    padded[:len(audio), :] = audio
                audio = padded
            elif audio.ndim == 1:
                # Convert mono to stereo
                audio = np.column_stack([audio, audio])
            
            # Apply conservative stem-specific gain with scaling
            base_gain = stem_gains.get(stem_type, 0.6)
            final_gain = base_gain * stem_scale_factor
            final_mix += audio * final_gain
        
        # Measure peak after summing
        peak = np.max(np.abs(final_mix))
        print(f"    Peak after stem summing: {peak:.3f} ({20*np.log10(peak):.1f} dBFS)")
        
        # Apply makeup gain to bring level back up while staying safe
        target_peak = 0.8  # Target -1.9 dBFS peak
        if peak > 0.001:  # Avoid division by zero
            makeup_gain = min(target_peak / peak, 1.5)  # Limit makeup gain
            final_mix *= makeup_gain
            print(f"    Applied makeup gain: {20*np.log10(makeup_gain):.1f} dB")
        
        # Light bus compression for glue if still too hot
        final_peak = np.max(np.abs(final_mix))
        if final_peak > 0.85:
            compression_ratio = 0.85 / final_peak
            final_mix *= compression_ratio
            print(f"    Applied bus compression: {20*np.log10(compression_ratio):.1f} dB")
        
        # Final safety clip
        final_mix = np.clip(final_mix, -0.99, 0.99)  # Slightly under ±1.0 for safety
        
        return final_mix
