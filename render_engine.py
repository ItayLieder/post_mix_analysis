

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
        Supports both basic dial-based processing and advanced processing chains.
        
        Args:
            base_outdir: Base output directory
            stem_combinations: List of (variant_name, combination_key) tuples
            opts: Render options
            
        Returns:
            Dict mapping variant names to their processing metadata
        """
        from stem_mastering import get_stem_variant_for_combination, STEM_COMBINATIONS
        from config import CONFIG
        
        os.makedirs(base_outdir, exist_ok=True)
        all_results = {}
        
        for variant_name, combination_key in stem_combinations:
            print(f"🎛️ Processing stem combination: {variant_name}")
            
            variant_dir = os.path.join(base_outdir, variant_name)
            os.makedirs(variant_dir, exist_ok=True)
            
            # Check variant type
            is_advanced = combination_key.startswith("advanced:")
            is_extreme = combination_key.startswith("extreme:")
            is_depth = combination_key.startswith("depth:")
            is_musical = combination_key.startswith("musical:")
            is_hybrid = combination_key.startswith("hybrid:")
            
            advanced_variant = None
            extreme_variant = None
            depth_style = None
            musical_style = None
            hybrid_style = None
            
            if is_advanced and CONFIG.pipeline.use_advanced_stem_processing:
                # Use advanced processing pipeline
                from advanced_stem_processing import apply_advanced_processing, ADVANCED_VARIANTS
                
                advanced_name = combination_key.replace("advanced:", "")
                # Find the matching variant
                for av in ADVANCED_VARIANTS:
                    if av.name == advanced_name:
                        advanced_variant = av
                        break
                
                if not advanced_variant:
                    print(f"  ⚠️ Unknown advanced variant: {advanced_name}, using basic processing")
                    is_advanced = False
            
            elif is_extreme and CONFIG.pipeline.use_extreme_stem_processing:
                # Use extreme processing pipeline
                from extreme_stem_processing import apply_extreme_processing, EXTREME_VARIANTS
                
                extreme_name = combination_key.replace("extreme:", "")
                # Find the matching variant
                for ev in EXTREME_VARIANTS:
                    if ev.name == extreme_name:
                        extreme_variant = ev
                        break
                
                if not extreme_variant:
                    print(f"  ⚠️ Unknown extreme variant: {extreme_name}, using basic processing")
                    is_extreme = False
                    
            elif is_depth and CONFIG.pipeline.use_depth_processing:
                # Use depth processing
                depth_style = combination_key.replace("depth:", "")
                valid_depth_styles = ["natural", "dramatic", "intimate", "stadium", "focused"]
                
                if depth_style not in valid_depth_styles:
                    print(f"  ⚠️ Unknown depth style: {depth_style}, using natural")
                    depth_style = "natural"
                    
            elif is_musical and CONFIG.pipeline.use_musical_depth_processing:
                # Use musical depth processing
                musical_style = combination_key.replace("musical:", "")
                valid_musical_styles = ["balanced", "vocal_forward", "warm", "clear", "polished"]
                
                if musical_style not in valid_musical_styles:
                    print(f"  ⚠️ Unknown musical style: {musical_style}, using balanced")
                    musical_style = "balanced"
                    
            elif is_hybrid and CONFIG.pipeline.use_hybrid_processing:
                # Use hybrid processing (advanced + depth)
                hybrid_style = combination_key.replace("hybrid:", "")
                valid_hybrid_styles = ["RadioReady_depth", "Aggressive_depth", "PunchyMix_depth"]
                
                if hybrid_style not in valid_hybrid_styles:
                    print(f"  ⚠️ Unknown hybrid style: {hybrid_style}, using RadioReady_depth")
                    hybrid_style = "RadioReady_depth"
            
            if is_advanced and CONFIG.pipeline.use_advanced_stem_processing and advanced_variant:
                # Advanced processing path
                print(f"  🎨 Using advanced processing: {advanced_variant.description}")
                
                # Load raw stems for advanced processing
                raw_stems = {}
                for stem_type, engine in self.stem_engines.items():
                    # Get raw audio from engine
                    raw_stems[stem_type] = engine.x
                
                # Apply advanced processing chain
                processed_stems = apply_advanced_processing(
                    raw_stems,
                    list(self.stem_engines.values())[0].sr,  # Get sample rate
                    advanced_variant
                )
                
                # Save processed stems
                stem_results = {}
                for stem_type, processed_audio in processed_stems.items():
                    stem_out_path = os.path.join(variant_dir, f"{stem_type}_processed.wav")
                    sf.write(stem_out_path, processed_audio, 
                           list(self.stem_engines.values())[0].sr, subtype=opts.bit_depth if opts else "PCM_24")
                    stem_results[stem_type] = {"out_path": stem_out_path, "advanced": True}
            
            elif is_extreme and CONFIG.pipeline.use_extreme_stem_processing and extreme_variant:
                # Extreme processing path
                print(f"  🔮 Using EXTREME processing: {extreme_variant.description}")
                print(f"      ⚠️ This may take significantly longer...")
                
                # Load raw stems for extreme processing
                raw_stems = {}
                for stem_type, engine in self.stem_engines.items():
                    # Get raw audio from engine
                    raw_stems[stem_type] = engine.x
                
                # Apply extreme processing chain
                processed_stems = apply_extreme_processing(
                    raw_stems,
                    list(self.stem_engines.values())[0].sr,  # Get sample rate
                    extreme_variant,
                    bpm=CONFIG.pipeline.default_bpm
                )
                
                # Save processed stems
                stem_results = {}
                for stem_type, processed_audio in processed_stems.items():
                    stem_out_path = os.path.join(variant_dir, f"{stem_type}_processed.wav")
                    sf.write(stem_out_path, processed_audio, 
                           list(self.stem_engines.values())[0].sr, subtype=opts.bit_depth if opts else "PCM_24")
                    stem_results[stem_type] = {"out_path": stem_out_path, "extreme": True}
                    
            elif is_depth and CONFIG.pipeline.use_depth_processing and depth_style:
                # Depth processing path
                print(f"  🏞️ Using depth processing: {depth_style} positioning")
                
                # Load raw stems for depth processing  
                raw_stems = {}
                for stem_type, engine in self.stem_engines.items():
                    # Get raw audio from engine
                    raw_stems[stem_type] = engine.x
                
                # Apply depth processing
                from depth_processing import create_depth_variant
                processed_stems = create_depth_variant(
                    raw_stems,
                    list(self.stem_engines.values())[0].sr,  # Get sample rate
                    depth_style
                )
                
                # Save processed stems
                stem_results = {}
                for stem_type, processed_audio in processed_stems.items():
                    stem_out_path = os.path.join(variant_dir, f"{stem_type}_processed.wav")
                    sf.write(stem_out_path, processed_audio, 
                           list(self.stem_engines.values())[0].sr, subtype=opts.bit_depth if opts else "PCM_24")
                    stem_results[stem_type] = {"out_path": stem_out_path, "depth": True}
                    
            elif is_musical and CONFIG.pipeline.use_musical_depth_processing and musical_style:
                # Musical depth processing path
                print(f"  🎵 Using musical depth processing: {musical_style} (subtle & professional)")
                
                # Load raw stems for musical depth processing
                raw_stems = {}
                for stem_type, engine in self.stem_engines.items():
                    # Get raw audio from engine
                    raw_stems[stem_type] = engine.x
                
                # Apply musical depth processing
                from subtle_depth_processing import create_musical_depth
                processed_stems = create_musical_depth(
                    raw_stems,
                    list(self.stem_engines.values())[0].sr,  # Get sample rate
                    musical_style
                )
                
                # Save processed stems
                stem_results = {}
                for stem_type, processed_audio in processed_stems.items():
                    stem_out_path = os.path.join(variant_dir, f"{stem_type}_processed.wav")
                    sf.write(stem_out_path, processed_audio, 
                           list(self.stem_engines.values())[0].sr, subtype=opts.bit_depth if opts else "PCM_24")
                    stem_results[stem_type] = {"out_path": stem_out_path, "musical": True}
                    
            elif is_hybrid and CONFIG.pipeline.use_hybrid_processing and hybrid_style:
                # Hybrid processing path (advanced + depth)
                print(f"  🔄 Using hybrid processing: {hybrid_style}")
                
                # Load raw stems for hybrid processing
                raw_stems = {}
                for stem_type, engine in self.stem_engines.items():
                    # Get raw audio from engine
                    raw_stems[stem_type] = engine.x
                
                # Apply hybrid processing
                from hybrid_processing import create_hybrid_variant
                processed_stems = create_hybrid_variant(
                    raw_stems,
                    list(self.stem_engines.values())[0].sr,  # Get sample rate
                    hybrid_style
                )
                
                # Save processed stems
                stem_results = {}
                for stem_type, processed_audio in processed_stems.items():
                    stem_out_path = os.path.join(variant_dir, f"{stem_type}_processed.wav")
                    sf.write(stem_out_path, processed_audio, 
                           list(self.stem_engines.values())[0].sr, subtype=opts.bit_depth if opts else "PCM_24")
                    stem_results[stem_type] = {"out_path": stem_out_path, "hybrid": True}
                
            else:
                # Basic dial-based processing path
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
            
            # Intelligent stem summing (works for both basic and advanced)
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
        Now uses configurable stem gains from CONFIG for better balance control.
        
        Args:
            processed_stems: Dict of stem_type -> processed audio arrays
            
        Returns:
            Final mixed stereo array
        """
        from config import CONFIG
        
        if not processed_stems:
            raise ValueError("No processed stems to sum")
        
        # Ensure all stems have the same length (pad shorter ones)
        max_length = max(len(audio) for audio in processed_stems.values())
        
        # Initialize final mix
        final_mix = np.zeros((max_length, 2), dtype=np.float32)
        
        # Calculate conservative gains based on number of active stems
        num_stems = len(processed_stems)
        
        # Use configured stem gains (now user-adjustable!)
        stem_gains = CONFIG.pipeline.get_stem_gains()  # This supports env vars too
        
        # Apply auto-gain compensation if enabled to prevent clipping
        if CONFIG.pipeline.auto_gain_compensation:
            # Smart scaling based on number of stems and their combined gain
            total_gain = sum(stem_gains.get(stem_type, 0.7) for stem_type in processed_stems.keys())
            if total_gain > 1.5:  # If combined gains might clip
                stem_scale_factor = 1.5 / total_gain
                print(f"    Auto-gain compensation: {20*np.log10(stem_scale_factor):.1f} dB")
            else:
                stem_scale_factor = 1.0
        else:
            stem_scale_factor = 1.0
        
        # Sum stems with intelligent gain handling (supports detailed stems)
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
            
            # Get gain for this specific stem (detailed stems get their own gain, otherwise fallback to category)
            if stem_type in stem_gains:
                # Use specific gain if available (for detailed stems)
                base_gain = stem_gains[stem_type]
            else:
                # Fallback to main category gain
                stem_mapping = {
                    'kick': 'drums', 'snare': 'drums', 'hats': 'drums',
                    'backvocals': 'vocals', 'leadvocals': 'vocals',
                    'guitar': 'music', 'keys': 'music', 'strings': 'music'
                }
                category = stem_mapping.get(stem_type, 'music')  # Default to music
                base_gain = stem_gains.get(category, 0.7)  # Use category gain or default
                print(f"      {stem_type} → {category} category gain")
            
            final_gain = base_gain * stem_scale_factor
            final_mix += audio * final_gain
            print(f"      {stem_type}: gain={base_gain:.2f} (scaled={final_gain:.2f})")
        
        # Measure peak after summing
        peak = np.max(np.abs(final_mix))
        print(f"    Peak after stem summing: {peak:.3f} ({20*np.log10(peak):.1f} dBFS)")
        
        # Apply makeup gain to bring level back up while staying safe
        target_peak = CONFIG.pipeline.stem_sum_target_peak  # Use configured target
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
