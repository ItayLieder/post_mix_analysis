

# Render Engine — Notebook Layer
# Requires previous cells (I/O, Analysis, DSP Primitives, Processors) to be loaded.

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List, Tuple
import os
import numpy as np
import soundfile as sf
from processors import build_preview_cache, render_from_cache  # Import the missing functions
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
        from utils import sanitize_audio
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
            is_big = combination_key.startswith("big:")
            is_advanced = combination_key.startswith("advanced:")
            is_extreme = combination_key.startswith("extreme:")
            is_depth = combination_key.startswith("depth:")
            is_musical = combination_key.startswith("musical:")
            is_hybrid = combination_key.startswith("hybrid:")
            
            big_variant_profile = None
            advanced_variant = None
            extreme_variant = None
            depth_style = None
            musical_style = None
            hybrid_style = None
            
            if is_big:
                # BIG variant processing - the AMAZING sound!
                big_variant_name = combination_key.replace("big:", "")
                
                # SPECIAL CASE: BIG_Exact_Match uses exact processing but with SAME STEMS as other variants
                if big_variant_name == "BIG_Exact_Match":
                    print(f"    💯 EXACT MATCH: Using exact processing recipe with current stems!")
                    # Note: This will use the exact processing (3.0 drums, 2.8 bass, 4.0 vocals, 2.0 music)
                    # but applied to the SAME stems as all other variants (not Reference_mix)
                    # Fall through to normal BIG variants processing
                
                # Normal BIG variants processing
                try:
                    from big_variants_system import get_big_variant_profile
                    big_variant_profile = get_big_variant_profile(big_variant_name)
                    print(f"    🚀 Using BIG variant: {big_variant_profile.description}")
                    
                except ImportError:
                    print(f"    ⚠️ BIG variants system not found, using standard BIG processing")
                    is_big = False
                    
            elif is_advanced and CONFIG.pipeline.use_advanced_stem_processing:
                # Use advanced processing pipeline if module exists
                try:
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
                except ImportError:
                    print(f"  ⚠️ Advanced processing module not found, using basic processing")
                    is_advanced = False
            
            elif is_extreme and CONFIG.pipeline.use_extreme_stem_processing:
                # Use extreme processing pipeline if module exists
                try:
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
                except ImportError:
                    print(f"  ⚠️ Extreme processing module not found, using basic processing")
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
                    stem_out_path, file_created = self._save_individual_stem_if_enabled(
                        stem_type, processed_audio, variant_dir, 
                        list(self.stem_engines.values())[0].sr, opts)
                    stem_results[stem_type] = {"out_path": stem_out_path, "advanced": True, "individual_file_created": file_created}
            
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
                    stem_out_path, file_created = self._save_individual_stem_if_enabled(
                        stem_type, processed_audio, variant_dir, 
                        list(self.stem_engines.values())[0].sr, opts)
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
                    stem_out_path, file_created = self._save_individual_stem_if_enabled(
                        stem_type, processed_audio, variant_dir, 
                        list(self.stem_engines.values())[0].sr, opts)
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
                    stem_out_path, file_created = self._save_individual_stem_if_enabled(
                        stem_type, processed_audio, variant_dir, 
                        list(self.stem_engines.values())[0].sr, opts)
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
                    stem_out_path, file_created = self._save_individual_stem_if_enabled(
                        stem_type, processed_audio, variant_dir, 
                        list(self.stem_engines.values())[0].sr, opts)
                    stem_results[stem_type] = {"out_path": stem_out_path, "hybrid": True}
                
            elif is_big and big_variant_profile:
                # BIG VARIANT PROCESSING PATH - Specific variant of amazing sound!
                stem_results = {}
                processed_stems = {}
                
                print(f"  🚀 Using {big_variant_profile.name} processing!")
                
                # Import BIG variant processing
                from big_variants_system import apply_big_variant_processing
                
                for stem_type, engine in self.stem_engines.items():
                    # Get raw audio from engine
                    raw_audio = engine.x
                    sample_rate = engine.sr
                    
                    # Apply specific BIG variant processing
                    processed_audio = apply_big_variant_processing(stem_type, raw_audio, sample_rate, big_variant_profile)
                    
                    # BIG variants now handle CONFIG gains internally - no additional gain needed!
                    processed_stems[stem_type] = processed_audio
                    print(f"      🎚️ {stem_type}: BIG processing applied with CONFIG gains!")
                    
                    # Save processed stem (only if enabled)
                    stem_out_path, file_created = self._save_individual_stem_if_enabled(
                        stem_type, processed_audio, variant_dir, sample_rate, opts)
                    
                    # Create metadata
                    stem_results[stem_type] = {
                        "out_path": stem_out_path,
                        "processing": f"big_variant_{big_variant_profile.name}",
                        "peak_dbfs": 20*np.log10(np.max(np.abs(processed_audio))) if np.max(np.abs(processed_audio)) > 0 else -100,
                        "variant_profile": big_variant_profile.name
                    }
                    
                    print(f"        ✅ {stem_type}: {big_variant_profile.name} applied!")
            
            else:
                # STANDARD BIG PROCESSING PATH - Original amazing approach (fallback)
                stem_results = {}
                processed_stems = {}
                
                print(f"  🚀 Using standard BIG processing (AMAZING!)")
                
                for stem_type, engine in self.stem_engines.items():
                    # Get raw audio from engine (bypass broken processing)
                    raw_audio = engine.x
                    sample_rate = engine.sr
                    
                    # Apply standard BIG processing 
                    processed_audio = self._apply_minimal_stem_processing(stem_type, raw_audio, sample_rate)
                    processed_stems[stem_type] = processed_audio
                    
                    # Save processed stem for inspection/debugging (only if enabled)
                    stem_out_path, file_created = self._save_individual_stem_if_enabled(
                        stem_type, processed_audio, variant_dir, sample_rate, opts)
                    
                    # Create metadata
                    stem_results[stem_type] = {
                        "out_path": stem_out_path,  # Will be None if not created
                        "processing": "big_standard_processing",
                        "peak_dbfs": 20*np.log10(np.max(np.abs(processed_audio))) if np.max(np.abs(processed_audio)) > 0 else -100,
                        "individual_file_created": file_created
                    }
                    
                    print(f"      ✅ {stem_type}: Standard BIG processing applied!")
            
            # Intelligent stem summing (works for both basic and advanced)
            # For BIG variants, gains were already applied during processing
            gains_already_applied = is_big and big_variant_profile is not None
            final_mix = self._sum_stems_intelligently(processed_stems, gains_already_applied)
            
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
    
    def _apply_minimal_stem_processing(self, stem_type: str, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Apply BIG, impressive stem processing that makes everything sound AMAZING!
        
        This creates BIGGER, more powerful, and more impressive stems:
        - Massive frequency enhancements for impact
        - Parallel compression for thickness and power  
        - Stereo widening for grandeur and scale
        - Harmonic excitement for richness
        - Sounds AMAZING and MUCH BIGGER!
        """
        from dsp_premitives import peaking_eq, shelf_filter, compressor, stereo_widener
        
        # Start with original audio
        processed = audio.copy()
        
        try:
            if stem_type == 'drums':
                print(f"      🥁 Drums: BIG, POWERFUL processing")
                
                # MASSIVE kick and snare power
                processed = peaking_eq(processed, sample_rate, f0=50, gain_db=3.5, Q=1.2)    # MASSIVE kick
                processed = peaking_eq(processed, sample_rate, f0=80, gain_db=2.5, Q=0.8)    # Kick body
                processed = peaking_eq(processed, sample_rate, f0=200, gain_db=1.5, Q=1.0)   # Drum body
                processed = peaking_eq(processed, sample_rate, f0=3500, gain_db=4.0, Q=1.2)  # HUGE snare crack
                processed = peaking_eq(processed, sample_rate, f0=6000, gain_db=2.5, Q=0.8)  # Snare sizzle
                
                # BIGNESS - Add depth and width
                processed = peaking_eq(processed, sample_rate, f0=35, gain_db=2.8, Q=1.5)    # Deep sub power
                processed = peaking_eq(processed, sample_rate, f0=10000, gain_db=3.0, Q=0.6) # Cymbal sparkle
                processed = peaking_eq(processed, sample_rate, f0=15000, gain_db=2.0, Q=0.4) # Ultra-high air
                
                # STEREO IMPACT - Make drums WIDE and impressive
                if audio.ndim == 2:
                    processed = stereo_widener(processed, width=1.6)  # Much wider drums
                    
            elif stem_type == 'bass':
                print(f"      🎸 Bass: MASSIVE, foundation-shaking processing")
                
                # MASSIVE LOW END - Make bass HUGE
                processed = peaking_eq(processed, sample_rate, f0=35, gain_db=4.5, Q=1.8)    # MASSIVE sub
                processed = peaking_eq(processed, sample_rate, f0=60, gain_db=3.8, Q=1.2)    # Huge fundamental
                processed = peaking_eq(processed, sample_rate, f0=100, gain_db=2.5, Q=1.0)   # Bass body
                
                # DEFINITION and PRESENCE - Cut through the mix
                processed = peaking_eq(processed, sample_rate, f0=800, gain_db=2.0, Q=1.0)   # Bass definition
                processed = peaking_eq(processed, sample_rate, f0=1500, gain_db=1.5, Q=0.8)  # String presence
                processed = peaking_eq(processed, sample_rate, f0=2500, gain_db=1.0, Q=0.6)  # Pick attack
                
                # Remove mud while keeping power
                processed = peaking_eq(processed, sample_rate, f0=250, gain_db=-1.0, Q=2.0)  # Clean mud
                
            elif stem_type == 'vocals':
                print(f"      🎤 Vocals: HUGE, commanding presence")
                
                # MASSIVE PRESENCE - Make vocals DOMINATE
                processed = peaking_eq(processed, sample_rate, f0=1200, gain_db=2.5, Q=0.8)  # Vocal power
                processed = peaking_eq(processed, sample_rate, f0=2800, gain_db=4.5, Q=1.0)  # HUGE presence
                processed = peaking_eq(processed, sample_rate, f0=4200, gain_db=3.0, Q=0.8)  # Vocal clarity
                
                # BIGNESS - Add depth and air
                processed = peaking_eq(processed, sample_rate, f0=200, gain_db=1.8, Q=0.8)   # Vocal body/warmth
                processed = peaking_eq(processed, sample_rate, f0=8000, gain_db=3.5, Q=0.6)  # HUGE air
                processed = peaking_eq(processed, sample_rate, f0=12000, gain_db=2.5, Q=0.4) # Sparkle
                
                # Clean up harsh frequencies while keeping power
                processed = peaking_eq(processed, sample_rate, f0=6000, gain_db=-0.8, Q=2.5) # Gentle de-ess
                
                # STEREO WIDTH for bigger vocal presence
                if audio.ndim == 2:
                    processed = stereo_widener(processed, width=1.3)  # Wider vocals
                    
            elif stem_type == 'music':
                print(f"      🎵 Music: BIG, cinematic, impressive")
                
                # HUGE FREQUENCY SPECTRUM - Make everything bigger
                processed = shelf_filter(processed, sample_rate, cutoff_hz=80, gain_db=2.0, kind='low')   # Huge low end
                processed = peaking_eq(processed, sample_rate, f0=150, gain_db=1.5, Q=0.8)   # Low warmth
                processed = peaking_eq(processed, sample_rate, f0=2000, gain_db=2.0, Q=0.7)  # Presence
                processed = peaking_eq(processed, sample_rate, f0=6000, gain_db=1.8, Q=0.6)  # Clarity
                processed = shelf_filter(processed, sample_rate, cutoff_hz=8000, gain_db=2.8, kind='high') # HUGE air
                processed = peaking_eq(processed, sample_rate, f0=15000, gain_db=2.0, Q=0.4) # Ultra-high
                
                # STEREO GRANDEUR - Make music WIDE and cinematic
                if audio.ndim == 2:
                    processed = stereo_widener(processed, width=1.8)  # VERY wide music
            
            # UNIVERSAL BIGNESS ENHANCEMENTS:
            
            # 1. PARALLEL COMPRESSION for thickness and power
            compressed = compressor(processed, sample_rate, threshold_db=-25, ratio=8.0, 
                                  attack_ms=1.0, release_ms=50.0, makeup_db=6.0)
            # Blend 80% original + 20% heavily compressed for thickness
            processed = processed * 0.8 + compressed * 0.2
            
            # 2. HARMONIC EXCITEMENT for bigger sound
            excitement_amount = 0.15  # More aggressive than minimal
            harmonic_content = np.tanh(processed * 1.5) * excitement_amount
            processed = processed + harmonic_content * 0.3
            
            # 3. DYNAMIC ENHANCEMENT - Make transients more impressive
            envelope = np.abs(processed)
            if processed.ndim == 2:
                envelope = np.mean(envelope, axis=1, keepdims=True)
            
            # Create dynamic enhancement
            enhancement = np.where(envelope > np.percentile(envelope, 70), 1.2, 0.95)
            if processed.ndim == 2 and enhancement.ndim == 2:
                processed = processed * enhancement
            
            # Safety checks - but allow bigger changes since we WANT impressive results
            peak_before = np.max(np.abs(audio))
            peak_after = np.max(np.abs(processed))
            
            if peak_after > peak_before * 6:  # Allow much bigger changes
                print(f"        ⚠️ Processing very aggressive for {stem_type}, reducing by 25%")
                processed = audio + (processed - audio) * 0.75
                
            if peak_after > peak_before * 10:  # Extreme safety check
                print(f"        ⚠️ Processing too extreme for {stem_type}, using 50% blend")
                processed = audio * 0.5 + processed * 0.5
                
        except Exception as e:
            print(f"        ⚠️ BIG processing failed for {stem_type}: {e}, using raw audio")
            return audio
            
        return processed
    
    def _save_individual_stem_if_enabled(self, stem_type: str, processed_audio: np.ndarray, 
                                       variant_dir: str, sample_rate: int, opts) -> tuple[str, bool]:
        """Save individual stem file if enabled by config flag"""
        from config import CONFIG
        if CONFIG.pipeline.create_individual_stem_files:
            stem_out_path = os.path.join(variant_dir, f"{stem_type}_processed.wav")
            
            # Ensure audio is in valid format for writing
            safe_audio = np.clip(processed_audio.astype(np.float32), -0.99, 0.99)
            if np.any(np.isnan(safe_audio)) or np.any(np.isinf(safe_audio)):
                print(f"        ⚠️ Invalid audio data for {stem_type}, using fallback")
                safe_audio = np.zeros_like(processed_audio, dtype=np.float32)
            
            sf.write(stem_out_path, safe_audio, sample_rate, subtype=opts.bit_depth if opts else "PCM_24")
            print(f"      💾 Saved individual stem: {os.path.basename(stem_out_path)}")
            return stem_out_path, True
        else:
            print(f"      ⏭️ Skipped individual stem file for {stem_type} (disabled)")
            return None, False
    
    def _sum_stems_intelligently(self, processed_stems: Dict[str, np.ndarray], gains_already_applied: bool = False) -> np.ndarray:
        """
        Combine processed stems with intelligent gain staging and bus processing.
        Now uses configurable stem gains from CONFIG for better balance control.
        
        Args:
            processed_stems: Dict of stem_type -> processed audio arrays
            gains_already_applied: If True, skip applying stem gains (for BIG variants)
            
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
        
        # Use configured stem gains (exact match to amazing BIG_POWERFUL_STEM_MIX.wav!)
        stem_gains = CONFIG.pipeline.get_stem_gains()  # This supports env vars too
        
        # Apply INTELLIGENT auto-gain compensation
        if CONFIG.pipeline.auto_gain_compensation:
            # More realistic scaling - stems don't all peak simultaneously
            total_gain = sum(stem_gains.get(stem_type, 0.7) for stem_type in processed_stems.keys())
            
            # Use MUCH higher threshold - stems have natural dynamics
            if total_gain > 4.5:  # Only reduce if extremely high (was 1.5)
                # More conservative reduction - don't kill the energy
                stem_scale_factor = 3.5 / total_gain  # Target 3.5 instead of 1.5
                print(f"    Auto-gain compensation: {20*np.log10(stem_scale_factor):.1f} dB")
            else:
                # No reduction needed for normal levels
                stem_scale_factor = 1.0
                print("    No auto-gain compensation needed - stems have natural dynamics")
        else:
            stem_scale_factor = 1.0
        
        # Sum stems with intelligent gain handling (supports detailed stems)
        # Note: For BIG variants, gains were already applied during processing
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
            
            # Apply gains only if they weren't already applied (e.g., by BIG variants processing)
            if gains_already_applied:
                # Gains already applied, just sum with scale factor
                final_gain = stem_scale_factor
                final_mix += audio * final_gain
                print(f"      {stem_type}: gains pre-applied, scale={final_gain:.2f}")
            else:
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
        
        # Apply AGGRESSIVE makeup gain to restore proper loudness
        target_peak = CONFIG.pipeline.stem_sum_target_peak  # Use configured target
        if peak > 0.001:  # Avoid division by zero
            # Check for extreme makeup gain mode
            import os
            extreme_mode = os.getenv('EXTREME_MAKEUP_GAIN', 'false').lower() == 'true'
            max_makeup_gain = float(os.getenv('MAX_MAKEUP_GAIN', '3.0'))
            
            if extreme_mode:
                print(f"    🚀 EXTREME MAKEUP GAIN MODE - allowing up to {max_makeup_gain}x gain!")
                makeup_gain = min(target_peak / peak, max_makeup_gain)  # Use extreme limit
            else:
                # Much more aggressive makeup gain - stems should be LOUD
                makeup_gain = min(target_peak / peak, 3.0)  # Increased from 1.5 to 3.0
                
            final_mix *= makeup_gain
            print(f"    Applied makeup gain: {20*np.log10(makeup_gain):.1f} dB")
            
            # Additional loudness boost if still too quiet (more aggressive in extreme mode)
            final_peak_after_makeup = np.max(np.abs(final_mix))
            quiet_threshold = 0.3 if extreme_mode else 0.5  # Lower threshold in extreme mode
            max_extra_boost = max_makeup_gain if extreme_mode else 2.0
            
            if final_peak_after_makeup < quiet_threshold:  # Still too quiet
                extra_boost = min(0.7 / final_peak_after_makeup, max_extra_boost)
                final_mix *= extra_boost
                print(f"    Applied extra loudness boost: {20*np.log10(extra_boost):.1f} dB")
                if extreme_mode:
                    print(f"    🔥 EXTREME MODE: Final result should be MASSIVELY more powerful!")
        
        # Light bus compression for glue if still too hot (more conservative for stems)
        final_peak = np.max(np.abs(final_mix))
        
        # Check if we're in extreme mode - be much more permissive
        import os
        extreme_mode = os.getenv('EXTREME_MAKEUP_GAIN', 'false').lower() == 'true'
        
        # Use higher threshold in extreme mode to preserve power
        compression_threshold = 0.95 if extreme_mode else 0.85
        compression_target = 0.92 if extreme_mode else 0.85
        
        if final_peak > compression_threshold:
            compression_ratio = compression_target / final_peak
            final_mix *= compression_ratio
            mode_text = " (EXTREME MODE)" if extreme_mode else ""
            print(f"    Applied bus compression: {20*np.log10(compression_ratio):.1f} dB{mode_text}")
        else:
            if extreme_mode:
                print(f"    ⚡ EXTREME MODE: No bus compression (peak {final_peak:.3f} ≤ {compression_threshold})")
        
        # Final safety clip
        final_mix = np.clip(final_mix, -0.99, 0.99)  # Slightly under ±1.0 for safety
        
        return final_mix
