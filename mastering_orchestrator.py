

# Mastering Orchestrator — Cleaned Up Version
# Handles mastering processing with local DSP and external service providers

import os
import time
import numpy as np
import soundfile as sf
from scipy import signal
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List
from config import CONFIG
from audio_utils import (
    sanitize_audio, db_to_linear, true_peak_db, normalize_true_peak,
    validate_audio, to_mono
)
from data_handler import register_artifact


class MasteringError(Exception):
    """Raised when mastering processing fails."""
    pass

# --- Data structures ---

@dataclass
class MasterRequest:
    input_path: str
    style: str = "neutral"
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




# --- EQ filters ---

# --- EQ filters ---

def _shelf_filter(x: np.ndarray, sr: int, freq: float, gain_db: float, shelf_type: str) -> np.ndarray:
    """Generic shelf filter implementation with error handling."""
    try:
        validate_audio(x, "shelf filter input")
        
        A = 10**(gain_db/40.0)
        w0 = 2 * np.pi * freq / sr
        cosw0 = np.cos(w0)
        sinw0 = np.sin(w0)
        S = CONFIG.processing.shelf_s_factor
        alpha = sinw0 / 2 * np.sqrt((A + 1/A) * (1/S - 1) + 2)
        
        if shelf_type == "low":
            b0 = A * ((A + 1) - (A - 1) * cosw0 + 2 * np.sqrt(A) * alpha)
            b1 = 2 * A * ((A - 1) - (A + 1) * cosw0)
            b2 = A * ((A + 1) - (A - 1) * cosw0 - 2 * np.sqrt(A) * alpha)
            a0 = (A + 1) + (A - 1) * cosw0 + 2 * np.sqrt(A) * alpha
            a1 = -2 * ((A - 1) + (A + 1) * cosw0)
            a2 = (A + 1) + (A - 1) * cosw0 - 2 * np.sqrt(A) * alpha
        else:  # high shelf
            b0 = A * ((A + 1) + (A - 1) * cosw0 + 2 * np.sqrt(A) * alpha)
            b1 = -2 * A * ((A - 1) + (A + 1) * cosw0)
            b2 = A * ((A + 1) + (A - 1) * cosw0 - 2 * np.sqrt(A) * alpha)
            a0 = (A + 1) - (A - 1) * cosw0 + 2 * np.sqrt(A) * alpha
            a1 = 2 * ((A - 1) - (A + 1) * cosw0)
            a2 = (A + 1) - (A - 1) * cosw0 - 2 * np.sqrt(A) * alpha
        
        # Check for stability
        if abs(a0) < 1e-12:
            raise MasteringError(f"Unstable {shelf_type} shelf filter at {freq} Hz")
        
        sos = signal.tf2sos([b0/a0, b1/a0, b2/a0], [1.0, a1/a0, a2/a0])
        clean_input = sanitize_audio(x)
        return signal.sosfilt(sos, clean_input, axis=0 if clean_input.ndim > 1 else 0).astype(np.float32)
        
    except Exception as e:
        raise MasteringError(f"Shelf filter failed: {e}")

def _lowshelf(x: np.ndarray, sr: int, freq: float, gain_db: float) -> np.ndarray:
    """Low shelf filter."""
    return _shelf_filter(x, sr, freq, gain_db, "low")

def _highshelf(x: np.ndarray, sr: int, freq: float, gain_db: float) -> np.ndarray:
    """High shelf filter."""
    return _shelf_filter(x, sr, freq, gain_db, "high")

def _peaking_eq(x: np.ndarray, sr: int, freq: float, gain_db: float, Q: float = None) -> np.ndarray:
    """Peaking EQ filter with configurable Q factor."""
    if Q is None:
        Q = CONFIG.processing.default_q_factor
    
    try:
        validate_audio(x, "peaking EQ input")
        
        A = 10**(gain_db/40.0)
        w0 = 2 * np.pi * freq / sr
        alpha = np.sin(w0) / (2 * Q)
        cosw0 = np.cos(w0)
        
        b0 = 1 + alpha * A
        b1 = -2 * cosw0
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * cosw0
        a2 = 1 - alpha / A
        
        if abs(a0) < 1e-12:
            raise MasteringError(f"Unstable peaking EQ at {freq} Hz")
        
        sos = signal.tf2sos([b0/a0, b1/a0, b2/a0], [1.0, a1/a0, a2/a0])
        clean_input = sanitize_audio(x)
        return signal.sosfilt(sos, clean_input, axis=0 if clean_input.ndim > 1 else 0).astype(np.float32)
        
    except Exception as e:
        raise MasteringError(f"Peaking EQ failed: {e}")

# --- Dynamics processing ---

def _lookahead_limiter(x: np.ndarray, sr: int,
                      ceiling_dbfs: float = None,
                      lookahead_ms: float = None,
                      attack_ms: float = None,
                      release_ms: float = None,
                      knee_db: float = None) -> np.ndarray:
    """
    Soft-knee lookahead limiter for transparent peak control.
    """
    # Use config defaults if not provided
    if ceiling_dbfs is None:
        ceiling_dbfs = CONFIG.audio.render_peak_target_dbfs
    if lookahead_ms is None:
        lookahead_ms = CONFIG.processing.limiter_lookahead_ms
    if attack_ms is None:
        attack_ms = CONFIG.processing.limiter_attack_ms
    if release_ms is None:
        release_ms = CONFIG.processing.limiter_release_ms
    if knee_db is None:
        knee_db = CONFIG.processing.limiter_knee_db
    
    try:
        validate_audio(x, "limiter input")
        
        clean_input = sanitize_audio(x)
        la_samples = max(1, int(sr * lookahead_ms / 1000.0))
        ceiling_linear = db_to_linear(ceiling_dbfs)
        knee_linear = db_to_linear(-abs(knee_db))

        # Create lookahead buffer
        if clean_input.ndim == 1:
            pad = np.zeros(la_samples, dtype=clean_input.dtype)
            x_delayed = np.concatenate([pad, clean_input])
            x_detector = np.concatenate([clean_input, pad])
        else:
            pad = np.zeros((la_samples, clean_input.shape[1]), dtype=clean_input.dtype)
            x_delayed = np.vstack([pad, clean_input])
            x_detector = np.vstack([clean_input, pad])

        # Envelope detector
        attack_coeff = np.exp(-1.0 / max(1, int(sr * attack_ms / 1000.0)))
        release_coeff = np.exp(-1.0 / max(1, int(sr * release_ms / 1000.0)))
        envelope = np.zeros_like(x_detector, dtype=np.float32)
        magnitude = np.abs(x_detector)
        
        if clean_input.ndim == 1:
            env_state = 0.0
            for i in range(len(magnitude)):
                coeff = attack_coeff if magnitude[i] > env_state else release_coeff
                env_state = max(magnitude[i], env_state * coeff)
                envelope[i] = env_state
        else:
            env_state = np.zeros(clean_input.shape[1], dtype=np.float32)
            for i in range(len(magnitude)):
                current_mag = magnitude[i]
                coeff = attack_coeff if np.any(current_mag > env_state) else release_coeff
                env_state = np.maximum(current_mag, env_state * coeff)
                envelope[i] = env_state

        # Gain reduction with soft knee
        eps = 1e-12
        knee_ratio = envelope / (envelope + knee_linear * ceiling_linear + eps)
        target_gain = ceiling_linear / (envelope + eps)
        gain = 1.0 - knee_ratio + knee_ratio * np.minimum(1.0, target_gain)

        # Apply gain and trim to original length
        limited = (x_delayed * gain[:len(x_delayed)]).astype(np.float32)
        return limited[la_samples:la_samples + len(clean_input)]
        
    except Exception as e:
        raise MasteringError(f"Limiter failed: {e}")

# --- Provider implementations ---

class MasteringProvider:
    """Base class for mastering providers."""
    
    def __init__(self, name: str = "base", bit_depth: str = "PCM_24"):
        self.name = name
        self.bit_depth = bit_depth
    
    def submit(self, req: MasterRequest) -> str:
        """Submit mastering request, return job ID."""
        raise NotImplementedError
    
    def poll(self, job_id: str) -> str:
        """Check job status."""
        raise NotImplementedError
    
    def download(self, job_id: str, out_path: str) -> MasterResult:
        """Download result to out_path."""
        raise NotImplementedError

class LocalMasterProvider(MasteringProvider):
    """Local DSP mastering with multiple styles."""
    
    def __init__(self, bit_depth: str = "PCM_24"):
        super().__init__("local", bit_depth)
    
    def _apply_style(self, audio: np.ndarray, sr: int, style: str, strength: float) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply mastering style with given strength using configuration values."""
        strength = np.clip(strength, CONFIG.mastering.min_strength, CONFIG.mastering.max_strength)
        
        try:
            validate_audio(audio, f"mastering input for {style} style")
            
            # Pre-process: normalize to safe level
            processed = normalize_true_peak(audio, sr, target_dbtp=-2.5)
            
            # Apply style-specific processing using config values
            if style == "warm":
                processed = _lowshelf(processed, sr, CONFIG.mastering.warm_low_shelf_hz, 1.2 * strength)
                processed = _peaking_eq(processed, sr, CONFIG.mastering.warm_cut_freq_hz, -0.6 * strength, Q=1.0)
                glue_amount = CONFIG.mastering.warm_glue_base + CONFIG.mastering.warm_glue_scale * strength
                
            elif style == "bright":
                processed = _highshelf(processed, sr, CONFIG.mastering.bright_high_shelf_hz, 1.4 * strength)
                processed = _peaking_eq(processed, sr, CONFIG.mastering.bright_cut_freq_hz, -0.5 * strength, Q=0.9)
                glue_amount = CONFIG.mastering.bright_glue_base + CONFIG.mastering.bright_glue_scale * strength
                
            elif style == "loud":
                processed = _highshelf(processed, sr, CONFIG.mastering.neutral_high_shelf_hz, 1.0 * strength)
                processed = _lowshelf(processed, sr, CONFIG.mastering.neutral_low_shelf_hz, 0.8 * strength)
                glue_amount = CONFIG.mastering.loud_glue_base + CONFIG.mastering.loud_glue_scale * strength
                
            elif style == "optimized_youtube":
                # YouTube optimization: midrange focus, controlled bass, aggressive limiting
                # Midrange presence boost for phone speakers and perceived loudness
                processed = _peaking_eq(processed, sr, 2500.0, 1.8 * strength, Q=0.8)  # Midrange presence
                processed = _peaking_eq(processed, sr, 4000.0, 1.2 * strength, Q=1.2)  # Upper midrange clarity
                # Controlled low-end to avoid wasting headroom
                processed = _peaking_eq(processed, sr, 60.0, -0.8 * strength, Q=0.7)   # Tighten sub-bass
                processed = _lowshelf(processed, sr, 120.0, 0.6 * strength)            # Controlled bass shelf
                # High-frequency management for AAC codec survival
                processed = _highshelf(processed, sr, 8000.0, 0.8 * strength)          # Gentle high shelf
                processed = _peaking_eq(processed, sr, 12000.0, -0.4 * strength, Q=1.0) # De-harsh for codec
                # More aggressive glue for YouTube's harsh penalty (-14 LUFS target → aim for -13 LUFS)
                glue_amount = 0.25 + 0.35 * strength  # More aggressive than other styles
                
            else:  # neutral or default
                processed = _highshelf(processed, sr, CONFIG.mastering.neutral_high_shelf_hz, 0.8 * strength)
                processed = _lowshelf(processed, sr, CONFIG.mastering.neutral_low_shelf_hz, 0.5 * strength)
                glue_amount = CONFIG.mastering.neutral_glue_base + CONFIG.mastering.neutral_glue_scale * strength
                style = "neutral"
            
            # Apply "glue" compression via parallel limiting
            if style == "optimized_youtube":
                # More aggressive limiting for YouTube optimization
                limited = _lookahead_limiter(processed, sr, ceiling_dbfs=-0.8)  # Tighter ceiling
                processed = (1.0 - glue_amount) * processed + glue_amount * limited
                # Final peak control with tighter true peak for YouTube's AAC codec
                processed = normalize_true_peak(processed, sr, target_dbtp=-1.5)  # Safer for AAC
            else:
                limited = _lookahead_limiter(processed, sr, ceiling_dbfs=-1.2)
                processed = (1.0 - glue_amount) * processed + glue_amount * limited
                # Final peak control
                processed = normalize_true_peak(processed, sr, target_dbtp=CONFIG.audio.render_peak_target_dbfs)
            
            params = {
                "style": style,
                "strength": strength,
                "glue_amount": round(glue_amount, 3),
                "final_true_peak_dbtp": round(true_peak_db(processed, sr), 3),
                "config_version": "2.0"
            }
            
            return processed.astype(np.float32), params
            
        except Exception as e:
            raise MasteringError(f"Style '{style}' processing failed: {e}")
    
    def submit(self, req: MasterRequest) -> str:
        return "local-sync"  # Local processing is synchronous
    
    def run_sync(self, req: MasterRequest, out_path: str) -> MasterResult:
        """Process audio synchronously and save result."""
        try:
            audio, sr = sf.read(req.input_path)
            processed_audio, params = self._apply_style(audio, sr, req.style, req.strength)
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
            sf.write(out_path, processed_audio, sr, subtype=self.bit_depth)
            
            return MasterResult(
                provider=self.name,
                style=req.style,
                strength=req.strength,
                out_path=os.path.abspath(out_path),
                sr=sr,
                bit_depth=self.bit_depth,
                params=params
            )
        except Exception as e:
            raise RuntimeError(f"Local mastering failed: {str(e)}")

class LandrProvider(MasteringProvider):
    """Stub for LANDR API integration."""
    
    def __init__(self, api_key: Optional[str] = None, bit_depth: str = "PCM_24"):
        super().__init__("landr", bit_depth)
        self.api_key = api_key or os.environ.get("LANDR_API_KEY")
        if not self.api_key:
            raise ValueError("LANDR API key required")
    
    def submit(self, req: MasterRequest) -> str:
        raise NotImplementedError("LANDR integration not implemented")
    
    def poll(self, job_id: str) -> str:
        raise NotImplementedError("LANDR integration not implemented")
    
    def download(self, job_id: str, out_path: str) -> MasterResult:
        raise NotImplementedError("LANDR integration not implemented")

# --- Orchestrator ---

class MasteringOrchestrator:
    """
    Orchestrates mastering across multiple providers and styles.
    Handles job submission, monitoring, and artifact registration.
    """
    
    def __init__(self, workspace_paths, manifest):
        self.paths = workspace_paths
        self.manifest = manifest
    
    def run(self, 
            premaster_path: str,
            providers: List[MasteringProvider],
            styles: List[Tuple[str, float]],
            out_tag: str = "master",
            level_match_preview_lufs: Optional[float] = None) -> List[MasterResult]:
        """
        Run mastering across all provider/style combinations.
        
        Args:
            premaster_path: Path to pre-mastered audio file
            providers: List of mastering providers to use
            styles: List of (style_name, strength) tuples
            out_tag: Output directory name
            level_match_preview_lufs: If set, create level-matched preview copies
            
        Returns:
            List of MasterResult objects
        """
        if not os.path.exists(premaster_path):
            raise FileNotFoundError(f"Premaster file not found: {premaster_path}")
        
        results = []
        output_dir = os.path.join(self.paths.outputs, out_tag)
        os.makedirs(output_dir, exist_ok=True)
        
        for provider in providers:
            for style, strength in styles:
                try:
                    result = self._process_single_job(
                        provider, premaster_path, style, strength, 
                        output_dir, out_tag, level_match_preview_lufs
                    )
                    if result:
                        results.append(result)
                except Exception as e:
                    print(f"Error processing {provider.name}/{style}: {e}")
                    continue
        
        return results
    
    def run_all_variants(self,
                        variant_metadata: List[Dict[str, Any]],
                        providers: List[MasteringProvider],
                        styles: List[Tuple[str, float]],
                        out_tag: str = "masters",
                        level_match_preview_lufs: Optional[float] = None) -> Dict[str, List[MasterResult]]:
        """
        Run mastering for ALL variants, creating subfolders for each variant.
        
        Args:
            variant_metadata: List of variant metadata from RenderEngine.commit_variants()
            providers: List of mastering providers to use
            styles: List of (style_name, strength) tuples
            out_tag: Base output directory name
            level_match_preview_lufs: If set, create level-matched preview copies
            
        Returns:
            Dict mapping variant names to their mastered results
        """
        all_results = {}
        base_output_dir = os.path.join(self.paths.outputs, out_tag)
        os.makedirs(base_output_dir, exist_ok=True)
        
        for variant_meta in variant_metadata:
            variant_name = os.path.splitext(os.path.basename(variant_meta["out_path"]))[0]
            variant_path = variant_meta["out_path"]
            
            if not os.path.exists(variant_path):
                print(f"⚠️ Skipping missing variant: {variant_path}")
                continue
            
            print(f"🎭 Mastering variant: {variant_name}")
            
            # Create variant-specific subfolder
            variant_output_dir = os.path.join(base_output_dir, variant_name)
            os.makedirs(variant_output_dir, exist_ok=True)
            
            variant_results = []
            
            for provider in providers:
                for style, strength in styles:
                    try:
                        result = self._process_variant_job(
                            provider, variant_path, style, strength,
                            variant_output_dir, out_tag, level_match_preview_lufs,
                            variant_name
                        )
                        if result:
                            variant_results.append(result)
                    except Exception as e:
                        print(f"  ❌ Error processing {provider.name}/{style} for {variant_name}: {e}")
                        continue
            
            all_results[variant_name] = variant_results
            print(f"  ✅ Created {len(variant_results)} mastered versions for {variant_name}")
        
        total_files = sum(len(results) for results in all_results.values())
        print(f"\n🎉 Successfully created {total_files} total mastered files across {len(all_results)} variants")
        
        return all_results
    
    def _process_variant_job(self, provider: MasteringProvider, variant_path: str,
                           style: str, strength: float, output_dir: str, out_tag: str,
                           level_match_preview_lufs: Optional[float], 
                           variant_name: str) -> Optional[MasterResult]:
        """Process a single mastering job for a specific variant."""
        job_name = f"{provider.name}_{style}_{int(round(strength*100))}"
        out_path = os.path.join(output_dir, f"{job_name}.wav")
        
        # Create request
        request = MasterRequest(variant_path, style=style, strength=strength)
        
        # Process based on provider type
        if isinstance(provider, LocalMasterProvider):
            result = provider.run_sync(request, out_path)
        else:
            result = self._handle_async_provider(provider, request, out_path)
        
        if not result:
            return None
        
        # Register artifact with variant context
        register_artifact(self.manifest, result.out_path, kind=f"{out_tag}_variant", params={
            "provider": result.provider,
            "style": result.style,
            "strength": result.strength,
            "variant_name": variant_name,
            **result.params
        }, stage=f"{variant_name}__{job_name}")
        
        # Create level-matched preview if requested
        if level_match_preview_lufs is not None:
            self._create_variant_level_matched_preview(
                result, output_dir, job_name, level_match_preview_lufs, out_tag, variant_name
            )
        
        return result
    
    def _process_single_job(self, provider: MasteringProvider, premaster_path: str,
                           style: str, strength: float, output_dir: str, out_tag: str,
                           level_match_preview_lufs: Optional[float]) -> Optional[MasterResult]:
        """Process a single mastering job."""
        job_name = f"{provider.name}_{style}_{int(round(strength*100))}"
        out_path = os.path.join(output_dir, f"{job_name}.wav")
        
        # Create request
        request = MasterRequest(premaster_path, style=style, strength=strength)
        
        # Process based on provider type
        if isinstance(provider, LocalMasterProvider):
            result = provider.run_sync(request, out_path)
        else:
            result = self._handle_async_provider(provider, request, out_path)
        
        if not result:
            return None
        
        # Register artifact
        register_artifact(self.manifest, result.out_path, kind=out_tag, params={
            "provider": result.provider,
            "style": result.style,
            "strength": result.strength,
            **result.params
        }, stage=job_name)
        
        # Create level-matched preview if requested
        if level_match_preview_lufs is not None:
            self._create_level_matched_preview(
                result, output_dir, job_name, level_match_preview_lufs, out_tag
            )
        
        return result
    
    def _handle_async_provider(self, provider: MasteringProvider, 
                              request: MasterRequest, out_path: str) -> Optional[MasterResult]:
        """Handle asynchronous provider processing."""
        try:
            job_id = provider.submit(request)
            
            # Poll for completion (with timeout)
            max_polls = CONFIG.mastering.async_job_timeout_seconds
            poll_count = 0
            
            while poll_count < max_polls:
                status = provider.poll(job_id)
                if status == "done":
                    return provider.download(job_id, out_path)
                elif status == "error":
                    print(f"Job {job_id} failed")
                    return None
                
                time.sleep(CONFIG.mastering.async_poll_interval_seconds)
                poll_count += 1
            
            print(f"Job {job_id} timed out")
            return None
            
        except NotImplementedError:
            print(f"Provider {provider.name} not fully implemented")
            return None
    
    def _create_level_matched_preview(self, result: MasterResult, output_dir: str,
                                    job_name: str, target_lufs: float, out_tag: str):
        """Create a level-matched preview copy for A/B comparison."""
        try:
            # Load processed audio
            audio, sr = sf.read(result.out_path)
            
            # Simple LUFS approximation
            mono = audio if audio.ndim == 1 else np.mean(audio, axis=1)
            
            # K-weighting approximation using config values
            sos_hp = signal.butter(2, CONFIG.analysis.k_weight_hp_freq/(sr*0.5), btype='highpass', output='sos')
            mono_weighted = signal.sosfilt(sos_hp, mono)
            
            # Estimate current LUFS using config offset
            mean_square = np.mean(mono_weighted**2)
            current_lufs = CONFIG.audio.lufs_bs1770_offset + 10 * np.log10(max(1e-12, mean_square))
            
            # Apply level matching
            gain_db = target_lufs - current_lufs
            level_matched = (audio * db_to_linear(gain_db)).astype(np.float32)
            
            # Ensure safe peak levels
            level_matched = normalize_true_peak(level_matched, sr, target_dbtp=CONFIG.audio.render_peak_target_dbfs)
            
            # Save preview
            preview_path = os.path.join(output_dir, f"{job_name}__LM{int(target_lufs)}LUFS.wav")
            sf.write(preview_path, level_matched, sr, subtype=result.bit_depth)
            
            # Register preview artifact
            register_artifact(self.manifest, preview_path, kind=f"{out_tag}_preview", params={
                "provider": result.provider,
                "style": result.style, 
                "strength": result.strength,
                "level_matched_lufs": target_lufs,
                "gain_applied_db": round(gain_db, 2)
            }, stage=f"{job_name}__preview")
            
        except Exception as e:
            raise MasteringError(f"Failed to create level-matched preview: {e}")
    
    def _create_variant_level_matched_preview(self, result: MasterResult, output_dir: str,
                                            job_name: str, target_lufs: float, out_tag: str, variant_name: str):
        """Create a level-matched preview copy for A/B comparison for a specific variant."""
        try:
            # Load processed audio
            audio, sr = sf.read(result.out_path)
            
            # Simple LUFS approximation
            mono = audio if audio.ndim == 1 else np.mean(audio, axis=1)
            
            # K-weighting approximation using config values
            sos_hp = signal.butter(2, CONFIG.analysis.k_weight_hp_freq/(sr*0.5), btype='highpass', output='sos')
            mono_weighted = signal.sosfilt(sos_hp, mono)
            
            # Estimate current LUFS using config offset
            mean_square = np.mean(mono_weighted**2)
            current_lufs = CONFIG.audio.lufs_bs1770_offset + 10 * np.log10(max(1e-12, mean_square))
            
            # Apply level matching
            gain_db = target_lufs - current_lufs
            level_matched = (audio * db_to_linear(gain_db)).astype(np.float32)
            
            # Ensure safe peak levels
            level_matched = normalize_true_peak(level_matched, sr, target_dbtp=CONFIG.audio.render_peak_target_dbfs)
            
            # Save preview
            preview_path = os.path.join(output_dir, f"{job_name}__LM{int(target_lufs)}LUFS.wav")
            sf.write(preview_path, level_matched, sr, subtype=result.bit_depth)
            
            # Register preview artifact with variant context
            register_artifact(self.manifest, preview_path, kind=f"{out_tag}_variant_preview", params={
                "provider": result.provider,
                "style": result.style, 
                "strength": result.strength,
                "variant_name": variant_name,
                "level_matched_lufs": target_lufs,
                "gain_applied_db": round(gain_db, 2)
            }, stage=f"{variant_name}__{job_name}__preview")
            
        except Exception as e:
            raise MasteringError(f"Failed to create variant level-matched preview: {e}")
