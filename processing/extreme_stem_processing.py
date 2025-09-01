#!/usr/bin/env python3
"""
Extreme Stem Processing - Push the boundaries of stem separation advantages
Includes 3D spatial audio, psychoacoustic processing, AI-inspired patterns, and more
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from scipy import signal, interpolate
from utils import ensure_stereo, to_mono, db_to_linear, linear_to_db
import warnings
warnings.filterwarnings('ignore')  # Suppress scipy warnings for experimental processing


# ============================================
# 3D SPATIAL AUDIO PROCESSING
# ============================================

def create_3d_soundfield(stems: Dict[str, np.ndarray], sr: int, variant: str = "immersive") -> Dict[str, np.ndarray]:
    """
    Create 3D spatial positioning using binaural and depth cues.
    
    Variants:
    - immersive: Full 360° soundfield with height
    - cinema: Movie theater-style positioning
    - binaural: Headphone-optimized 3D
    - dome: Planetarium-style overhead space
    - vr_space: Virtual reality audio positioning
    """
    processed = {}
    
    # 3D position maps (azimuth, elevation, distance)
    position_maps = {
        "immersive": {
            "kick": (0, -2, 0.98),          # Center, barely below, barely closer
            "snare": (5, 0, 1.05),          # Just slightly right, ear level, barely back
            "hats": (-10, 8, 1.15),         # Left, slightly above, just slightly back
            "bass": (0, -5, 0.92),          # Center, slightly below, just slightly close
            "leadvocals": (0, 1, 0.75),     # Center, barely above, MUCH closer for audibility
            "backvocals": ((-10, 3, 0.95), (10, 3, 0.95)),  # Very narrow spread, close for strength
            "guitar": (12, 2, 1.20),        # Right, barely up, just slightly back
            "keys": (-12, 2, 1.20),         # Left, barely up, just slightly back
            "strings": (0, 12, 1.35),       # Center, slightly above, moderate distance
        },
        "cinema": {
            "kick": (0, -1, 0.98),
            "bass": (0, -3, 0.95),
            "leadvocals": (0, 0, 0.75),     # Front, MUCH closer for strength
            "backvocals": ((-20, 0, 1.4), (20, 0, 1.4)),  # Subtle surround
            "music": ((-10, 8, 1.3), (10, 8, 1.3)),       # Subtle overhead
        },
        "binaural": {
            # Optimized for headphone listening - ULTRA CONSERVATIVE
            "kick": (0, 0, 0.95),
            "snare": (3, 1, 1.08),
            "hats": (-8, 6, 1.18),          # Minimal left and above
            "bass": (0, -6, 0.90),          # Slightly below but minimal
            "leadvocals": (0, 2, 0.75),     # MUCH closer for strength
            "backvocals": ((-25, 0, 1.3), (25, 0, 1.3)),  # Subtle left/right
            "guitar": (15, 4, 1.3),
            "keys": (-15, 4, 1.3),
        },
        "dome": {
            # Planetarium-style with emphasis on height - ULTRA CONSERVATIVE
            "drums": (0, -6, 1.15),         # Slightly below but minimal
            "bass": (0, -10, 0.98),         # Slightly below but barely
            "vocals": (0, 5, 0.75),         # Slightly above, MUCH closer for strength
            "music": (0, 15, 1.4),          # Moderate dome height (much reduced)
            "strings": (0, 20, 1.6),        # Above but much more conservative
        },
        "vr_space": {
            # Virtual reality positioning with movement - ULTRA CONSERVATIVE
            "kick": (0, 0, 0.95),
            "snare": (8, 0, 1.05),
            "hats": ((-12, 6, 1.2), (-8, 12, 1.3)),    # Minimal movement
            "bass": (0, -6, 0.88),
            "leadvocals": (0, 2, 0.75),     # MUCH closer for strength
            "guitar": ((15, 0, 1.3), (15, 8, 1.4)),    # Minimal movement
            "keys": ((-15, 0, 1.3), (-15, 8, 1.4)),    # Minimal movement
        }
    }
    
    positions = position_maps.get(variant, position_maps["immersive"])
    
    for stem_name, audio in stems.items():
        if stem_name in positions:
            pos_data = positions[stem_name]
            
            if isinstance(pos_data[0], tuple):
                # Multiple positions - create movement
                processed[stem_name] = apply_3d_movement(audio, sr, pos_data)
            else:
                # Static position
                processed[stem_name] = apply_3d_position(audio, sr, *pos_data)
        else:
            # Default center position
            processed[stem_name] = apply_3d_position(audio, sr, 0, 0, 1.0)
    
    return processed


def apply_3d_position(audio: np.ndarray, sr: int, azimuth: float, elevation: float, distance: float) -> np.ndarray:
    """
    Apply 3D positioning using HRTF-inspired processing.
    
    Args:
        azimuth: Horizontal angle (-180 to 180, 0=front)
        elevation: Vertical angle (-90 to 90, 0=ear level)
        distance: Distance factor (0.5=close, 2.0=far)
    """
    stereo = ensure_stereo(audio)
    
    # Interaural Time Difference (ITD) for azimuth
    head_radius = 0.09  # meters
    speed_of_sound = 343  # m/s
    max_itd = head_radius / speed_of_sound  # seconds
    
    itd = max_itd * np.sin(np.radians(azimuth))
    itd_samples = int(abs(itd * sr))
    
    # Apply ITD
    if itd_samples > 0:
        if azimuth > 0:  # Sound from right
            # Delay left channel
            stereo[:, 0] = np.pad(stereo[:, 0], (itd_samples, 0), mode='constant')[:len(stereo)]
        else:  # Sound from left
            # Delay right channel
            stereo[:, 1] = np.pad(stereo[:, 1], (itd_samples, 0), mode='constant')[:len(stereo)]
    
    # Interaural Level Difference (ILD) for azimuth - ULTRA REDUCED for subtlety
    ild_db = 3 * np.sin(np.radians(azimuth))  # Max ±3dB difference (was ±8dB)
    if azimuth > 0:
        stereo[:, 0] *= db_to_linear(-abs(ild_db) / 2)  # Reduce left
        stereo[:, 1] *= db_to_linear(abs(ild_db) / 2)   # Boost right
    else:
        stereo[:, 0] *= db_to_linear(abs(ild_db) / 2)   # Boost left
        stereo[:, 1] *= db_to_linear(-abs(ild_db) / 2)  # Reduce right
    
    # Elevation cues (spectral filtering)
    if elevation != 0:
        # Higher elevation = more high frequency emphasis
        # Lower elevation = more low frequency emphasis
        from dsp_premitives import shelf_filter
        
        if elevation > 0:  # Above - ULTRA GENTLER EQ changes
            # Boost highs, cut lows - barely perceptible
            stereo = shelf_filter(stereo, sr, 8000, elevation/50, kind="high")  # Was /20
            stereo = shelf_filter(stereo, sr, 200, -elevation/100, kind="low")  # Was /40
        else:  # Below - ULTRA GENTLER EQ changes
            # Boost lows, cut highs - barely perceptible
            stereo = shelf_filter(stereo, sr, 200, abs(elevation)/80, kind="low")  # Was /30
            stereo = shelf_filter(stereo, sr, 8000, elevation/80, kind="high")     # Was /30
    
    # Distance cues
    if distance != 1.0:
        # Volume attenuation - ULTRA GENTLE (barely noticeable)
        stereo *= 1.0 / (distance ** 0.3)  # Was ** 0.8 - now ultra gentle
        
        # High frequency roll-off for distance - MORE CONSERVATIVE
        if distance > 1.3:  # Start rolloff later (was 1.5)
            cutoff = 16000 / (distance ** 0.7)  # Gentler rolloff curve
            cutoff = max(cutoff, 6000)  # Don't cut too much (preserve clarity)
            from dsp_premitives import lowpass_filter
            stereo = lowpass_filter(stereo, sr, cutoff, order=1)  # Gentler slope (was order=2)
        
        # Add early reflections for distance - ULTRA SUBTLE
        if distance > 1.6:  # Start much later (was 1.4)
            reflection_delay = int(0.004 * distance * sr)  # 4ms per unit distance (was 8ms)
            reflection = np.pad(stereo, ((reflection_delay, 0), (0, 0)), mode='constant')[:len(stereo)]
            stereo += reflection * 0.06 * (distance - 1.0)  # Much reduced level (was 0.12)
    
    return stereo.astype(np.float32)


def apply_3d_movement(audio: np.ndarray, sr: int, positions: List[Tuple[float, float, float]]) -> np.ndarray:
    """Apply moving 3D position over time"""
    stereo = ensure_stereo(audio)
    num_positions = len(positions)
    segment_length = len(stereo) // num_positions
    
    processed = np.zeros_like(stereo)
    
    for i, (az, el, dist) in enumerate(positions):
        start = i * segment_length
        end = min(start + segment_length, len(stereo))
        segment = stereo[start:end]
        processed[start:end] = apply_3d_position(segment, sr, az, el, dist)
    
    return processed


# ============================================
# PSYCHOACOUSTIC ENHANCEMENT
# ============================================

def psychoacoustic_enhancement(stems: Dict[str, np.ndarray], sr: int, variant: str = "clarity") -> Dict[str, np.ndarray]:
    """
    Apply psychoacoustic principles for enhanced perception.
    
    Variants:
    - clarity: Maximize intelligibility using masking curves
    - presence: Enhance phantom center and depth
    - energy: Optimize perceived loudness without level increase
    - hypnotic: Repetitive patterns for trance-like effect
    - subliminal: Subtle enhancements below conscious threshold
    """
    processed = {}
    
    for stem_name, audio in stems.items():
        if variant == "clarity":
            processed[stem_name] = enhance_clarity_psycho(audio, sr, stem_name)
        elif variant == "presence":
            processed[stem_name] = enhance_presence_psycho(audio, sr, stem_name)
        elif variant == "energy":
            processed[stem_name] = enhance_energy_psycho(audio, sr, stem_name)
        elif variant == "hypnotic":
            processed[stem_name] = create_hypnotic_effect(audio, sr, stem_name)
        elif variant == "subliminal":
            processed[stem_name] = add_subliminal_enhancement(audio, sr, stem_name)
        else:
            processed[stem_name] = audio
    
    return processed


def enhance_clarity_psycho(audio: np.ndarray, sr: int, stem_type: str) -> np.ndarray:
    """Use masking curves to enhance clarity"""
    from dsp_premitives import peaking_eq
    
    stereo = ensure_stereo(audio)
    
    # Critical bands for different stem types
    critical_bands = {
        "vocals": [(2800, 3.0, 1.2), (5000, 2.5, 1.0)],  # Formants
        "kick": [(60, 3.0, 0.8), (4000, 2.0, 1.5)],      # Fundamental + click
        "snare": [(200, 2.5, 1.0), (5000, 2.0, 1.2)],    # Body + snap
        "bass": [(80, 2.5, 0.7), (1200, 1.5, 1.5)],      # Sub + harmonics
    }
    
    # Get appropriate bands or use default
    bands = critical_bands.get(stem_type, [(1000, 1.5, 1.0)])
    
    # Apply critical band enhancement
    enhanced = stereo.copy()
    for freq, gain, q in bands:
        enhanced = peaking_eq(enhanced, sr, freq, gain, q)
    
    # Add missing fundamental illusion for bass content
    if stem_type in ["kick", "bass"]:
        # Generate harmonics that imply lower fundamental
        fundamental = 50 if stem_type == "kick" else 40
        for harmonic in [2, 3, 4]:
            freq = fundamental * harmonic
            enhanced = peaking_eq(enhanced, sr, freq, 1.0, 2.0)
    
    return enhanced


def enhance_presence_psycho(audio: np.ndarray, sr: int, stem_type: str) -> np.ndarray:
    """Enhance phantom center and depth perception"""
    stereo = ensure_stereo(audio)
    
    # Haas effect for depth
    if stem_type in ["vocals", "leadvocals"]:
        # Very subtle delay difference creates forward presence
        stereo[:, 1] = np.pad(stereo[:, 1], (20, 0), mode='constant')[:len(stereo)]
    
    # Blumlein shuffling for width
    mid = (stereo[:, 0] + stereo[:, 1]) * 0.5
    side = (stereo[:, 0] - stereo[:, 1]) * 0.5
    
    # Enhance side information at specific frequencies
    from dsp_premitives import bandpass_filter
    side_enhanced = bandpass_filter(side, sr, 1000, 8000, order=2) * 1.5
    side = side * 0.7 + side_enhanced * 0.3
    
    # Reconstruct
    enhanced = np.column_stack([mid + side, mid - side])
    
    return enhanced.astype(np.float32)


def enhance_energy_psycho(audio: np.ndarray, sr: int, stem_type: str) -> np.ndarray:
    """Optimize perceived loudness using psychoacoustic curves"""
    stereo = ensure_stereo(audio)
    
    # Fletcher-Munson curve approximation
    # Boost frequencies where ear is less sensitive
    from dsp_premitives import peaking_eq
    
    # Equal loudness contour compensation
    enhanced = peaking_eq(stereo, sr, 100, 2.0, 0.7)   # Low bass boost
    enhanced = peaking_eq(enhanced, sr, 3500, 1.5, 1.0)  # Presence boost
    enhanced = peaking_eq(enhanced, sr, 12000, 2.5, 0.8)  # Air boost
    
    # Add psychoacoustic excitement through micro-dynamics
    envelope = np.abs(signal.hilbert(to_mono(enhanced)))
    envelope_smooth = signal.savgol_filter(envelope, 1001, 3)
    micro_dynamics = (envelope / (envelope_smooth + 1e-6)) - 1.0
    micro_dynamics = np.clip(micro_dynamics * 0.1, -0.1, 0.1)
    
    if enhanced.ndim == 1:
        enhanced *= (1 + micro_dynamics)
    else:
        enhanced *= (1 + micro_dynamics)[:, np.newaxis]
    
    return enhanced.astype(np.float32)


def create_hypnotic_effect(audio: np.ndarray, sr: int, stem_type: str) -> np.ndarray:
    """Create repetitive, trance-inducing patterns"""
    stereo = ensure_stereo(audio)
    
    # Rhythmic amplitude modulation
    if stem_type in ["drums", "kick", "snare"]:
        # Sync to typical BPM (120-130)
        bpm = 125
        beat_hz = bpm / 60
        t = np.arange(len(stereo)) / sr
        
        # Create polyrhythmic modulation
        mod1 = (np.sin(2 * np.pi * beat_hz * t) + 1) * 0.5
        mod2 = (np.sin(2 * np.pi * beat_hz * 1.5 * t) + 1) * 0.5
        mod_combined = mod1 * 0.7 + mod2 * 0.3
        
        # Apply subtle modulation
        stereo *= (0.85 + 0.15 * mod_combined)[:, np.newaxis]
    
    # Rotating pan for hypnotic movement
    if stem_type in ["hats", "keys", "backvocals"]:
        pan_rate = 0.25  # Hz (slow rotation)
        t = np.arange(len(stereo)) / sr
        pan = np.sin(2 * np.pi * pan_rate * t)
        
        # Apply auto-pan
        left_gain = (1 - pan) * 0.5
        right_gain = (1 + pan) * 0.5
        stereo[:, 0] *= left_gain
        stereo[:, 1] *= right_gain
    
    return stereo.astype(np.float32)


def add_subliminal_enhancement(audio: np.ndarray, sr: int, stem_type: str) -> np.ndarray:
    """Add barely perceptible enhancements"""
    stereo = ensure_stereo(audio)
    
    # Ultra-subtle harmonic excitement (just below threshold)
    from advanced_stem_processing import add_harmonics
    excited = add_harmonics(stereo, drive=0.1, sat_type="tube", mix=0.05)
    
    # Subliminal width enhancement
    mid = (stereo[:, 0] + stereo[:, 1]) * 0.5
    side = (stereo[:, 0] - stereo[:, 1]) * 0.5
    side *= 1.05  # 5% width increase (barely noticeable)
    enhanced = np.column_stack([mid + side, mid - side])
    
    # Mix enhanced with original (90% original, 10% enhanced)
    return (stereo * 0.9 + enhanced * 0.1).astype(np.float32)


# ============================================
# AI-INSPIRED INTERACTION PATTERNS
# ============================================

def ai_stem_interaction(stems: Dict[str, np.ndarray], sr: int, variant: str = "adaptive") -> Dict[str, np.ndarray]:
    """
    AI-inspired adaptive interaction between stems.
    
    Variants:
    - adaptive: Self-adjusting based on content analysis
    - neural: Mimics neural network activation patterns
    - predictive: Anticipatory ducking and boosting
    - swarm: Collective behavior patterns
    - quantum: Probabilistic processing inspired by quantum mechanics
    """
    
    if variant == "adaptive":
        return adaptive_interaction(stems, sr)
    elif variant == "neural":
        return neural_activation_pattern(stems, sr)
    elif variant == "predictive":
        return predictive_ducking(stems, sr)
    elif variant == "swarm":
        return swarm_behavior(stems, sr)
    elif variant == "quantum":
        return quantum_probability_processing(stems, sr)
    else:
        return stems


def adaptive_interaction(stems: Dict[str, np.ndarray], sr: int) -> Dict[str, np.ndarray]:
    """Stems adapt to each other based on content analysis"""
    processed = {}
    
    # Analyze energy distribution of each stem
    stem_energies = {}
    for name, audio in stems.items():
        stem_energies[name] = np.mean(np.abs(audio))
    
    # Find dominant stem
    dominant = max(stem_energies, key=stem_energies.get)
    
    for stem_name, audio in stems.items():
        if stem_name == dominant:
            # Dominant stem stays mostly unchanged
            processed[stem_name] = audio * 1.05
        else:
            # Other stems adapt to dominant
            ratio = stem_energies[stem_name] / stem_energies[dominant]
            
            # More adaptation for quieter stems
            adaptation_factor = 1.0 - (ratio * 0.3)
            
            # Apply frequency-dependent adaptation
            from dsp_premitives import bandpass_filter
            low = bandpass_filter(audio, sr, 20, 250, order=2)
            mid = bandpass_filter(audio, sr, 250, 4000, order=2)
            high = bandpass_filter(audio, sr, 4000, 20000, order=2)
            
            # Adapt differently per band
            adapted = (low * (2 - adaptation_factor) +
                      mid * adaptation_factor +
                      high * (1 + adaptation_factor * 0.5))
            
            processed[stem_name] = adapted.astype(np.float32)
    
    return processed


def neural_activation_pattern(stems: Dict[str, np.ndarray], sr: int) -> Dict[str, np.ndarray]:
    """Apply neural network-like activation patterns"""
    processed = {}
    
    for stem_name, audio in stems.items():
        # Sigmoid-like activation for smooth limiting
        x = audio / (np.max(np.abs(audio)) + 1e-6)
        
        # Different activation functions for different stems
        if stem_name in ["kick", "snare"]:
            # ReLU-like for transients
            activated = np.maximum(0, x) * 1.2
            activated = np.minimum(activated, 1.0)
        elif stem_name in ["bass"]:
            # Tanh for smooth saturation
            activated = np.tanh(x * 1.5) * 0.9
        elif stem_name in ["vocals", "leadvocals"]:
            # Soft sigmoid for presence
            activated = x / (1 + np.abs(x) * 0.3)
        else:
            # Leaky ReLU for general content
            activated = np.where(x > 0, x, x * 0.1)
        
        # Restore scale
        processed[stem_name] = (activated * np.max(np.abs(audio))).astype(np.float32)
    
    return processed


def predictive_ducking(stems: Dict[str, np.ndarray], sr: int) -> Dict[str, np.ndarray]:
    """Anticipate transients and duck preemptively"""
    from processors import _envelope_detector
    
    processed = stems.copy()
    
    # Detect upcoming transients
    if "kick" in stems or "drums" in stems:
        trigger = stems.get("kick", stems.get("drums", None))
        if trigger is not None:
            # Get envelope with fast attack
            env = _envelope_detector(to_mono(trigger), sr, attack_ms=0.5, release_ms=50)
            
            # Shift envelope earlier (predictive)
            lookahead_samples = int(0.005 * sr)  # 5ms lookahead
            env_predictive = np.pad(env, (lookahead_samples, 0), mode='edge')[:-lookahead_samples]
            
            # Duck other elements predictively
            for stem_name in ["bass", "music", "keys", "guitar"]:
                if stem_name in processed:
                    duck_amount = 0.3 if stem_name == "bass" else 0.15
                    gain = 1.0 - (env_predictive * duck_amount)
                    
                    if processed[stem_name].ndim == 1:
                        processed[stem_name] *= gain
                    else:
                        processed[stem_name] *= gain[:, np.newaxis]
    
    return processed


def swarm_behavior(stems: Dict[str, np.ndarray], sr: int) -> Dict[str, np.ndarray]:
    """Stems behave like a swarm, following and avoiding each other"""
    processed = {}
    
    # Calculate center of mass (audio centroid)
    centroids = {}
    for name, audio in stems.items():
        mono = to_mono(audio)
        # Simple spectral centroid
        fft = np.fft.rfft(mono[:8192])
        freqs = np.fft.rfftfreq(8192, 1/sr)
        centroid = np.sum(freqs * np.abs(fft)) / np.sum(np.abs(fft))
        centroids[name] = centroid
    
    mean_centroid = np.mean(list(centroids.values()))
    
    for stem_name, audio in stems.items():
        # Move towards or away from swarm center
        distance = centroids[stem_name] - mean_centroid
        
        if abs(distance) > 1000:  # Far from center
            # Move towards center (compression)
            from dsp_premitives import peaking_eq
            processed[stem_name] = peaking_eq(audio, sr, mean_centroid, 2.0, 1.0)
        else:  # Close to center
            # Move away (expansion)
            target_freq = mean_centroid + (distance * 2)
            from dsp_premitives import peaking_eq
            processed[stem_name] = peaking_eq(audio, sr, target_freq, 1.5, 0.8)
    
    return processed


def quantum_probability_processing(stems: Dict[str, np.ndarray], sr: int) -> Dict[str, np.ndarray]:
    """Probabilistic processing inspired by quantum superposition"""
    processed = {}
    
    for stem_name, audio in stems.items():
        # Create multiple "quantum states" of the audio
        states = []
        
        # State 1: Original
        states.append(audio)
        
        # State 2: Phase-shifted
        hilbert = signal.hilbert(to_mono(audio))
        phase_shift = np.imag(hilbert)
        if audio.ndim == 2:
            phase_shift = np.column_stack([phase_shift, phase_shift])
        states.append(phase_shift * 0.5)
        
        # State 3: Time-stretched micro-variations
        stretch_factor = 1.001  # Very subtle
        indices = np.arange(len(audio)) * stretch_factor
        indices = np.clip(indices, 0, len(audio) - 1).astype(int)
        stretched = audio[indices] if audio.ndim == 1 else audio[indices, :]
        states.append(stretched)
        
        # Collapse wavefunction (combine states with random weights)
        np.random.seed(hash(stem_name) % 2**32)  # Consistent randomness per stem
        weights = np.random.dirichlet([1, 0.5, 0.3])  # Random but normalized weights
        
        combined = sum(state * weight for state, weight in zip(states, weights))
        processed[stem_name] = combined.astype(np.float32)
    
    return processed


# ============================================
# TEMPO-SYNCED MODULATION
# ============================================

def tempo_synced_effects(stems: Dict[str, np.ndarray], sr: int, variant: str = "rhythmic", bpm: float = 120) -> Dict[str, np.ndarray]:
    """
    Apply tempo-synchronized modulation effects.
    
    Variants:
    - rhythmic: Beat-synced filters and gates
    - polyrhythmic: Complex rhythmic patterns
    - breakbeat: Glitchy, chopped effects
    - trance_gate: Classic trance gating
    - dubstep_wobble: Bass wobble effects
    """
    processed = {}
    
    # Calculate beat grid
    beat_duration = 60.0 / bpm  # seconds per beat
    samples_per_beat = int(beat_duration * sr)
    
    for stem_name, audio in stems.items():
        if variant == "rhythmic":
            processed[stem_name] = apply_rhythmic_filter(audio, sr, bpm, stem_name)
        elif variant == "polyrhythmic":
            processed[stem_name] = apply_polyrhythm(audio, sr, bpm, stem_name)
        elif variant == "breakbeat":
            processed[stem_name] = apply_breakbeat_chops(audio, sr, bpm, stem_name)
        elif variant == "trance_gate":
            processed[stem_name] = apply_trance_gate(audio, sr, bpm, stem_name)
        elif variant == "dubstep_wobble":
            processed[stem_name] = apply_wobble_bass(audio, sr, bpm, stem_name)
        else:
            processed[stem_name] = audio
    
    return processed


def apply_rhythmic_filter(audio: np.ndarray, sr: int, bpm: float, stem_type: str) -> np.ndarray:
    """Beat-synced filter sweeps"""
    from dsp_premitives import peaking_eq
    
    stereo = ensure_stereo(audio)
    
    # Different patterns for different stems
    if stem_type in ["bass", "keys"]:
        # 4-bar filter sweep
        sweep_duration = (60.0 / bpm) * 16  # 16 beats = 4 bars
        t = np.arange(len(stereo)) / sr
        
        # Sine wave LFO for filter frequency
        lfo = np.sin(2 * np.pi * t / sweep_duration)
        
        # Apply time-varying filter
        chunk_size = 512
        filtered = np.zeros_like(stereo)
        
        for i in range(0, len(stereo) - chunk_size, chunk_size):
            chunk = stereo[i:i+chunk_size]
            # Map LFO to frequency (200Hz to 2000Hz)
            freq = 1100 + lfo[i] * 900
            filtered[i:i+chunk_size] = peaking_eq(chunk, sr, freq, 3.0, 2.0)
        
        return filtered.astype(np.float32)
    
    return stereo


def apply_polyrhythm(audio: np.ndarray, sr: int, bpm: float, stem_type: str) -> np.ndarray:
    """Complex polyrhythmic patterns"""
    stereo = ensure_stereo(audio)
    
    if stem_type in ["hats", "percussion"]:
        # 3 against 4 polyrhythm
        t = np.arange(len(stereo)) / sr
        beat_hz = bpm / 60
        
        rhythm3 = (np.sin(2 * np.pi * beat_hz * 0.75 * t) > 0).astype(float)
        rhythm4 = (np.sin(2 * np.pi * beat_hz * t) > 0).astype(float)
        
        combined_rhythm = rhythm3 * 0.6 + rhythm4 * 0.4
        stereo *= combined_rhythm[:, np.newaxis]
    
    return stereo.astype(np.float32)


def apply_breakbeat_chops(audio: np.ndarray, sr: int, bpm: float, stem_type: str) -> np.ndarray:
    """Glitchy chopped effects"""
    stereo = ensure_stereo(audio)
    
    if stem_type in ["drums", "kick", "snare"]:
        samples_per_16th = int((60.0 / bpm / 4) * sr)
        
        # Random pattern of chops
        np.random.seed(42)  # Consistent randomness
        for i in range(0, len(stereo) - samples_per_16th * 4, samples_per_16th * 4):
            if np.random.random() > 0.7:  # 30% chance of chop
                # Repeat a 16th note
                segment = stereo[i:i+samples_per_16th]
                for j in range(1, 4):
                    stereo[i+samples_per_16th*j:i+samples_per_16th*(j+1)] = segment
    
    return stereo.astype(np.float32)


def apply_trance_gate(audio: np.ndarray, sr: int, bpm: float, stem_type: str) -> np.ndarray:
    """Classic trance gate effect"""
    stereo = ensure_stereo(audio)
    
    if stem_type in ["pads", "strings", "keys", "music"]:
        # 16th note gate pattern
        gate_rate = (bpm / 60) * 4  # 16th notes
        t = np.arange(len(stereo)) / sr
        
        # Square wave gate
        gate = (np.sin(2 * np.pi * gate_rate * t) > 0).astype(float)
        
        # Smooth the edges
        gate = signal.savgol_filter(gate, 101, 3)
        
        stereo *= gate[:, np.newaxis]
    
    return stereo.astype(np.float32)


def apply_wobble_bass(audio: np.ndarray, sr: int, bpm: float, stem_type: str) -> np.ndarray:
    """Dubstep-style bass wobble"""
    from dsp_premitives import lowpass_filter
    
    stereo = ensure_stereo(audio)
    
    if stem_type == "bass":
        # Wobble rate (typically 1/8 or 1/16 notes)
        wobble_rate = (bpm / 60) / 2  # 8th notes
        t = np.arange(len(stereo)) / sr
        
        # LFO for filter cutoff
        lfo = (np.sin(2 * np.pi * wobble_rate * t) + 1) * 0.5
        
        # Apply time-varying lowpass
        chunk_size = 256
        wobbled = np.zeros_like(stereo)
        
        for i in range(0, len(stereo) - chunk_size, chunk_size):
            chunk = stereo[i:i+chunk_size]
            # Map LFO to cutoff frequency
            cutoff = 100 + lfo[i] * 400  # 100Hz to 500Hz
            wobbled[i:i+chunk_size] = lowpass_filter(chunk, sr, cutoff, order=4)
        
        return wobbled.astype(np.float32)
    
    return stereo


# ============================================
# SPECTRAL MORPHING
# ============================================

def spectral_morphing(stems: Dict[str, np.ndarray], sr: int, variant: str = "hybrid") -> Dict[str, np.ndarray]:
    """
    Morph spectral characteristics between stems.
    
    Variants:
    - hybrid: Create hybrid instruments
    - vocoder: Use one stem to modulate another
    - spectral_swap: Swap frequency content
    - harmonic_fusion: Merge harmonic structures
    - formant_transfer: Transfer formants between stems
    """
    processed = {}
    
    if variant == "hybrid":
        return create_hybrid_instruments(stems, sr)
    elif variant == "vocoder":
        return vocoder_processing(stems, sr)
    elif variant == "spectral_swap":
        return swap_spectral_content(stems, sr)
    elif variant == "harmonic_fusion":
        return fuse_harmonics(stems, sr)
    elif variant == "formant_transfer":
        return transfer_formants(stems, sr)
    else:
        return stems


def create_hybrid_instruments(stems: Dict[str, np.ndarray], sr: int) -> Dict[str, np.ndarray]:
    """Blend spectral characteristics of different stems"""
    processed = stems.copy()
    
    # Create drum-bass hybrid
    if "kick" in stems and "bass" in stems:
        kick_spectrum = np.fft.rfft(to_mono(stems["kick"])[:8192])
        bass_spectrum = np.fft.rfft(to_mono(stems["bass"])[:8192])
        
        # Take low frequencies from kick, high from bass
        hybrid_spectrum = np.zeros_like(kick_spectrum)
        split_point = len(hybrid_spectrum) // 4
        hybrid_spectrum[:split_point] = kick_spectrum[:split_point]
        hybrid_spectrum[split_point:] = bass_spectrum[split_point:] * 0.5
        
        # Convert back to time domain
        hybrid = np.fft.irfft(hybrid_spectrum)
        if stems["kick"].ndim == 2:
            hybrid = np.column_stack([hybrid, hybrid])
        
        processed["kick_bass_hybrid"] = hybrid.astype(np.float32)
    
    # Create vocal-synth hybrid
    if "vocals" in stems and "keys" in stems:
        vocal_spectrum = np.fft.rfft(to_mono(stems["vocals"])[:8192])
        keys_spectrum = np.fft.rfft(to_mono(stems["keys"])[:8192])
        
        # Use vocal magnitude with keys phase
        hybrid_spectrum = np.abs(vocal_spectrum) * np.exp(1j * np.angle(keys_spectrum))
        
        hybrid = np.fft.irfft(hybrid_spectrum)
        if stems["vocals"].ndim == 2:
            hybrid = np.column_stack([hybrid, hybrid])
        
        processed["vocal_synth"] = hybrid.astype(np.float32)
    
    return processed


def vocoder_processing(stems: Dict[str, np.ndarray], sr: int) -> Dict[str, np.ndarray]:
    """Use one stem as modulator for another"""
    processed = stems.copy()
    
    # Vocals modulate synths
    if "vocals" in stems and "keys" in stems:
        modulator = to_mono(stems["vocals"])
        carrier = stems["keys"]
        
        # Simple vocoder using envelope following
        from processors import _envelope_detector
        envelope = _envelope_detector(modulator, sr, attack_ms=5, release_ms=20)
        
        # Normalize and apply
        envelope = envelope / (np.max(envelope) + 1e-6)
        
        if carrier.ndim == 1:
            vocoded = carrier * envelope
        else:
            vocoded = carrier * envelope[:, np.newaxis]
        
        processed["vocoded_keys"] = vocoded.astype(np.float32)
    
    return processed


def swap_spectral_content(stems: Dict[str, np.ndarray], sr: int) -> Dict[str, np.ndarray]:
    """Swap frequency content between stems"""
    processed = stems.copy()
    
    # Swap high frequencies between hats and vocals
    if "hats" in stems and "vocals" in stems:
        from dsp_premitives import lowpass_filter, highpass_filter
        
        # Split at 4kHz
        hats_low = lowpass_filter(stems["hats"], sr, 4000, order=4)
        hats_high = highpass_filter(stems["hats"], sr, 4000, order=4)
        
        vocals_low = lowpass_filter(stems["vocals"], sr, 4000, order=4)
        vocals_high = highpass_filter(stems["vocals"], sr, 4000, order=4)
        
        # Swap highs
        processed["hats"] = (hats_low + vocals_high * 0.5).astype(np.float32)
        processed["vocals"] = (vocals_low + hats_high * 0.3).astype(np.float32)
    
    return processed


def fuse_harmonics(stems: Dict[str, np.ndarray], sr: int) -> Dict[str, np.ndarray]:
    """Merge harmonic structures of different stems"""
    processed = stems.copy()
    
    # Fuse bass and kick harmonics
    if "kick" in stems and "bass" in stems:
        # Extract fundamental frequencies
        kick_fft = np.fft.rfft(to_mono(stems["kick"])[:4096])
        bass_fft = np.fft.rfft(to_mono(stems["bass"])[:4096])
        
        # Find peaks (harmonics)
        kick_peaks = signal.find_peaks(np.abs(kick_fft), height=np.max(np.abs(kick_fft)) * 0.1)[0]
        bass_peaks = signal.find_peaks(np.abs(bass_fft), height=np.max(np.abs(bass_fft)) * 0.1)[0]
        
        # Create fused spectrum
        fused_spectrum = np.zeros_like(kick_fft)
        fused_spectrum[kick_peaks] = kick_fft[kick_peaks] * 0.6
        fused_spectrum[bass_peaks] += bass_fft[bass_peaks] * 0.4
        
        fused = np.fft.irfft(fused_spectrum)
        if stems["kick"].ndim == 2:
            fused = np.column_stack([fused, fused])
        
        processed["kick_bass_fused"] = fused.astype(np.float32)
    
    return processed


def transfer_formants(stems: Dict[str, np.ndarray], sr: int) -> Dict[str, np.ndarray]:
    """Transfer formant characteristics between stems"""
    processed = stems.copy()
    
    if "vocals" in stems:
        # Extract vocal formants
        vocal_mono = to_mono(stems["vocals"])
        
        # Simple formant detection using LPC
        # This is simplified - real formant extraction would use proper LPC analysis
        vocal_fft = np.fft.rfft(vocal_mono[:4096])
        formant_envelope = signal.savgol_filter(np.abs(vocal_fft), 51, 3)
        
        # Apply to other stems
        for stem_name in ["guitar", "keys", "strings"]:
            if stem_name in stems:
                stem_mono = to_mono(stems[stem_name])
                stem_fft = np.fft.rfft(stem_mono[:4096])
                
                # Apply formant envelope
                modified_fft = stem_fft * (formant_envelope / (np.abs(stem_fft) + 1e-6)) ** 0.3
                
                modified = np.fft.irfft(modified_fft)
                if stems[stem_name].ndim == 2:
                    modified = np.column_stack([modified, modified])
                
                processed[f"{stem_name}_vocalized"] = modified.astype(np.float32)
    
    return processed


# ============================================
# EXTREME VARIANT DEFINITIONS
# ============================================

@dataclass
class ExtremeVariant:
    """Container for extreme processing parameters"""
    name: str
    spatial_3d: str = "immersive"
    psychoacoustic: str = "clarity"
    ai_interaction: str = "adaptive"
    tempo_sync: str = "rhythmic"
    spectral: str = "hybrid"
    description: str = ""


EXTREME_VARIANTS = [
    ExtremeVariant(
        "3D_Immersive",
        spatial_3d="immersive", psychoacoustic="presence",
        ai_interaction="adaptive", tempo_sync=None, spectral=None,
        description="Full 3D soundfield with adaptive AI interaction"
    ),
    ExtremeVariant(
        "Cinematic_AI",
        spatial_3d="cinema", psychoacoustic="energy",
        ai_interaction="predictive", tempo_sync=None, spectral="hybrid",
        description="Movie theater spatialization with predictive dynamics"
    ),
    ExtremeVariant(
        "Binaural_Psycho",
        spatial_3d="binaural", psychoacoustic="hypnotic",
        ai_interaction="neural", tempo_sync="polyrhythmic", spectral=None,
        description="Headphone-optimized with hypnotic psychoacoustic effects"
    ),
    ExtremeVariant(
        "VR_Experience",
        spatial_3d="vr_space", psychoacoustic="presence",
        ai_interaction="swarm", tempo_sync=None, spectral="vocoder",
        description="Virtual reality audio with swarm behavior"
    ),
    ExtremeVariant(
        "Quantum_Club",
        spatial_3d="dome", psychoacoustic="energy",
        ai_interaction="quantum", tempo_sync="dubstep_wobble", spectral="harmonic_fusion",
        description="Quantum probability processing with club-ready wobbles"
    ),
    ExtremeVariant(
        "Neural_Trance",
        spatial_3d="immersive", psychoacoustic="hypnotic",
        ai_interaction="neural", tempo_sync="trance_gate", spectral="formant_transfer",
        description="Neural network patterns with trance gating"
    ),
    ExtremeVariant(
        "Breakbeat_Morph",
        spatial_3d="vr_space", psychoacoustic="energy",
        ai_interaction="predictive", tempo_sync="breakbeat", spectral="spectral_swap",
        description="Glitchy breakbeats with spectral morphing"
    ),
    ExtremeVariant(
        "Subliminal_Adaptive",
        spatial_3d="binaural", psychoacoustic="subliminal",
        ai_interaction="adaptive", tempo_sync="rhythmic", spectral="hybrid",
        description="Subtle enhancements below conscious perception"
    ),
]


def apply_extreme_processing(stems: Dict[str, np.ndarray], sr: int, 
                            variant: ExtremeVariant, bpm: float = 120) -> Dict[str, np.ndarray]:
    """Apply extreme processing chain with safety checks"""
    
    print(f"    🔮 Applying {variant.name}: {variant.description}")
    
    processed = stems
    
    # Apply each processing stage with safety checks
    try:
        if variant.spatial_3d:
            processed = create_3d_soundfield(processed, sr, variant.spatial_3d)
            processed = _sanitize_processed_stems(processed, "3D Spatial")
            print(f"      ✓ 3D Spatial: {variant.spatial_3d}")
        
        if variant.psychoacoustic:
            processed = psychoacoustic_enhancement(processed, sr, variant.psychoacoustic)
            processed = _sanitize_processed_stems(processed, "Psychoacoustic")
            print(f"      ✓ Psychoacoustic: {variant.psychoacoustic}")
        
        if variant.ai_interaction:
            processed = ai_stem_interaction(processed, sr, variant.ai_interaction)
            processed = _sanitize_processed_stems(processed, "AI Interaction")
            print(f"      ✓ AI Interaction: {variant.ai_interaction}")
        
        if variant.tempo_sync:
            processed = tempo_synced_effects(processed, sr, variant.tempo_sync, bpm)
            processed = _sanitize_processed_stems(processed, "Tempo Sync")
            print(f"      ✓ Tempo Sync: {variant.tempo_sync}")
        
        if variant.spectral:
            processed = spectral_morphing(processed, sr, variant.spectral)
            processed = _sanitize_processed_stems(processed, "Spectral")
            print(f"      ✓ Spectral: {variant.spectral}")
            
    except Exception as e:
        print(f"      ⚠️ Processing error: {e}")
        # Return safe processed stems
        processed = _sanitize_processed_stems(stems, "Fallback")
    
    return processed


def _sanitize_processed_stems(stems: Dict[str, np.ndarray], stage: str) -> Dict[str, np.ndarray]:
    """Sanitize processed stems to prevent file writing errors"""
    sanitized = {}
    
    for stem_name, audio in stems.items():
        # Convert to numpy array if needed
        audio = np.asarray(audio, dtype=np.float32)
        
        # Replace NaN and Inf values
        audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Clip to safe range
        audio = np.clip(audio, -4.0, 4.0)  # Allow some headroom but prevent extreme values
        
        # Ensure proper shape
        if audio.ndim == 1:
            # Keep mono
            sanitized[stem_name] = audio.astype(np.float32)
        elif audio.ndim == 2:
            # Keep stereo
            sanitized[stem_name] = audio.astype(np.float32)
        else:
            # Convert weird shapes to mono
            sanitized[stem_name] = np.mean(audio, axis=tuple(range(1, audio.ndim))).astype(np.float32)
        
        # Check for remaining issues
        if not np.isfinite(sanitized[stem_name]).all():
            print(f"      ⚠️ {stage}: Still has invalid values in {stem_name}, zeroing out")
            sanitized[stem_name] = np.zeros_like(sanitized[stem_name])
    
    return sanitized


print("🔮 Extreme Stem Processing loaded!")
print(f"   • {len(EXTREME_VARIANTS)} extreme variants")
print("   • 3D Spatial Audio (5 modes)")
print("   • Psychoacoustic Enhancement (5 modes)")
print("   • AI-Inspired Interactions (5 modes)")
print("   • Tempo-Synced Effects (5 modes)")
print("   • Spectral Morphing (5 modes)")
print("   • Quantum, Neural, Swarm behaviors!")
print("   • Binaural, VR, and Cinematic spatial modes!")
print("   🎧 These push stem separation to the absolute limit!")