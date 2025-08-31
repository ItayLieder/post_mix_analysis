#!/usr/bin/env python3
"""
Advanced DSP Functions for Professional Mixing
Implements the techniques used by top mixing engineers
"""

import numpy as np
from scipy import signal, interpolate
from typing import Tuple, Optional, Dict, List
from dsp_premitives import (
    compressor, highpass_filter, lowpass_filter, 
    peaking_eq, shelf_filter, _db_to_lin, _lin_to_db
)

# ============================================================================
# SIDECHAIN COMPRESSION
# ============================================================================

def sidechain_compressor(
    signal_audio: np.ndarray,
    sidechain_audio: np.ndarray, 
    sr: int,
    threshold_db: float = -20,
    ratio: float = 4,
    attack_ms: float = 1,
    release_ms: float = 50,
    knee_db: float = 2,
    amount: float = 1.0
) -> np.ndarray:
    """
    Sidechain compression - duck signal based on sidechain input
    Used for kick/bass interaction, vocal/music ducking, etc.
    """
    # Ensure same length
    min_len = min(len(signal_audio), len(sidechain_audio))
    signal_audio = signal_audio[:min_len]
    sidechain_audio = sidechain_audio[:min_len]
    
    # Get envelope from sidechain
    sidechain_mono = sidechain_audio if sidechain_audio.ndim == 1 else np.mean(sidechain_audio, axis=1)
    
    # Envelope follower
    attack_samples = int(attack_ms * sr / 1000)
    release_samples = int(release_ms * sr / 1000)
    
    envelope = np.abs(sidechain_mono)
    
    # Smooth envelope
    if attack_samples > 1:
        envelope = signal.filtfilt(
            np.ones(attack_samples) / attack_samples,
            1, envelope
        )
    
    # Convert to dB
    envelope_db = 20 * np.log10(envelope + 1e-10)
    
    # Calculate gain reduction
    gain_reduction_db = np.zeros_like(envelope_db)
    
    # Above threshold
    above_thresh = envelope_db > threshold_db
    
    if knee_db > 0:
        # Soft knee
        knee_start = threshold_db - knee_db
        knee_end = threshold_db + knee_db
        
        # Linear interpolation in knee
        knee_mask = (envelope_db > knee_start) & (envelope_db <= knee_end)
        knee_factor = (envelope_db[knee_mask] - knee_start) / (2 * knee_db)
        
        gain_reduction_db[knee_mask] = (
            knee_factor * (envelope_db[knee_mask] - threshold_db) * (1 - 1/ratio)
        )
    
    # Hard compression above threshold
    gain_reduction_db[above_thresh] = (
        (envelope_db[above_thresh] - threshold_db) * (1 - 1/ratio)
    )
    
    # Apply attack/release
    gain_reduction_linear = _db_to_lin(-gain_reduction_db)
    
    # Expand to match signal shape
    if signal_audio.ndim == 2:
        gain_reduction_linear = np.stack([gain_reduction_linear, gain_reduction_linear], axis=-1)
    
    # Apply compression with amount control
    compressed = signal_audio * (1 - amount + amount * gain_reduction_linear)
    
    return compressed

# ============================================================================
# PARALLEL COMPRESSION
# ============================================================================

def parallel_compression(
    audio: np.ndarray,
    sr: int,
    threshold_db: float = -30,
    ratio: float = 10,
    attack_ms: float = 0.5,
    release_ms: float = 100,
    mix: float = 0.5,
    eq_before: Optional[Dict] = None,
    eq_after: Optional[Dict] = None
) -> np.ndarray:
    """
    New York style parallel compression
    Adds punch and energy without destroying dynamics
    """
    # Apply pre-EQ if specified
    processed = audio.copy()
    if eq_before:
        for freq, params in eq_before.items():
            processed = peaking_eq(processed, sr, freq, params['gain'], params.get('q', 0.7))
    
    # Heavy compression
    compressed = compressor(
        processed, sr,
        threshold_db=threshold_db,
        ratio=ratio,
        attack_ms=attack_ms,
        release_ms=release_ms,
        knee_db=0.5  # Hard knee for aggressive compression
    )
    
    # Apply post-EQ if specified
    if eq_after:
        for freq, params in eq_after.items():
            compressed = peaking_eq(compressed, sr, freq, params['gain'], params.get('q', 0.7))
    
    # Mix with original
    return audio * (1 - mix) + compressed * mix

# ============================================================================
# MULTIBAND COMPRESSION
# ============================================================================

def multiband_compressor(
    audio: np.ndarray,
    sr: int,
    bands: List[Dict],
    crossover_freqs: List[float]
) -> np.ndarray:
    """
    Multiband compression for frequency-specific dynamics control
    
    bands: List of dicts with compression settings per band
    crossover_freqs: Frequency boundaries between bands
    """
    if len(bands) != len(crossover_freqs) + 1:
        raise ValueError("Number of bands must be len(crossover_freqs) + 1")
    
    result = np.zeros_like(audio)
    
    for i, band_settings in enumerate(bands):
        # Extract band
        if i == 0:
            # Low band
            band_audio = lowpass_filter(audio, sr, crossover_freqs[0], order=4)
        elif i == len(bands) - 1:
            # High band
            band_audio = highpass_filter(audio, sr, crossover_freqs[-1], order=4)
        else:
            # Mid band
            band_audio = highpass_filter(audio, sr, crossover_freqs[i-1], order=4)
            band_audio = lowpass_filter(band_audio, sr, crossover_freqs[i], order=4)
        
        # Compress band
        if band_settings.get('enabled', True):
            band_audio = compressor(
                band_audio, sr,
                threshold_db=band_settings.get('threshold', -20),
                ratio=band_settings.get('ratio', 3),
                attack_ms=band_settings.get('attack', 10),
                release_ms=band_settings.get('release', 100),
                knee_db=band_settings.get('knee', 2)
            )
        
        # Apply band gain
        band_audio *= band_settings.get('gain', 1.0)
        
        # Sum bands
        result += band_audio
    
    return result

# ============================================================================
# SATURATION & HARMONICS
# ============================================================================

def tape_saturation(
    audio: np.ndarray,
    drive: float = 0.5,
    warmth: float = 0.5,
    bias: float = 0.0
) -> np.ndarray:
    """
    Tape saturation emulation
    Adds warmth, compression, and harmonic richness
    """
    # Input gain
    driven = audio * (1 + drive * 4)
    
    # Asymmetric saturation (tape characteristic)
    positive = driven > 0
    negative = driven <= 0
    
    # Different saturation curves for positive and negative
    saturated = np.zeros_like(driven)
    
    # Positive half - softer saturation
    saturated[positive] = np.tanh(driven[positive] * (1 - bias * 0.2))
    
    # Negative half - slightly harder saturation
    saturated[negative] = np.tanh(driven[negative] * (1 + bias * 0.2))
    
    # Add even harmonics for warmth
    if warmth > 0:
        # Generate 2nd harmonic
        second_harmonic = saturated ** 2 * np.sign(saturated) * warmth * 0.1
        saturated += second_harmonic
    
    # Compensate for level
    saturated *= 0.7 / (1 + drive * 0.5)
    
    return saturated

def tube_saturation(
    audio: np.ndarray,
    drive: float = 0.5,
    warmth: float = 0.5,
    presence: float = 0.5
) -> np.ndarray:
    """
    Tube saturation emulation
    Adds warmth and musical harmonics
    """
    # Input gain
    driven = audio * (1 + drive * 3)
    
    # Tube-like soft clipping
    saturated = np.sign(driven) * (1 - np.exp(-np.abs(driven * 2)))
    
    # Add odd harmonics for presence
    if presence > 0:
        third_harmonic = saturated ** 3 * presence * 0.05
        saturated += third_harmonic
    
    # Add even harmonics for warmth
    if warmth > 0:
        second_harmonic = saturated ** 2 * np.sign(saturated) * warmth * 0.08
        saturated += second_harmonic
    
    # Output compensation
    saturated *= 0.5 / (1 + drive * 0.3)
    
    return saturated

def analog_console_saturation(
    audio: np.ndarray,
    sr: int,
    drive: float = 0.3,
    character: str = 'clean'  # 'clean', 'warm', 'aggressive'
) -> np.ndarray:
    """
    Console channel saturation
    Emulates analog mixing console characteristics
    """
    # Console input transformer saturation
    saturated = audio * (1 + drive)
    
    # Character-specific processing
    if character == 'warm':
        # SSL-style: subtle compression and warmth
        saturated = np.tanh(saturated * 0.7) * 1.2
        # Subtle low-mid boost
        saturated = peaking_eq(saturated, sr, 200, 0.5, 0.7)
        
    elif character == 'aggressive':
        # API-style: punchy with harmonic edge
        saturated = np.sign(saturated) * np.abs(saturated) ** 0.7
        # Add presence
        saturated = peaking_eq(saturated, sr, 3000, 1.0, 0.8)
        
    else:  # clean
        # Neve-style: transparent with subtle coloration
        saturated = saturated / (1 + np.abs(saturated) * 0.2)
    
    # Console output transformer
    saturated = np.tanh(saturated * 0.9)
    
    # Compensate level
    saturated *= 0.85
    
    return saturated

# ============================================================================
# TRANSIENT DESIGN
# ============================================================================

def advanced_transient_shaper(
    audio: np.ndarray,
    sr: int,
    attack_boost: float = 0,  # -1 to +1
    sustain_boost: float = 0,  # -1 to +1
    attack_time_ms: float = 5,
    release_time_ms: float = 100
) -> np.ndarray:
    """
    Advanced transient shaping for punch and clarity
    """
    # Detect envelope
    envelope = np.abs(audio if audio.ndim == 1 else np.mean(audio, axis=1))
    
    # Fast envelope for transients
    attack_samples = max(1, int(attack_time_ms * sr / 1000))
    from scipy.ndimage import maximum_filter1d
    fast_envelope = maximum_filter1d(envelope, size=attack_samples)
    
    # Slow envelope for sustain
    release_samples = max(1, int(release_time_ms * sr / 1000))
    slow_envelope = signal.filtfilt(
        np.ones(release_samples) / release_samples,
        1, envelope
    )
    
    # Transient is the difference
    transients = fast_envelope - slow_envelope
    transients = np.maximum(0, transients)
    
    # Create gain curves
    transient_gain = 1 + transients * attack_boost * 2
    sustain_gain = 1 + slow_envelope * sustain_boost
    
    # Combine gains
    total_gain = transient_gain * sustain_gain
    
    # Smooth to avoid clicks
    total_gain = signal.filtfilt(
        np.ones(100) / 100, 1, total_gain
    )
    
    # Apply to signal
    if audio.ndim == 2:
        total_gain = np.stack([total_gain, total_gain], axis=-1)
    
    return audio * total_gain

# ============================================================================
# STEREO ENHANCEMENT
# ============================================================================

def haas_effect(
    audio: np.ndarray,
    sr: int,
    delay_ms: float = 20,
    amount: float = 0.5
) -> np.ndarray:
    """
    Haas effect for stereo widening
    Creates width through micro-delays
    """
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=-1)
    
    delay_samples = int(delay_ms * sr / 1000)
    
    # Delay right channel
    right_delayed = np.pad(audio[:, 1], (delay_samples, 0), mode='constant')[:-delay_samples]
    
    # Mix delayed with original
    result = audio.copy()
    result[:, 1] = audio[:, 1] * (1 - amount) + right_delayed * amount
    
    return result

def stereo_spreader(
    audio: np.ndarray,
    sr: int,
    width: float = 1.5,
    bass_mono_freq: float = 120
) -> np.ndarray:
    """
    Frequency-dependent stereo spreading
    Keeps bass mono while widening highs
    """
    if audio.ndim == 1:
        return audio
    
    # Split into mono and stereo
    mid = (audio[:, 0] + audio[:, 1]) / 2
    side = (audio[:, 0] - audio[:, 1]) / 2
    
    # Keep bass mono
    side_highpassed = highpass_filter(side, sr, bass_mono_freq, order=2)
    
    # Enhance stereo
    side_enhanced = side_highpassed * width
    
    # Reconstruct
    left = mid + side_enhanced
    right = mid - side_enhanced
    
    return np.stack([left, right], axis=-1)

# ============================================================================
# INTELLIGENT MIXING
# ============================================================================

def auto_gain_staging(
    channels: Dict[str, np.ndarray],
    target_headroom_db: float = -6
) -> Dict[str, float]:
    """
    Automatically set optimal gain staging
    Returns gain multipliers for each channel
    """
    gains = {}
    
    # First pass: analyze levels
    peak_levels = {}
    for name, audio in channels.items():
        peak = np.max(np.abs(audio))
        peak_levels[name] = peak
    
    # Find the loudest element
    max_peak = max(peak_levels.values())
    
    # Calculate gains to optimize headroom
    target_peak = _db_to_lin(target_headroom_db)
    
    for name, peak in peak_levels.items():
        if peak > 0:
            # Scale so loudest element hits target
            gains[name] = (target_peak / max_peak)
        else:
            gains[name] = 1.0
    
    return gains

def frequency_slot_eq(
    channels: Dict[str, np.ndarray],
    sr: int
) -> Dict[str, List[Dict]]:
    """
    Automatically carve EQ slots for each instrument
    Reduces masking and improves clarity
    """
    eq_settings = {}
    
    # Define frequency slots for common instruments
    slots = {
        'kick': {'boost': [60, 3000], 'cut': []},
        'bass': {'boost': [100], 'cut': [60]},  # Cut where kick lives
        'snare': {'boost': [200, 5000], 'cut': []},
        'vocal': {'boost': [2500, 10000], 'cut': [300]},
        'guitar': {'boost': [2000], 'cut': [200, 2500]},  # Cut where vocals live
        'keys': {'boost': [1000, 8000], 'cut': [500]},
    }
    
    for name, audio in channels.items():
        eq_bands = []
        
        # Match instrument type
        name_lower = name.lower()
        for inst_type, freqs in slots.items():
            if inst_type in name_lower:
                # Add boost frequencies
                for freq in freqs['boost']:
                    eq_bands.append({
                        'freq': freq,
                        'gain': 2,
                        'q': 0.7
                    })
                
                # Add cut frequencies
                for freq in freqs['cut']:
                    eq_bands.append({
                        'freq': freq,
                        'gain': -2,
                        'q': 0.8
                    })
                break
        
        eq_settings[name] = eq_bands
    
    return eq_settings