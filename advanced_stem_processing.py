#!/usr/bin/env python3
"""
Advanced Stem Processing Module
Professional studio-quality processing techniques for individual stems
Emulates classic hardware and professional mixing approaches
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
from scipy import signal
import soundfile as sf


@dataclass
class AdvancedVariant:
    """Defines an advanced processing variant"""
    name: str
    description: str
    drum_settings: Dict
    bass_settings: Dict
    vocal_settings: Dict
    music_settings: Dict


# Define all advanced processing variants
ADVANCED_VARIANTS = [
    AdvancedVariant(
        name="SSL_Console",
        description="SSL 4000-style console processing with bus compression and EQ",
        drum_settings={
            "comp_ratio": 4,
            "comp_attack": 3,
            "comp_release": 100,
            "comp_threshold": -15,
            "eq_high_shelf": {"freq": 10000, "gain": 2.0, "q": 0.7},
            "eq_low_shelf": {"freq": 60, "gain": 1.5, "q": 0.7},
            "bus_comp_ratio": 2,
            "saturation_type": "console",
            "saturation_amount": 0.3
        },
        bass_settings={
            "comp_ratio": 3,
            "comp_attack": 10,
            "comp_release": 200,
            "eq_bands": [
                {"freq": 80, "gain": 2.0, "q": 0.8},
                {"freq": 700, "gain": -1.0, "q": 0.6},
                {"freq": 2500, "gain": 1.5, "q": 0.7}
            ],
            "saturation_amount": 0.2
        },
        vocal_settings={
            "comp_ratio": 3,
            "comp_attack": 5,
            "comp_release": 150,
            "eq_bands": [
                {"freq": 200, "gain": -1.0, "q": 0.7},
                {"freq": 3000, "gain": 2.5, "q": 0.8},
                {"freq": 12000, "gain": 1.5, "q": 0.5}
            ],
            "de_esser": True,
            "saturation_amount": 0.1
        },
        music_settings={
            "comp_ratio": 2,
            "comp_attack": 10,
            "eq_high_shelf": {"freq": 8000, "gain": 1.0, "q": 0.6},
            "stereo_width": 1.2,
            "saturation_amount": 0.15
        }
    ),
    
    AdvancedVariant(
        name="Neve_Warmth",
        description="Neve 1073-style warm transformer saturation and musical EQ",
        drum_settings={
            "comp_ratio": 3,
            "comp_attack": 8,
            "comp_release": 150,
            "eq_bands": [
                {"freq": 60, "gain": 3.0, "q": 0.6},
                {"freq": 220, "gain": 1.5, "q": 0.7},
                {"freq": 7200, "gain": 2.0, "q": 0.5}
            ],
            "saturation_type": "transformer",
            "saturation_amount": 0.5,
            "harmonic_enhancement": 0.3
        },
        bass_settings={
            "comp_ratio": 2.5,
            "comp_attack": 15,
            "eq_bands": [
                {"freq": 110, "gain": 2.5, "q": 0.7},
                {"freq": 440, "gain": 1.0, "q": 0.6}
            ],
            "saturation_type": "transformer",
            "saturation_amount": 0.6
        },
        vocal_settings={
            "comp_ratio": 2.5,
            "comp_attack": 12,
            "eq_bands": [
                {"freq": 270, "gain": -0.5, "q": 0.8},
                {"freq": 3300, "gain": 3.0, "q": 0.7},
                {"freq": 12000, "gain": 2.0, "q": 0.4}
            ],
            "saturation_type": "tube",
            "saturation_amount": 0.4
        },
        music_settings={
            "comp_ratio": 2,
            "eq_bands": [
                {"freq": 440, "gain": 0.8, "q": 0.6},
                {"freq": 7200, "gain": 1.5, "q": 0.5}
            ],
            "saturation_type": "transformer",
            "saturation_amount": 0.3
        }
    ),
    
    AdvancedVariant(
        name="API_Punch",
        description="API 2500-style aggressive punch with forward midrange",
        drum_settings={
            "comp_ratio": 6,
            "comp_attack": 0.5,
            "comp_release": 50,
            "comp_knee": "hard",
            "comp_thrust": True,  # High-pass sidechain
            "eq_bands": [
                {"freq": 50, "gain": 2.5, "q": 0.9},
                {"freq": 500, "gain": 2.0, "q": 0.7},
                {"freq": 5000, "gain": 3.0, "q": 0.6}
            ],
            "transient_attack": 0.7,
            "saturation_amount": 0.4
        },
        bass_settings={
            "comp_ratio": 4,
            "comp_attack": 3,
            "comp_release": 100,
            "eq_bands": [
                {"freq": 80, "gain": 3.0, "q": 0.8},
                {"freq": 800, "gain": 1.5, "q": 0.6}
            ],
            "saturation_amount": 0.3
        },
        vocal_settings={
            "comp_ratio": 4,
            "comp_attack": 2,
            "comp_release": 80,
            "eq_bands": [
                {"freq": 600, "gain": 1.5, "q": 0.7},
                {"freq": 2500, "gain": 3.0, "q": 0.8}
            ],
            "presence_boost": 0.4
        },
        music_settings={
            "comp_ratio": 3,
            "comp_attack": 5,
            "eq_bands": [
                {"freq": 800, "gain": 1.0, "q": 0.6}
            ],
            "transient_attack": 0.3
        }
    ),
    
    AdvancedVariant(
        name="Fairchild_Glue",
        description="Fairchild 670-style tube compression for smooth glue",
        drum_settings={
            "comp_ratio": 2,
            "comp_attack": 20,
            "comp_release": 300,
            "comp_knee": "soft",
            "tube_warmth": 0.6,
            "parallel_blend": 0.5,
            "saturation_type": "tube",
            "saturation_amount": 0.5
        },
        bass_settings={
            "comp_ratio": 1.8,
            "comp_attack": 25,
            "comp_release": 400,
            "tube_warmth": 0.7,
            "saturation_type": "tube",
            "saturation_amount": 0.4
        },
        vocal_settings={
            "comp_ratio": 2.5,
            "comp_attack": 15,
            "comp_release": 250,
            "tube_warmth": 0.5,
            "saturation_type": "tube",
            "saturation_amount": 0.3,
            "smoothing": 0.4
        },
        music_settings={
            "comp_ratio": 1.5,
            "comp_attack": 30,
            "comp_release": 500,
            "tube_warmth": 0.4,
            "glue_factor": 0.6
        }
    ),
    
    AdvancedVariant(
        name="LA2A_Vocals",
        description="LA-2A-style optical compression optimized for smooth vocals",
        drum_settings={
            "comp_ratio": 3,
            "comp_attack": 10,
            "comp_release": 200,
            "optical_smoothness": 0.3
        },
        bass_settings={
            "comp_ratio": 3,
            "comp_attack": 15,
            "comp_release": 300,
            "optical_smoothness": 0.4
        },
        vocal_settings={
            "comp_ratio": 4,
            "comp_attack": 10,
            "comp_release": 500,
            "comp_knee": "soft",
            "optical_smoothness": 0.7,
            "frequency_dependent": True,
            "presence_control": 0.3,
            "saturation_type": "tube",
            "saturation_amount": 0.2
        },
        music_settings={
            "comp_ratio": 2,
            "comp_attack": 20,
            "comp_release": 400,
            "optical_smoothness": 0.5
        }
    ),
    
    AdvancedVariant(
        name="1176_Drums",
        description="1176-style FET compression for explosive drums",
        drum_settings={
            "comp_ratio": 8,
            "comp_attack": 0.02,  # Ultra-fast
            "comp_release": 50,
            "comp_all_buttons": True,  # All-buttons mode
            "fet_character": 0.7,
            "transient_emphasis": 0.8,
            "saturation_amount": 0.4
        },
        bass_settings={
            "comp_ratio": 4,
            "comp_attack": 1,
            "comp_release": 100,
            "fet_character": 0.5
        },
        vocal_settings={
            "comp_ratio": 4,
            "comp_attack": 0.5,
            "comp_release": 150,
            "fet_character": 0.3
        },
        music_settings={
            "comp_ratio": 3,
            "comp_attack": 2,
            "comp_release": 200,
            "fet_character": 0.2
        }
    ),
    
    AdvancedVariant(
        name="Pultec_EQ",
        description="Pultec-style passive EQ with simultaneous boost/cut",
        drum_settings={
            "pultec_low_boost": {"freq": 60, "gain": 3.0},
            "pultec_low_cut": {"freq": 30, "gain": -1.0},
            "pultec_high_boost": {"freq": 10000, "gain": 2.5},
            "pultec_high_cut": {"freq": 20000, "gain": -0.5},
            "tube_saturation": 0.3,
            "comp_ratio": 2.5
        },
        bass_settings={
            "pultec_low_boost": {"freq": 100, "gain": 4.0},
            "pultec_low_cut": {"freq": 40, "gain": -1.5},
            "pultec_mid_dip": {"freq": 500, "gain": -1.0},
            "tube_saturation": 0.4
        },
        vocal_settings={
            "pultec_low_cut": {"freq": 80, "gain": -2.0},
            "pultec_high_boost": {"freq": 12000, "gain": 3.0},
            "pultec_presence": {"freq": 3000, "gain": 2.0},
            "tube_saturation": 0.2
        },
        music_settings={
            "pultec_low_boost": {"freq": 80, "gain": 1.5},
            "pultec_high_boost": {"freq": 8000, "gain": 2.0},
            "tube_saturation": 0.25
        }
    ),
    
    AdvancedVariant(
        name="Multiband_Dynamics",
        description="Sophisticated multiband compression per stem",
        drum_settings={
            "multiband_low": {"freq": 200, "ratio": 4, "attack": 5},
            "multiband_mid": {"freq": 2000, "ratio": 3, "attack": 2},
            "multiband_high": {"freq": 8000, "ratio": 2.5, "attack": 0.5},
            "crossover_slope": 24,
            "band_interaction": 0.3
        },
        bass_settings={
            "multiband_low": {"freq": 150, "ratio": 3, "attack": 10},
            "multiband_mid": {"freq": 800, "ratio": 2.5, "attack": 5},
            "multiband_high": {"freq": 3000, "ratio": 2, "attack": 3}
        },
        vocal_settings={
            "multiband_low": {"freq": 250, "ratio": 2.5, "attack": 8},
            "multiband_mid": {"freq": 2500, "ratio": 3, "attack": 3},
            "multiband_high": {"freq": 8000, "ratio": 3.5, "attack": 1},
            "de_esser_band": {"freq": 6000, "ratio": 6}
        },
        music_settings={
            "multiband_low": {"freq": 300, "ratio": 2, "attack": 10},
            "multiband_mid": {"freq": 3000, "ratio": 2.5, "attack": 5},
            "multiband_high": {"freq": 10000, "ratio": 2, "attack": 2}
        }
    ),
    
    AdvancedVariant(
        name="MS_Processing",
        description="Mid/Side processing for precise width control",
        drum_settings={
            "mid_comp_ratio": 3,
            "side_comp_ratio": 2,
            "mid_eq_boost": {"freq": 100, "gain": 2.0},
            "side_eq_boost": {"freq": 10000, "gain": 2.5},
            "width_control": 1.1,
            "mono_bass_freq": 120
        },
        bass_settings={
            "mid_comp_ratio": 3.5,
            "side_comp_ratio": 2,
            "center_focus": 0.8,
            "mono_below": 200
        },
        vocal_settings={
            "mid_comp_ratio": 3,
            "side_comp_ratio": 2.5,
            "center_presence": {"freq": 3000, "gain": 2.0},
            "side_air": {"freq": 12000, "gain": 1.5},
            "width_control": 0.9
        },
        music_settings={
            "mid_comp_ratio": 2,
            "side_comp_ratio": 2.5,
            "side_enhancement": 0.4,
            "width_control": 1.3
        }
    ),
    
    AdvancedVariant(
        name="Parallel_NY",
        description="New York parallel compression with precise blend",
        drum_settings={
            "dry_level": 0.6,
            "parallel_level": 0.4,
            "parallel_ratio": 20,
            "parallel_attack": 0.1,
            "parallel_release": 30,
            "parallel_saturation": 0.6,
            "parallel_eq": {"freq": 100, "gain": 3.0}
        },
        bass_settings={
            "dry_level": 0.7,
            "parallel_level": 0.3,
            "parallel_ratio": 10,
            "parallel_attack": 1,
            "parallel_release": 50
        },
        vocal_settings={
            "dry_level": 0.8,
            "parallel_level": 0.2,
            "parallel_ratio": 8,
            "parallel_attack": 2,
            "parallel_release": 100,
            "parallel_excitement": 0.3
        },
        music_settings={
            "dry_level": 0.75,
            "parallel_level": 0.25,
            "parallel_ratio": 6,
            "parallel_attack": 5,
            "parallel_release": 150
        }
    )
]


def apply_advanced_processing(audio: np.ndarray, 
                             stem_type: str,
                             variant: AdvancedVariant,
                             sample_rate: int = 44100) -> np.ndarray:
    """
    Apply advanced processing to a stem based on the selected variant
    
    Args:
        audio: Input audio array
        stem_type: Type of stem ('drums', 'bass', 'vocals', 'music')
        variant: The AdvancedVariant to apply
        sample_rate: Sample rate
        
    Returns:
        Processed audio array
    """
    
    # Get settings for this stem type
    if stem_type == 'drums':
        settings = variant.drum_settings
    elif stem_type == 'bass':
        settings = variant.bass_settings
    elif stem_type == 'vocals':
        settings = variant.vocal_settings
    elif stem_type == 'music':
        settings = variant.music_settings
    else:
        return audio  # No processing for unknown stem types
    
    processed = audio.copy()
    
    # Apply compression if specified
    if 'comp_ratio' in settings:
        processed = apply_compression(
            processed,
            ratio=settings.get('comp_ratio', 3),
            threshold_db=settings.get('comp_threshold', -20),
            attack_ms=settings.get('comp_attack', 5),
            release_ms=settings.get('comp_release', 100),
            knee=settings.get('comp_knee', 'soft'),
            sample_rate=sample_rate
        )
    
    # Apply EQ if specified
    if 'eq_bands' in settings:
        for band in settings['eq_bands']:
            processed = apply_eq_band(
                processed,
                freq=band['freq'],
                gain_db=band['gain'],
                q=band.get('q', 0.7),
                sample_rate=sample_rate
            )
    
    # Apply specialized processing based on variant name
    if 'SSL' in variant.name:
        processed = apply_ssl_character(processed, settings, sample_rate)
    elif 'Neve' in variant.name:
        processed = apply_neve_warmth(processed, settings, sample_rate)
    elif 'API' in variant.name:
        processed = apply_api_punch(processed, settings, sample_rate)
    elif 'Fairchild' in variant.name:
        processed = apply_fairchild_glue(processed, settings, sample_rate)
    elif 'LA2A' in variant.name:
        processed = apply_la2a_optical(processed, settings, sample_rate)
    elif '1176' in variant.name:
        processed = apply_1176_fet(processed, settings, sample_rate)
    elif 'Pultec' in variant.name:
        processed = apply_pultec_eq(processed, settings, sample_rate)
    elif 'Multiband' in variant.name:
        processed = apply_multiband_dynamics(processed, settings, sample_rate)
    elif 'MS' in variant.name:
        processed = apply_ms_processing(processed, settings, sample_rate)
    elif 'Parallel' in variant.name:
        processed = apply_parallel_compression(processed, settings, sample_rate)
    
    # Apply saturation if specified
    if 'saturation_amount' in settings and settings['saturation_amount'] > 0:
        saturation_type = settings.get('saturation_type', 'tape')
        processed = apply_saturation(processed, saturation_type, settings['saturation_amount'])
    
    return processed


def apply_compression(audio: np.ndarray, ratio: float, threshold_db: float,
                      attack_ms: float, release_ms: float, knee: str,
                      sample_rate: int) -> np.ndarray:
    """Apply compression with specified parameters"""
    
    # Convert to mono for processing if stereo
    if len(audio.shape) > 1:
        mono = np.mean(audio, axis=1)
    else:
        mono = audio.copy()
    
    # Convert parameters
    threshold = 10 ** (threshold_db / 20)
    attack_samples = int(attack_ms * sample_rate / 1000)
    release_samples = int(release_ms * sample_rate / 1000)
    
    # Simple compression implementation
    output = audio.copy()
    envelope = 0
    
    for i in range(len(mono)):
        input_level = abs(mono[i])
        
        # Update envelope
        if input_level > envelope:
            envelope += (input_level - envelope) / max(1, attack_samples)
        else:
            envelope += (input_level - envelope) / max(1, release_samples)
        
        # Apply compression
        if envelope > threshold:
            if knee == 'hard':
                gain_reduction = threshold + (envelope - threshold) / ratio
            else:  # soft knee
                knee_width = 0.1
                if envelope < threshold - knee_width:
                    gain_reduction = envelope
                elif envelope > threshold + knee_width:
                    gain_reduction = threshold + (envelope - threshold) / ratio
                else:
                    # Transition region
                    x = (envelope - (threshold - knee_width)) / (2 * knee_width)
                    gain_reduction = envelope * (1 - x) + (threshold + (envelope - threshold) / ratio) * x
            
            gain = gain_reduction / max(envelope, 1e-10)
        else:
            gain = 1.0
        
        # Apply gain to all channels
        if len(audio.shape) > 1:
            output[i] *= gain
        else:
            output[i] *= gain
    
    return output


def apply_eq_band(audio: np.ndarray, freq: float, gain_db: float, 
                  q: float, sample_rate: int) -> np.ndarray:
    """Apply a single EQ band"""
    
    if gain_db == 0:
        return audio
    
    # Design peaking EQ filter
    nyquist = sample_rate / 2
    normalized_freq = freq / nyquist
    
    if normalized_freq >= 1.0:
        normalized_freq = 0.99
    
    # Use scipy to design the filter
    if gain_db > 0:
        # Boost
        b, a = signal.iirpeak(normalized_freq, q)
        gain_linear = 10 ** (gain_db / 20)
        b = b * gain_linear
    else:
        # Cut
        b, a = signal.iirnotch(normalized_freq, q)
        # Blend with original
        blend = 1 - abs(gain_db) / 20  # Scale the cut amount
        
    # Apply filter
    if len(audio.shape) > 1:
        output = np.zeros_like(audio)
        for ch in range(audio.shape[1]):
            output[:, ch] = signal.filtfilt(b, a, audio[:, ch])
    else:
        output = signal.filtfilt(b, a, audio)
    
    if gain_db < 0:
        # Blend for cuts
        output = audio * blend + output * (1 - blend)
    
    return output


def apply_saturation(audio: np.ndarray, sat_type: str, amount: float) -> np.ndarray:
    """Apply saturation/harmonic distortion"""
    
    if amount == 0:
        return audio
    
    output = audio.copy()
    
    if sat_type == 'tape':
        # Tape saturation - soft clipping with even harmonics
        drive = 1 + amount * 4
        output = np.tanh(output * drive) / drive
        # Add subtle even harmonics
        output = output + 0.02 * amount * np.sin(2 * np.pi * output)
        
    elif sat_type == 'tube':
        # Tube saturation - asymmetric with odd harmonics
        drive = 1 + amount * 3
        output = output * drive
        # Asymmetric clipping
        positive = output > 0
        output[positive] = np.tanh(output[positive] * 0.7) / 0.7
        output[~positive] = np.tanh(output[~positive] * 0.9) / 0.9
        output = output / drive
        
    elif sat_type == 'console':
        # Console saturation - subtle with complex harmonics
        drive = 1 + amount * 2
        output = output * drive
        # Soft knee compression-like saturation
        output = np.sign(output) * np.log1p(np.abs(output) * 10) / 10
        output = output / drive * 1.5
        
    elif sat_type == 'transformer':
        # Transformer saturation - smooth with predominantly 3rd harmonic
        drive = 1 + amount * 2.5
        output = output * drive
        # Generate 3rd harmonic
        third_harmonic = output ** 3
        output = output * 0.9 + third_harmonic * 0.1 * amount
        output = np.tanh(output * 0.8) / 0.8
        output = output / drive
    
    # Ensure no clipping
    max_val = np.max(np.abs(output))
    if max_val > 0.99:
        output = output * 0.99 / max_val
    
    return output


# Specialized processing functions for each variant type

def apply_ssl_character(audio: np.ndarray, settings: Dict, sample_rate: int) -> np.ndarray:
    """Apply SSL console character"""
    # SSL-style bus compression and EQ coloration
    output = audio.copy()
    
    # Add SSL-style harmonics
    if 'saturation_amount' in settings:
        output = apply_saturation(output, 'console', settings['saturation_amount'] * 0.5)
    
    # SSL-style "glue"
    if 'bus_comp_ratio' in settings:
        output = apply_compression(output, settings['bus_comp_ratio'], -10, 30, 100, 'soft', sample_rate)
    
    return output


def apply_neve_warmth(audio: np.ndarray, settings: Dict, sample_rate: int) -> np.ndarray:
    """Apply Neve-style warmth and transformer saturation"""
    output = audio.copy()
    
    # Neve transformer saturation
    if 'harmonic_enhancement' in settings:
        # Add 2nd and 3rd harmonics
        second = output ** 2 * np.sign(output)
        third = output ** 3
        enhancement = settings['harmonic_enhancement']
        output = output * (1 - enhancement * 0.2) + second * enhancement * 0.15 + third * enhancement * 0.05
    
    return output


def apply_api_punch(audio: np.ndarray, settings: Dict, sample_rate: int) -> np.ndarray:
    """Apply API-style punch and forward midrange"""
    output = audio.copy()
    
    # API-style transient enhancement
    if 'transient_attack' in settings:
        # Simple transient shaper
        envelope = np.abs(output)
        transients = np.diff(envelope, prepend=envelope[0])
        transients = np.where(transients > 0, transients, 0)
        output = output + transients * settings['transient_attack'] * 2
    
    # API thrust (high-pass sidechain simulation)
    if settings.get('comp_thrust', False):
        # Boost low-mid punch
        output = apply_eq_band(output, 500, 1.5, 0.7, sample_rate)
    
    return output


def apply_fairchild_glue(audio: np.ndarray, settings: Dict, sample_rate: int) -> np.ndarray:
    """Apply Fairchild-style tube compression glue"""
    output = audio.copy()
    
    # Tube warmth
    if 'tube_warmth' in settings:
        warmth = settings['tube_warmth']
        output = apply_saturation(output, 'tube', warmth * 0.5)
        
    # Parallel blend for glue
    if 'parallel_blend' in settings:
        compressed = apply_compression(output, 10, -25, 0.5, 50, 'soft', sample_rate)
        blend = settings['parallel_blend']
        output = output * (1 - blend) + compressed * blend
    
    return output


def apply_la2a_optical(audio: np.ndarray, settings: Dict, sample_rate: int) -> np.ndarray:
    """Apply LA-2A-style optical compression character"""
    output = audio.copy()
    
    # Optical smoothness (frequency-dependent compression)
    if 'optical_smoothness' in settings:
        smoothness = settings['optical_smoothness']
        # Slower attack/release for lower frequencies
        low_freq = apply_eq_band(output, 200, 0, 0.7, sample_rate)
        high_freq = output - low_freq
        
        # Different compression for different frequency ranges
        low_compressed = apply_compression(low_freq, 3, -15, 20, 500, 'soft', sample_rate)
        high_compressed = apply_compression(high_freq, 2, -18, 5, 200, 'soft', sample_rate)
        
        output = low_compressed + high_compressed
        
    # Presence control for vocals
    if 'presence_control' in settings and settings['presence_control'] > 0:
        output = apply_eq_band(output, 3500, settings['presence_control'] * 3, 0.8, sample_rate)
    
    return output


def apply_1176_fet(audio: np.ndarray, settings: Dict, sample_rate: int) -> np.ndarray:
    """Apply 1176-style FET compression character"""
    output = audio.copy()
    
    # FET character (fast and aggressive)
    if 'fet_character' in settings:
        character = settings['fet_character']
        # Very fast compression
        output = apply_compression(output, 8, -12, 0.05, 50, 'hard', sample_rate)
        
    # All-buttons mode (extreme compression)
    if settings.get('comp_all_buttons', False):
        # Extreme parallel compression
        smashed = apply_compression(output, 20, -30, 0.01, 10, 'hard', sample_rate)
        output = output * 0.3 + smashed * 0.7
        
    # Transient emphasis
    if 'transient_emphasis' in settings:
        # Enhance initial transients
        diff = np.diff(output, prepend=output[0])
        transients = np.where(diff > 0, diff, 0)
        output = output + transients * settings['transient_emphasis']
    
    return output


def apply_pultec_eq(audio: np.ndarray, settings: Dict, sample_rate: int) -> np.ndarray:
    """Apply Pultec-style passive EQ curves"""
    output = audio.copy()
    
    # Pultec low boost and cut (simultaneous)
    if 'pultec_low_boost' in settings:
        boost = settings['pultec_low_boost']
        output = apply_eq_band(output, boost['freq'], boost['gain'], 0.5, sample_rate)
        
    if 'pultec_low_cut' in settings:
        cut = settings['pultec_low_cut']
        # High-pass filter effect
        b, a = signal.butter(1, cut['freq'] / (sample_rate/2), 'high')
        if len(output.shape) > 1:
            for ch in range(output.shape[1]):
                output[:, ch] = signal.filtfilt(b, a, output[:, ch])
        else:
            output = signal.filtfilt(b, a, output)
    
    # Pultec high boost
    if 'pultec_high_boost' in settings:
        boost = settings['pultec_high_boost']
        output = apply_eq_band(output, boost['freq'], boost['gain'], 0.4, sample_rate)
        
    # Tube saturation
    if 'tube_saturation' in settings:
        output = apply_saturation(output, 'tube', settings['tube_saturation'])
    
    return output


def apply_multiband_dynamics(audio: np.ndarray, settings: Dict, sample_rate: int) -> np.ndarray:
    """Apply multiband compression"""
    output = np.zeros_like(audio)
    
    # Define crossover frequencies
    bands = []
    if 'multiband_low' in settings:
        bands.append(settings['multiband_low'])
    if 'multiband_mid' in settings:
        bands.append(settings['multiband_mid'])
    if 'multiband_high' in settings:
        bands.append(settings['multiband_high'])
    
    if len(bands) < 2:
        return audio  # Need at least 2 bands
    
    # Create filters for band splitting
    prev_freq = 0
    for i, band in enumerate(bands):
        freq = band['freq']
        
        # Create bandpass filter for this band
        if i == 0:
            # Low band - lowpass
            b, a = signal.butter(2, freq / (sample_rate/2), 'low')
        elif i == len(bands) - 1:
            # High band - highpass
            b, a = signal.butter(2, prev_freq / (sample_rate/2), 'high')
        else:
            # Mid band - bandpass
            b, a = signal.butter(2, [prev_freq / (sample_rate/2), freq / (sample_rate/2)], 'band')
        
        # Filter and compress this band
        if len(audio.shape) > 1:
            band_audio = np.zeros_like(audio)
            for ch in range(audio.shape[1]):
                band_audio[:, ch] = signal.filtfilt(b, a, audio[:, ch])
        else:
            band_audio = signal.filtfilt(b, a, audio)
        
        # Compress this band
        compressed = apply_compression(
            band_audio,
            band.get('ratio', 3),
            -20,
            band.get('attack', 5),
            100,
            'soft',
            sample_rate
        )
        
        output += compressed
        prev_freq = freq
    
    # Normalize to prevent gain buildup
    output = output * 0.9
    
    return output


def apply_ms_processing(audio: np.ndarray, settings: Dict, sample_rate: int) -> np.ndarray:
    """Apply Mid/Side processing"""
    
    if len(audio.shape) == 1:
        return audio  # Can't do M/S on mono
    
    # Convert to M/S
    mid = (audio[:, 0] + audio[:, 1]) / 2
    side = (audio[:, 0] - audio[:, 1]) / 2
    
    # Process mid
    if 'mid_comp_ratio' in settings:
        mid = apply_compression(mid, settings['mid_comp_ratio'], -18, 10, 150, 'soft', sample_rate)
    
    if 'mid_eq_boost' in settings:
        boost = settings['mid_eq_boost']
        mid = apply_eq_band(mid.reshape(-1, 1), boost['freq'], boost['gain'], 0.7, sample_rate).flatten()
    
    # Process side
    if 'side_comp_ratio' in settings:
        side = apply_compression(side, settings['side_comp_ratio'], -20, 15, 200, 'soft', sample_rate)
    
    if 'side_eq_boost' in settings:
        boost = settings['side_eq_boost']
        side = apply_eq_band(side.reshape(-1, 1), boost['freq'], boost['gain'], 0.6, sample_rate).flatten()
    
    # Width control
    if 'width_control' in settings:
        side = side * settings['width_control']
    
    # Mono bass (sum to mono below frequency)
    if 'mono_bass_freq' in settings or 'mono_below' in settings:
        mono_freq = settings.get('mono_bass_freq', settings.get('mono_below', 120))
        b, a = signal.butter(2, mono_freq / (sample_rate/2), 'low')
        side_low = signal.filtfilt(b, a, side)
        side_high = side - side_low
        side = side_high  # Remove low frequencies from side
    
    # Convert back to L/R
    output = np.zeros_like(audio)
    output[:, 0] = mid + side
    output[:, 1] = mid - side
    
    return output


def apply_parallel_compression(audio: np.ndarray, settings: Dict, sample_rate: int) -> np.ndarray:
    """Apply New York style parallel compression"""
    
    # Get dry and wet levels
    dry_level = settings.get('dry_level', 0.7)
    parallel_level = settings.get('parallel_level', 0.3)
    
    # Create heavily compressed version
    compressed = apply_compression(
        audio,
        settings.get('parallel_ratio', 10),
        -25,
        settings.get('parallel_attack', 0.5),
        settings.get('parallel_release', 50),
        'hard',
        sample_rate
    )
    
    # Add saturation to parallel path
    if 'parallel_saturation' in settings:
        compressed = apply_saturation(compressed, 'console', settings['parallel_saturation'])
    
    # EQ the parallel path
    if 'parallel_eq' in settings:
        eq = settings['parallel_eq']
        compressed = apply_eq_band(compressed, eq['freq'], eq['gain'], 0.7, sample_rate)
    
    # Add excitement to parallel path
    if 'parallel_excitement' in settings:
        # Add harmonics
        compressed = apply_saturation(compressed, 'tube', settings['parallel_excitement'] * 0.5)
    
    # Blend
    output = audio * dry_level + compressed * parallel_level
    
    # Normalize
    max_val = np.max(np.abs(output))
    if max_val > 0.95:
        output = output * 0.95 / max_val
    
    return output


# Helper function to get variant by name
def get_variant_by_name(name: str) -> Optional[AdvancedVariant]:
    """Get an advanced variant by name"""
    for variant in ADVANCED_VARIANTS:
        if variant.name == name:
            return variant
    return None