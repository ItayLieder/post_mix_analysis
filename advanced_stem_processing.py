#!/usr/bin/env python3
"""
Advanced Stem Processing - Creative and intelligent stem manipulation
Provides unique processing options that leverage stem separation advantages
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from scipy import signal
from audio_utils import ensure_stereo, to_mono, db_to_linear, linear_to_db


# ============================================
# SMART PANNING SYSTEM
# ============================================

def smart_pan_stems(stems: Dict[str, np.ndarray], sr: int, variant: str = "natural") -> Dict[str, np.ndarray]:
    """
    Intelligently pan stems to create width and separation.
    
    Variants:
    - natural: Typical band positioning
    - wide: Extreme panning for spaciousness
    - focused: Center-focused with subtle width
    - lopsided: Asymmetric creative panning
    - orchestral: Classical positioning
    """
    panned_stems = {}
    
    # Panning positions for each variant (L/R balance, 0=center, -1=left, +1=right)
    pan_maps = {
        "natural": {
            "kick": 0.0,           # Dead center
            "snare": 0.05,         # Slightly right
            "hats": -0.3,          # Left
            "drums": 0.0,          # Center
            "bass": 0.0,           # Center
            "leadvocals": 0.0,     # Center
            "backvocals": (-0.4, 0.4),  # Stereo spread
            "vocals": 0.0,         # Center
            "guitar": 0.35,        # Right
            "keys": -0.35,         # Left
            "strings": (-0.5, 0.5),  # Wide stereo
            "music": (-0.2, 0.2),    # Slight spread
        },
        "wide": {
            "kick": 0.0,
            "snare": 0.15,
            "hats": -0.7,          # Hard left
            "drums": (-0.2, 0.2),
            "bass": 0.0,
            "leadvocals": 0.0,
            "backvocals": (-0.8, 0.8),  # Very wide
            "vocals": 0.0,
            "guitar": 0.7,         # Hard right
            "keys": -0.6,          # Far left
            "strings": (-0.9, 0.9),  # Extreme width
            "music": (-0.5, 0.5),
        },
        "focused": {
            "kick": 0.0,
            "snare": 0.0,
            "hats": -0.15,
            "drums": 0.0,
            "bass": 0.0,
            "leadvocals": 0.0,
            "backvocals": (-0.2, 0.2),
            "vocals": 0.0,
            "guitar": 0.2,
            "keys": -0.2,
            "strings": (-0.25, 0.25),
            "music": (-0.1, 0.1),
        },
        "lopsided": {  # Creative asymmetric
            "kick": -0.1,          # Slightly left
            "snare": 0.4,          # Right heavy
            "hats": -0.6,
            "drums": 0.1,
            "bass": 0.05,          # Slightly right
            "leadvocals": -0.05,   # Slightly left
            "backvocals": (-0.7, 0.3),  # More left
            "vocals": -0.05,
            "guitar": 0.8,         # Far right
            "keys": -0.4,
            "strings": (-0.3, 0.6),  # Right heavy
            "music": 0.2,
        },
        "orchestral": {  # Classical positioning
            "kick": 0.0,
            "snare": -0.1,
            "hats": 0.3,
            "drums": (-0.15, 0.15),
            "bass": 0.1,           # Slightly right (like double bass)
            "leadvocals": 0.0,
            "backvocals": (-0.3, 0.3),
            "vocals": 0.0,
            "guitar": 0.4,         # Like classical guitar position
            "keys": -0.2,          # Like piano position
            "strings": (-0.6, 0.6),  # Wide orchestral strings
            "music": (-0.3, 0.3),
        }
    }
    
    positions = pan_maps.get(variant, pan_maps["natural"])
    
    for stem_name, audio in stems.items():
        stereo = ensure_stereo(audio)
        
        if stem_name in positions:
            pan_val = positions[stem_name]
            
            if isinstance(pan_val, tuple):
                # Stereo spread panning
                left_pan, right_pan = pan_val
                panned = apply_stereo_spread(stereo, left_pan, right_pan)
            else:
                # Simple pan
                panned = apply_pan(stereo, pan_val)
        else:
            panned = stereo  # No panning for unknown stems
            
        panned_stems[stem_name] = panned
    
    return panned_stems


def apply_pan(audio: np.ndarray, position: float) -> np.ndarray:
    """Apply constant power panning. Position: -1=left, 0=center, +1=right"""
    stereo = ensure_stereo(audio)
    
    # Constant power panning law (3dB center attenuation)
    position = np.clip(position, -1.0, 1.0)
    angle = (position + 1.0) * np.pi / 4  # 0 to π/2
    
    left_gain = np.cos(angle)
    right_gain = np.sin(angle)
    
    panned = stereo.copy()
    panned[:, 0] *= left_gain
    panned[:, 1] *= right_gain
    
    return panned


def apply_stereo_spread(audio: np.ndarray, left_pos: float, right_pos: float) -> np.ndarray:
    """Apply different panning to L/R channels for width"""
    stereo = ensure_stereo(audio)
    
    # Pan left channel to left_pos, right channel to right_pos
    left_panned = apply_pan(stereo[:, 0:1], left_pos)
    right_panned = apply_pan(stereo[:, 1:2], right_pos)
    
    return np.column_stack([left_panned[:, 0], right_panned[:, 1]])


# ============================================
# FREQUENCY SLOTTING (SPECTRAL CARVING)
# ============================================

def frequency_slot_stems(stems: Dict[str, np.ndarray], sr: int, variant: str = "clean") -> Dict[str, np.ndarray]:
    """
    Carve out frequency space for each stem to reduce masking.
    
    Variants:
    - clean: Gentle carving for clarity
    - surgical: Aggressive carving for maximum separation
    - vintage: Emulate console EQ curves
    - modern: Hyped modern production curves
    """
    from dsp_premitives import peaking_eq, shelf_filter
    
    processed = {}
    
    # Frequency carving presets (freq_hz, gain_db, Q)
    carve_presets = {
        "clean": {
            "kick": [
                (80, 2.0, 0.7),      # Boost fundamental
                (250, -2.0, 0.8),    # Cut mud
                (4000, 1.5, 1.0),    # Click
            ],
            "snare": [
                (200, 2.0, 0.8),     # Body
                (800, -1.5, 1.0),    # Cut boxiness
                (5000, 1.5, 0.9),    # Crack
            ],
            "bass": [
                (60, 1.5, 0.6),      # Sub
                (400, -2.0, 0.7),    # Cut low-mids
                (1200, 1.0, 1.2),    # Definition
            ],
            "leadvocals": [
                (150, -1.5, 0.8),    # Cut low mud
                (3000, 2.0, 0.8),    # Presence
                (8000, 1.5, 0.7),    # Air
            ],
            "backvocals": [
                (200, -2.5, 0.7),    # More low cut
                (2000, -1.0, 1.0),   # Make room for lead
                (10000, 2.0, 0.6),   # Airy top
            ],
            "guitar": [
                (100, -2.0, 0.6),    # Cut lows
                (500, -1.5, 0.9),    # Reduce mud
                (2500, 1.5, 0.8),    # Bite
            ],
            "keys": [
                (250, -1.5, 0.7),    # Thin out lows
                (1000, 1.0, 1.0),    # Mid presence
                (6000, 1.5, 0.8),    # Sparkle
            ],
        },
        "surgical": {
            "kick": [
                (60, 4.0, 1.2),      # Heavy sub boost
                (200, -4.0, 1.5),    # Deep cut
                (800, -3.0, 1.2),    # Remove mids
                (4500, 3.0, 1.5),    # Sharp click
            ],
            "snare": [
                (80, -3.0, 1.0),     # Remove sub
                (220, 3.0, 1.2),     # Focused body
                (600, -3.5, 1.5),    # Deep scoop
                (6000, 3.5, 1.3),    # Bright crack
            ],
            "bass": [
                (50, 3.0, 1.0),      # Sub emphasis
                (350, -4.0, 1.2),    # Deep cut
                (1500, 2.5, 1.5),    # Attack
            ],
            "vocals": [
                (100, -4.0, 1.0),    # Heavy HP
                (500, -2.5, 1.2),    # Cut mud
                (3500, 3.5, 1.2),    # Strong presence
                (12000, 3.0, 0.8),   # Extended air
            ],
        },
        "vintage": {
            "drums": [
                (100, 2.5, 0.5),     # Broad low boost
                (800, -1.8, 0.6),    # Gentle mid cut
                (10000, 2.0, 0.4),   # Smooth highs
            ],
            "bass": [
                (80, 3.0, 0.4),      # Wide low boost
                (2000, 1.5, 0.5),    # Gentle presence
            ],
            "vocals": [
                (300, 1.5, 0.5),     # Warm body
                (5000, 2.5, 0.6),    # Smooth presence
            ],
        },
        "modern": {
            "kick": [
                (40, 4.0, 0.8),      # Massive sub
                (300, -5.0, 1.0),    # Scooped
                (5000, 4.0, 1.2),    # Hyped click
            ],
            "vocals": [
                (150, -3.0, 1.0),    # Clean lows
                (4000, 4.0, 1.0),    # Hyped presence
                (12000, 3.5, 0.7),   # Crispy air
            ],
        }
    }
    
    preset = carve_presets.get(variant, carve_presets["clean"])
    
    for stem_name, audio in stems.items():
        processed_audio = audio.copy()
        
        # Get carving for specific stem or fallback to category
        stem_mapping = {
            'hats': 'snare',  # Use snare carving for hats
            'leadvocals': 'vocals',
            'backvocals': 'vocals',
            'guitar': 'guitar',
            'strings': 'keys',  # Use keys carving for strings
        }
        
        carve_key = stem_mapping.get(stem_name, stem_name)
        
        if carve_key in preset:
            for freq, gain, q in preset[carve_key]:
                processed_audio = peaking_eq(processed_audio, sr, freq, gain, q)
        
        processed[stem_name] = processed_audio
    
    return processed


# ============================================
# DYNAMIC INTERACTION (SIDECHAIN/DUCKING)
# ============================================

def dynamic_interaction_stems(stems: Dict[str, np.ndarray], sr: int, variant: str = "subtle") -> Dict[str, np.ndarray]:
    """
    Create dynamic relationships between stems (sidechain compression, ducking).
    
    Variants:
    - subtle: Light interaction for glue
    - pumping: Classic EDM pumping
    - groovy: Musical ducking for groove
    - extreme: Heavy ducking for effect
    """
    processed = stems.copy()
    
    # Interaction parameters
    interactions = {
        "subtle": [
            # (trigger, target, amount, attack_ms, release_ms)
            ("kick", "bass", 0.15, 5, 50),      # Kick ducks bass slightly
            ("kick", "music", 0.08, 10, 60),    # Kick slightly affects music
            ("snare", "music", 0.05, 8, 40),    # Snare barely touches music
            ("leadvocals", "music", 0.12, 20, 100),  # Vocals duck music
        ],
        "pumping": [
            ("kick", "bass", 0.4, 2, 80),       # Heavy kick->bass pumping
            ("kick", "music", 0.3, 3, 90),      # Kick pumps music
            ("kick", "keys", 0.35, 3, 85),      # Kick pumps keys
            ("snare", "music", 0.2, 5, 60),     # Snare pumps too
        ],
        "groovy": [
            ("kick", "bass", 0.25, 4, 60),      # Musical kick/bass interaction
            ("snare", "hats", 0.15, 3, 35),     # Snare affects hats
            ("kick", "guitar", 0.2, 8, 70),     # Kick grooves guitar
            ("leadvocals", "guitar", 0.2, 15, 120),  # Vocals push guitar back
            ("leadvocals", "keys", 0.18, 15, 120),   # Vocals push keys back
        ],
        "extreme": [
            ("kick", "bass", 0.6, 1, 100),      # Extreme kick/bass ducking
            ("kick", "music", 0.5, 2, 110),     # Heavy pumping
            ("snare", "music", 0.4, 2, 70),     # Snare pumps hard
            ("leadvocals", "everything", 0.4, 10, 150),  # Vocals duck everything
        ]
    }
    
    if variant not in interactions:
        return processed
    
    for trigger_name, target_name, amount, attack_ms, release_ms in interactions[variant]:
        # Check if both stems exist
        trigger_audio = None
        target_audio = None
        
        # Find trigger
        if trigger_name in stems:
            trigger_audio = stems[trigger_name]
        elif trigger_name == "vocals" and "leadvocals" in stems:
            trigger_audio = stems["leadvocals"]
        
        # Find target(s)
        if target_name == "everything":
            # Duck all non-trigger stems
            for stem_name in stems:
                if stem_name != trigger_name and trigger_audio is not None:
                    processed[stem_name] = apply_ducking(
                        processed[stem_name], trigger_audio, sr,
                        amount, attack_ms, release_ms
                    )
        elif target_name in stems:
            target_audio = stems[target_name]
            if trigger_audio is not None and target_audio is not None:
                processed[target_name] = apply_ducking(
                    target_audio, trigger_audio, sr,
                    amount, attack_ms, release_ms
                )
    
    return processed


def apply_ducking(target: np.ndarray, trigger: np.ndarray, sr: int,
                  amount: float = 0.3, attack_ms: float = 5, release_ms: float = 50) -> np.ndarray:
    """Apply sidechain compression (ducking) to target based on trigger envelope"""
    from processors import _envelope_detector
    from audio_utils import to_mono
    
    # Get trigger envelope
    trigger_mono = to_mono(trigger)
    envelope = _envelope_detector(trigger_mono, sr, attack_ms, release_ms)
    
    # Normalize envelope
    env_max = np.max(envelope) if np.max(envelope) > 0 else 1.0
    envelope = envelope / env_max
    
    # Create gain reduction curve
    gain_reduction = 1.0 - (envelope * amount)
    gain_reduction = np.clip(gain_reduction, 0.3, 1.0)  # Never duck more than -10dB
    
    # Apply to target
    if target.ndim == 1:
        return target * gain_reduction
    else:
        return target * gain_reduction[:, np.newaxis]


# ============================================
# SPATIAL ENHANCEMENT
# ============================================

def spatial_enhancement_stems(stems: Dict[str, np.ndarray], sr: int, variant: str = "natural") -> Dict[str, np.ndarray]:
    """
    Add depth and space to stems using reverb, delays, and spatial processing.
    
    Variants:
    - natural: Realistic space
    - stadium: Big reverberant space
    - intimate: Close, dry with subtle ambience
    - psychedelic: Creative spatial effects
    """
    processed = {}
    
    # Spatial parameters for each variant
    spatial_params = {
        "natural": {
            "drums": {"reverb": 0.1, "predelay": 5, "width": 1.1},
            "snare": {"reverb": 0.15, "predelay": 8, "width": 1.2},
            "bass": {"reverb": 0.05, "predelay": 0, "width": 0.8},
            "leadvocals": {"reverb": 0.12, "predelay": 15, "width": 1.0},
            "backvocals": {"reverb": 0.25, "predelay": 20, "width": 1.4},
            "guitar": {"reverb": 0.15, "predelay": 10, "width": 1.3},
            "keys": {"reverb": 0.2, "predelay": 12, "width": 1.4},
            "strings": {"reverb": 0.3, "predelay": 25, "width": 1.5},
        },
        "stadium": {
            "drums": {"reverb": 0.22, "predelay": 15, "width": 1.2},
            "vocals": {"reverb": 0.25, "predelay": 20, "width": 1.15},
            "guitar": {"reverb": 0.28, "predelay": 18, "width": 1.3},
        },
        "intimate": {
            "vocals": {"reverb": 0.08, "predelay": 5, "width": 0.9},
            "guitar": {"reverb": 0.06, "predelay": 3, "width": 0.85},
            "keys": {"reverb": 0.1, "predelay": 8, "width": 1.0},
        },
        "psychedelic": {
            "vocals": {"reverb": 0.25, "predelay": 25, "width": 1.4, "modulation": True},
            "guitar": {"reverb": 0.30, "predelay": 22, "width": 1.5, "modulation": True},
            "keys": {"reverb": 0.28, "predelay": 30, "width": 1.6, "modulation": True},
        }
    }
    
    params = spatial_params.get(variant, spatial_params["natural"])
    
    for stem_name, audio in stems.items():
        if stem_name in params:
            p = params[stem_name]
            processed_audio = add_space(
                audio, sr,
                reverb_amount=p.get("reverb", 0.1),
                predelay_ms=p.get("predelay", 10),
                width_factor=p.get("width", 1.0),
                modulation=p.get("modulation", False)
            )
        else:
            # Default minimal processing
            processed_audio = add_space(audio, sr, reverb_amount=0.05)
        
        processed[stem_name] = processed_audio
    
    return processed


def add_space(audio: np.ndarray, sr: int, reverb_amount: float = 0.1,
              predelay_ms: float = 10, width_factor: float = 1.0,
              modulation: bool = False) -> np.ndarray:
    """Simplified spatial processing (would use convolution reverb in production)"""
    from dsp_premitives import stereo_widener
    
    stereo = ensure_stereo(audio)
    
    # Simple reverb simulation using comb filters and delays
    # This is a placeholder - production would use proper convolution
    predelay_samples = int(sr * predelay_ms / 1000)
    
    if predelay_samples > 0:
        delayed = np.pad(stereo, ((predelay_samples, 0), (0, 0)), mode='constant')
        delayed = delayed[:len(stereo)]
    else:
        delayed = stereo
    
    # Mix dry and "wet" (delayed)
    wet = delayed * reverb_amount
    dry = stereo * (1.0 - reverb_amount * 0.5)  # Reduce dry when adding reverb
    mixed = dry + wet
    
    # Apply width adjustment
    if width_factor != 1.0:
        mixed = stereo_widener(mixed, width_factor)
    
    # Optional modulation for psychedelic effect
    if modulation:
        # Simple chorus-like modulation
        mod_hz = 0.5  # Modulation frequency
        mod_depth = 0.002  # 2ms modulation depth
        t = np.arange(len(mixed)) / sr
        mod = np.sin(2 * np.pi * mod_hz * t) * mod_depth
        # This would need proper implementation with variable delay
        mixed = mixed * (1 + mod[:, np.newaxis] * 0.1)
    
    return mixed.astype(np.float32)


# ============================================
# HARMONIC ENHANCEMENT
# ============================================

def harmonic_enhancement_stems(stems: Dict[str, np.ndarray], sr: int, variant: str = "warm") -> Dict[str, np.ndarray]:
    """
    Add harmonic richness through saturation and excitation.
    
    Variants:
    - warm: Tube-like even harmonics
    - crispy: Bright odd harmonics
    - vintage: Tape-like saturation
    - aggressive: Heavy distortion
    """
    processed = {}
    
    saturation_params = {
        "warm": {
            "bass": {"drive": 0.3, "type": "tube", "mix": 0.4},
            "vocals": {"drive": 0.2, "type": "tube", "mix": 0.3},
            "guitar": {"drive": 0.25, "type": "tube", "mix": 0.35},
        },
        "crispy": {
            "drums": {"drive": 0.4, "type": "transistor", "mix": 0.3},
            "snare": {"drive": 0.5, "type": "transistor", "mix": 0.35},
            "vocals": {"drive": 0.3, "type": "exciter", "mix": 0.25},
        },
        "vintage": {
            "drums": {"drive": 0.35, "type": "tape", "mix": 0.4},
            "bass": {"drive": 0.4, "type": "tape", "mix": 0.45},
            "vocals": {"drive": 0.25, "type": "tape", "mix": 0.3},
        },
        "aggressive": {
            "drums": {"drive": 0.6, "type": "clip", "mix": 0.5},
            "bass": {"drive": 0.7, "type": "fuzz", "mix": 0.4},
            "guitar": {"drive": 0.8, "type": "distortion", "mix": 0.45},
        }
    }
    
    params = saturation_params.get(variant, saturation_params["warm"])
    
    for stem_name, audio in stems.items():
        if stem_name in params:
            p = params[stem_name]
            processed_audio = add_harmonics(
                audio,
                drive=p["drive"],
                sat_type=p["type"],
                mix=p["mix"]
            )
        else:
            processed_audio = audio  # No processing
        
        processed[stem_name] = processed_audio
    
    return processed


def add_harmonics(audio: np.ndarray, drive: float = 0.3, 
                  sat_type: str = "tube", mix: float = 0.3) -> np.ndarray:
    """Add harmonic saturation"""
    
    driven = audio * db_to_linear(drive * 12)  # Drive up to +12dB
    
    if sat_type == "tube":
        # Soft clipping with even harmonics
        saturated = np.tanh(driven * 0.7) * 1.3
    elif sat_type == "transistor":
        # Harder clipping with odd harmonics
        saturated = np.sign(driven) * np.minimum(np.abs(driven), 0.9)
    elif sat_type == "tape":
        # Soft saturation with compression
        saturated = np.tanh(driven * 0.5) + np.tanh(driven * 2.0) * 0.1
    elif sat_type == "exciter":
        # High frequency harmonic generation
        # Would need proper implementation with band-splitting
        saturated = driven + np.diff(np.pad(driven, ((1, 0), (0, 0)), mode='edge'), axis=0) * 0.2
    elif sat_type == "clip":
        # Hard clipping
        saturated = np.clip(driven, -0.8, 0.8) * 1.25
    elif sat_type == "fuzz":
        # Extreme clipping
        saturated = np.clip(driven * 3, -0.5, 0.5) * 2
    else:  # distortion
        # Wave shaping distortion
        saturated = np.sign(driven) * (1 - np.exp(-np.abs(driven * 2)))
    
    # Mix dry and wet
    return (audio * (1 - mix) + saturated * mix).astype(np.float32)


# ============================================
# MASTER PROCESSING COMBINATIONS
# ============================================

@dataclass
class AdvancedStemVariant:
    """Container for advanced processing parameters"""
    name: str
    panning: str = "natural"
    frequency: str = "clean"  
    dynamics: str = "subtle"
    spatial: str = "natural"
    harmonics: str = "warm"
    description: str = ""


# Pre-defined advanced variants
ADVANCED_VARIANTS = [
    AdvancedStemVariant(
        "RadioReady",
        panning="focused", frequency="surgical", dynamics="subtle",
        spatial="natural", harmonics="crispy",
        description="Optimized for radio/streaming: clear separation, controlled dynamics"
    ),
    AdvancedStemVariant(
        "LiveBand", 
        panning="orchestral", frequency="vintage", dynamics="groovy",
        spatial="stadium", harmonics="warm",
        description="Live band feel: natural positioning, musical interaction"
    ),
    AdvancedStemVariant(
        "EDM_Club",
        panning="wide", frequency="modern", dynamics="pumping",
        spatial="stadium", harmonics="aggressive",
        description="EDM/Club: wide, pumping, aggressive"
    ),
    AdvancedStemVariant(
        "Intimate_Acoustic",
        panning="natural", frequency="clean", dynamics="subtle",
        spatial="intimate", harmonics="warm",
        description="Intimate acoustic: close, warm, minimal processing"
    ),
    AdvancedStemVariant(
        "Experimental",
        panning="lopsided", frequency="surgical", dynamics="extreme",
        spatial="psychedelic", harmonics="aggressive",
        description="Creative/experimental: asymmetric, heavily processed"
    ),
    AdvancedStemVariant(
        "Vintage_Soul",
        panning="natural", frequency="vintage", dynamics="groovy",
        spatial="natural", harmonics="vintage",
        description="Vintage soul: tape saturation, musical dynamics"
    ),
    AdvancedStemVariant(
        "Modern_Pop",
        panning="wide", frequency="modern", dynamics="subtle",
        spatial="natural", harmonics="crispy",
        description="Modern pop: wide, bright, polished"
    ),
    AdvancedStemVariant(
        "Heavy_Rock",
        panning="wide", frequency="surgical", dynamics="groovy",
        spatial="stadium", harmonics="aggressive",
        description="Heavy rock: wide guitars, aggressive, stadium sound"
    ),
]


def apply_advanced_processing(stems: Dict[str, np.ndarray], sr: int, 
                            variant: AdvancedStemVariant) -> Dict[str, np.ndarray]:
    """Apply a complete advanced processing chain to stems with safety checks"""
    
    print(f"    🎛️ Applying {variant.name}: {variant.description}")
    
    # Apply processing chain in order
    processed = stems
    
    try:
        # 1. Frequency slotting first (EQ before spatial)
        processed = frequency_slot_stems(processed, sr, variant.frequency)
        processed = _sanitize_stems(processed)
        print(f"      ✓ Frequency: {variant.frequency}")
        
        # 2. Dynamic interaction
        processed = dynamic_interaction_stems(processed, sr, variant.dynamics)
        processed = _sanitize_stems(processed)
        print(f"      ✓ Dynamics: {variant.dynamics}")
        
        # 3. Harmonic enhancement
        processed = harmonic_enhancement_stems(processed, sr, variant.harmonics)
        processed = _sanitize_stems(processed)
        print(f"      ✓ Harmonics: {variant.harmonics}")
        
        # 4. Spatial enhancement
        processed = spatial_enhancement_stems(processed, sr, variant.spatial)
        processed = _sanitize_stems(processed)
        print(f"      ✓ Spatial: {variant.spatial}")
        
        # 5. Panning last (after all processing)
        processed = smart_pan_stems(processed, sr, variant.panning)
        processed = _sanitize_stems(processed)
        print(f"      ✓ Panning: {variant.panning}")
        
    except Exception as e:
        print(f"      ⚠️ Advanced processing error: {e}")
        # Return safely processed stems
        processed = _sanitize_stems(stems)
    
    return processed


def _sanitize_stems(stems: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Sanitize stems to prevent audio errors"""
    sanitized = {}
    
    for stem_name, audio in stems.items():
        # Convert to numpy array
        audio = np.asarray(audio, dtype=np.float32)
        
        # Replace NaN and Inf
        audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Clip to reasonable range
        audio = np.clip(audio, -3.0, 3.0)
        
        sanitized[stem_name] = audio.astype(np.float32)
    
    return sanitized


print("🎯 Advanced Stem Processing loaded!")
print(f"   • {len(ADVANCED_VARIANTS)} pre-configured variants")
print("   • Smart panning (5 modes)")
print("   • Frequency slotting (4 modes)")
print("   • Dynamic interaction (4 modes)")
print("   • Spatial enhancement (4 modes)")
print("   • Harmonic saturation (4 modes)")