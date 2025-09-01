#!/usr/bin/env python3
"""
Hybrid Processing - Combine advanced stem processing with subtle depth

Creates variants that apply advanced processing first, then add subtle depth on top.
Perfect for getting the best of both worlds: advanced sonic shaping + dimensional enhancement.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from utils import ensure_stereo, to_mono, db_to_linear, linear_to_db


def create_hybrid_variant(stems: Dict[str, np.ndarray], sr: int, hybrid_style: str) -> Dict[str, np.ndarray]:
    """
    Create hybrid processing by combining advanced variants with subtle depth.
    
    Args:
        hybrid_style: "RadioReady_depth", "Aggressive_depth", "PunchyMix_depth"
    """
    
    print(f"    🔄 Creating hybrid variant: {hybrid_style}")
    
    # Step 1: Apply the base advanced processing
    if hybrid_style == "RadioReady_depth":
        processed = apply_radioready_with_depth(stems, sr)
    elif hybrid_style == "Aggressive_depth":
        processed = apply_aggressive_with_depth(stems, sr)
    elif hybrid_style == "PunchyMix_depth":
        processed = apply_punchymix_with_depth(stems, sr)
    else:
        print(f"      ⚠️ Unknown hybrid style: {hybrid_style}, using basic processing")
        processed = stems.copy()
    
    return processed


def apply_radioready_with_depth(stems: Dict[str, np.ndarray], sr: int) -> Dict[str, np.ndarray]:
    """
    RadioReady processing + subtle depth enhancement.
    
    RadioReady: Optimized for radio/streaming with clear separation and controlled dynamics
    + Subtle depth: Just enough dimensional enhancement to sound more professional
    """
    print(f"      📻 Applying RadioReady processing...")
    
    # Step 1: Apply RadioReady advanced processing
    from advanced_stem_processing import apply_advanced_processing, ADVANCED_VARIANTS
    
    # Find RadioReady variant
    radioready_variant = None
    for av in ADVANCED_VARIANTS:
        if av.name == "RadioReady":
            radioready_variant = av
            break
    
    if radioready_variant:
        processed = apply_advanced_processing(stems, sr, radioready_variant)
        print(f"        ✓ Applied RadioReady: focused panning, surgical EQ, subtle dynamics")
    else:
        processed = stems.copy()
        print(f"        ⚠️ RadioReady variant not found, using original stems")
    
    # Step 2: Add subtle depth enhancement (like musical:balanced but even more subtle)
    print(f"      🏞️ Adding RadioReady-compatible depth...")
    processed = add_radioready_depth(processed, sr)
    print(f"        ✓ Applied minimal depth: maintains RadioReady balance")
    
    return processed


def apply_aggressive_with_depth(stems: Dict[str, np.ndarray], sr: int) -> Dict[str, np.ndarray]:
    """
    Aggressive processing + subtle depth enhancement.
    
    Aggressive: Wide, pumping, in-your-face energy
    + Subtle depth: Adds dimension without losing the aggressive character
    """
    print(f"      🔥 Applying Aggressive processing...")
    
    try:
        # Step 1: Apply Aggressive basic processing using advanced system
        from advanced_stem_processing import apply_advanced_processing, ADVANCED_VARIANTS
        
        # Create an aggressive-like variant using Heavy_Rock (closest to aggressive)
        heavy_rock_variant = None
        for av in ADVANCED_VARIANTS:
            if av.name == "Heavy_Rock":
                heavy_rock_variant = av
                break
        
        if heavy_rock_variant:
            processed = apply_advanced_processing(stems, sr, heavy_rock_variant)
            print(f"        ✓ Applied Heavy_Rock (aggressive-style): wide guitars, aggressive, stadium sound")
        else:
            processed = stems.copy()
            print(f"        ⚠️ Heavy_Rock variant not found, using original stems")
    except Exception as e:
        print(f"        ⚠️ Error in aggressive processing: {e}")
        processed = stems.copy()
    
    # Step 2: Add depth that complements the aggressive character
    print(f"      🏞️ Adding aggressive-compatible depth...")
    processed = add_aggressive_depth(processed, sr)
    print(f"        ✓ Applied dramatic depth: maintains aggressive energy with added dimension")
    
    return processed


def apply_punchymix_with_depth(stems: Dict[str, np.ndarray], sr: int) -> Dict[str, np.ndarray]:
    """
    PunchyMix processing + subtle depth enhancement.
    
    PunchyMix: Tight, impactful, punchy character with strong transients
    + Subtle depth: Adds space without losing the punch
    """
    print(f"      👊 Applying PunchyMix processing...")
    
    try:
        # Step 1: Apply PunchyMix-style processing using advanced system
        from advanced_stem_processing import apply_advanced_processing, ADVANCED_VARIANTS
        
        # Create a punchy-like variant using EDM_Club (has punchy, wide characteristics)
        edm_club_variant = None
        for av in ADVANCED_VARIANTS:
            if av.name == "EDM_Club":
                edm_club_variant = av
                break
        
        if edm_club_variant:
            processed = apply_advanced_processing(stems, sr, edm_club_variant)
            print(f"        ✓ Applied EDM_Club (punchy-style): wide, pumping, aggressive - perfect for punch")
        else:
            processed = stems.copy()
            print(f"        ⚠️ EDM_Club variant not found, using original stems")
    except Exception as e:
        print(f"        ⚠️ Error in punchy processing: {e}")
        processed = stems.copy()
    
    # Step 2: Add depth that preserves the punchy character
    print(f"      🏞️ Adding punch-compatible depth...")
    processed = add_punchy_depth(processed, sr)
    print(f"        ✓ Applied natural depth: maintains punch with subtle dimensional enhancement")
    
    return processed


def add_radioready_depth(stems: Dict[str, np.ndarray], sr: int) -> Dict[str, np.ndarray]:
    """
    Add extremely subtle depth that preserves RadioReady's balance and clarity.
    Even more conservative than musical:balanced to maintain radio compatibility.
    """
    
    # Ultra-conservative depth settings - maintains RadioReady balance
    depth_map = {
        "kick": 0.96,          # Almost neutral
        "snare": 0.98,         # Almost neutral
        "hats": 1.02,          # Tiny bit back
        "drums": 0.98,         # Almost neutral
        "bass": 0.95,          # Tiny bit forward for radio impact
        "leadvocals": 0.92,    # Slightly forward for radio clarity
        "backvocals": 1.05,    # Just slightly back
        "vocals": 0.92,        # Slightly forward for radio clarity
        "guitar": 1.01,        # Almost neutral
        "keys": 1.03,          # Tiny bit back
        "strings": 1.06,       # Slightly back
        "music": 1.02,         # Tiny bit back
    }
    
    processed = {}
    
    for stem_name, audio in stems.items():
        if stem_name in depth_map:
            distance = depth_map[stem_name]
            processed[stem_name] = apply_radioready_positioning(audio, sr, distance)
        else:
            processed[stem_name] = audio
    
    return processed


def add_aggressive_depth(stems: Dict[str, np.ndarray], sr: int) -> Dict[str, np.ndarray]:
    """
    Add dramatic depth that complements aggressive processing.
    More contrast than RadioReady but still maintains the energy.
    """
    
    # More dramatic depth settings to match aggressive character
    depth_map = {
        "kick": 0.88,          # Forward for impact
        "snare": 0.92,         # Forward for crack
        "hats": 1.12,          # Back for width
        "drums": 0.90,         # Forward for power
        "bass": 0.85,          # Forward for aggression
        "leadvocals": 0.87,    # Forward for presence
        "backvocals": 1.20,    # Back for contrast
        "vocals": 0.87,        # Forward for presence
        "guitar": 1.10,        # Back for width
        "keys": 1.15,          # Back for space
        "strings": 1.25,       # Back for atmosphere
        "music": 1.12,         # Back for contrast
    }
    
    processed = {}
    
    for stem_name, audio in stems.items():
        if stem_name in depth_map:
            distance = depth_map[stem_name]
            processed[stem_name] = apply_aggressive_positioning(audio, sr, distance)
        else:
            processed[stem_name] = audio
    
    return processed


def add_punchy_depth(stems: Dict[str, np.ndarray], sr: int) -> Dict[str, np.ndarray]:
    """
    Add natural depth that preserves punch and transients.
    Conservative depth that maintains the tight, impactful character.
    """
    
    # Natural depth settings that preserve punch
    depth_map = {
        "kick": 0.90,          # Slightly forward for punch
        "snare": 0.93,         # Slightly forward for impact
        "hats": 1.08,          # Just slightly back
        "drums": 0.92,         # Slightly forward for power
        "bass": 0.88,          # Forward for tight low end
        "leadvocals": 0.89,    # Forward but not extreme
        "backvocals": 1.15,    # Back but not too far
        "vocals": 0.89,        # Forward but not extreme
        "guitar": 1.06,        # Slightly back
        "keys": 1.10,          # Slightly back
        "strings": 1.18,       # Back but moderate
        "music": 1.08,         # Slightly back
    }
    
    processed = {}
    
    for stem_name, audio in stems.items():
        if stem_name in depth_map:
            distance = depth_map[stem_name]
            processed[stem_name] = apply_punchy_positioning(audio, sr, distance)
        else:
            processed[stem_name] = audio
    
    return processed


def apply_radioready_positioning(audio: np.ndarray, sr: int, distance: float) -> np.ndarray:
    """
    Apply ultra-subtle positioning that preserves RadioReady characteristics.
    Minimal processing to maintain radio/streaming compatibility.
    """
    from dsp_premitives import shelf_filter, lowpass_filter
    
    stereo = ensure_stereo(audio)
    processed = stereo.copy()
    
    # 1. Ultra-minimal volume scaling (almost no change)
    volume_scale = 1.0 / (distance ** 0.1)  # Extremely gentle
    processed *= volume_scale
    
    # 2. Very subtle high frequency adjustment (only for distant elements)
    if distance > 1.15:
        # Very gentle high-frequency roll-off to maintain radio clarity
        cutoff_freq = 20000 / (distance ** 0.2)  # Very gentle
        cutoff_freq = max(cutoff_freq, 15000)  # Keep radio brightness
        processed = lowpass_filter(processed, sr, cutoff_freq, order=1)
    
    # 3. Minimal spectral tilt (barely perceptible)
    if distance > 1.2:
        tilt_amount = (distance - 1.0) * -0.15  # Ultra-gentle
        processed = shelf_filter(processed, sr, 5000, tilt_amount, kind="high")
    elif distance < 0.95:
        presence_boost = (0.95 - distance) * 0.1  # Ultra-gentle
        processed = shelf_filter(processed, sr, 3000, presence_boost, kind="high")
    
    return processed.astype(np.float32)


def apply_aggressive_positioning(audio: np.ndarray, sr: int, distance: float) -> np.ndarray:
    """
    Apply more dramatic positioning that maintains aggressive character.
    More processing to create dramatic depth without losing energy.
    """
    from dsp_premitives import shelf_filter, lowpass_filter
    
    stereo = ensure_stereo(audio)
    processed = stereo.copy()
    
    # 1. More noticeable volume scaling (maintains aggressive dynamics)
    volume_scale = 1.0 / (distance ** 0.4)  # More pronounced than RadioReady
    processed *= volume_scale
    
    # 2. Moderate high frequency roll-off for distance
    if distance > 1.2:
        cutoff_freq = 18000 / (distance ** 0.6)  # Moderate rolloff
        cutoff_freq = max(cutoff_freq, 8000)  # Maintain presence
        processed = lowpass_filter(processed, sr, cutoff_freq, order=1)
    
    # 3. Moderate spectral tilt for depth impression
    if distance > 1.15:
        tilt_amount = (distance - 1.0) * -0.6  # More noticeable
        processed = shelf_filter(processed, sr, 4000, tilt_amount, kind="high")
    elif distance < 0.92:
        presence_boost = (0.92 - distance) * 0.4  # Enhance forward elements
        processed = shelf_filter(processed, sr, 3000, presence_boost, kind="high")
    
    # 4. Add subtle early reflections for aggressive elements in back
    if distance > 1.25:
        processed = add_aggressive_reflections(processed, sr, distance)
    
    return processed.astype(np.float32)


def apply_punchy_positioning(audio: np.ndarray, sr: int, distance: float) -> np.ndarray:
    """
    Apply natural positioning that preserves punch and transient clarity.
    Balanced processing that adds depth without softening the impact.
    """
    from dsp_premitives import shelf_filter, lowpass_filter
    
    stereo = ensure_stereo(audio)
    processed = stereo.copy()
    
    # 1. Gentle volume scaling (preserves punch)
    volume_scale = 1.0 / (distance ** 0.25)  # Between RadioReady and Aggressive
    processed *= volume_scale
    
    # 2. Conservative high frequency handling (maintains transients)
    if distance > 1.25:
        cutoff_freq = 19000 / (distance ** 0.4)  # Conservative rolloff
        cutoff_freq = max(cutoff_freq, 10000)  # Preserve transient clarity
        processed = lowpass_filter(processed, sr, cutoff_freq, order=1)
    
    # 3. Natural spectral tilt (subtle but present)
    if distance > 1.18:
        tilt_amount = (distance - 1.0) * -0.4  # Natural amount
        processed = shelf_filter(processed, sr, 4500, tilt_amount, kind="high")
    elif distance < 0.95:
        presence_boost = (0.95 - distance) * 0.25  # Subtle punch enhancement
        processed = shelf_filter(processed, sr, 2500, presence_boost, kind="high")
    
    return processed.astype(np.float32)


def add_aggressive_reflections(audio: np.ndarray, sr: int, distance: float) -> np.ndarray:
    """Add subtle early reflections for aggressive back elements"""
    from dsp_premitives import lowpass_filter
    
    # Simple early reflection for depth
    delay_samples = int(0.006 * distance * sr)  # 6ms per unit distance
    if delay_samples > 0 and delay_samples < len(audio):
        delayed = np.zeros_like(audio)
        delayed[delay_samples:] = audio[:-delay_samples]
        
        # Filter the reflection (walls absorb highs)
        delayed = lowpass_filter(delayed, sr, 10000, order=1)
        
        # Mix in reflection (subtle)
        reflection_level = 0.08 * (distance - 1.0)  # Up to 8% wet
        audio += delayed * reflection_level
    
    return audio


def _sanitize_hybrid_stems(stems: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Sanitize hybrid processed stems"""
    sanitized = {}
    
    for stem_name, audio in stems.items():
        # Convert to numpy array
        audio = np.asarray(audio, dtype=np.float32)
        
        # Replace NaN and Inf
        audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Clip to safe range
        audio = np.clip(audio, -2.0, 2.0)
        
        sanitized[stem_name] = audio.astype(np.float32)
    
    return sanitized


# Pre-configured hybrid variants
HYBRID_VARIANTS = {
    "RadioReady_depth": "RadioReady + subtle depth - maintains radio clarity with dimension",
    "Aggressive_depth": "Aggressive + dramatic depth - wide, punchy with spatial dimension",
    "PunchyMix_depth": "PunchyMix + natural depth - tight punch with balanced spatial enhancement",
}


print("🔄 Hybrid Processing loaded!")
print("   • RadioReady + Depth: Radio clarity with subtle dimension")
print("   • Aggressive + Depth: Aggressive energy with dramatic space") 
print("   • PunchyMix + Depth: Tight punch with natural depth")
print("   → Best of both worlds: advanced processing + spatial enhancement!")