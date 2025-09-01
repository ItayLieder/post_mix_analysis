#!/usr/bin/env python3
"""
Subtle Depth Processing - Musical depth enhancement that respects your mix balance

This creates depth without going overboard - think "RadioReady with dimension"
"""

import numpy as np
from typing import Dict, Tuple, Optional
from utils import ensure_stereo, to_mono, db_to_linear, linear_to_db
from scipy import signal


def add_subtle_depth(stems: Dict[str, np.ndarray], sr: int, variant: str = "balanced") -> Dict[str, np.ndarray]:
    """
    Add subtle, musical depth that enhances rather than transforms the mix.
    
    Variants:
    - balanced: Gentle depth that maintains mix balance
    - vocal_forward: Slight vocal emphasis with natural backing
    - warm: Cozy depth with gentle warmth
    - clear: Clarity-focused with subtle separation
    - polished: Professional polish with dimension
    """
    
    # Much more conservative depth settings
    depth_maps = {
        "balanced": {
            # Very subtle distance differences - barely noticeable individually
            "kick": 0.85,          # Just slightly forward
            "snare": 0.90,         # Neutral
            "hats": 1.10,          # Just slightly back
            "drums": 0.90,         # Neutral
            "bass": 0.80,          # Slightly forward for impact
            "leadvocals": 0.75,    # A bit forward (not extreme)
            "backvocals": 1.20,    # Gently back
            "vocals": 0.75,        # A bit forward
            "guitar": 1.05,        # Slightly back
            "keys": 1.15,          # A bit back
            "strings": 1.30,       # Moderately back (not extreme)
            "music": 1.10,         # Slightly back
        },
        "vocal_forward": {
            # Vocals get a bit more presence, but nothing crazy
            "kick": 0.90,
            "snare": 0.95,
            "hats": 1.15,
            "bass": 0.85,
            "leadvocals": 0.65,    # More forward but not in-your-face
            "backvocals": 1.35,    # More contrast with lead
            "vocals": 0.65,
            "guitar": 1.15,
            "keys": 1.20,
            "strings": 1.40,
            "music": 1.15,
        },
        "warm": {
            # Cozy, intimate feel
            "kick": 0.80,
            "snare": 0.85,
            "hats": 1.00,
            "bass": 0.75,
            "leadvocals": 0.70,
            "backvocals": 1.10,
            "vocals": 0.70,
            "guitar": 0.95,        # Keep instruments closer
            "keys": 1.05,
            "strings": 1.20,       # Not too far
            "music": 1.00,
        },
        "clear": {
            # Subtle separation for clarity
            "kick": 0.85,
            "snare": 0.90,
            "hats": 1.15,
            "bass": 0.80,
            "leadvocals": 0.70,
            "backvocals": 1.25,
            "vocals": 0.70,
            "guitar": 1.10,
            "keys": 1.20,
            "strings": 1.35,
            "music": 1.15,
        },
        "polished": {
            # Professional dimension without being dramatic
            "kick": 0.82,
            "snare": 0.88,
            "hats": 1.12,
            "bass": 0.78,
            "leadvocals": 0.72,
            "backvocals": 1.28,
            "vocals": 0.72,
            "guitar": 1.08,
            "keys": 1.18,
            "strings": 1.32,
            "music": 1.12,
        }
    }
    
    positions = depth_maps.get(variant, depth_maps["balanced"])
    processed = {}
    
    print(f"    🎵 Adding subtle depth: {variant}")
    
    for stem_name, audio in stems.items():
        if stem_name in positions:
            distance = positions[stem_name]
            processed[stem_name] = apply_subtle_positioning(audio, sr, distance, stem_name)
        else:
            # Default very subtle positioning for unknown stems
            processed[stem_name] = apply_subtle_positioning(audio, sr, 1.0, stem_name)
    
    return processed


def apply_subtle_positioning(audio: np.ndarray, sr: int, distance: float, stem_type: str) -> np.ndarray:
    """
    Apply very subtle depth cues that enhance rather than transform.
    Much gentler than the extreme depth processing.
    """
    from dsp_premitives import shelf_filter
    
    stereo = ensure_stereo(audio)
    processed = stereo.copy()
    
    # 1. Very gentle volume changes (barely noticeable)
    # Much less aggressive than inverse square law
    volume_scale = 1.0 / (distance ** 0.3)  # Very gentle scaling
    volume_scale = np.clip(volume_scale, 0.85, 1.15)  # Limit to ±15%
    processed *= volume_scale
    
    # 2. Very subtle high frequency adjustments
    # Only for more extreme distance differences
    if distance > 1.3:
        # Gentle high-frequency reduction for distant elements
        tilt_amount = (distance - 1.3) * -1.0  # Max -1dB
        processed = shelf_filter(processed, sr, 8000, tilt_amount, kind="high")
    elif distance < 0.7:
        # Subtle presence boost for very close elements  
        presence_boost = (0.7 - distance) * 1.0  # Max +1dB
        processed = shelf_filter(processed, sr, 4000, presence_boost, kind="high")
    
    # 3. Minimal early reflections (only for clearly distant elements)
    if distance > 1.2:
        processed = add_gentle_reflections(processed, sr, distance, stem_type)
    
    return processed.astype(np.float32)


def add_gentle_reflections(audio: np.ndarray, sr: int, distance: float, stem_type: str) -> np.ndarray:
    """Add very subtle early reflections - barely noticeable"""
    
    # Much smaller and fewer reflections than extreme processing
    reflection_delay = 0.008 * (distance - 1.0)  # 8ms max for 2.0x distance
    reflection_gain = 0.08 * (distance - 1.0)    # Max 8% reflection
    
    # Cap the effect
    reflection_delay = min(reflection_delay, 0.015)  # Max 15ms
    reflection_gain = min(reflection_gain, 0.10)     # Max 10%
    
    delay_samples = int(reflection_delay * sr)
    
    if delay_samples > 0:
        # Single gentle reflection
        reflected = np.zeros_like(audio)
        reflected[delay_samples:] = audio[:-delay_samples]
        
        # Very gentle filtering (walls absorb some highs)
        from dsp_premitives import lowpass_filter
        reflected = lowpass_filter(reflected, sr, 15000, order=1)  # Gentle rolloff
        
        # Mix in very quietly
        audio = audio + reflected * reflection_gain
    
    return audio


def enhance_subtle_stereo_width(stems: Dict[str, np.ndarray], sr: int) -> Dict[str, np.ndarray]:
    """Very gentle stereo width enhancement for depth perception"""
    
    processed = {}
    
    for stem_name, audio in stems.items():
        stereo = ensure_stereo(audio)
        
        # Mid-side decomposition
        left, right = stereo[:, 0], stereo[:, 1]
        mid = (left + right) * 0.5
        side = (left - right) * 0.5
        
        # Very subtle side adjustments
        if stem_name in ["backvocals", "strings", "keys"]:
            # Gentle width increase for background elements
            side *= 1.08  # 8% increase (barely noticeable)
        elif stem_name in ["hats"]:
            # Slight width for hats
            side *= 1.05  # 5% increase
        # Lead elements stay unchanged
        
        # Reconstruct stereo
        new_left = mid + side
        new_right = mid - side
        
        processed[stem_name] = np.column_stack([new_left, new_right]).astype(np.float32)
    
    return processed


def add_gentle_haas_effect(stems: Dict[str, np.ndarray], sr: int) -> Dict[str, np.ndarray]:
    """Very subtle Haas effect - professional polish without weirdness"""
    
    processed = {}
    
    for stem_name, audio in stems.items():
        stereo = ensure_stereo(audio)
        
        # Much smaller delays than extreme processing
        if stem_name in ["leadvocals", "vocals"]:
            # Tiny delay for forward presence (0.5ms max)
            delay_samples = int(0.0005 * sr)  
            if delay_samples > 0:
                stereo[:, 1] = np.pad(stereo[:, 1], (delay_samples, 0), mode='constant')[:len(stereo)]
                stereo[:, 1] *= 0.98  # 2% quieter
                
        elif stem_name in ["backvocals"]:
            # Tiny delay for background sense (1ms max)
            delay_samples = int(0.001 * sr)
            if delay_samples > 0:
                stereo[:, 0] = np.pad(stereo[:, 0], (delay_samples, 0), mode='constant')[:len(stereo)]
                stereo[:, 0] *= 0.97  # 3% quieter
        
        processed[stem_name] = stereo
    
    return processed


def create_musical_depth(stems: Dict[str, np.ndarray], sr: int, depth_style: str = "balanced") -> Dict[str, np.ndarray]:
    """
    Create musical depth enhancement that sounds professional and natural.
    Think "RadioReady with dimension" not "experimental effects".
    """
    
    print(f"    🎵 Creating musical depth: {depth_style}")
    
    try:
        # Stage 1: Apply subtle distance positioning (main effect)
        processed = add_subtle_depth(stems, sr, depth_style)
        processed = _sanitize_musical_stems(processed)
        print(f"      ✓ Applied gentle positioning")
        
        # Stage 2: Very subtle stereo enhancement
        processed = enhance_subtle_stereo_width(processed, sr)
        processed = _sanitize_musical_stems(processed)
        print(f"      ✓ Enhanced stereo subtly")
        
        # Stage 3: Gentle Haas effect
        processed = add_gentle_haas_effect(processed, sr)
        processed = _sanitize_musical_stems(processed)
        print(f"      ✓ Applied gentle positioning cues")
        
    except Exception as e:
        print(f"      ⚠️ Depth processing error: {e}")
        processed = _sanitize_musical_stems(stems)
    
    return processed


def _sanitize_musical_stems(stems: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Sanitize with very conservative limits for musical processing"""
    sanitized = {}
    
    for stem_name, audio in stems.items():
        audio = np.asarray(audio, dtype=np.float32)
        
        # Replace NaN and Inf
        audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Very conservative clipping (musical range)
        audio = np.clip(audio, -1.5, 1.5)
        
        sanitized[stem_name] = audio.astype(np.float32)
    
    return sanitized


# Musical depth variants
MUSICAL_DEPTH_VARIANTS = {
    "Musical_Balanced": "balanced",      # Gentle depth maintaining balance
    "Musical_VocalForward": "vocal_forward",  # Subtle vocal emphasis  
    "Musical_Warm": "warm",              # Cozy intimate depth
    "Musical_Clear": "clear",            # Clarity-focused separation
    "Musical_Polished": "polished",      # Professional dimension
}


print("🎵 Musical Depth Processing loaded!")
print("   • 5 subtle depth modes")
print("   • Gentle positioning (max ±30% distance)")
print("   • Conservative EQ adjustments (max ±1dB)")
print("   • Minimal reflections (max 10% wet)")
print("   • Subtle stereo enhancement (max 8% width)")
print("   • Professional polish without weirdness")
print("   → Perfect for 'RadioReady with depth' sound!")