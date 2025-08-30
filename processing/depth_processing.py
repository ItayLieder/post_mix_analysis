#!/usr/bin/env python3
"""
Depth Processing - Add dimension and space to flat-sounding mixes
Focuses specifically on creating front-to-back depth perception
"""

import numpy as np
from typing import Dict, Tuple, Optional
from audio_utils import ensure_stereo, to_mono, db_to_linear, linear_to_db
from scipy import signal


def add_mix_depth(stems: Dict[str, np.ndarray], sr: int, variant: str = "natural") -> Dict[str, np.ndarray]:
    """
    Add depth to stems by positioning them at different distances from the listener.
    
    Variants:
    - natural: Realistic depth positioning
    - dramatic: Exaggerated depth for effect
    - intimate: Close positioning with subtle depth
    - stadium: Large space with distant elements
    - focused: Vocals forward, everything else back
    """
    
    depth_maps = {
        "natural": {
            # RADIOREADY-LIKE - barely any change, just slight depth impression
            "kick": 0.98,          # Almost neutral
            "snare": 0.99,         # Almost neutral
            "hats": 1.03,          # Tiny bit back
            "drums": 0.99,         # Almost neutral
            "bass": 0.96,          # Tiny bit forward
            "leadvocals": 0.94,    # Tiny bit forward
            "backvocals": 1.06,    # Tiny bit back
            "vocals": 0.94,        # Tiny bit forward
            "guitar": 1.02,        # Tiny bit back
            "keys": 1.04,          # Tiny bit back
            "strings": 1.08,       # Slightly back (minimal)
            "music": 1.03,         # Tiny bit back
        },
        "dramatic": {
            # RADIOREADY-LIKE "dramatic" - just slightly more contrast than natural
            "kick": 0.94,          # Just barely forward
            "snare": 0.96,
            "hats": 1.10,          # Just a bit back
            "bass": 0.92,          # Just barely forward
            "leadvocals": 0.90,    # Slightly forward but natural
            "backvocals": 1.15,    # Just a bit back
            "vocals": 0.90,
            "guitar": 1.08,
            "keys": 1.12,          # Just a bit back
            "strings": 1.18,       # Moderate but not extreme
            "music": 1.09,
        },
        "intimate": {
            # Everything very close together - RADIOREADY-LIKE with minimal differences
            "kick": 0.96,
            "snare": 0.98,
            "hats": 1.01,
            "bass": 0.94,
            "leadvocals": 0.92,    # Just slightly close
            "backvocals": 1.04,    # Barely back
            "vocals": 0.92,
            "guitar": 0.98,        # Almost neutral
            "keys": 1.01,
            "strings": 1.05,       # Just barely back
            "music": 0.99,
        },
        "stadium": {
            # Big space feeling - RADIOREADY-LIKE with just a hint of space
            "kick": 1.02,          # Barely pushed back
            "snare": 1.04,
            "hats": 1.10,
            "bass": 1.01,
            "leadvocals": 0.94,    # Just slightly forward
            "backvocals": 1.12,    # Just a bit back
            "vocals": 0.94,
            "guitar": 1.10,        # Just a bit back
            "keys": 1.12,
            "strings": 1.18,       # Moderate but not far
            "music": 1.09,
        },
        "focused": {
            # Vocal clarity - RADIOREADY-LIKE with just slight vocal emphasis
            "kick": 1.04,
            "snare": 1.06,
            "hats": 1.10,
            "bass": 1.04,
            "leadvocals": 0.92,    # Just slightly forward
            "backvocals": 1.10,    # Just slightly back
            "vocals": 0.92,
            "guitar": 1.09,        # Just a bit back
            "keys": 1.12,
            "strings": 1.15,
            "music": 1.08,
        }
    }
    
    positions = depth_maps.get(variant, depth_maps["natural"])
    processed = {}
    
    print(f"    🏞️ Adding depth: {variant} positioning")
    
    for stem_name, audio in stems.items():
        if stem_name in positions:
            distance = positions[stem_name]
            processed[stem_name] = apply_depth_positioning(audio, sr, distance, stem_name)
            print(f"      • {stem_name}: {distance:.1f}x distance")
        else:
            # Default medium distance for unknown stems
            processed[stem_name] = apply_depth_positioning(audio, sr, 1.0, stem_name)
    
    return processed


def apply_depth_positioning(audio: np.ndarray, sr: int, distance: float, stem_type: str) -> np.ndarray:
    """
    Apply depth cues to position audio at specified distance.
    
    Depth cues used:
    1. Volume attenuation (inverse square law)
    2. High frequency roll-off (air absorption)
    3. Early reflections (room ambience)
    4. Reverb/ambience amount
    5. Spectral tilt (distance = less highs)
    """
    from dsp_premitives import shelf_filter, lowpass_filter
    
    stereo = ensure_stereo(audio)
    processed = stereo.copy()
    
    # 1. Volume attenuation - RADIOREADY-LIKE (almost no volume change)
    # Barely any volume difference to maintain balance like RadioReady
    volume_scale = 1.0 / (distance ** 0.15)  # Extremely gentle (was 0.3)
    processed *= volume_scale
    
    # 2. High frequency roll-off - RADIOREADY-LIKE (barely any filtering)
    if distance > 1.4:  # Start much later
        # Very subtle high-frequency change like RadioReady
        cutoff_freq = 20000 / (distance ** 0.3)  # Extremely gradual
        cutoff_freq = max(cutoff_freq, 12000)  # Keep almost all highs
        processed = lowpass_filter(processed, sr, cutoff_freq, order=1)  # Very gentle
    
    # 3. Spectral tilt for distance - RADIOREADY-LIKE (barely any EQ change)
    if distance > 1.25:  # Start much later - only for very distant elements
        # Barely any brightness change - like RadioReady's subtlety
        tilt_amount = (distance - 1.0) * -0.3  # Extremely gentle (was -0.8dB)
        processed = shelf_filter(processed, sr, 4000, tilt_amount, kind="high")
    elif distance < 0.9:  # Start later - only for very close elements
        # Barely any presence boost - maintain RadioReady balance
        presence_boost = (0.9 - distance) * 0.25  # Extremely gentle (was 0.6dB)
        processed = shelf_filter(processed, sr, 3000, presence_boost, kind="high")
    
    # 4. Add early reflections - RADIOREADY-LIKE (minimal reflections)
    if distance > 1.35:  # Start very late - only for clearly distant elements
        processed = add_early_reflections(processed, sr, distance, stem_type)
    
    # 5. Add subtle ambience/reverb - RADIOREADY-LIKE (minimal reverb)
    if distance > 1.4:  # Start very late - barely any reverb like RadioReady
        processed = add_depth_reverb(processed, sr, distance, stem_type)
    
    return processed.astype(np.float32)


def add_early_reflections(audio: np.ndarray, sr: int, distance: float, stem_type: str) -> np.ndarray:
    """Add early reflections to simulate room depth"""
    
    # Early reflection patterns depend on stem type and distance
    reflection_delays = []
    reflection_gains = []
    
    if stem_type in ["kick", "snare", "drums"]:
        # Drums get floor and wall reflections
        base_delay = 0.015 * distance  # 15ms base delay scaled by distance
        reflection_delays = [base_delay, base_delay * 1.7, base_delay * 2.3]
        reflection_gains = [0.15, 0.08, 0.05]
    elif stem_type in ["vocals", "leadvocals"]:
        # Vocals get subtle room reflections
        base_delay = 0.012 * distance
        reflection_delays = [base_delay, base_delay * 1.5]
        reflection_gains = [0.10, 0.06]
    elif stem_type in ["guitar", "keys", "strings", "music"]:
        # Instruments get diffuse reflections
        base_delay = 0.020 * distance
        reflection_delays = [base_delay, base_delay * 1.4, base_delay * 2.1, base_delay * 2.8]
        reflection_gains = [0.12, 0.08, 0.05, 0.03]
    else:
        # Default reflections
        base_delay = 0.015 * distance
        reflection_delays = [base_delay, base_delay * 1.6]
        reflection_gains = [0.12, 0.07]
    
    # Apply reflections
    result = audio.copy()
    
    for delay, gain in zip(reflection_delays, reflection_gains):
        delay_samples = int(delay * sr)
        if delay_samples > 0 and delay_samples < len(audio):
            # Create delayed version
            delayed = np.zeros_like(audio)
            delayed[delay_samples:] = audio[:-delay_samples]
            
            # Add slight filtering to reflection (walls absorb highs)
            from dsp_premitives import lowpass_filter
            delayed = lowpass_filter(delayed, sr, 12000, order=1)
            
            # Mix in reflection
            result += delayed * gain * (distance - 0.5)  # More reflection for distant sounds
    
    return result


def add_depth_reverb(audio: np.ndarray, sr: int, distance: float, stem_type: str) -> np.ndarray:
    """Add subtle reverb/ambience for depth impression"""
    
    # Different reverb characteristics - RADIOREADY-LIKE (barely any reverb)
    reverb_amount = min(0.04, (distance - 1.0) * 0.03)  # Max 4% wet - like RadioReady subtlety
    
    if reverb_amount <= 0:
        return audio
    
    # Simple reverb using multiple delays and filtering
    # This is a simplified reverb - production would use convolution
    
    delays = [0.030, 0.047, 0.071, 0.089, 0.107, 0.127]  # Prime number delays
    gains = [0.4, 0.35, 0.3, 0.25, 0.2, 0.15]
    
    reverb_sum = np.zeros_like(audio)
    
    for delay, gain in zip(delays, gains):
        delay_samples = int(delay * sr)
        if delay_samples < len(audio):
            # Create delayed version
            delayed = np.zeros_like(audio)
            delayed[delay_samples:] = audio[:-delay_samples]
            
            # Filter the delay (reverb gets darker over time)
            from dsp_premitives import lowpass_filter
            cutoff = 8000 / (gain * 2)  # Later delays are darker
            delayed = lowpass_filter(delayed, sr, cutoff, order=1)
            
            reverb_sum += delayed * gain
    
    # Normalize reverb
    if np.max(np.abs(reverb_sum)) > 0:
        reverb_sum = reverb_sum / np.max(np.abs(reverb_sum)) * 0.3
    
    # Mix dry and reverb
    dry_amount = 1.0 - reverb_amount
    result = audio * dry_amount + reverb_sum * reverb_amount
    
    return result


def enhance_stereo_depth(stems: Dict[str, np.ndarray], sr: int) -> Dict[str, np.ndarray]:
    """Enhance stereo depth using mid-side processing"""
    
    processed = {}
    
    for stem_name, audio in stems.items():
        stereo = ensure_stereo(audio)
        
        # Mid-side decomposition
        left, right = stereo[:, 0], stereo[:, 1]
        mid = (left + right) * 0.5
        side = (left - right) * 0.5
        
        # Enhance depth through side channel processing - RADIOREADY-LIKE
        if stem_name in ["backvocals", "strings", "keys", "hats"]:
            # Just a hint more stereo width - like RadioReady's subtlety
            side *= 1.03  # Barely perceptible (was 1.1)
        elif stem_name in ["kick", "snare", "bass", "leadvocals"]:
            # Keep centered like RadioReady - almost no change
            side *= 0.99  # Almost neutral (was 0.97)
        
        # Subtle delay between mid and side for depth
        if len(side) > 20:
            side_delayed = np.zeros_like(side)
            side_delayed[20:] = side[:-20]  # 0.5ms delay at 44.1kHz
            side = side * 0.7 + side_delayed * 0.3
        
        # Reconstruct stereo
        new_left = mid + side
        new_right = mid - side
        
        processed[stem_name] = np.column_stack([new_left, new_right]).astype(np.float32)
    
    return processed


def add_haas_effect_depth(stems: Dict[str, np.ndarray], sr: int) -> Dict[str, np.ndarray]:
    """Use Haas effect (precedence effect) to create depth"""
    
    processed = {}
    
    for stem_name, audio in stems.items():
        stereo = ensure_stereo(audio)
        
        # Apply Haas effect for depth - RADIOREADY-LIKE
        if stem_name in ["leadvocals", "vocals"]:
            # Vocals: ultra-minimal delay like RadioReady's balance preservation
            delay_samples = int(0.0002 * sr)  # 0.2ms - barely detectable
            if delay_samples > 0:
                stereo[:, 1] = np.pad(stereo[:, 1], (delay_samples, 0), mode='constant')[:len(stereo)]
                stereo[:, 1] *= 0.995  # Almost no level difference
        
        elif stem_name in ["backvocals", "strings"]:
            # Background elements: ultra-minimal delay - RadioReady-like balance
            delay_samples = int(0.0003 * sr)  # 0.3ms - barely detectable
            if delay_samples > 0:
                stereo[:, 0] = np.pad(stereo[:, 0], (delay_samples, 0), mode='constant')[:len(stereo)]
                stereo[:, 0] *= 0.99  # Almost no level difference
        
        elif stem_name in ["hats", "keys"]:
            # High frequency elements: RadioReady-like minimal effect
            delay_samples = int(0.0001 * sr)  # 0.1ms - almost imperceptible
            if delay_samples > 0:
                # Alternate which channel gets delayed - almost no effect
                if hash(stem_name) % 2 == 0:
                    stereo[:, 1] = np.pad(stereo[:, 1], (delay_samples, 0), mode='constant')[:len(stereo)]
                    stereo[:, 1] *= 0.998  # Almost no level difference
                else:
                    stereo[:, 0] = np.pad(stereo[:, 0], (delay_samples, 0), mode='constant')[:len(stereo)]
                    stereo[:, 0] *= 0.998  # Almost no level difference
        
        processed[stem_name] = stereo
    
    return processed


def create_depth_variant(stems: Dict[str, np.ndarray], sr: int, depth_style: str = "natural") -> Dict[str, np.ndarray]:
    """
    Complete depth processing pipeline with safety checks.
    
    Args:
        depth_style: "natural", "dramatic", "intimate", "stadium", "focused"
    """
    
    print(f"    🏞️ Creating depth variant: {depth_style}")
    
    try:
        # Stage 1: Apply distance positioning (main depth effect)
        processed = add_mix_depth(stems, sr, depth_style)
        processed = _sanitize_depth_stems(processed)
        print(f"      ✓ Applied distance positioning")
        
        # Stage 2: Enhance stereo depth
        processed = enhance_stereo_depth(processed, sr)
        processed = _sanitize_depth_stems(processed)
        print(f"      ✓ Enhanced stereo width for depth")
        
        # Stage 3: Add Haas effect for front-back positioning
        processed = add_haas_effect_depth(processed, sr)
        processed = _sanitize_depth_stems(processed)
        print(f"      ✓ Applied Haas effect positioning")
        
    except Exception as e:
        print(f"      ⚠️ Depth processing error: {e}")
        processed = _sanitize_depth_stems(stems)
    
    return processed


def _sanitize_depth_stems(stems: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Sanitize depth processed stems"""
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


# Pre-configured depth variants for easy use
DEPTH_VARIANTS = {
    "Depth_Natural": "natural",      # Realistic depth positioning
    "Depth_Dramatic": "dramatic",    # Exaggerated depth for effect
    "Depth_Intimate": "intimate",    # Close and personal
    "Depth_Stadium": "stadium",      # Large space feeling
    "Depth_VocalFocus": "focused",   # Vocals forward, rest back
}


print("🏞️ Depth Processing loaded!")
print("   • 5 depth positioning modes")  
print("   • Distance-based volume/EQ changes")
print("   • Early reflections simulation")
print("   • Subtle reverb for ambience") 
print("   • Mid-side depth enhancement")
print("   • Haas effect positioning")
print("   → Transforms flat mixes into dimensional soundscapes!")