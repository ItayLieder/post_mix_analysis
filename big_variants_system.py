"""
BIG Variants System - Create multiple variants based on the amazing BIG processing approach.
Each variant uses the BIG processing foundation with different flavor profiles.
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

@dataclass
class BigVariantProfile:
    """Profile for BIG processing variants with different characteristics"""
    name: str
    description: str
    
    # EQ boost multipliers (1.0 = standard BIG processing)
    kick_boost_mult: float = 1.0
    snare_boost_mult: float = 1.0
    bass_boost_mult: float = 1.0
    vocal_presence_mult: float = 1.0
    air_boost_mult: float = 1.0
    
    # Stereo width multipliers
    drums_width_mult: float = 1.0
    vocals_width_mult: float = 1.0  
    music_width_mult: float = 1.0
    
    # Processing intensity
    parallel_compression_mult: float = 1.0
    harmonic_excitement_mult: float = 1.0
    dynamic_enhancement_mult: float = 1.0
    
    # Stem gain multipliers (applied to BIG gains)
    drums_gain_mult: float = 1.0
    bass_gain_mult: float = 1.0
    vocals_gain_mult: float = 1.0
    music_gain_mult: float = 1.0

# Define BIG variant profiles
BIG_VARIANTS = [
    # CORE BIG VARIANT - The original amazing sound
    BigVariantProfile(
        name="Standard_Mix",
        description="Standard balanced processing with full frequency enhancement",
        # All multipliers at 1.0 = standard BIG processing
    ),
    
    # REFERENCE VARIANT - Reference processing
    BigVariantProfile(
        name="Reference_Mix",
        description="Reference processing matching the original mix specifications",
        # All multipliers at 1.0 = baseline processing
    ),
    
    # ENHANCED BIG VARIANTS
    BigVariantProfile(
        name="Drum_Heavy", 
        description="BIG processing with even MORE powerful drums - for drum-focused tracks",
        kick_boost_mult=1.5,          # Even MORE kick
        snare_boost_mult=1.4,         # Even MORE snare
        drums_width_mult=1.3,         # Wider drums
        drums_gain_mult=1.2,          # Louder drums
        parallel_compression_mult=1.3, # More thickness
    ),
    
    BigVariantProfile(
        name="Bass_Heavy",
        description="BIG processing with EXTREME bass power - foundation-shaking low end",
        bass_boost_mult=1.6,          # EXTREME bass boosts
        kick_boost_mult=1.3,          # Support kick too
        bass_gain_mult=1.3,           # Much louder bass
        parallel_compression_mult=1.2, # Extra thickness for low end
    ),
    
    BigVariantProfile(
        name="Vocal_Forward", 
        description="BIG processing with commanding vocal presence - vocals cut through everything",
        vocal_presence_mult=1.4,      # HUGE vocal presence
        air_boost_mult=1.3,           # More vocal air
        vocals_width_mult=1.4,        # Wider vocals
        vocals_gain_mult=1.2,         # Louder vocals
        # Slightly reduce other elements to make room
        drums_gain_mult=0.9,
        music_gain_mult=0.9,
    ),
    
    BigVariantProfile(
        name="Wide_Stereo",
        description="BIG processing with EXTREME width - huge, cinematic soundstage", 
        drums_width_mult=1.5,         # Much wider drums
        vocals_width_mult=1.6,        # Much wider vocals  
        music_width_mult=1.4,         # Much wider music
        air_boost_mult=1.3,           # More air for spaciousness
        music_gain_mult=1.1,          # Slightly louder music for fullness
    ),
    
    BigVariantProfile(
        name="Radio_Ready",
        description="BIG processing optimized for radio - punchy, loud, and impressive",
        kick_boost_mult=1.2,          # Punchy kick
        snare_boost_mult=1.3,         # Punchy snare  
        vocal_presence_mult=1.2,      # Clear vocals
        parallel_compression_mult=1.4, # Extra punch
        dynamic_enhancement_mult=1.3,  # More punch
        # Balanced gains for radio consistency
    ),
    
    BigVariantProfile(
        name="Club_Energy", 
        description="BIG processing for club/EDM - massive low end with energy",
        kick_boost_mult=1.6,          # HUGE kick for club
        bass_boost_mult=1.5,          # MASSIVE bass
        snare_boost_mult=1.4,         # Punchy snare
        drums_gain_mult=1.3,          # Louder drums
        bass_gain_mult=1.4,           # Much louder bass
        parallel_compression_mult=1.3, # Club thickness
        # Reduce vocals slightly for instrumental focus
        vocals_gain_mult=0.8,
    ),
    
    BigVariantProfile(
        name="Modern_Pop",
        description="BIG processing for modern pop - balanced power with vocal focus",
        vocal_presence_mult=1.2,      # Clear pop vocals
        air_boost_mult=1.2,           # Modern air
        kick_boost_mult=1.1,          # Solid but not overpowering kick
        vocals_width_mult=1.2,        # Wider vocals
        music_width_mult=1.2,         # Wider music
        vocals_gain_mult=1.1,         # Slightly forward vocals
    ),
    
    BigVariantProfile(
        name="Rock_Power",
        description="BIG processing for rock - aggressive, powerful, and driving",
        kick_boost_mult=1.3,          # Powerful rock kick
        snare_boost_mult=1.5,         # HUGE rock snare
        bass_boost_mult=1.2,          # Solid rock bass
        drums_gain_mult=1.2,          # Louder drums
        parallel_compression_mult=1.4, # Rock aggression
        dynamic_enhancement_mult=1.4,  # Rock punch
        harmonic_excitement_mult=1.2,  # Rock saturation
    ),
    
    BigVariantProfile(
        name="Intimate_Power",
        description="BIG processing with intimate feel but powerful impact - best of both worlds",
        vocal_presence_mult=1.1,      # Clear but intimate vocals
        air_boost_mult=1.1,           # Gentle air
        # Reduce stereo width for intimacy
        drums_width_mult=0.8,
        vocals_width_mult=0.9, 
        music_width_mult=0.9,
        # But keep the power
        kick_boost_mult=1.1,
        bass_boost_mult=1.1,
        parallel_compression_mult=1.1,
    ),
    
    BigVariantProfile(
        name="Maximum_Impact",
        description="BIG processing pushed to the absolute maximum - most impressive possible",
        # EVERYTHING cranked up!
        kick_boost_mult=1.8,
        snare_boost_mult=1.6, 
        bass_boost_mult=1.7,
        vocal_presence_mult=1.5,
        air_boost_mult=1.4,
        drums_width_mult=1.6,
        vocals_width_mult=1.5,
        music_width_mult=1.5,
        parallel_compression_mult=1.5,
        harmonic_excitement_mult=1.3,
        dynamic_enhancement_mult=1.5,
        # Extreme gains
        drums_gain_mult=1.3,
        bass_gain_mult=1.4,  
        vocals_gain_mult=1.3,
        music_gain_mult=1.2,
    ),
]

def get_big_stem_combinations() -> List[Tuple[str, str]]:
    """Get all BIG variant combinations for the pipeline"""
    combinations = []
    for variant in BIG_VARIANTS:
        combinations.append((variant.name, f"big:{variant.name}"))
    return combinations

def get_big_variant_profile(variant_name: str) -> BigVariantProfile:
    """Get the profile for a specific BIG variant"""
    for variant in BIG_VARIANTS:
        if variant.name == variant_name:
            return variant
    # Fallback to original BIG
    return BIG_VARIANTS[0]  # Standard_Mix

def apply_big_variant_processing(stem_type: str, audio: np.ndarray, sample_rate: int, 
                               variant_profile: BigVariantProfile) -> np.ndarray:
    """
    Apply BIG processing with variant-specific modifications.
    This takes the amazing BIG processing and applies the variant multipliers.
    Now respects CONFIG.pipeline.stem_gains for user control!
    """
    from dsp_premitives import peaking_eq, shelf_filter, compressor, stereo_widener, apply_gain_db
    from config import CONFIG
    
    processed = audio.copy()
    
    # FIRST: Apply overall stem gain from CONFIG (this is what the user actually wants!)
    stem_gain = CONFIG.pipeline.get_stem_gains().get(stem_type, 1.0)
    if stem_gain != 1.0:
        gain_db = 20 * np.log10(max(0.001, stem_gain))  # Convert to dB, prevent log(0)
        processed = apply_gain_db(processed, gain_db)
        print(f"      🎚️ Applied {stem_gain}x gain ({gain_db:.1f} dB) to {stem_type}")
    
    try:
        if stem_type == 'drums':
            print(f"      🥁 Drums: {variant_profile.name} processing")
            
            # Get drums gain from CONFIG (user controllable!)
            base_drums_gain = CONFIG.pipeline.get_stem_gains().get('drums', 3.0)
            
            # Scale base processing with CONFIG gain and variant multipliers
            kick_gain = (base_drums_gain * 1.17) * variant_profile.kick_boost_mult  # was 3.5
            snare_gain = (base_drums_gain * 1.33) * variant_profile.snare_boost_mult  # was 4.0
            
            processed = peaking_eq(processed, sample_rate, f0=50, gain_db=kick_gain, Q=1.2)
            processed = peaking_eq(processed, sample_rate, f0=80, gain_db=(base_drums_gain * 0.83) * variant_profile.kick_boost_mult, Q=0.8)  # was 2.5
            processed = peaking_eq(processed, sample_rate, f0=200, gain_db=1.5, Q=1.0)
            processed = peaking_eq(processed, sample_rate, f0=3500, gain_db=snare_gain, Q=1.2)
            processed = peaking_eq(processed, sample_rate, f0=6000, gain_db=(base_drums_gain * 0.83) * variant_profile.snare_boost_mult, Q=0.8)  # was 2.5
            
            # Sub and highs
            processed = peaking_eq(processed, sample_rate, f0=35, gain_db=(base_drums_gain * 0.93) * variant_profile.kick_boost_mult, Q=1.5)  # was 2.8
            processed = peaking_eq(processed, sample_rate, f0=10000, gain_db=3.0, Q=0.6)
            processed = peaking_eq(processed, sample_rate, f0=15000, gain_db=2.0, Q=0.4)
            
            # Variant stereo width
            if audio.ndim == 2:
                width = 1.6 * variant_profile.drums_width_mult
                width = min(width, 2.0)  # Safety limit
                processed = stereo_widener(processed, width=width)
                
        elif stem_type == 'bass':
            print(f"      🎸 Bass: {variant_profile.name} processing")
            
            # Get bass gain from CONFIG (user controllable!)
            base_bass_gain = CONFIG.pipeline.get_stem_gains().get('bass', 2.8)
            
            # Scale base processing with CONFIG gain and variant multipliers
            sub_gain = (base_bass_gain * 1.61) * variant_profile.bass_boost_mult  # was 4.5
            fundamental_gain = (base_bass_gain * 1.36) * variant_profile.bass_boost_mult  # was 3.8
            
            processed = peaking_eq(processed, sample_rate, f0=35, gain_db=sub_gain, Q=1.8)
            processed = peaking_eq(processed, sample_rate, f0=60, gain_db=fundamental_gain, Q=1.2)
            processed = peaking_eq(processed, sample_rate, f0=100, gain_db=(base_bass_gain * 0.89) * variant_profile.bass_boost_mult, Q=1.0)  # was 2.5
            
            # Definition
            processed = peaking_eq(processed, sample_rate, f0=800, gain_db=2.0, Q=1.0)
            processed = peaking_eq(processed, sample_rate, f0=1500, gain_db=1.5, Q=0.8)
            processed = peaking_eq(processed, sample_rate, f0=2500, gain_db=1.0, Q=0.6)
            processed = peaking_eq(processed, sample_rate, f0=250, gain_db=-1.0, Q=2.0)
            
        elif stem_type == 'vocals':
            print(f"      🎤 Vocals: {variant_profile.name} processing")
            
            # Get vocals gain from CONFIG (user controllable!)
            base_vocals_gain = CONFIG.pipeline.get_stem_gains().get('vocals', 4.0)
            
            # Scale base processing with CONFIG gain and variant multipliers
            presence_gain = (base_vocals_gain * 1.13) * variant_profile.vocal_presence_mult  # was 4.5
            air_gain = (base_vocals_gain * 0.88) * variant_profile.air_boost_mult  # was 3.5
            
            processed = peaking_eq(processed, sample_rate, f0=1200, gain_db=2.5, Q=0.8)
            processed = peaking_eq(processed, sample_rate, f0=2800, gain_db=presence_gain, Q=1.0)
            processed = peaking_eq(processed, sample_rate, f0=4200, gain_db=(base_vocals_gain * 0.75) * variant_profile.vocal_presence_mult, Q=0.8)  # was 3.0
            
            # Air and body
            processed = peaking_eq(processed, sample_rate, f0=200, gain_db=1.8, Q=0.8)
            processed = peaking_eq(processed, sample_rate, f0=8000, gain_db=air_gain, Q=0.6)
            processed = peaking_eq(processed, sample_rate, f0=12000, gain_db=(base_vocals_gain * 0.63) * variant_profile.air_boost_mult, Q=0.4)  # was 2.5
            processed = peaking_eq(processed, sample_rate, f0=6000, gain_db=-0.8, Q=2.5)
            
            # Variant stereo width  
            if audio.ndim == 2:
                width = 1.3 * variant_profile.vocals_width_mult
                width = min(width, 2.0)  # Safety limit
                processed = stereo_widener(processed, width=width)
                
        elif stem_type == 'music':
            print(f"      🎵 Music: {variant_profile.name} processing")
            
            # Get music gain from CONFIG (user controllable!)
            base_music_gain = CONFIG.pipeline.get_stem_gains().get('music', 2.0)
            
            # Scale base processing with CONFIG gain and variant multipliers  
            air_mult = variant_profile.air_boost_mult
            
            processed = shelf_filter(processed, sample_rate, cutoff_hz=80, gain_db=2.0, kind='low')
            processed = peaking_eq(processed, sample_rate, f0=150, gain_db=1.5, Q=0.8)
            processed = peaking_eq(processed, sample_rate, f0=2000, gain_db=2.0, Q=0.7)
            processed = peaking_eq(processed, sample_rate, f0=6000, gain_db=1.8, Q=0.6)
            processed = shelf_filter(processed, sample_rate, cutoff_hz=8000, gain_db=(base_music_gain * 1.4) * air_mult, kind='high')  # was 2.8
            processed = peaking_eq(processed, sample_rate, f0=15000, gain_db=(base_music_gain * 1.0) * air_mult, Q=0.4)  # was 2.0
            
            # Variant stereo width
            if audio.ndim == 2:
                width = 1.8 * variant_profile.music_width_mult
                width = min(width, 2.5)  # Safety limit
                processed = stereo_widener(processed, width=width)
        
        # Universal enhancements with variant multipliers
        
        # Parallel compression
        comp_ratio = 8.0 * variant_profile.parallel_compression_mult
        comp_ratio = min(comp_ratio, 12.0)  # Safety limit
        compressed = compressor(processed, sample_rate, threshold_db=-25, ratio=comp_ratio,
                              attack_ms=1.0, release_ms=50.0, makeup_db=6.0)
        processed = processed * 0.8 + compressed * (0.2 * variant_profile.parallel_compression_mult)
        
        # Harmonic excitement
        excitement_amount = 0.15 * variant_profile.harmonic_excitement_mult
        harmonic_content = np.tanh(processed * 1.5) * excitement_amount
        processed = processed + harmonic_content * 0.3
        
        # Dynamic enhancement
        if variant_profile.dynamic_enhancement_mult != 1.0:
            envelope = np.abs(processed)
            if processed.ndim == 2:
                envelope = np.mean(envelope, axis=1, keepdims=True)
                
            enhancement_strength = 1.2 * variant_profile.dynamic_enhancement_mult
            enhancement = np.where(envelope > np.percentile(envelope, 70), enhancement_strength, 0.95)
            if processed.ndim == 2 and enhancement.ndim == 2:
                processed = processed * enhancement
        
        # Safety checks
        peak_before = np.max(np.abs(audio))
        peak_after = np.max(np.abs(processed))
        
        if peak_after > peak_before * 8:  # Allow big changes for variants
            processed = audio + (processed - audio) * 0.7
            
    except Exception as e:
        print(f"        ⚠️ {variant_profile.name} processing failed for {stem_type}: {e}, using raw audio")
        return audio
        
    return processed

if __name__ == "__main__":
    print("🚀 BIG VARIANTS SYSTEM")
    print("=" * 50)
    print(f"Available BIG variants: {len(BIG_VARIANTS)}")
    print()
    for variant in BIG_VARIANTS:
        print(f"🎵 {variant.name}")
        print(f"   {variant.description}")
        print()
    
    print("🎯 Ready to integrate into pipeline!")