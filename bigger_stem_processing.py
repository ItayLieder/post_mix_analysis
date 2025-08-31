"""
BIGGER stem processing - Make stems sound more impressive, powerful, and grand.
Focus on SCALE, IMPACT, and PRESENCE rather than subtle polish.
"""
import numpy as np
import soundfile as sf
from dsp_premitives import peaking_eq, shelf_filter, compressor, stereo_widener
import os

def apply_bigger_stem_processing(stem_type: str, audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Make stems sound BIGGER, more impressive, and more powerful:
    1. Aggressive frequency shaping for maximum impact
    2. Parallel processing for thickness and power
    3. Stereo enhancement for width and grandeur
    4. Dynamic enhancement for punch and presence
    
    This is about making things sound IMPRESSIVE and BIG!
    """
    
    # Start with original audio
    processed = audio.copy()
    
    try:
        if stem_type == 'drums':
            print(f"    🥁 Drums: BIG, POWERFUL processing")
            
            # CORE POWER - Make kick and snare HUGE
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
            print(f"    🎸 Bass: MASSIVE, foundation-shaking processing")
            
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
            print(f"    🎤 Vocals: HUGE, commanding presence")
            
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
            print(f"    🎵 Music: BIG, cinematic, impressive")
            
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
        # Create a heavily compressed version and blend it
        compressed = compressor(processed, sample_rate, threshold_db=-25, ratio=8.0, 
                              attack_ms=1.0, release_ms=50.0, makeup_db=6.0)
        
        # Blend 80% original + 20% heavily compressed for thickness
        processed = processed * 0.8 + compressed * 0.2
        
        # 2. HARMONIC EXCITEMENT for bigger sound
        # Gentle harmonic distortion for richness and power
        excitement_amount = 0.15  # More aggressive than minimal
        harmonic_content = np.tanh(processed * 1.5) * excitement_amount
        processed = processed + harmonic_content * 0.3
        
        # 3. DYNAMIC ENHANCEMENT - Make transients more impressive
        # Enhance the difference between loud and quiet parts
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
            print(f"      ⚠️ Processing very aggressive for {stem_type}, reducing by 25%")
            processed = audio + (processed - audio) * 0.75
            
        if peak_after > peak_before * 10:  # Extreme safety check
            print(f"      ⚠️ Processing too extreme for {stem_type}, using 50% blend")
            processed = audio * 0.5 + processed * 0.5
            
        # Final limiting for power (not safety - we want LOUD)
        final_peak = np.max(np.abs(processed))
        if final_peak > 0.95:
            processed = processed * (0.92 / final_peak)  # Keep it hot but not clipping
            
    except Exception as e:
        print(f"      ⚠️ BIG processing failed for {stem_type}: {e}, using raw audio")
        return audio
        
    return processed

def create_bigger_stem_mix():
    """Create stem mix using BIG, impressive processing"""
    
    print("🚀 BIG STEM PROCESSING PIPELINE")
    print("💥 MAXIMUM IMPACT, SCALE, AND PRESENCE!")
    print("=" * 60)
    
    # Raw stem paths
    stem_paths = {
        'drums': '/Users/itay/Documents/post_mix_data/Reference_mix/stems/drums.wav',
        'bass': '/Users/itay/Documents/post_mix_data/Reference_mix/stems/bass.wav', 
        'vocals': '/Users/itay/Documents/post_mix_data/Reference_mix/stems/vocals.wav',
        'music': '/Users/itay/Documents/post_mix_data/Reference_mix/stems/music.wav'
    }
    
    # Use EXTREME stem gains for HUGE sound
    stem_gains = {
        'drums': 3.0,   # HUGE drums
        'bass': 2.8,    # MASSIVE bass  
        'vocals': 4.0,  # COMMANDING vocals
        'music': 2.0    # BIG musical content
    }
    
    # Load and process stems for BIGNESS
    processed_stems = {}
    max_length = 0
    sample_rate = None
    
    print("🔄 Loading and applying BIG processing...")
    
    for stem_type, path in stem_paths.items():
        if os.path.exists(path):
            # Load raw stem
            raw_audio, sr = sf.read(path)
            if sample_rate is None:
                sample_rate = sr
                
            # Apply BIG processing
            processed_audio = apply_bigger_stem_processing(stem_type, raw_audio, sr)
            
            processed_stems[stem_type] = processed_audio
            max_length = max(max_length, len(processed_audio))
            
            # Show the impressive changes
            raw_rms = np.sqrt(np.mean(raw_audio**2))
            proc_rms = np.sqrt(np.mean(processed_audio**2))
            rms_change = 20*np.log10(proc_rms/raw_rms) if raw_rms > 0 else 0
            
            impact_level = '🚀🚀🚀' if rms_change > 5 else '🚀🚀' if rms_change > 3 else '🚀'
            print(f"      IMPACT: {rms_change:+.1f} dB change {impact_level}")
            
    # Sum with EXTREME balancing for HUGE sound
    print(f"\n🎚️ EXTREME stem summing for MAXIMUM IMPACT...")
    final_mix = np.zeros((max_length, 2), dtype=np.float32)
    
    for stem_type, audio in processed_stems.items():
        gain = stem_gains.get(stem_type, 1.0)
        
        # Ensure proper format
        if len(audio) < max_length:
            if audio.ndim == 1:
                padded = np.zeros((max_length, 2), dtype=np.float32)
                padded[:len(audio), :] = np.column_stack([audio, audio])
            else:
                padded = np.zeros((max_length, 2), dtype=np.float32) 
                padded[:len(audio), :] = audio
            audio = padded
        elif audio.ndim == 1:
            audio = np.column_stack([audio, audio])
            
        # Apply EXTREME gains for HUGE impact
        weighted_audio = audio * gain
        final_mix += weighted_audio
        
        peak = np.max(np.abs(weighted_audio))
        power_level = '💥💥💥' if gain > 3.5 else '💥💥' if gain > 2.5 else '💥'
        print(f"  {stem_type}: {gain}x gain {power_level}, peak={peak:.3f}")
    
    # Check the HUGE result
    peak = np.max(np.abs(final_mix))
    rms = np.sqrt(np.mean(final_mix**2))
    peak_db = 20*np.log10(peak) if peak > 0 else -100
    rms_db = 20*np.log10(rms) if rms > 0 else -100
    
    print(f"\n📊 BIG PROCESSING RESULTS:")
    print(f"  Peak: {peak:.4f} ({peak_db:.1f} dBFS) 💥")
    print(f"  RMS:  {rms:.4f} ({rms_db:.1f} dBFS) 🚀")
    
    # Aggressive but controlled limiting for MAXIMUM impact
    if peak > 0.90:
        powerful_gain = 0.85 / peak  # Keep it VERY hot
        final_mix *= powerful_gain
        print(f"  Applied POWER limiting: {20*np.log10(powerful_gain):.1f} dB")
    
    # Save the HUGE result
    output_path = "/Users/itay/Documents/post_mix_data/BIG_POWERFUL_STEM_MIX.wav"
    sf.write(output_path, final_mix, sample_rate, subtype="PCM_24")
    
    print(f"\n✅ BIG STEM MIX SAVED:")
    print(f"📁 {output_path}")
    print(f"🚀 HUGE, impressive, powerful processing!")
    print(f"💥 This should sound MUCH BIGGER and more impressive!")
    
    return output_path

if __name__ == "__main__":
    create_bigger_stem_mix()