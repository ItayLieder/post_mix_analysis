"""
MINIMAL stem processing that preserves audio quality.
Only applies gentle, musical processing that enhances rather than destroys.
"""
import numpy as np
import soundfile as sf
import os
from dsp_premitives import peaking_eq, shelf_filter, compressor
from config import CONFIG

def apply_minimal_stem_processing(stem_type, audio, sample_rate):
    """
    Apply MINIMAL, gentle processing that enhances without destroying.
    Only the most essential processing that actually improves the sound.
    """
    
    # Start with original audio
    processed = audio.copy()
    
    if stem_type == 'drums':
        print(f"    🥁 Drums: Gentle enhancement only")
        # ONLY gentle punch enhancement - no filtering!
        # Slight kick enhancement
        processed = peaking_eq(processed, sample_rate, f0=60, gain_db=1.5, Q=0.8)
        # Slight snare presence  
        processed = peaking_eq(processed, sample_rate, f0=3000, gain_db=1.2, Q=1.0)
        
    elif stem_type == 'bass':
        print(f"    🎸 Bass: Gentle low-end enhancement")
        # ONLY gentle low-end enhancement
        processed = shelf_filter(processed, sample_rate, cutoff_hz=80, gain_db=1.0, kind='low')
        # Slight definition in upper bass
        processed = peaking_eq(processed, sample_rate, f0=150, gain_db=0.8, Q=1.0)
        
    elif stem_type == 'vocals':
        print(f"    🎤 Vocals: Gentle presence boost")
        # ONLY gentle presence - no harsh processing
        processed = peaking_eq(processed, sample_rate, f0=2500, gain_db=1.5, Q=0.8)
        # Gentle air
        processed = shelf_filter(processed, sample_rate, cutoff_hz=8000, gain_db=1.0, kind='high')
        
    elif stem_type == 'music':
        print(f"    🎵 Music: Gentle polish")
        # ONLY gentle overall polish
        processed = shelf_filter(processed, sample_rate, cutoff_hz=100, gain_db=0.5, kind='low')
        processed = shelf_filter(processed, sample_rate, cutoff_hz=10000, gain_db=0.8, kind='high')
    
    # Check for artifacts
    peak_before = np.max(np.abs(audio))
    peak_after = np.max(np.abs(processed))
    
    if peak_after > peak_before * 3:  # Sanity check
        print(f"      ⚠️ Processing seems too aggressive, using raw audio")
        return audio
        
    return processed

def create_minimal_processed_stem_mix():
    """Create stem mix using MINIMAL processing that preserves quality"""
    
    print("🛠️  MINIMAL STEM PROCESSING PIPELINE")
    print("✨ Only gentle, musical enhancements - no destruction!")
    print("=" * 60)
    
    # Raw stem paths
    stem_paths = {
        'drums': '/Users/itay/Documents/post_mix_data/Reference_mix/stems/drums.wav',
        'bass': '/Users/itay/Documents/post_mix_data/Reference_mix/stems/bass.wav', 
        'vocals': '/Users/itay/Documents/post_mix_data/Reference_mix/stems/vocals.wav',
        'music': '/Users/itay/Documents/post_mix_data/Reference_mix/stems/music.wav'
    }
    
    # Use extreme stem gains (since they work)
    stem_gains = {
        'drums': 2.5,   # Powerful drums
        'bass': 2.2,    # Strong bass  
        'vocals': 3.5,  # Forward vocals
        'music': 1.5    # Supporting music
    }
    
    # Load and process stems
    processed_stems = {}
    max_length = 0
    sample_rate = None
    
    print("🔄 Loading and processing stems...")
    
    for stem_type, path in stem_paths.items():
        if os.path.exists(path):
            # Load raw stem
            raw_audio, sr = sf.read(path)
            if sample_rate is None:
                sample_rate = sr
                
            # Apply MINIMAL processing
            processed_audio = apply_minimal_stem_processing(stem_type, raw_audio, sr)
            
            processed_stems[stem_type] = processed_audio
            max_length = max(max_length, len(processed_audio))
            
            # Quality check
            raw_rms = np.sqrt(np.mean(raw_audio**2))
            proc_rms = np.sqrt(np.mean(processed_audio**2))
            rms_change = 20*np.log10(proc_rms/raw_rms) if raw_rms > 0 else 0
            
            print(f"      Quality: {rms_change:+.1f} dB RMS change (should be minimal)")
            
    # Sum with intelligent balancing
    print(f"\n🎚️ Intelligent stem summing...")
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
            
        # Apply gain and sum
        weighted_audio = audio * gain
        final_mix += weighted_audio
        
        peak = np.max(np.abs(weighted_audio))
        print(f"  {stem_type}: gain={gain}x, peak={peak:.3f}")
    
    # Gentle final limiting if needed (preserve dynamics)
    peak = np.max(np.abs(final_mix))
    rms = np.sqrt(np.mean(final_mix**2))
    peak_db = 20*np.log10(peak) if peak > 0 else -100
    rms_db = 20*np.log10(rms) if rms > 0 else -100
    
    print(f"\n📊 MINIMAL PROCESSING RESULTS:")
    print(f"  Peak: {peak:.4f} ({peak_db:.1f} dBFS)")
    print(f"  RMS:  {rms:.4f} ({rms_db:.1f} dBFS)")
    
    # Only gentle limiting if really needed
    if peak > 0.92:
        gentle_gain = 0.88 / peak  # Very conservative
        final_mix *= gentle_gain
        print(f"  Applied gentle limiting: {20*np.log10(gentle_gain):.1f} dB")
    
    # Save result
    output_path = "/Users/itay/Documents/post_mix_data/MINIMAL_PROCESSED_STEM_MIX.wav"
    sf.write(output_path, final_mix, sample_rate, subtype="PCM_24")
    
    print(f"\n✅ MINIMAL PROCESSED MIX SAVED:")
    print(f"📁 {output_path}")
    print(f"✨ Gentle processing that enhances without destroying")
    print(f"🎵 Should sound much better than heavy processing!")
    
    return output_path

if __name__ == "__main__":
    create_minimal_processed_stem_mix()