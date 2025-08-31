"""
EMERGENCY: Bypass ALL processing and just do raw stem summing
This will test if the raw stems sound good when just balanced and summed
"""
import numpy as np
import soundfile as sf
import os
from config import CONFIG

def create_raw_stem_mix():
    """Create a mix using ONLY raw stems + balancing, NO processing"""
    
    print("🚨 EMERGENCY: BYPASSING ALL PROCESSING!")
    print("📊 Testing raw stem summing only...")
    print("=" * 50)
    
    # Raw stem paths
    stem_paths = {
        'drums': '/Users/itay/Documents/post_mix_data/Reference_mix/stems/drums.wav',
        'bass': '/Users/itay/Documents/post_mix_data/Reference_mix/stems/bass.wav', 
        'vocals': '/Users/itay/Documents/post_mix_data/Reference_mix/stems/vocals.wav',
        'music': '/Users/itay/Documents/post_mix_data/Reference_mix/stems/music.wav'
    }
    
    # Use extreme stem gains
    stem_gains = {
        'drums': 2.5,   # EXTREME drums boost
        'bass': 2.2,    # EXTREME bass boost  
        'vocals': 3.5,  # EXTREME vocals boost
        'music': 1.5    # Strong music boost
    }
    
    # Load all stems
    stems = {}
    max_length = 0
    sample_rate = None
    
    for stem_type, path in stem_paths.items():
        if os.path.exists(path):
            audio, sr = sf.read(path)
            stems[stem_type] = audio
            max_length = max(max_length, len(audio))
            if sample_rate is None:
                sample_rate = sr
            print(f"✓ Loaded {stem_type}: {len(audio)/sr:.1f}s, peak={np.max(np.abs(audio)):.3f}")
        else:
            print(f"❌ Missing: {stem_type}")
    
    if not stems:
        print("❌ No stems found!")
        return None
        
    # Initialize mix
    final_mix = np.zeros((max_length, 2), dtype=np.float32)
    
    print(f"\n🔄 RAW STEM SUMMING (NO PROCESSING):")
    
    # Sum stems with only gain adjustments - NO OTHER PROCESSING
    for stem_type, audio in stems.items():
        gain = stem_gains.get(stem_type, 1.0)
        
        # Pad if needed
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
            
        # Apply ONLY gain - no EQ, no compression, no filtering
        final_mix += audio * gain
        print(f"  {stem_type}: gain={gain}x, peak after={np.max(np.abs(audio * gain)):.3f}")
    
    # Check result
    peak = np.max(np.abs(final_mix))
    rms = np.sqrt(np.mean(final_mix**2))
    peak_db = 20*np.log10(peak) if peak > 0 else -100
    rms_db = 20*np.log10(rms) if rms > 0 else -100
    
    print(f"\n📊 RAW MIX RESULTS:")
    print(f"  Peak: {peak:.4f} ({peak_db:.1f} dBFS)")
    print(f"  RMS:  {rms:.4f} ({rms_db:.1f} dBFS)")
    
    # Light safety limiting if needed
    if peak > 0.95:
        safety_gain = 0.9 / peak
        final_mix *= safety_gain
        print(f"  Applied safety gain: {20*np.log10(safety_gain):.1f} dB")
    
    # Save result
    output_path = "/Users/itay/Documents/post_mix_data/EMERGENCY_RAW_STEM_MIX.wav"
    sf.write(output_path, final_mix, sample_rate, subtype="PCM_24")
    
    print(f"\n✅ RAW STEM MIX SAVED:")
    print(f"📁 {output_path}")
    print(f"🎵 This is ONLY balanced raw stems - NO processing whatsoever")
    print(f"🧪 Test this to see if raw stems sound good before processing")
    
    return output_path

if __name__ == "__main__":
    create_raw_stem_mix()