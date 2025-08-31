"""
Enhanced minimal processing - gentle improvements that preserve quality.
Only adds proven, safe enhancements that make stems sound more professional.
"""
import numpy as np
import soundfile as sf
from dsp_premitives import peaking_eq, shelf_filter, compressor, stereo_widener

def apply_enhanced_minimal_stem_processing(stem_type: str, audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Enhanced minimal processing with additional gentle improvements:
    1. Gentle stereo widening for spaciousness
    2. Very light compression for cohesion (only if needed)
    3. Subtle harmonic enhancement frequencies
    4. Gentle transient preservation
    
    All processing is conservative and has fallbacks to preserve quality.
    """
    
    # Start with original audio
    processed = audio.copy()
    
    try:
        if stem_type == 'drums':
            print(f"    🥁 Drums: Enhanced minimal processing")
            # Core minimal processing
            processed = peaking_eq(processed, sample_rate, f0=60, gain_db=1.5, Q=0.8)    # Kick foundation
            processed = peaking_eq(processed, sample_rate, f0=3000, gain_db=1.2, Q=1.0)  # Snare presence
            
            # SAFE ENHANCEMENTS:
            # 1. Gentle sub-harmonic enhancement for deeper kick
            processed = peaking_eq(processed, sample_rate, f0=45, gain_db=0.8, Q=1.2)    # Sub-kick
            # 2. Slight attack enhancement for punch
            processed = peaking_eq(processed, sample_rate, f0=5000, gain_db=0.6, Q=0.8)  # Attack clarity
            # 3. Very gentle stereo widening for spaciousness
            if audio.ndim == 2:
                processed = stereo_widener(processed, width=1.15)  # Very subtle widening
                
        elif stem_type == 'bass':
            print(f"    🎸 Bass: Enhanced minimal processing")
            # Core minimal processing  
            processed = shelf_filter(processed, sample_rate, cutoff_hz=80, gain_db=1.0, kind='low')
            processed = peaking_eq(processed, sample_rate, f0=150, gain_db=0.8, Q=1.0)
            
            # SAFE ENHANCEMENTS:
            # 1. Gentle sub-bass enhancement for depth
            processed = peaking_eq(processed, sample_rate, f0=35, gain_db=1.0, Q=1.5)    # Deep sub
            # 2. Slight string/pick definition
            processed = peaking_eq(processed, sample_rate, f0=1200, gain_db=0.5, Q=0.8)  # String definition
            # 3. Keep bass mono below 100Hz for focus
            if audio.ndim == 2:
                # Simple bass-focused processing - keep low end centered
                pass  # Bass stays natural stereo
                
        elif stem_type == 'vocals':
            print(f"    🎤 Vocals: Enhanced minimal processing") 
            # Core minimal processing
            processed = peaking_eq(processed, sample_rate, f0=2500, gain_db=1.5, Q=0.8)
            processed = shelf_filter(processed, sample_rate, cutoff_hz=8000, gain_db=1.0, kind='high')
            
            # SAFE ENHANCEMENTS:
            # 1. Gentle de-essing region control
            processed = peaking_eq(processed, sample_rate, f0=6500, gain_db=-0.5, Q=2.0)  # Gentle de-ess
            # 2. Warmth enhancement 
            processed = peaking_eq(processed, sample_rate, f0=400, gain_db=0.3, Q=0.6)   # Gentle warmth
            # 3. Very light compression for consistency (only if dynamic range is huge)
            peak = np.max(np.abs(processed))
            rms = np.sqrt(np.mean(processed**2))
            if peak > 0 and rms > 0:
                dynamic_range = 20 * np.log10(peak / rms)
                if dynamic_range > 25:  # Only compress if very dynamic
                    processed = compressor(processed, sample_rate, threshold=-20, ratio=2.0, 
                                         attack=10.0, release=100.0, makeup_gain=1.0)
                    
        elif stem_type == 'music':
            print(f"    🎵 Music: Enhanced minimal processing")
            # Core minimal processing
            processed = shelf_filter(processed, sample_rate, cutoff_hz=100, gain_db=0.5, kind='low')
            processed = shelf_filter(processed, sample_rate, cutoff_hz=10000, gain_db=0.8, kind='high')
            
            # SAFE ENHANCEMENTS:
            # 1. Gentle midrange clarity
            processed = peaking_eq(processed, sample_rate, f0=2000, gain_db=0.4, Q=0.7)   # Clarity
            # 2. Subtle stereo enhancement for width
            if audio.ndim == 2:
                processed = stereo_widener(processed, width=1.2)  # Gentle widening
            # 3. Very gentle "air" enhancement
            processed = peaking_eq(processed, sample_rate, f0=12000, gain_db=0.3, Q=0.5)  # Air
        
        # UNIVERSAL SAFE ENHANCEMENTS for all stems:
        
        # 1. Gentle saturation/warmth (very subtle)
        # Simple soft-clipping for harmonic warmth
        saturation_amount = 0.05  # Very gentle
        processed = processed + saturation_amount * np.tanh(processed * 2) * 0.1
        
        # 2. Safety checks - if processing changed things too much, reduce or revert
        peak_before = np.max(np.abs(audio))
        peak_after = np.max(np.abs(processed))
        
        if peak_after > peak_before * 2.5:  # Conservative check
            print(f"      ⚠️ Processing too aggressive for {stem_type}, using gentler version")
            # Reduce processing strength by 50%
            processed = audio + (processed - audio) * 0.5
            
        if peak_after > peak_before * 4:  # Extreme check
            print(f"      ⚠️ Processing failed for {stem_type}, using raw audio")
            return audio
            
        # 3. Final gentle limiting only if clipping
        final_peak = np.max(np.abs(processed))
        if final_peak > 0.98:
            processed = processed * (0.95 / final_peak)
            
    except Exception as e:
        print(f"      ⚠️ Enhanced processing failed for {stem_type}: {e}, using raw audio")
        return audio
        
    return processed

def compare_processing_approaches():
    """Test different processing levels on the same stems"""
    
    stem_paths = {
        'drums': '/Users/itay/Documents/post_mix_data/Reference_mix/stems/drums.wav',
        'vocals': '/Users/itay/Documents/post_mix_data/Reference_mix/stems/vocals.wav'
    }
    
    print("🧪 COMPARING PROCESSING APPROACHES")
    print("=" * 50)
    
    for stem_type, path in stem_paths.items():
        if os.path.exists(path):
            print(f"\n{stem_type.upper()} COMPARISON:")
            
            # Load raw
            raw_audio, sr = sf.read(path)
            raw_rms = np.sqrt(np.mean(raw_audio**2))
            print(f"  Raw RMS: {20*np.log10(raw_rms):.1f} dBFS")
            
            # Test minimal processing
            from minimal_stem_processing import apply_minimal_stem_processing
            minimal = apply_minimal_stem_processing(stem_type, raw_audio, sr)
            minimal_rms = np.sqrt(np.mean(minimal**2))
            minimal_change = 20*np.log10(minimal_rms/raw_rms) if raw_rms > 0 else 0
            print(f"  Minimal RMS: {20*np.log10(minimal_rms):.1f} dBFS ({minimal_change:+.1f} dB)")
            
            # Test enhanced minimal
            enhanced = apply_enhanced_minimal_stem_processing(stem_type, raw_audio, sr)
            enhanced_rms = np.sqrt(np.mean(enhanced**2))
            enhanced_change = 20*np.log10(enhanced_rms/raw_rms) if raw_rms > 0 else 0
            print(f"  Enhanced RMS: {20*np.log10(enhanced_rms):.1f} dBFS ({enhanced_change:+.1f} dB)")
            
            # Save for comparison
            base_name = f"/Users/itay/Documents/post_mix_data/{stem_type}_processing_comparison"
            sf.write(f"{base_name}_minimal.wav", minimal, sr, subtype="PCM_24")
            sf.write(f"{base_name}_enhanced.wav", enhanced, sr, subtype="PCM_24")
            print(f"  💾 Saved comparison files for listening test")

if __name__ == "__main__":
    import os
    compare_processing_approaches()