#!/usr/bin/env python3
"""Analyze the current mix to identify specific problems"""

import soundfile as sf
import numpy as np
from scipy import signal

def analyze_mix(file_path):
    """Analyze a mix file for common issues"""
    audio, sr = sf.read(file_path)
    
    if len(audio.shape) == 2:
        mono = np.mean(audio, axis=1)
    else:
        mono = audio
    
    # Frequency analysis
    freqs, psd = signal.welch(mono, sr, nperseg=4096)
    
    # Find frequency bands
    sub_bass = np.mean(psd[(freqs >= 20) & (freqs < 60)])
    bass = np.mean(psd[(freqs >= 60) & (freqs < 250)])
    low_mids = np.mean(psd[(freqs >= 250) & (freqs < 500)])
    mids = np.mean(psd[(freqs >= 500) & (freqs < 2000)])
    upper_mids = np.mean(psd[(freqs >= 2000) & (freqs < 4000)])
    highs = np.mean(psd[(freqs >= 4000) & (freqs < 8000)])
    air = np.mean(psd[(freqs >= 8000) & (freqs < 20000)])
    
    # Convert to dB
    def to_db(x): return 10 * np.log10(x + 1e-10)
    
    print(f"Analyzing: {file_path}")
    print("="*50)
    print("\nFREQUENCY BALANCE (relative levels):")
    print(f"  Sub-bass (20-60 Hz):    {to_db(sub_bass):.1f} dB")
    print(f"  Bass (60-250 Hz):       {to_db(bass):.1f} dB")
    print(f"  Low-mids (250-500 Hz):  {to_db(low_mids):.1f} dB")
    print(f"  Mids (500-2k Hz):       {to_db(mids):.1f} dB")
    print(f"  Upper-mids (2-4k Hz):   {to_db(upper_mids):.1f} dB")
    print(f"  Highs (4-8k Hz):        {to_db(highs):.1f} dB")
    print(f"  Air (8-20k Hz):         {to_db(air):.1f} dB")
    
    # Dynamic analysis
    peak = np.max(np.abs(audio))
    rms = np.sqrt(np.mean(audio**2))
    crest_factor = peak / (rms + 1e-10)
    
    print(f"\nDYNAMICS:")
    print(f"  Peak level:      {20*np.log10(peak):.1f} dBFS")
    print(f"  RMS level:       {20*np.log10(rms):.1f} dBFS")
    print(f"  Crest factor:    {20*np.log10(crest_factor):.1f} dB")
    
    # Identify problems
    print("\nPOTENTIAL ISSUES:")
    if to_db(low_mids) > to_db(mids) + 3:
        print("  ⚠️ Muddy low-mids (250-500 Hz too loud)")
    if to_db(upper_mids) > to_db(mids) + 6:
        print("  ⚠️ Harsh upper-mids (2-4k Hz too loud)")
    if to_db(air) < to_db(highs) - 6:
        print("  ⚠️ Lacks air/sparkle (8-20k Hz too quiet)")
    if crest_factor < 1.4:
        print("  ⚠️ Over-compressed (low dynamic range)")
    if peak > 0.99:
        print("  ⚠️ Hitting limiter hard (may cause distortion)")
    
    return {
        'sub_bass': to_db(sub_bass),
        'bass': to_db(bass),
        'low_mids': to_db(low_mids),
        'mids': to_db(mids),
        'upper_mids': to_db(upper_mids),
        'highs': to_db(highs),
        'air': to_db(air),
        'peak': 20*np.log10(peak),
        'rms': 20*np.log10(rms),
        'crest_factor': 20*np.log10(crest_factor)
    }

# Compare impressive vs current
print("\n" + "="*60)
print("IMPRESSIVE MIX (session_20250830_151135):")
print("="*60)
impressive = analyze_mix('/Users/itay/Documents/post_mix_data/mixing_sessions/session_20250830_151135/full_mix.wav')

print("\n" + "="*60)
print("LATEST MIX (session_20250830_172016):")
print("="*60)
latest = analyze_mix('/Users/itay/Documents/post_mix_data/mixing_sessions/session_20250830_172016/full_mix.wav')

print("\n" + "="*60)
print("COMPARISON (Impressive - Latest):")
print("="*60)
for key in impressive:
    diff = impressive[key] - latest[key]
    if abs(diff) > 0.5:
        symbol = "🔴" if abs(diff) > 2 else "🟡"
        print(f"{symbol} {key:12}: {diff:+.1f} dB difference")