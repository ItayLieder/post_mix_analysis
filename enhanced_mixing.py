#!/usr/bin/env python3
"""Enhanced mixing functions that actually work with the current system"""

import numpy as np
from dsp_premitives import (
    shelf_filter, peaking_eq, highpass_filter, 
    compressor, transient_shaper
)
from audio_utils import db_to_linear, ensure_stereo

def enhance_mix_settings(session):
    """Apply WORKING enhancements to the mix session"""
    
    print("🎛️ APPLYING REAL ENHANCEMENTS")
    print("="*50)
    
    # 1. FIX FREQUENCY BALANCE
    print("\n📊 FREQUENCY BALANCE FIXES:")
    
    for ch_id, strip in session.channel_strips.items():
        category = strip.category.lower()
        
        # Initialize EQ if needed
        if not strip.eq_bands:
            strip.eq_bands = {}
        
        # DRUMS - Add punch and clarity
        if category == 'drums':
            if 'kick' in ch_id.lower():
                # More sub punch
                strip.eq_bands['50hz'] = {'gain': 4, 'q': 0.7}  # Sub punch
                strip.eq_bands['80hz'] = {'gain': 2, 'q': 0.8}  # Body
                strip.eq_bands['300hz'] = {'gain': -3, 'q': 0.9}  # Remove mud
                strip.eq_bands['3500hz'] = {'gain': 3, 'q': 0.8}  # Click
                print(f"  ✓ {ch_id}: Enhanced sub + click")
                
            elif 'snare' in ch_id.lower():
                strip.eq_bands['200hz'] = {'gain': 2, 'q': 0.9}  # Body
                strip.eq_bands['350hz'] = {'gain': -2, 'q': 0.8}  # Remove box
                strip.eq_bands['4500hz'] = {'gain': 4, 'q': 0.7}  # Crack
                strip.eq_bands['8000hz'] = {'gain': 3, 'q': 0.6}  # Sizzle
                print(f"  ✓ {ch_id}: Enhanced crack + sizzle")
                
            elif 'hihat' in ch_id.lower() or 'cymbal' in ch_id.lower():
                strip.eq_bands['400hz'] = {'gain': -4, 'q': 0.7}  # Remove mud
                strip.eq_bands['6000hz'] = {'gain': 3, 'q': 0.6}  # Shimmer
                strip.eq_bands['12000hz'] = {'gain': 4, 'q': 0.5}  # Air
                print(f"  ✓ {ch_id}: Added shimmer + air")
        
        # BASS - Tighten and clarify
        elif category == 'bass':
            strip.eq_bands['40hz'] = {'gain': 2, 'q': 0.7}  # Sub extension
            strip.eq_bands['100hz'] = {'gain': 1, 'q': 0.8}  # Fundamental
            strip.eq_bands['250hz'] = {'gain': -2, 'q': 0.8}  # Remove mud
            strip.eq_bands['800hz'] = {'gain': 2, 'q': 0.9}  # Definition
            strip.eq_bands['2500hz'] = {'gain': 1, 'q': 0.7}  # Presence
            print(f"  ✓ {ch_id}: Tightened low end + added definition")
        
        # VOCALS - Clarity and presence
        elif 'vocal' in category:
            if 'lead' in ch_id.lower() or 'vocals' in category:
                strip.eq_bands['150hz'] = {'gain': -2, 'q': 0.7}  # Remove mud
                strip.eq_bands['350hz'] = {'gain': -1, 'q': 0.8}  # Remove box
                strip.eq_bands['2500hz'] = {'gain': 3, 'q': 0.8}  # Presence
                strip.eq_bands['5000hz'] = {'gain': 2, 'q': 0.7}  # Clarity
                strip.eq_bands['10000hz'] = {'gain': 3, 'q': 0.5}  # Air
                print(f"  ✓ {ch_id}: Enhanced presence + air")
        
        # GUITARS - Cut through mix
        elif 'guitar' in category:
            strip.eq_bands['100hz'] = {'gain': -3, 'q': 0.7}  # Remove boom
            strip.eq_bands['400hz'] = {'gain': -2, 'q': 0.8}  # Remove mud
            strip.eq_bands['2000hz'] = {'gain': 2, 'q': 0.8}  # Presence
            strip.eq_bands['4000hz'] = {'gain': 2, 'q': 0.7}  # Edge
            strip.eq_bands['8000hz'] = {'gain': 2, 'q': 0.6}  # Sparkle
            print(f"  ✓ {ch_id}: Cleaned mud + added edge")
        
        # KEYS/SYNTHS - Space and clarity
        elif category in ['keys', 'synths']:
            strip.eq_bands['200hz'] = {'gain': -2, 'q': 0.7}  # Remove mud
            strip.eq_bands['1500hz'] = {'gain': 1, 'q': 0.8}  # Presence
            strip.eq_bands['5000hz'] = {'gain': 2, 'q': 0.6}  # Brightness
            strip.eq_bands['12000hz'] = {'gain': 3, 'q': 0.5}  # Air
            print(f"  ✓ {ch_id}: Added brightness + air")
    
    # 2. OPTIMIZE COMPRESSION
    print("\n🎚️ COMPRESSION OPTIMIZATION:")
    
    for ch_id, strip in session.channel_strips.items():
        category = strip.category.lower()
        
        if category == 'drums':
            if 'kick' in ch_id.lower():
                strip.comp_enabled = True
                strip.comp_threshold = -12  # Tighter control
                strip.comp_ratio = 6  # More punch
                strip.comp_attack = 5  # Fast attack
                strip.comp_release = 50  # Quick release
                print(f"  ✓ {ch_id}: Punchy compression")
                
            elif 'snare' in ch_id.lower():
                strip.comp_enabled = True
                strip.comp_threshold = -10
                strip.comp_ratio = 4
                strip.comp_attack = 3
                strip.comp_release = 80
                print(f"  ✓ {ch_id}: Snappy compression")
        
        elif 'vocal' in category:
            strip.comp_enabled = True
            strip.comp_threshold = -15
            strip.comp_ratio = 3
            strip.comp_attack = 10
            strip.comp_release = 100
            print(f"  ✓ {ch_id}: Smooth vocal compression")
        
        elif category == 'bass':
            strip.comp_enabled = True
            strip.comp_threshold = -12
            strip.comp_ratio = 4
            strip.comp_attack = 10
            strip.comp_release = 100
            print(f"  ✓ {ch_id}: Consistent bass compression")
    
    # 3. ADJUST GAINS FOR BETTER BALANCE
    print("\n📈 GAIN OPTIMIZATION:")
    
    # Add brightness boost to everything
    brightness_boost = 1.15  # 15% boost to match impressive mix's brightness
    
    for ch_id, strip in session.channel_strips.items():
        category = strip.category.lower()
        
        # Boost highs more
        if 'hihat' in ch_id.lower() or 'cymbal' in ch_id.lower():
            strip.gain *= brightness_boost * 1.2  # Extra boost for cymbals
            print(f"  ✓ {ch_id}: +20% brightness boost")
        elif category in ['keys', 'synths']:
            strip.gain *= brightness_boost
            print(f"  ✓ {ch_id}: +15% brightness boost")
    
    print("\n✅ ENHANCEMENTS COMPLETE!")
    print("  • Fixed muddy low-mids")
    print("  • Added missing highs and air")
    print("  • Optimized compression for punch")
    print("  • Boosted brightness to match impressive mix")
    
    return session

def create_parallel_compression(audio, sr, amount=0.5):
    """Simple parallel compression that actually works"""
    # Heavy compression
    compressed = compressor(
        audio, sr,
        threshold_db=-20,
        ratio=10,
        attack_ms=1,
        release_ms=50
    )
    
    # Mix with original
    return audio * (1 - amount) + compressed * amount

def add_harmonic_enhancement(audio, sr, amount=0.2):
    """Simple harmonic enhancement using available tools"""
    # Use transient shaper to add harmonics
    enhanced = transient_shaper(
        audio, sr,
        attack_boost=amount * 2,
        sustain_boost=amount
    )
    
    # Add some high-frequency emphasis
    enhanced = shelf_filter(enhanced, sr, 8000, amount * 3, 'high')
    
    # Mix with original
    return audio * (1 - amount) + enhanced * amount