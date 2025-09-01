#!/usr/bin/env python3
"""
FIXED Professional Mixing Engine
Actually sounds good - less processing, more musicality
"""

import numpy as np
import soundfile as sf
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import os
from pathlib import Path
from datetime import datetime
import time

# Import DSP functions
from dsp_premitives import (
    peaking_eq, shelf_filter, highpass_filter, lowpass_filter,
    compressor, transient_shaper, stereo_widener, 
    _db_to_lin, _lin_to_db, measure_peak, measure_rms
)

# Simple stubs for missing advanced DSP functions
def analog_console_saturation(audio, sample_rate_or_drive=44100, drive=0.3, character='warm', **kwargs):
    """Simple console saturation stub - gentle harmonic enhancement"""
    # Handle both old and new calling conventions
    if isinstance(sample_rate_or_drive, (int, float)) and sample_rate_or_drive > 100:
        # Called with (audio, sample_rate, drive, character)
        actual_drive = drive if isinstance(drive, (int, float)) else 0.3
    else:
        # Called with (audio, drive, ...)
        actual_drive = sample_rate_or_drive if isinstance(sample_rate_or_drive, (int, float)) else 0.3
    
    return audio + np.tanh(audio * actual_drive * 0.5) * 0.1

def tape_saturation(audio, drive=0.3, warmth=0.5, **kwargs):
    """Simple tape saturation stub - warm harmonic distortion"""
    return audio + np.tanh(audio * drive) * 0.08

def tube_saturation(audio, drive=0.3, warmth=0.5, **kwargs):
    """Simple tube saturation stub - smooth harmonic enhancement"""
    return audio + np.tanh(audio * drive * 0.8) * 0.12

def multiband_compressor(audio, sample_rate, **kwargs):
    """Simple multiband compressor stub - uses single band compressor"""
    return compressor(audio, sample_rate, threshold_db=-12, ratio=3.0, attack_ms=10, release_ms=100)


@dataclass
class SimpleProChannel:
    """Simplified professional channel - less is more"""
    name: str
    category: str
    audio: np.ndarray
    sample_rate: int = 44100
    
    # Basic level and pan
    gain: float = 1.0
    pan: float = 0.0
    
    # Simple EQ - just what's needed
    eq_bands: List[Dict] = field(default_factory=list)
    
    # Gentle compression - musical, not crushing
    comp_enabled: bool = False
    comp_threshold: float = -20.0
    comp_ratio: float = 2.0  # Never more than 4:1 except for limiting
    comp_attack: float = 10.0  # Slower attacks preserve transients
    comp_release: float = 100.0
    
    # Light saturation - just for warmth
    saturation_enabled: bool = False
    saturation_amount: float = 0.1  # Very subtle
    
    def process(self) -> np.ndarray:
        """Process with restraint"""
        processed = self.audio.copy()
        
        # 1. Simple EQ
        for band in self.eq_bands:
            if abs(band['gain']) > 0.1:  # Only apply if significant
                processed = peaking_eq(
                    processed, self.sample_rate,
                    band['freq'], band['gain'], band.get('q', 0.7)
                )
        
        # 2. Gentle compression if needed
        if self.comp_enabled and self.comp_ratio <= 4:
            processed = compressor(
                processed, self.sample_rate,
                threshold_db=self.comp_threshold,
                ratio=self.comp_ratio,
                attack_ms=self.comp_attack,
                release_ms=self.comp_release,
                knee_db=2.0  # Always soft knee for musicality
            )
        
        # 3. Very subtle saturation
        if self.saturation_enabled and self.saturation_amount < 0.5:
            if self.category == 'drums':
                processed = analog_console_saturation(
                    processed, self.sample_rate, 
                    self.saturation_amount, 'warm'
                )
            elif self.category == 'bass':
                processed = tape_saturation(
                    processed, self.saturation_amount, 0.3
                )
            elif 'vocal' in self.category:
                processed = tube_saturation(
                    processed, self.saturation_amount * 0.5, 0.5
                )
            else:
                # Very light for everything else
                processed = analog_console_saturation(
                    processed, self.sample_rate,
                    self.saturation_amount * 0.5, 'clean'
                )
        
        # 4. Apply gain and pan
        processed *= self.gain
        
        # Simple pan if stereo
        if processed.ndim == 2 and abs(self.pan) > 0.01:
            # Equal power panning
            angle = self.pan * np.pi / 4
            left_gain = np.cos(angle - np.pi/4) * np.sqrt(2)
            right_gain = np.sin(angle + np.pi/4) * np.sqrt(2)
            processed[:, 0] *= left_gain
            processed[:, 1] *= right_gain
        
        # 5. MODERN PRODUCTION: Per-channel limiting for kick/snare
        if self.category == 'drums' and ('kick' in self.name.lower() or 'snare' in self.name.lower()):
            print(f"    🔒 Modern production limiting: {self.name}")
            
            # Stage 1: Transparent peak control
            processed = compressor(
                processed, self.sample_rate,
                threshold_db=-6.0,     # Higher threshold for transparency
                ratio=8.0,             # Moderate ratio for musicality
                attack_ms=0.3,         # Preserve transient character
                release_ms=25.0,       # Quick but natural
                knee_db=1.0            # Soft knee for smoothness
            )
            
            # Stage 2: Transparent clipping prevention
            processed = np.tanh(processed * 0.95) / 0.95
            
            # Stage 3: Modern peak ceiling
            peak = np.max(np.abs(processed))
            if peak > 0.9:  # Higher ceiling for more punch
                processed = processed * (0.88 / peak)
        
        return processed


class FixedProMixingSession:
    """Fixed professional mixing - sounds actually good"""
    
    def __init__(self, channels: Dict, sample_rate: int = 44100):
        self.raw_channels = channels
        self.sample_rate = sample_rate
        self.channel_strips = {}
        
        # Load channels with MUCH simpler defaults
        self._load_channels_simple()
    
    def _load_channels_simple(self):
        """Load with minimal processing"""
        print("🎵 Loading channels with musical processing...")
        
        for category, tracks in self.raw_channels.items():
            if not tracks:
                continue
                
            for name, path in tracks.items():
                try:
                    audio, sr = sf.read(path)
                    
                    # Resample if needed
                    if sr != self.sample_rate:
                        from scipy import signal as sp
                        num_samples = int(len(audio) * self.sample_rate / sr)
                        audio = sp.resample(audio, num_samples)
                    
                    # Create simple channel
                    strip = SimpleProChannel(
                        name=name,
                        category=category,
                        audio=audio,
                        sample_rate=self.sample_rate
                    )
                    
                    # Apply MINIMAL intelligent defaults
                    self._apply_minimal_defaults(strip)
                    
                    channel_id = f"{category}.{name}"
                    self.channel_strips[channel_id] = strip
                    
                    print(f"  ✓ {channel_id} loaded")
                    
                except Exception as e:
                    print(f"  ✗ Failed to load {name}: {e}")
    
    def _apply_minimal_defaults(self, strip: SimpleProChannel):
        """Minimal, musical defaults only"""
        name_lower = strip.name.lower()
        category_lower = strip.category.lower()
        
        # DRUMS - Keep them punchy and natural
        if category_lower == 'drums':
            if 'kick' in name_lower:
                strip.eq_bands = [
                    {'freq': 60, 'gain': 1.5, 'q': 0.7},   # Gentle sub boost
                    {'freq': 3500, 'gain': 1.0, 'q': 0.8}, # Slight click
                ]
                # Very gentle compression
                strip.comp_enabled = True
                strip.comp_threshold = -12  # Only catch peaks
                strip.comp_ratio = 2.5      # Gentle
                strip.comp_attack = 10      # Let transient through
                strip.comp_release = 50
                
            elif 'snare' in name_lower:
                # TIGHT MODERN SNARE - Clean, punchy, controlled
                strip.eq_bands = [
                    {'freq': 100, 'gain': -2.0, 'q': 0.7},  # Clean up mud
                    {'freq': 200, 'gain': 1.5, 'q': 2.0},   # Tight fundamental
                    {'freq': 400, 'gain': -1.5, 'q': 1.5},  # Remove boxiness
                    {'freq': 800, 'gain': 1.0, 'q': 1.2},   # Snare wire rattle
                    {'freq': 2000, 'gain': 2.5, 'q': 1.5},  # Main crack
                    {'freq': 3500, 'gain': 2.0, 'q': 1.2},  # Attack definition
                    {'freq': 5500, 'gain': 1.0, 'q': 1.0},  # Presence (controlled)
                    {'freq': 8000, 'gain': -1.0, 'q': 0.8}, # Gentle high control
                ]
                strip.comp_enabled = True
                strip.comp_threshold = -10  # Standard threshold
                strip.comp_ratio = 2.5      # Moderate compression
                strip.comp_attack = 1.0     # Fast but not instant
                strip.comp_release = 40     # Natural decay
                
                # Light saturation for cohesion
                strip.saturation_enabled = True
                strip.saturation_amount = 0.1  # Subtle warmth
                
            elif 'hihat' in name_lower or 'hat' in name_lower:
                strip.eq_bands = [
                    {'freq': 200, 'gain': -1.0, 'q': 0.8},  # Remove mud
                    {'freq': 8000, 'gain': 1.0, 'q': 0.5},  # Slight brightness
                ]
                # No compression on hats!
                
            elif 'tom' in name_lower:
                strip.eq_bands = [
                    {'freq': 100, 'gain': 1.0, 'q': 0.7},   # Body
                ]
                # Very light compression
                strip.comp_enabled = True
                strip.comp_threshold = -15
                strip.comp_ratio = 1.5
                strip.comp_attack = 10
                strip.comp_release = 100
                
            elif 'cymbal' in name_lower or 'crash' in name_lower:
                strip.eq_bands = [
                    {'freq': 300, 'gain': -1.5, 'q': 0.7},  # Remove boxiness
                    {'freq': 10000, 'gain': 1.0, 'q': 0.4}, # Air
                ]
                # No compression on cymbals!
        
        # BASS - Keep it solid but not overprocessed
        elif category_lower == 'bass':
            strip.eq_bands = [
                {'freq': 80, 'gain': 1.0, 'q': 0.8},    # Fundamental support
                {'freq': 700, 'gain': 0.5, 'q': 0.7},   # Definition
            ]
            strip.comp_enabled = True
            strip.comp_threshold = -15
            strip.comp_ratio = 3.0  # Controlled
            strip.comp_attack = 10
            strip.comp_release = 100
            # Tiny bit of saturation for warmth
            strip.saturation_enabled = True
            strip.saturation_amount = 0.15
        
        # VOCALS - Clear but natural
        elif 'vocal' in category_lower:
            if 'lead' in name_lower or category_lower == 'vocals':
                strip.eq_bands = [
                    {'freq': 150, 'gain': -0.5, 'q': 0.7},  # Slight low cut
                    {'freq': 3000, 'gain': 1.0, 'q': 0.8},  # Presence
                    {'freq': 10000, 'gain': 0.5, 'q': 0.5}, # Air
                ]
                strip.comp_enabled = True
                strip.comp_threshold = -18
                strip.comp_ratio = 2.5
                strip.comp_attack = 5
                strip.comp_release = 80
            else:  # Backing vocals
                strip.eq_bands = [
                    {'freq': 200, 'gain': -1.0, 'q': 0.7},  # More low cut
                    {'freq': 5000, 'gain': 0.5, 'q': 0.6},  # Slight presence
                ]
                strip.comp_enabled = True
                strip.comp_threshold = -15
                strip.comp_ratio = 2.0
                strip.comp_attack = 8
                strip.comp_release = 100
        
        # GUITARS - Warm and present
        elif category_lower == 'guitars':
            strip.eq_bands = [
                {'freq': 250, 'gain': -0.5, 'q': 0.7},  # Reduce mud
                {'freq': 2000, 'gain': 0.5, 'q': 0.7},  # Presence
            ]
            # Very light saturation for warmth
            strip.saturation_enabled = True
            strip.saturation_amount = 0.1
        
        # KEYS/SYNTHS - Clean and spacious
        elif category_lower in ['keys', 'synths']:
            strip.eq_bands = [
                {'freq': 500, 'gain': -0.5, 'q': 0.6},  # Slight mid scoop
                {'freq': 8000, 'gain': 0.5, 'q': 0.5},  # Air
            ]
            # No compression needed usually
        
        # FX - Just clean up
        elif category_lower == 'fx':
            strip.eq_bands = [
                {'freq': 100, 'gain': -1.0, 'q': 0.7},  # Remove rumble
            ]
    
    def apply_sidechain_compression(self):
        """Simple sidechain - kick ducks bass gently"""
        # Find kick and bass
        kick_strip = None
        bass_strips = []
        
        for ch_id, strip in self.channel_strips.items():
            if 'kick' in ch_id.lower():
                kick_strip = strip
            elif 'bass' in ch_id.lower():
                bass_strips.append(strip)
        
        if kick_strip and bass_strips:
            print("🔗 Setting up gentle sidechain (kick → bass)")
            # We'll apply this during processing
            self.sidechain_amount = 0.3  # 30% duck - subtle
    
    def apply_frequency_slotting(self):
        """Apply modern frequency slotting for clean, impressive mix"""
        print("\n🎯 APPLYING MODERN MIX CLARITY")
        print("=" * 50)
        print("🎵 Creating clean, impressive production sound...")
        
        # KICK vs BASS - Fundamental separation
        for ch_id, strip in self.channel_strips.items():
            if 'kick' in ch_id.lower():
                # Kick owns 50-80Hz
                strip.eq_bands.append({'freq': 60, 'gain': 2.0, 'q': 1.5})  # Boost fundamental
                strip.eq_bands.append({'freq': 150, 'gain': -1.5, 'q': 1.2})  # Cut upper bass
                print(f"  • {ch_id}: Owns 50-80Hz (fundamental)")
                
            elif 'bass' in ch_id.lower():
                # Bass owns 80-200Hz
                strip.eq_bands.append({'freq': 50, 'gain': -2.0, 'q': 1.5})  # Cut sub (kick's area)
                strip.eq_bands.append({'freq': 120, 'gain': 1.5, 'q': 1.0})  # Boost bass body
                strip.eq_bands.append({'freq': 800, 'gain': 1.0, 'q': 1.2})  # Definition
                print(f"  • {ch_id}: Owns 80-200Hz (body)")
        
        # VOCALS vs GUITARS/KEYS - Midrange clarity
        for ch_id, strip in self.channel_strips.items():
            if 'vocal' in ch_id.lower() and 'lead' in ch_id.lower():
                # Modern vocal clarity
                strip.eq_bands.append({'freq': 100, 'gain': -2.0, 'q': 0.7})  # Clean low cut
                strip.eq_bands.append({'freq': 2500, 'gain': 2.0, 'q': 0.8})  # Strong presence
                strip.eq_bands.append({'freq': 4000, 'gain': 1.5, 'q': 1.0})  # Definition
                strip.eq_bands.append({'freq': 8000, 'gain': 1.0, 'q': 0.6})  # Modern air
                strip.eq_bands.append({'freq': 800, 'gain': -1.5, 'q': 1.5})  # Remove mud
                print(f"  • {ch_id}: Modern vocal clarity")
                
            elif 'guitar' in ch_id.lower():
                # Guitars avoid vocal frequencies
                strip.eq_bands.append({'freq': 2500, 'gain': -1.5, 'q': 1.2})  # Duck vocal area
                strip.eq_bands.append({'freq': 500, 'gain': 1.0, 'q': 1.0})  # Body elsewhere
                strip.eq_bands.append({'freq': 5000, 'gain': 1.0, 'q': 0.8})  # Sparkle above vocals
                print(f"  • {ch_id}: Carved around vocals")
                
            elif 'keys' in ch_id.lower() or 'piano' in ch_id.lower():
                # Keys sit lower/higher than vocals
                strip.eq_bands.append({'freq': 2000, 'gain': -1.5, 'q': 1.0})  # Duck vocal area
                strip.eq_bands.append({'freq': 400, 'gain': 1.0, 'q': 1.0})  # Low mids
                strip.eq_bands.append({'freq': 7000, 'gain': 1.0, 'q': 0.7})  # High sparkle
                print(f"  • {ch_id}: Sits around vocals")
        
        # DRUMS - Each drum owns its space
        for ch_id, strip in self.channel_strips.items():
            if 'snare' in ch_id.lower():
                # Snare owns 200Hz and 2kHz
                strip.eq_bands.append({'freq': 150, 'gain': -1.0, 'q': 1.5})  # Avoid kick
                print(f"  • {ch_id}: Owns 200Hz + 2kHz")
                
            elif 'hihat' in ch_id.lower() or 'cymbal' in ch_id.lower():
                # Modern hi-end sparkle
                strip.eq_bands.append({'freq': 300, 'gain': -4.0, 'q': 0.7})  # Clean low cut
                strip.eq_bands.append({'freq': 8000, 'gain': 2.0, 'q': 0.6})  # Modern sparkle
                strip.eq_bands.append({'freq': 12000, 'gain': 1.5, 'q': 0.4}) # Expensive air
                strip.eq_bands.append({'freq': 15000, 'gain': 1.0, 'q': 0.3}) # Ultra-high sheen
                print(f"  • {ch_id}: Modern hi-end sparkle")
        
        print("\n✅ Frequency slotting complete - no more masking!")
    
    def process_mix(self, output_dir: str) -> Dict:
        """Process the mix with musicality"""
        os.makedirs(output_dir, exist_ok=True)
        start_time = time.time()
        
        print("🎚️ Processing musical mix...")
        
        # Process channels
        processed_channels = {}
        for ch_id, strip in self.channel_strips.items():
            print(f"  Processing {strip.category}/{strip.name}...")
            processed = strip.process()
            processed_channels[ch_id] = processed
        
        # Group into buses (just for organization, minimal processing)
        drum_mix = self._mix_drums(processed_channels)
        bass_mix = self._mix_bass(processed_channels)
        vocal_mix = self._mix_vocals(processed_channels)
        music_mix = self._mix_music(processed_channels)
        
        print("  Creating full mix...")
        
        # Simple mixing - no over-processing
        max_len = max(
            len(drum_mix) if drum_mix is not None else 0,
            len(bass_mix) if bass_mix is not None else 0,
            len(vocal_mix) if vocal_mix is not None else 0,
            len(music_mix) if music_mix is not None else 0
        )
        
        # Initialize mix
        if drum_mix is not None and drum_mix.ndim == 2:
            full_mix = np.zeros((max_len, 2))
        else:
            full_mix = np.zeros(max_len)
        
        # Mix with musical balance
        if drum_mix is not None:
            full_mix[:len(drum_mix)] += drum_mix * 0.9  # Drums present but not overpowering
        if bass_mix is not None:
            full_mix[:len(bass_mix)] += bass_mix * 0.8  # Solid bass
        if vocal_mix is not None:
            full_mix[:len(vocal_mix)] += vocal_mix * 1.0  # Vocals upfront
        if music_mix is not None:
            full_mix[:len(music_mix)] += music_mix * 0.7  # Supporting instruments
        
        # VERY gentle master bus processing
        print("  Applying gentle master processing...")
        
        # 1. Gentle glue compression
        full_mix = compressor(
            full_mix, self.sample_rate,
            threshold_db=-12,  # Only catch peaks
            ratio=1.5,         # Very gentle
            attack_ms=30,      # Slow attack
            release_ms=200,    # Natural release
            knee_db=4.0        # Very soft knee
        )
        
        # 2. Musical EQ - just enhancement
        full_mix = shelf_filter(full_mix, self.sample_rate, 60, 0.5, 'low')    # Gentle low shelf
        full_mix = shelf_filter(full_mix, self.sample_rate, 10000, 0.5, 'high') # Gentle high shelf
        
        # 3. MODERN CLEAN PRODUCTION - Transparent loudness
        # Stage 1: Clean compression for density
        full_mix = compressor(
            full_mix, self.sample_rate,
            threshold_db=-8.0,     # Gentle threshold for transparency
            ratio=3.0,             # Musical ratio
            attack_ms=2.0,         # Preserve transient character
            release_ms=60.0,       # Natural release
            knee_db=2.0            # Soft knee for smoothness
        )
        
        # Stage 2: Professional peak limiting
        full_mix = compressor(
            full_mix, self.sample_rate,
            threshold_db=-2.0,     # Clean limiting threshold
            ratio=12.0,            # Controlled limiting
            attack_ms=0.1,         # Fast but not harsh
            release_ms=30.0,       # Quick but natural
            knee_db=0.5            # Slightly soft knee
        )
        
        # Stage 3: Transparent saturation for warmth
        full_mix = np.tanh(full_mix * 0.96) / 0.96  # Subtle warmth
        
        # Stage 4: Modern streaming loudness target
        peak = np.max(np.abs(full_mix))
        if peak > 0:
            ceiling_linear = _db_to_lin(-0.3)  # Modern streaming ceiling
            full_mix = full_mix * (ceiling_linear / peak)
        
        # Save
        mix_path = os.path.join(output_dir, 'mix.wav')
        sf.write(mix_path, full_mix, self.sample_rate, subtype='FLOAT')
        
        # Simple mastering
        print("  Simple mastering...")
        mastered = self._simple_master(full_mix)
        master_path = os.path.join(output_dir, 'master.wav')
        sf.write(master_path, mastered, self.sample_rate, subtype='FLOAT')
        
        # Create stems
        print("  Creating stems...")
        stem_paths = self._create_stems(
            output_dir, drum_mix, bass_mix, vocal_mix, music_mix
        )
        
        # Results
        peak_db = _lin_to_db(np.max(np.abs(mastered)))
        rms_db = _lin_to_db(np.sqrt(np.mean(mastered**2)))
        
        print(f"\n✅ Musical mix complete!")
        print(f"  Peak: {peak_db:.1f} dBFS")
        print(f"  RMS: {rms_db:.1f} dBFS")
        print(f"  Time: {time.time() - start_time:.1f}s")
        
        return {
            'peak_db': peak_db,
            'rms_db': rms_db,
            'time': time.time() - start_time,
            'output_file': master_path,
            'mix_file': mix_path,
            'stem_files': stem_paths
        }
    
    def _mix_drums(self, processed_channels: Dict) -> Optional[np.ndarray]:
        """Mix drums with minimal bus processing"""
        drum_channels = [ch for ch_id, ch in processed_channels.items() if 'drums.' in ch_id]
        if not drum_channels:
            return None
        
        # Find the longest
        max_len = max(len(ch) for ch in drum_channels)
        if drum_channels[0].ndim == 2:
            drum_bus = np.zeros((max_len, 2))
        else:
            drum_bus = np.zeros(max_len)
        
        # Simple summing
        for ch in drum_channels:
            drum_bus[:len(ch)] += ch
        
        # VERY gentle bus compression for glue
        drum_bus = compressor(
            drum_bus, self.sample_rate,
            threshold_db=-10,  # Only loud peaks
            ratio=1.8,         # Barely compressing
            attack_ms=10,      # Let transients through
            release_ms=100,
            knee_db=4.0
        )
        
        # Tiny bit of saturation for cohesion
        drum_bus = analog_console_saturation(
            drum_bus, self.sample_rate, 0.1, 'warm'
        )
        
        # MODERN CLEAN DRUMS: Professional bus processing
        print("    🥁 Applying modern drum bus processing...")
        drum_bus = compressor(
            drum_bus, self.sample_rate,
            threshold_db=-4.0,     # Higher threshold for transparency
            ratio=6.0,             # Moderate ratio for punch
            attack_ms=0.5,         # Preserve transient character
            release_ms=40.0,       # Natural release
            knee_db=1.5            # Soft knee for smoothness
        )
        
        # Clean peak control (no clipping sound)
        peak = np.max(np.abs(drum_bus))
        if peak > 0.9:
            drum_bus = drum_bus * (0.85 / peak)  # Clean reduction
        
        return drum_bus
    
    def _mix_bass(self, processed_channels: Dict) -> Optional[np.ndarray]:
        """Mix bass - keep it solid"""
        bass_channels = [ch for ch_id, ch in processed_channels.items() if 'bass.' in ch_id]
        if not bass_channels:
            return None
        
        max_len = max(len(ch) for ch in bass_channels)
        if bass_channels[0].ndim == 2:
            bass_bus = np.zeros((max_len, 2))
        else:
            bass_bus = np.zeros(max_len)
        
        for ch in bass_channels:
            bass_bus[:len(ch)] += ch
        
        # Just a touch of compression for consistency
        bass_bus = compressor(
            bass_bus, self.sample_rate,
            threshold_db=-12,
            ratio=2.0,
            attack_ms=10,
            release_ms=100,
            knee_db=2.0
        )
        
        return bass_bus
    
    def _mix_vocals(self, processed_channels: Dict) -> Optional[np.ndarray]:
        """Mix vocals - keep them clear"""
        vocal_channels = [ch for ch_id, ch in processed_channels.items() 
                         if 'vocal' in ch_id.lower()]
        if not vocal_channels:
            return None
        
        max_len = max(len(ch) for ch in vocal_channels)
        if vocal_channels[0].ndim == 2:
            vocal_bus = np.zeros((max_len, 2))
        else:
            vocal_bus = np.zeros(max_len)
        
        for ch in vocal_channels:
            vocal_bus[:len(ch)] += ch
        
        # Gentle compression for consistency
        vocal_bus = compressor(
            vocal_bus, self.sample_rate,
            threshold_db=-15,
            ratio=2.0,
            attack_ms=5,
            release_ms=80,
            knee_db=2.0
        )
        
        return vocal_bus
    
    def _mix_music(self, processed_channels: Dict) -> Optional[np.ndarray]:
        """Mix everything else"""
        music_channels = [ch for ch_id, ch in processed_channels.items() 
                         if not any(x in ch_id for x in ['drums.', 'bass.', 'vocal'])]
        if not music_channels:
            return None
        
        max_len = max(len(ch) for ch in music_channels)
        if music_channels[0].ndim == 2:
            music_bus = np.zeros((max_len, 2))
        else:
            music_bus = np.zeros(max_len)
        
        for ch in music_channels:
            music_bus[:len(ch)] += ch
        
        # No compression needed - keep it dynamic
        return music_bus
    
    def _simple_master(self, mix: np.ndarray) -> np.ndarray:
        """Simple, transparent mastering"""
        mastered = mix.copy()
        
        # 1. Gentle EQ enhancement
        mastered = peaking_eq(mastered, self.sample_rate, 40, -0.5, 0.7)    # Clean sub
        mastered = shelf_filter(mastered, self.sample_rate, 100, 0.5, 'low')   # Warmth
        mastered = peaking_eq(mastered, self.sample_rate, 3000, 0.5, 0.7)   # Presence
        mastered = shelf_filter(mastered, self.sample_rate, 12000, 0.5, 'high') # Air
        
        # 2. Multiband compression - VERY gentle
        mastered = multiband_compressor(
            mastered, self.sample_rate,
            bands=[
                {'threshold_db': -12, 'ratio': 1.5, 'attack_ms': 20, 'release_ms': 200},
                {'threshold_db': -15, 'ratio': 1.3, 'attack_ms': 10, 'release_ms': 150},
                {'threshold_db': -18, 'ratio': 1.2, 'attack_ms': 5, 'release_ms': 100}
            ],
            crossover_freqs=[200, 2000]
        )
        
        # 3. Maximum loudness mastering chain
        # Stage 1: Pre-limiting compression for density
        mastered = compressor(
            mastered, self.sample_rate,
            threshold_db=-8.0,     # Catch more signal for density
            ratio=4.0,             # Moderate ratio to preserve dynamics
            attack_ms=1.0,         # Preserve transients
            release_ms=50.0,       # Musical release
            knee_db=2.0            # Soft knee to preserve your balance
        )
        
        # Stage 2: Aggressive peak limiting for maximum loudness
        mastered = compressor(
            mastered, self.sample_rate,
            threshold_db=-0.5,     # VERY aggressive threshold
            ratio=100.0,           # Extreme ratio for brickwall limiting
            attack_ms=0.01,        # Fastest possible attack
            release_ms=15.0,       # Very fast release for transparency
            knee_db=0.01           # Hard knee for maximum efficiency
        )
        
        # Stage 3: Multi-stage saturation for perceived loudness
        mastered = np.tanh(mastered * 0.96) / 0.96  # First saturation
        mastered = np.tanh(mastered * 0.97) / 0.97  # Second saturation
        mastered = np.tanh(mastered * 0.985) / 0.985  # Final saturation
        
        # Stage 4: Maximize to -0.1dB ceiling (streaming/CD max)
        peak = np.max(np.abs(mastered))
        if peak > 0:
            ceiling_linear = _db_to_lin(-0.1)  # Maximum possible loudness
            mastered = mastered * (ceiling_linear / peak)
        
        return mastered
    
    def _create_stems(self, output_dir: str, drum_mix: Optional[np.ndarray],
                     bass_mix: Optional[np.ndarray], vocal_mix: Optional[np.ndarray],
                     music_mix: Optional[np.ndarray]) -> Dict[str, str]:
        """Save stems"""
        stem_paths = {}
        
        if drum_mix is not None:
            path = os.path.join(output_dir, 'drums.wav')
            sf.write(path, drum_mix, self.sample_rate, subtype='FLOAT')
            stem_paths['drums'] = path
            
        if bass_mix is not None:
            path = os.path.join(output_dir, 'bass.wav')
            sf.write(path, bass_mix, self.sample_rate, subtype='FLOAT')
            stem_paths['bass'] = path
            
        if vocal_mix is not None:
            path = os.path.join(output_dir, 'vocals.wav')
            sf.write(path, vocal_mix, self.sample_rate, subtype='FLOAT')
            stem_paths['vocals'] = path
            
        if music_mix is not None:
            path = os.path.join(output_dir, 'music.wav')
            sf.write(path, music_mix, self.sample_rate, subtype='FLOAT')
            stem_paths['music'] = path
            
        return stem_paths

    def _get_mix_configurations(self) -> Dict[str, Dict]:
        """Get meaningful mix configuration variants"""
        return {
            "Standard_Mix": {
                "description": "Balanced mix with moderate processing",
                "saturation_mult": 1.0,
                "compression_mult": 1.0,
                "eq_mult": 1.0,
                "stereo_mult": 1.0
            },
            "Vocal_Forward": {
                "description": "Vocals pushed forward, instruments pulled back",
                "vocal_boost": 1.3,
                "instrument_reduction": 0.8,
                "vocal_compression": 0.8
            },
            "Drum_Heavy": {
                "description": "Powerful drums with enhanced low end",
                "drum_boost": 1.4,
                "bass_boost": 1.2,
                "compression_mult": 1.3
            },
            "Intimate_Mix": {
                "description": "Close, warm sound with reduced dynamics",
                "saturation_mult": 1.5,
                "compression_mult": 1.4,
                "stereo_mult": 0.7
            },
            "Wide_Stereo": {
                "description": "Maximum stereo width and space",
                "stereo_mult": 1.6,
                "reverb_mult": 1.3,
                "compression_mult": 0.8
            },
            "Punchy_Rock": {
                "description": "Aggressive, powerful rock sound",
                "compression_mult": 1.5,
                "saturation_mult": 1.3,
                "drum_boost": 1.3
            },
            "Clean_Pop": {
                "description": "Clean, polished modern pop sound",
                "compression_mult": 0.7,
                "saturation_mult": 0.5,
                "eq_mult": 1.2
            },
            "Warm_Vintage": {
                "description": "Warm, analog-inspired vintage sound",
                "saturation_mult": 1.8,
                "compression_mult": 1.2,
                "eq_mult": 0.8
            },
            "Radio_Ready": {
                "description": "Loud, compressed for radio play",
                "compression_mult": 2.0,
                "loudness_mult": 1.5,
                "eq_mult": 1.1
            },
            "Dynamic_Master": {
                "description": "Preserve dynamics, minimal processing",
                "compression_mult": 0.5,
                "saturation_mult": 0.3,
                "loudness_mult": 0.8
            }
        }

    def apply_mix_configuration(self, config_name: str):
        """Apply a specific mix configuration"""
        configs = self._get_mix_configurations()
        
        if config_name not in configs:
            print(f"⚠️ Configuration '{config_name}' not found, using defaults")
            return
        
        config = configs[config_name]
        print(f"🎛️ Applying configuration: {config_name}")
        print(f"   {config['description']}")
        
        # Apply configuration parameters to all channel strips
        for ch_id, strip in self.channel_strips.items():
            # Apply saturation multipliers
            if 'saturation_mult' in config:
                strip.saturation_amount *= config['saturation_mult']
            
            # Apply compression multipliers  
            if 'compression_mult' in config:
                if strip.comp_enabled:
                    strip.comp_ratio = min(strip.comp_ratio * config['compression_mult'], 10.0)
            
            # Apply vocal-specific adjustments
            if 'vocal' in ch_id.lower():
                if 'vocal_boost' in config:
                    strip.gain *= config['vocal_boost']
                if 'vocal_compression' in config:
                    strip.comp_ratio *= config['vocal_compression']
            
            # Apply drum-specific adjustments
            if 'drums' in ch_id.lower():
                if 'drum_boost' in config:
                    strip.gain *= config['drum_boost']
            
            # Apply bass-specific adjustments  
            if 'bass' in ch_id.lower():
                if 'bass_boost' in config:
                    strip.gain *= config['bass_boost']
            
            # Apply instrument reduction
            if 'instrument_reduction' in config and 'vocal' not in ch_id.lower():
                strip.gain *= config['instrument_reduction']
            
            # Apply EQ multipliers
            if 'eq_mult' in config:
                for eq_band in strip.eq_bands:
                    eq_band['gain'] *= config['eq_mult']