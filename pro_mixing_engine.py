#!/usr/bin/env python3
"""
Professional Mixing Engine - Studio Quality Processing
Implements techniques used by world-class mixing engineers
"""

import numpy as np
import soundfile as sf
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import os
from dataclasses import dataclass, field
import json
import time

from utils import ensure_stereo, to_mono, db_to_linear, linear_to_db
from dsp_premitives import (
    peaking_eq, shelf_filter, highpass_filter, lowpass_filter,
    compressor, transient_shaper, stereo_widener
)
# from advanced_dsp import (
#     sidechain_compressor, parallel_compression, multiband_compressor,
#     tape_saturation, tube_saturation, analog_console_saturation,
#     advanced_transient_shaper, haas_effect, stereo_spreader,
#     auto_gain_staging, frequency_slot_eq
# )

# Simple stubs for missing advanced DSP functions
def sidechain_compressor(audio, sidechain, sample_rate, **kwargs):
    """Simple sidechain compressor stub - basic ducking effect"""
    # Simple envelope-based ducking
    envelope = np.abs(sidechain)
    reduction = np.where(envelope > 0.1, 0.7, 1.0)
    return audio * reduction

def parallel_compression(audio, sample_rate, **kwargs):
    """Simple parallel compression stub"""
    compressed = compressor(audio, sample_rate, threshold_db=-15, ratio=4.0, attack_ms=5, release_ms=50)
    return audio * 0.7 + compressed * 0.3

def advanced_transient_shaper(audio, attack=0.0, sustain=0.0, **kwargs):
    """Simple transient shaper stub"""
    return transient_shaper(audio, attack_gain_db=attack*6, sustain_gain_db=sustain*3)

def stereo_spreader(audio, sample_rate, width=1.0, **kwargs):
    """Simple stereo spreader stub"""
    return stereo_widener(audio, width=width) if audio.ndim == 2 else audio

def auto_gain_staging(audio, **kwargs):
    """Simple auto gain staging stub"""
    return audio

# Copy stubs from pro_mixing_engine_fixed
def analog_console_saturation(audio, drive=0.3, **kwargs):
    """Simple console saturation stub"""
    return audio + np.tanh(audio * drive * 0.5) * 0.1

def tape_saturation(audio, drive=0.3, **kwargs):
    """Simple tape saturation stub"""
    return audio + np.tanh(audio * drive) * 0.08

def tube_saturation(audio, drive=0.3, **kwargs):
    """Simple tube saturation stub"""
    return audio + np.tanh(audio * drive * 0.8) * 0.12

def multiband_compressor(audio, sample_rate, **kwargs):
    """Simple multiband compressor stub"""
    return compressor(audio, sample_rate, threshold_db=-12, ratio=3.0, attack_ms=10, release_ms=100)


@dataclass
class ProChannelStrip:
    """Professional channel strip with full processing chain"""
    name: str
    category: str
    audio: np.ndarray
    sample_rate: int
    
    # Basic Controls
    gain: float = 1.0
    pan: float = 0.0  # -1 to 1
    mute: bool = False
    solo: bool = False
    phase_invert: bool = False
    
    # Input Processing
    input_gain: float = 1.0
    highpass_freq: Optional[float] = None
    highpass_slope: int = 12  # dB/octave
    
    # Gate
    gate_enabled: bool = False
    gate_threshold: float = -40.0
    gate_range: float = -60.0  # How much to reduce
    gate_attack: float = 0.1
    gate_release: float = 100.0
    gate_hold: float = 10.0
    
    # EQ (up to 8 bands)
    eq_enabled: bool = True
    eq_bands: List[Dict] = field(default_factory=list)
    
    # Compression
    comp_enabled: bool = False
    comp_threshold: float = -20.0
    comp_ratio: float = 3.0
    comp_attack: float = 10.0
    comp_release: float = 100.0
    comp_knee: float = 2.0
    comp_makeup: float = 0.0
    
    # Parallel Compression
    parallel_comp_enabled: bool = False
    parallel_comp_send: float = 0.5
    parallel_comp_return: float = 0.5
    parallel_comp_settings: Dict = field(default_factory=dict)
    
    # Saturation
    saturation_enabled: bool = False
    saturation_type: str = 'tape'  # 'tape', 'tube', 'console'
    saturation_drive: float = 0.3
    saturation_character: float = 0.5
    saturation_mix: float = 0.5
    
    # Transient Shaping
    transient_enabled: bool = False
    transient_attack: float = 0.0
    transient_sustain: float = 0.0
    
    # Sends
    reverb_send: float = 0.0
    delay_send: float = 0.0
    parallel_send: float = 0.0
    
    # Automation
    automation_data: Dict = field(default_factory=dict)
    
    def process(self, sidechain_input: Optional[np.ndarray] = None) -> np.ndarray:
        """Process audio through complete channel strip"""
        
        if self.mute:
            return np.zeros_like(self.audio)
        
        processed = self.audio.copy()
        
        # 1. Input Stage
        processed *= self.input_gain
        
        if self.phase_invert:
            processed *= -1
        
        # 2. High-pass filter
        if self.highpass_freq:
            order = self.highpass_slope // 6  # Convert dB/oct to filter order
            processed = highpass_filter(processed, self.sample_rate, 
                                       self.highpass_freq, order=order)
        
        # 3. Gate (if enabled)
        if self.gate_enabled:
            processed = self._apply_gate(processed)
        
        # 4. EQ
        if self.eq_enabled and self.eq_bands:
            for band in self.eq_bands:
                if band.get('enabled', True):
                    processed = peaking_eq(
                        processed, self.sample_rate,
                        band['freq'], band['gain'], band.get('q', 0.7)
                    )
        
        # 5. Compression (with optional sidechain)
        if self.comp_enabled:
            if sidechain_input is not None:
                processed = sidechain_compressor(
                    processed, sidechain_input, self.sample_rate,
                    self.comp_threshold, self.comp_ratio,
                    self.comp_attack, self.comp_release, self.comp_knee
                )
            else:
                processed = compressor(
                    processed, self.sample_rate,
                    self.comp_threshold, self.comp_ratio,
                    self.comp_attack, self.comp_release, self.comp_knee
                )
            
            # Makeup gain
            if self.comp_makeup != 0:
                processed *= db_to_linear(self.comp_makeup)
        
        # 6. Parallel Compression
        if self.parallel_comp_enabled:
            parallel = parallel_compression(
                processed, self.sample_rate,
                mix=self.parallel_comp_return,
                **self.parallel_comp_settings
            )
            processed = processed * (1 - self.parallel_comp_send) + parallel * self.parallel_comp_send
        
        # 7. Saturation
        if self.saturation_enabled:
            saturated = self._apply_saturation(processed)
            processed = processed * (1 - self.saturation_mix) + saturated * self.saturation_mix
        
        # 8. Transient Shaping
        if self.transient_enabled:
            processed = advanced_transient_shaper(
                processed, self.sample_rate,
                self.transient_attack, self.transient_sustain
            )
        
        # 9. Output gain and pan
        processed *= self.gain
        processed = self._apply_pan(processed)
        
        return processed
    
    def _apply_gate(self, audio: np.ndarray) -> np.ndarray:
        """Apply noise gate"""
        envelope = np.abs(audio if audio.ndim == 1 else np.mean(audio, axis=1))
        
        # Simple gate for now
        threshold_linear = db_to_linear(self.gate_threshold)
        gate_gain = np.ones_like(envelope)
        gate_gain[envelope < threshold_linear] = db_to_linear(self.gate_range)
        
        # Smooth to avoid clicks
        from scipy.signal import filtfilt
        gate_gain = filtfilt(np.ones(100)/100, 1, gate_gain)
        
        if audio.ndim == 2:
            gate_gain = np.stack([gate_gain, gate_gain], axis=-1)
        
        return audio * gate_gain
    
    def _apply_saturation(self, audio: np.ndarray) -> np.ndarray:
        """Apply selected saturation type"""
        if self.saturation_type == 'tape':
            return tape_saturation(audio, self.saturation_drive, self.saturation_character)
        elif self.saturation_type == 'tube':
            return tube_saturation(audio, self.saturation_drive, self.saturation_character)
        else:  # console
            character_map = {0.0: 'clean', 0.5: 'warm', 1.0: 'aggressive'}
            character = character_map.get(
                round(self.saturation_character * 2) / 2, 'warm'
            )
            return analog_console_saturation(
                audio, self.sample_rate, self.saturation_drive, character
            )
    
    def _apply_pan(self, audio: np.ndarray) -> np.ndarray:
        """Apply panning"""
        if audio.ndim == 1 or self.pan == 0:
            return audio
        
        # Equal power panning
        angle = self.pan * np.pi / 4  # -45 to +45 degrees
        left_gain = np.cos(angle - np.pi/4) * np.sqrt(2)
        right_gain = np.sin(angle + np.pi/4) * np.sqrt(2)
        
        panned = audio.copy()
        panned[:, 0] *= left_gain
        panned[:, 1] *= right_gain
        
        return panned


@dataclass 
class ProMixBus:
    """Professional mix bus with advanced processing"""
    name: str
    channels: List[ProChannelStrip]
    
    # Bus Processing
    eq_enabled: bool = True
    eq_bands: List[Dict] = field(default_factory=list)
    
    # Bus Compression
    comp_enabled: bool = False
    comp_threshold: float = -15.0
    comp_ratio: float = 2.0
    comp_attack: float = 30.0
    comp_release: float = 100.0
    comp_knee: float = 2.0
    
    # Multiband Compression
    multiband_enabled: bool = False
    multiband_settings: List[Dict] = field(default_factory=list)
    multiband_crossovers: List[float] = field(default_factory=list)
    
    # Saturation
    saturation_enabled: bool = False
    saturation_amount: float = 0.2
    
    # Parallel Compression (Bus Level)
    parallel_comp_enabled: bool = False
    parallel_comp_threshold: float = -30.0
    parallel_comp_ratio: float = 10.0
    parallel_comp_attack: float = 0.5
    parallel_comp_release: float = 100.0
    parallel_comp_mix: float = 0.5
    
    # Stereo Processing
    width: float = 1.0
    stereo_spread_enabled: bool = False
    bass_mono_freq: float = 120.0
    
    def process(self) -> np.ndarray:
        """Process all channels through bus"""
        if not self.channels:
            return np.array([[0.0, 0.0]])
        
        # Sum channels
        bus_sum = None
        for channel in self.channels:
            if not channel.mute:
                channel_out = channel.process()
                if bus_sum is None:
                    bus_sum = channel_out
                else:
                    min_len = min(len(bus_sum), len(channel_out))
                    bus_sum = bus_sum[:min_len] + channel_out[:min_len]
        
        if bus_sum is None:
            return np.array([[0.0, 0.0]])
        
        # Bus EQ
        if self.eq_enabled:
            for band in self.eq_bands:
                if band.get('enabled', True):
                    bus_sum = peaking_eq(
                        bus_sum, self.channels[0].sample_rate,
                        band['freq'], band['gain'], band.get('q', 0.7)
                    )
        
        # Bus Compression
        if self.comp_enabled:
            bus_sum = compressor(
                bus_sum, self.channels[0].sample_rate,
                self.comp_threshold, self.comp_ratio,
                self.comp_attack, self.comp_release, self.comp_knee
            )
        
        # Multiband Compression
        if self.multiband_enabled and self.multiband_settings:
            bus_sum = multiband_compressor(
                bus_sum, self.channels[0].sample_rate,
                self.multiband_settings, self.multiband_crossovers
            )
        
        # Parallel Compression (Bus Level)
        if self.parallel_comp_enabled:
            bus_sum = parallel_compression(
                bus_sum, self.channels[0].sample_rate,
                threshold_db=self.parallel_comp_threshold,
                ratio=self.parallel_comp_ratio,
                attack_ms=self.parallel_comp_attack,
                release_ms=self.parallel_comp_release,
                mix=self.parallel_comp_mix
            )
        
        # Saturation
        if self.saturation_enabled:
            saturated = analog_console_saturation(
                bus_sum, self.channels[0].sample_rate,
                self.saturation_amount, 'warm'
            )
            bus_sum = bus_sum * (1 - self.saturation_amount) + saturated * self.saturation_amount
        
        # Stereo Processing
        if self.stereo_spread_enabled:
            bus_sum = stereo_spreader(
                bus_sum, self.channels[0].sample_rate,
                self.width, self.bass_mono_freq
            )
        elif self.width != 1.0:
            bus_sum = stereo_widener(bus_sum, self.width)
        
        return bus_sum


class ProMixingSession:
    """Professional mixing session with complete studio processing"""
    
    def __init__(self, channels: Dict, template: str = 'modern',
                 sample_rate: int = 44100, bit_depth: int = 24):
        self.raw_channels = channels
        self.template = template
        self.sample_rate = sample_rate
        self.bit_depth = bit_depth
        
        self.channel_strips = {}
        self.buses = {}
        self.master_bus = None
        
        # Reverb sends
        self.reverb_bus = None
        self.delay_bus = None
        
        # Load and prepare channels
        self._load_channels()
        self._setup_default_routing()
        
    def _load_channels(self):
        """Load audio files and create professional channel strips"""
        print("🎛️ Loading channels into professional mixing system...")
        
        for category, tracks in self.raw_channels.items():
            if not tracks:
                continue
            
            for name, path in tracks.items():
                try:
                    audio, sr = sf.read(path)
                    
                    # Resample if necessary
                    if sr != self.sample_rate:
                        from scipy import signal as sp
                        num_samples = int(len(audio) * self.sample_rate / sr)
                        audio = sp.resample(audio, num_samples)
                    
                    # Create professional channel strip
                    strip = ProChannelStrip(
                        name=name,
                        category=category,
                        audio=audio,
                        sample_rate=self.sample_rate
                    )
                    
                    # Apply intelligent defaults based on instrument type
                    self._apply_intelligent_defaults(strip)
                    
                    channel_id = f"{category}.{name}"
                    self.channel_strips[channel_id] = strip
                    
                    print(f"  ✓ {channel_id} loaded with pro processing")
                    
                except Exception as e:
                    print(f"  ✗ Failed to load {name}: {e}")
    
    def _apply_intelligent_defaults(self, strip: ProChannelStrip):
        """Apply intelligent default settings based on instrument type"""
        name_lower = strip.name.lower()
        category_lower = strip.category.lower()
        
        # DRUMS
        if category_lower == 'drums':
            if 'kick' in name_lower:
                strip.highpass_freq = 30
                strip.gate_enabled = True
                strip.gate_threshold = -35  # Gentle gate
                strip.eq_bands = [
                    {'freq': 60, 'gain': 2.5, 'q': 0.7},   # Sub punch
                    {'freq': 80, 'gain': 1.5, 'q': 0.8},   # Fundamental thump
                    {'freq': 150, 'gain': -1, 'q': 0.8},   # Clean mud
                    {'freq': 3500, 'gain': 2, 'q': 0.8},   # Beater click
                ]
                strip.comp_enabled = True
                strip.comp_threshold = -15   # Moderate sensitivity
                strip.comp_ratio = 3         # Controlled but not crushing
                strip.comp_attack = 3        # Fast enough for control, slow enough for punch
                strip.comp_release = 50      # Quick recovery
                
                # Professional saturation - controlled
                strip.saturation_enabled = True
                strip.saturation_type = 'console'
                strip.saturation_drive = 0.3
                strip.saturation_mix = 0.2   # Subtle blend
                
                # Enhanced transients - moderate
                strip.transient_enabled = True
                strip.transient_attack = 0.4     # Enhance but don't overdo
                strip.transient_sustain = -0.1   # Slight tightening
                
            elif 'snare' in name_lower:
                strip.highpass_freq = 80
                strip.gate_enabled = True
                strip.gate_threshold = -30  # Gentle gate
                strip.eq_bands = [
                    {'freq': 200, 'gain': 2, 'q': 0.8},   # Body/thump
                    {'freq': 800, 'gain': -1, 'q': 0.7},  # Control ring
                    {'freq': 2500, 'gain': 2.5, 'q': 0.8}, # Crack/snap
                    {'freq': 5000, 'gain': 2, 'q': 0.7},  # High crack
                ]
                strip.comp_enabled = True
                strip.comp_threshold = -12  # Controlled
                strip.comp_ratio = 3.5      # Moderate aggression
                strip.comp_attack = 2       # Fast but not crushing
                strip.comp_release = 70     # Quick recovery
                strip.reverb_send = 0.08
                
                # Professional saturation - controlled
                strip.saturation_enabled = True
                strip.saturation_type = 'console'
                strip.saturation_drive = 0.25
                strip.saturation_mix = 0.15
                
                # Enhanced transients - moderate
                strip.transient_enabled = True
                strip.transient_attack = 0.5     # Good crack enhancement
                strip.transient_sustain = -0.2   # Tighten the tail
                
            elif 'hihat' in name_lower or 'hat' in name_lower:
                strip.highpass_freq = 200
                strip.eq_bands = [
                    {'freq': 500, 'gain': -3, 'q': 0.7},  # Remove mud
                    {'freq': 8000, 'gain': 2, 'q': 0.6}   # Sizzle
                ]
                strip.pan = 0.3  # Slight right
                
        # BASS
        elif category_lower == 'bass':
            strip.highpass_freq = 25
            strip.eq_bands = [
                {'freq': 80, 'gain': 2, 'q': 0.7},    # Fundamental
                {'freq': 250, 'gain': -1, 'q': 0.8},  # Remove mud
                {'freq': 800, 'gain': 1.5, 'q': 0.8}, # Definition
            ]
            strip.comp_enabled = True
            strip.comp_threshold = -15
            strip.comp_ratio = 4
            strip.comp_attack = 10
            strip.comp_release = 100
            strip.saturation_enabled = True
            strip.saturation_type = 'tube'
            strip.saturation_drive = 0.3
            strip.saturation_mix = 0.3
            
        # VOCALS
        elif 'vocal' in category_lower:
            strip.highpass_freq = 80
            strip.gate_enabled = True
            strip.gate_threshold = -35
            strip.eq_bands = [
                {'freq': 150, 'gain': -1.5, 'q': 0.7}, # Remove mud
                {'freq': 2500, 'gain': 2, 'q': 0.8},   # Presence
                {'freq': 10000, 'gain': 2, 'q': 0.5}   # Air
            ]
            strip.comp_enabled = True
            strip.comp_threshold = -18
            strip.comp_ratio = 3
            strip.comp_attack = 10
            strip.comp_release = 100
            strip.reverb_send = 0.15
            
        # GUITARS
        elif 'guitar' in category_lower:
            strip.highpass_freq = 60
            strip.eq_bands = [
                {'freq': 200, 'gain': -2, 'q': 0.7},  # Remove mud
                {'freq': 2000, 'gain': 1.5, 'q': 0.8}, # Presence
                {'freq': 5000, 'gain': 1, 'q': 0.6}   # Brightness
            ]
            if 'electric' in name_lower:
                strip.saturation_enabled = True
                strip.saturation_type = 'tube'
                strip.saturation_drive = 0.4
                strip.saturation_mix = 0.4
                strip.pan = -0.3 if 'rhythm' in name_lower else 0.3
                
        # KEYS
        elif category_lower == 'keys':
            strip.highpass_freq = 50
            strip.eq_bands = [
                {'freq': 150, 'gain': -1, 'q': 0.7},   # Remove mud
                {'freq': 1000, 'gain': 1, 'q': 0.8},   # Definition
                {'freq': 8000, 'gain': 1.5, 'q': 0.6}  # Sparkle
            ]
            strip.comp_enabled = True
            strip.comp_threshold = -20
            strip.comp_ratio = 2
            strip.comp_attack = 20
            strip.comp_release = 150
            strip.reverb_send = 0.1
            if 'piano' in name_lower:
                strip.pan = -0.2 if '1' in name_lower else 0.2
                
        # SYNTHS
        elif category_lower == 'synths':
            strip.highpass_freq = 40
            strip.eq_bands = [
                {'freq': 120, 'gain': -1.5, 'q': 0.8}, # Clean up low end
                {'freq': 2000, 'gain': 1, 'q': 0.7},   # Presence
                {'freq': 12000, 'gain': 2, 'q': 0.5}   # Air/sparkle
            ]
            if 'pad' in name_lower:
                strip.comp_enabled = True
                strip.comp_threshold = -25
                strip.comp_ratio = 1.5
                strip.comp_attack = 50
                strip.comp_release = 200
                strip.reverb_send = 0.2
                strip.pan = -0.4 if '2' in name_lower else 0.4
            else:  # Rhythmic synths
                strip.comp_enabled = True
                strip.comp_threshold = -18
                strip.comp_ratio = 2.5
                strip.comp_attack = 5
                strip.comp_release = 80
                
        # FX
        elif category_lower == 'fx':
            strip.highpass_freq = 60
            # FX processing is usually minimal to preserve character
            strip.eq_bands = [
                {'freq': 200, 'gain': -0.5, 'q': 0.5}  # Gentle cleanup
            ]
            # Most FX don't need compression
            strip.comp_enabled = False
            # Spread FX across stereo field
            if 'fx1' in name_lower or 'fx3' in name_lower or 'fx5' in name_lower:
                strip.pan = -0.6
            elif 'fx2' in name_lower or 'fx4' in name_lower or 'fx6' in name_lower:
                strip.pan = 0.6
                
    def _setup_default_routing(self):
        """Setup default bus routing"""
        # Create drum bus
        drum_channels = [s for id, s in self.channel_strips.items() if 'drums.' in id]
        if drum_channels:
            self.buses['drums'] = ProMixBus(
                name='Drum Bus',
                channels=drum_channels
            )
            # Intelligent bus compression - glue without killing punch
            self.buses['drums'].comp_enabled = True
            self.buses['drums'].comp_threshold = -12   # Less sensitive - only catch peaks
            self.buses['drums'].comp_ratio = 2.5       # Gentle glue compression
            self.buses['drums'].comp_attack = 8.0      # Slow enough to preserve transients
            self.buses['drums'].comp_release = 80.0    # Natural release
            
            # Controlled parallel compression - punch without over-squashing
            self.buses['drums'].parallel_comp_enabled = True
            self.buses['drums'].parallel_comp_threshold = -25  # Only heavy hits
            self.buses['drums'].parallel_comp_ratio = 8        # Strong but not crushing
            self.buses['drums'].parallel_comp_attack = 0.1     # Fast attack
            self.buses['drums'].parallel_comp_release = 50     # Quick release
            self.buses['drums'].parallel_comp_mix = 0.35       # Balanced blend - punch without muddiness
            
            # Controlled saturation for harmonic glue
            self.buses['drums'].saturation_enabled = True
            self.buses['drums'].saturation_amount = 0.25       # Subtle harmonic enhancement
            
            # Intelligent bus EQ - enhance without over-processing
            self.buses['drums'].eq_bands = [
                {'freq': 80, 'gain': 1.0, 'q': 0.7},    # Subtle kick enhancement
                {'freq': 200, 'gain': 0.5, 'q': 0.6},   # Gentle snare body
                {'freq': 2500, 'gain': 1.0, 'q': 0.5},  # Snare presence
                {'freq': 8000, 'gain': 1.5, 'q': 0.4}   # Air and clarity
            ]
            
        # Create bass bus
        bass_channels = [s for id, s in self.channel_strips.items() if 'bass.' in id]
        if bass_channels:
            self.buses['bass'] = ProMixBus(
                name='Bass Bus',
                channels=bass_channels
            )
            self.buses['bass'].comp_enabled = True
            self.buses['bass'].multiband_enabled = True
            self.buses['bass'].multiband_crossovers = [100, 500]
            self.buses['bass'].multiband_settings = [
                {'threshold': -15, 'ratio': 4, 'attack': 10, 'release': 100, 'gain': 1.0},
                {'threshold': -18, 'ratio': 2, 'attack': 5, 'release': 80, 'gain': 1.0},
                {'threshold': -20, 'ratio': 1.5, 'attack': 3, 'release': 50, 'gain': 1.0}
            ]
            
        # Create vocal bus
        vocal_channels = [s for id, s in self.channel_strips.items() if 'vocal' in id]
        if vocal_channels:
            self.buses['vocals'] = ProMixBus(
                name='Vocal Bus',
                channels=vocal_channels
            )
            self.buses['vocals'].comp_enabled = True
            self.buses['vocals'].comp_threshold = -15
            self.buses['vocals'].comp_ratio = 2
            self.buses['vocals'].stereo_spread_enabled = True
            self.buses['vocals'].width = 1.2
            
        # Create guitar bus
        guitar_channels = [s for id, s in self.channel_strips.items() if 'guitars.' in id]
        if guitar_channels:
            self.buses['guitars'] = ProMixBus(
                name='Guitar Bus',
                channels=guitar_channels
            )
            self.buses['guitars'].comp_enabled = True
            self.buses['guitars'].comp_threshold = -18
            self.buses['guitars'].comp_ratio = 1.5
            self.buses['guitars'].saturation_enabled = True
            self.buses['guitars'].saturation_amount = 0.3
            
        # Create keys bus  
        keys_channels = [s for id, s in self.channel_strips.items() if 'keys.' in id]
        if keys_channels:
            self.buses['keys'] = ProMixBus(
                name='Keys Bus', 
                channels=keys_channels
            )
            self.buses['keys'].comp_enabled = True
            self.buses['keys'].comp_threshold = -20
            self.buses['keys'].comp_ratio = 1.8
            self.buses['keys'].stereo_spread_enabled = True
            self.buses['keys'].width = 1.4
            
        # Create synths bus
        synth_channels = [s for id, s in self.channel_strips.items() if 'synths.' in id]
        if synth_channels:
            self.buses['synths'] = ProMixBus(
                name='Synths Bus',
                channels=synth_channels
            )
            self.buses['synths'].comp_enabled = True
            self.buses['synths'].comp_threshold = -16
            self.buses['synths'].comp_ratio = 2.5
            self.buses['synths'].stereo_spread_enabled = True
            self.buses['synths'].width = 1.3
            
        # Create FX bus
        fx_channels = [s for id, s in self.channel_strips.items() if 'fx.' in id]
        if fx_channels:
            self.buses['fx'] = ProMixBus(
                name='FX Bus',
                channels=fx_channels
            )
            self.buses['fx'].comp_enabled = False  # FX usually don't need compression
            self.buses['fx'].stereo_spread_enabled = True
            self.buses['fx'].width = 1.5
            
    def apply_sidechain_compression(self):
        """Setup sidechain compression (kick ducking bass)"""
        kick_strip = None
        bass_strips = []
        
        for ch_id, strip in self.channel_strips.items():
            if 'kick' in ch_id.lower():
                kick_strip = strip
            elif 'bass' in ch_id.lower():
                bass_strips.append(strip)
        
        if kick_strip and bass_strips:
            print("🔗 Setting up sidechain compression (kick → bass)")
            # This will be used during processing
            self.sidechain_routing = {
                'kick': kick_strip,
                'targets': bass_strips
            }
            
    def process_mix(self, output_dir: str) -> Dict:
        """Process the complete professional mix"""
        print("\n🎚️ Processing professional mix...")
        start_time = time.time()
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Process buses
        bus_outputs = {}
        for bus_name, bus in self.buses.items():
            print(f"  Processing {bus_name}...")
            
            # Apply sidechain if configured
            if hasattr(self, 'sidechain_routing') and bus_name == 'bass':
                # Process with sidechain
                kick_audio = self.sidechain_routing['kick'].process()
                
                # Process each bass channel with sidechain
                bass_sum = None
                for bass_strip in self.sidechain_routing['targets']:
                    bass_audio = bass_strip.process(sidechain_input=kick_audio)
                    if bass_sum is None:
                        bass_sum = bass_audio
                    else:
                        min_len = min(len(bass_sum), len(bass_audio))
                        bass_sum = bass_sum[:min_len] + bass_audio[:min_len]
                
                bus_outputs[bus_name] = bass_sum
            else:
                bus_outputs[bus_name] = bus.process()
        
        # Create full mix (unmastered)
        print("  Creating full mix...")
        unmastered_mix = self._create_full_mix(bus_outputs)
        
        # Save unmastered mix
        print("  Saving unmastered mix...")
        mix_path = os.path.join(output_dir, "mix.wav")
        sf.write(mix_path, unmastered_mix, self.sample_rate, subtype='PCM_24')
        
        # Create and save mixed stems
        print("  Creating mixed stems...")
        stem_paths = self._create_mixed_stems(bus_outputs, output_dir)
        
        # Apply master processing
        print("  Applying master processing...")
        mastered_mix = self._apply_master_processing(unmastered_mix.copy())
        
        # Save mastered version
        print("  Saving mastered version...")
        master_path = os.path.join(output_dir, "master.wav")
        sf.write(master_path, mastered_mix, self.sample_rate, subtype='PCM_24')
        
        # Analysis (use mastered version for analysis)
        peak_db = linear_to_db(np.max(np.abs(mastered_mix)))
        rms_db = linear_to_db(np.sqrt(np.mean(mastered_mix**2)))
        
        results = {
            'peak_db': peak_db,
            'rms_db': rms_db,
            'time': time.time() - start_time,
            'mix_file': mix_path,
            'master_file': master_path,
            'stem_files': stem_paths,
            'output_file': master_path  # For compatibility
        }
        
        print(f"\n✅ Professional mix complete!")
        print(f"  Peak: {peak_db:.1f} dBFS")
        print(f"  RMS: {rms_db:.1f} dBFS")
        print(f"  Time: {results['time']:.1f}s")
        print(f"\n📁 Files created:")
        print(f"  • mix.wav (unmastered)")
        print(f"  • master.wav (final mastered)")
        print(f"  • Mixed stems:")
        for stem_name, stem_path in stem_paths.items():
            print(f"    - {stem_name}_stem.wav")
        
        return results
    
    def _create_full_mix(self, bus_outputs: Dict) -> np.ndarray:
        """Sum all buses to create full mix"""
        mix = None
        for bus_name, bus_audio in bus_outputs.items():
            if mix is None:
                mix = bus_audio
            else:
                min_len = min(len(mix), len(bus_audio))
                mix = mix[:min_len] + bus_audio[:min_len]
        
        return mix if mix is not None else np.array([[0.0, 0.0]])
    
    def _create_mixed_stems(self, bus_outputs: Dict, output_dir: str) -> Dict[str, str]:
        """Create mixed stems and save them"""
        stem_paths = {}
        
        # Create stem combinations
        stems = {
            'drums': ['drums'],
            'bass': ['bass'], 
            'vocals': ['vocals'],
            'music': ['guitars', 'keys', 'synths', 'fx']
        }
        
        for stem_name, bus_names in stems.items():
            stem_mix = None
            
            # Sum buses for this stem
            for bus_name in bus_names:
                if bus_name in bus_outputs:
                    bus_audio = bus_outputs[bus_name]
                    if stem_mix is None:
                        stem_mix = bus_audio
                    else:
                        min_len = min(len(stem_mix), len(bus_audio))
                        stem_mix = stem_mix[:min_len] + bus_audio[:min_len]
            
            # Save stem if we have audio
            if stem_mix is not None:
                stem_path = os.path.join(output_dir, f"{stem_name}_stem.wav")
                sf.write(stem_path, stem_mix, self.sample_rate, subtype='PCM_24')
                stem_paths[stem_name] = stem_path
                print(f"  ✓ {stem_name}_stem.wav saved")
            else:
                print(f"  ⚠️ No audio for {stem_name} stem")
        
        return stem_paths
    
    def _apply_master_processing(self, audio: np.ndarray) -> np.ndarray:
        """Apply intelligent mastering chain - enhance without destroying"""
        processed = audio.copy()
        
        # 1. Surgical Master EQ - subtle corrections only
        processed = shelf_filter(processed, self.sample_rate, 80, 0.3, 'high')  # Gentle low cleanup
        processed = peaking_eq(processed, self.sample_rate, 250, -0.3, 0.5)     # Subtle mud control
        processed = shelf_filter(processed, self.sample_rate, 12000, 0.8, 'high') # Air but not harsh
        
        # 2. Gentle Master Compression - glue without squashing
        processed = compressor(
            processed, self.sample_rate,
            threshold_db=-18, ratio=1.3, attack_ms=50,      # Much gentler
            release_ms=150, knee_db=3                        # Smooth knee
        )
        
        # 3. Subtle Master Saturation - warmth without distortion
        saturated = analog_console_saturation(
            processed, self.sample_rate, 0.1, 'clean'       # Clean console, minimal drive
        )
        processed = processed * 0.9 + saturated * 0.1       # Very subtle blend
        
        # 4. Conservative Stereo Enhancement - width without mono issues
        processed = stereo_spreader(processed, self.sample_rate, 1.05, 120)  # Slight width only
        
        # 5. Intelligent Limiter - catch peaks without destroying transients
        processed = compressor(
            processed, self.sample_rate,
            threshold_db=-1.0, ratio=10, attack_ms=0.1,      # Gentler limiting
            release_ms=80, knee_db=0.5                       # Soft knee
        )
        
        return processed
    
    def apply_mix_configuration(self, config_name: str):
        """Apply a specific mix configuration preset"""
        configs = self._get_mix_configurations()
        if config_name not in configs:
            raise ValueError(f"Configuration '{config_name}' not found")
        
        config = configs[config_name]
        
        # Apply channel-level settings
        if 'channels' in config:
            for pattern, settings in config['channels'].items():
                for ch_id, strip in self.channel_strips.items():
                    if pattern in ch_id.lower() or pattern == 'all':
                        for param, value in settings.items():
                            if hasattr(strip, param):
                                setattr(strip, param, value)
        
        # Apply bus-level settings  
        if 'buses' in config:
            for bus_name, settings in config['buses'].items():
                if bus_name in self.buses:
                    bus = self.buses[bus_name]
                    for param, value in settings.items():
                        if hasattr(bus, param):
                            setattr(bus, param, value)
    
    def _get_mix_configurations(self) -> Dict:
        """Define all mix configuration presets"""
        return {
            "modern_punchy": {
                "description": "Modern punchy mix with fast compression and bright EQ",
                "channels": {
                    "kick": {
                        "comp_attack": 0.5,
                        "comp_ratio": 6,
                        "transient_attack": 0.8,
                        "saturation_drive": 0.6
                    },
                    "snare": {
                        "comp_attack": 0.3,
                        "comp_ratio": 5,
                        "transient_attack": 0.9,
                        "saturation_drive": 0.5
                    },
                    "vocal": {
                        "comp_attack": 5,
                        "comp_ratio": 4,
                        "saturation_drive": 0.2
                    }
                },
                "buses": {
                    "drums": {
                        "comp_attack": 2.0,
                        "comp_ratio": 4,
                        "parallel_comp_mix": 0.7,
                        "saturation_amount": 0.5
                    }
                }
            },
            
            "vintage_warm": {
                "description": "Vintage warm mix with slow compression and warm saturation",
                "channels": {
                    "kick": {
                        "comp_attack": 10,
                        "comp_ratio": 3,
                        "transient_attack": 0.3,
                        "saturation_type": "tape",
                        "saturation_drive": 0.7
                    },
                    "snare": {
                        "comp_attack": 8,
                        "comp_ratio": 2.5,
                        "transient_attack": 0.4,
                        "saturation_type": "tape",
                        "saturation_drive": 0.6
                    },
                    "vocal": {
                        "comp_attack": 20,
                        "comp_ratio": 2,
                        "saturation_type": "tube",
                        "saturation_drive": 0.5
                    }
                },
                "buses": {
                    "drums": {
                        "comp_attack": 15.0,
                        "comp_ratio": 2,
                        "parallel_comp_mix": 0.3,
                        "saturation_amount": 0.6
                    }
                }
            },
            
            "clean_transparent": {
                "description": "Clean transparent mix with minimal processing",
                "channels": {
                    "all": {
                        "comp_ratio": 1.5,
                        "saturation_enabled": False,
                        "transient_attack": 0.1
                    }
                },
                "buses": {
                    "drums": {
                        "comp_ratio": 1.8,
                        "parallel_comp_mix": 0.2,
                        "saturation_amount": 0.1
                    },
                    "vocals": {
                        "comp_ratio": 1.5,
                        "saturation_enabled": False
                    }
                }
            },
            
            "heavy_compressed": {
                "description": "Heavy compressed mix with aggressive limiting",
                "channels": {
                    "kick": {
                        "comp_attack": 0.1,
                        "comp_ratio": 8,
                        "comp_threshold": -8
                    },
                    "snare": {
                        "comp_attack": 0.1,
                        "comp_ratio": 6,
                        "comp_threshold": -6
                    },
                    "vocal": {
                        "comp_attack": 2,
                        "comp_ratio": 6,
                        "comp_threshold": -12
                    }
                },
                "buses": {
                    "drums": {
                        "comp_ratio": 6,
                        "parallel_comp_ratio": 20,
                        "parallel_comp_mix": 0.8
                    }
                }
            },
            
            "wide_ambient": {
                "description": "Wide ambient mix with spacious reverbs",
                "channels": {
                    "vocal": {
                        "reverb_send": 0.3
                    },
                    "snare": {
                        "reverb_send": 0.2
                    }
                },
                "buses": {
                    "vocals": {
                        "width": 1.5,
                        "stereo_spread_enabled": True
                    },
                    "keys": {
                        "width": 1.6
                    },
                    "synths": {
                        "width": 1.8
                    },
                    "fx": {
                        "width": 2.0
                    }
                }
            },
            
            "tight_focused": {
                "description": "Tight focused mix with narrow stereo",
                "buses": {
                    "drums": {
                        "width": 0.8,
                        "comp_attack": 1.0,
                        "comp_ratio": 5
                    },
                    "vocals": {
                        "width": 0.9,
                        "stereo_spread_enabled": False
                    },
                    "keys": {
                        "width": 0.7
                    },
                    "synths": {
                        "width": 0.8
                    }
                }
            },
            
            "bright_modern": {
                "description": "Bright modern mix with high-end emphasis",
                "channels": {
                    "vocal": {
                        "eq_bands": [
                            {'freq': 150, 'gain': -2, 'q': 0.7},
                            {'freq': 3000, 'gain': 3, 'q': 0.8},
                            {'freq': 12000, 'gain': 4, 'q': 0.5}
                        ]
                    }
                },
                "buses": {
                    "drums": {
                        "eq_bands": [
                            {'freq': 80, 'gain': 2.0, 'q': 0.8},
                            {'freq': 200, 'gain': 1.5, 'q': 0.7},
                            {'freq': 3000, 'gain': 2.0, 'q': 0.6},
                            {'freq': 10000, 'gain': 3.0, 'q': 0.6}
                        ]
                    }
                }
            },
            
            "dark_moody": {
                "description": "Dark moody mix with low-mid focus",
                "buses": {
                    "drums": {
                        "eq_bands": [
                            {'freq': 60, 'gain': 3.0, 'q': 0.8},
                            {'freq': 150, 'gain': 2.0, 'q': 0.7},
                            {'freq': 8000, 'gain': -1.0, 'q': 0.6}
                        ]
                    },
                    "vocals": {
                        "eq_bands": [
                            {'freq': 200, 'gain': 1.0, 'q': 0.7},
                            {'freq': 10000, 'gain': -2.0, 'q': 0.5}
                        ]
                    }
                }
            },
            
            "dynamic_natural": {
                "description": "Dynamic natural mix preserving dynamics",
                "channels": {
                    "all": {
                        "comp_threshold": -25,
                        "comp_ratio": 1.8,
                        "comp_attack": 15
                    }
                },
                "buses": {
                    "drums": {
                        "comp_threshold": -15,
                        "comp_ratio": 2,
                        "parallel_comp_enabled": False
                    }
                }
            },
            
            "saturated_analog": {
                "description": "Heavily saturated analog-style mix",
                "channels": {
                    "all": {
                        "saturation_enabled": True,
                        "saturation_drive": 0.6,
                        "saturation_type": "tape"
                    },
                    "vocal": {
                        "saturation_type": "tube",
                        "saturation_drive": 0.7
                    }
                },
                "buses": {
                    "drums": {
                        "saturation_amount": 0.7
                    },
                    "bass": {
                        "saturation_enabled": True,
                        "saturation_amount": 0.5
                    }
                }
            },
            
            # NEW CONFIGURATIONS 11-20
            "radio_ready": {
                "description": "Radio-ready mix with optimized loudness and clarity",
                "channels": {
                    "kick": {
                        "comp_attack": 1.0,
                        "comp_ratio": 4.5,
                        "comp_threshold": -10,
                        "eq_bands": [
                            {'freq': 60, 'gain': 2.0, 'q': 0.8},
                            {'freq': 5000, 'gain': 1.5, 'q': 0.6}
                        ]
                    },
                    "vocal": {
                        "comp_attack": 3,
                        "comp_ratio": 3.5,
                        "eq_bands": [
                            {'freq': 2500, 'gain': 2.0, 'q': 0.7},
                            {'freq': 8000, 'gain': 1.5, 'q': 0.5}
                        ]
                    }
                },
                "buses": {
                    "drums": {
                        "comp_ratio": 3.5,
                        "parallel_comp_mix": 0.6
                    }
                }
            },
            
            "club_system": {
                "description": "Club system mix with powerful low end and punch",
                "channels": {
                    "kick": {
                        "comp_attack": 0.3,
                        "comp_ratio": 8,
                        "saturation_drive": 0.8,
                        "eq_bands": [
                            {'freq': 50, 'gain': 4.0, 'q': 1.0},
                            {'freq': 80, 'gain': 2.0, 'q': 0.8}
                        ]
                    },
                    "bass": {
                        "comp_attack": 2,
                        "comp_ratio": 6,
                        "saturation_drive": 0.6
                    }
                },
                "buses": {
                    "drums": {
                        "comp_ratio": 6,
                        "parallel_comp_ratio": 15,
                        "parallel_comp_mix": 0.8
                    },
                    "bass": {
                        "saturation_amount": 0.7
                    }
                }
            },
            
            "acoustic_natural": {
                "description": "Natural acoustic mix with minimal processing",
                "channels": {
                    "all": {
                        "comp_ratio": 1.8,
                        "comp_attack": 20,
                        "saturation_enabled": False,
                        "transient_attack": 0.0
                    },
                    "vocal": {
                        "comp_ratio": 2.5,
                        "comp_attack": 15,
                        "reverb_send": 0.15
                    }
                },
                "buses": {
                    "drums": {
                        "comp_ratio": 1.5,
                        "parallel_comp_enabled": False,
                        "saturation_amount": 0.0
                    },
                    "vocals": {
                        "width": 1.1,
                        "reverb_send": 0.2
                    }
                }
            },
            
            "metal_aggressive": {
                "description": "Metal mix with aggressive compression and saturation",
                "channels": {
                    "kick": {
                        "comp_attack": 0.1,
                        "comp_ratio": 10,
                        "saturation_drive": 0.9,
                        "transient_attack": 1.0
                    },
                    "snare": {
                        "comp_attack": 0.2,
                        "comp_ratio": 8,
                        "saturation_drive": 0.8,
                        "transient_attack": 0.9
                    },
                    "guitar": {
                        "comp_ratio": 4,
                        "saturation_drive": 0.7
                    }
                },
                "buses": {
                    "drums": {
                        "comp_ratio": 8,
                        "parallel_comp_ratio": 25,
                        "parallel_comp_mix": 0.9,
                        "saturation_amount": 0.8
                    },
                    "guitars": {
                        "saturation_amount": 0.6,
                        "width": 1.4
                    }
                }
            },
            
            "jazz_smooth": {
                "description": "Smooth jazz mix with warm compression and space",
                "channels": {
                    "all": {
                        "comp_attack": 25,
                        "comp_ratio": 2.2,
                        "saturation_type": "tube",
                        "saturation_drive": 0.3
                    },
                    "vocal": {
                        "comp_attack": 30,
                        "comp_ratio": 2.8,
                        "reverb_send": 0.25
                    }
                },
                "buses": {
                    "drums": {
                        "comp_attack": 20,
                        "comp_ratio": 2,
                        "parallel_comp_mix": 0.2
                    },
                    "keys": {
                        "width": 1.3,
                        "reverb_send": 0.15
                    },
                    "vocals": {
                        "width": 1.0,
                        "reverb_send": 0.3
                    }
                }
            },
            
            "hiphop_punchy": {
                "description": "Hip-hop mix with punchy drums and tight vocals",
                "channels": {
                    "kick": {
                        "comp_attack": 0.5,
                        "comp_ratio": 6,
                        "saturation_drive": 0.7,
                        "eq_bands": [
                            {'freq': 60, 'gain': 3.0, 'q': 0.9},
                            {'freq': 2500, 'gain': 1.0, 'q': 0.6}
                        ]
                    },
                    "snare": {
                        "comp_attack": 0.3,
                        "comp_ratio": 5,
                        "transient_attack": 0.8,
                        "eq_bands": [
                            {'freq': 200, 'gain': 2.0, 'q': 0.8},
                            {'freq': 5000, 'gain': 2.5, 'q': 0.7}
                        ]
                    },
                    "vocal": {
                        "comp_attack": 2,
                        "comp_ratio": 4.5,
                        "saturation_drive": 0.3
                    }
                },
                "buses": {
                    "drums": {
                        "comp_ratio": 5,
                        "parallel_comp_mix": 0.7
                    }
                }
            },
            
            "electronic_wide": {
                "description": "Electronic mix with wide stereo field and effects",
                "channels": {
                    "synth": {
                        "width": 1.6,
                        "comp_ratio": 3,
                        "saturation_drive": 0.4
                    },
                    "vocal": {
                        "reverb_send": 0.2,
                        "comp_ratio": 3.5
                    }
                },
                "buses": {
                    "synths": {
                        "width": 1.8,
                        "stereo_spread_enabled": True,
                        "reverb_send": 0.15
                    },
                    "fx": {
                        "width": 2.0,
                        "reverb_send": 0.25
                    },
                    "vocals": {
                        "width": 1.2,
                        "reverb_send": 0.3
                    },
                    "keys": {
                        "width": 1.5
                    }
                }
            },
            
            "vintage_70s": {
                "description": "70s vintage mix with classic tape saturation",
                "channels": {
                    "all": {
                        "saturation_type": "tape",
                        "saturation_drive": 0.8,
                        "comp_attack": 15,
                        "comp_ratio": 2.5
                    },
                    "kick": {
                        "comp_attack": 12,
                        "comp_ratio": 3,
                        "eq_bands": [
                            {'freq': 80, 'gain': 1.5, 'q': 0.8},
                            {'freq': 10000, 'gain': -1.0, 'q': 0.5}
                        ]
                    },
                    "vocal": {
                        "saturation_type": "tube",
                        "saturation_drive": 0.6,
                        "comp_attack": 20,
                        "reverb_send": 0.2
                    }
                },
                "buses": {
                    "drums": {
                        "saturation_amount": 0.9,
                        "comp_attack": 18,
                        "parallel_comp_mix": 0.4
                    }
                }
            },
            
            "broadcast_clear": {
                "description": "Broadcast-ready mix with speech clarity focus",
                "channels": {
                    "vocal": {
                        "comp_attack": 1,
                        "comp_ratio": 5,
                        "comp_threshold": -12,
                        "eq_bands": [
                            {'freq': 100, 'gain': -1.0, 'q': 0.7},
                            {'freq': 3000, 'gain': 3.0, 'q': 0.8},
                            {'freq': 6000, 'gain': 2.0, 'q': 0.6}
                        ]
                    },
                    "backing_vocal": {
                        "comp_ratio": 6,
                        "eq_bands": [
                            {'freq': 300, 'gain': -2.0, 'q': 0.8}
                        ]
                    }
                },
                "buses": {
                    "vocals": {
                        "comp_ratio": 3,
                        "width": 0.9
                    },
                    "music": {
                        "comp_ratio": 2,
                        "eq_bands": [
                            {'freq': 2500, 'gain': -1.5, 'q': 0.6}
                        ]
                    }
                }
            },
            
            "lo_fi_character": {
                "description": "Lo-fi mix with vintage character and warmth",
                "channels": {
                    "all": {
                        "saturation_enabled": True,
                        "saturation_type": "tape",
                        "saturation_drive": 0.9,
                        "eq_bands": [
                            {'freq': 12000, 'gain': -3.0, 'q': 0.5},
                            {'freq': 8000, 'gain': -2.0, 'q': 0.7}
                        ]
                    },
                    "vocal": {
                        "comp_attack": 10,
                        "comp_ratio": 3,
                        "saturation_drive": 0.7
                    }
                },
                "buses": {
                    "drums": {
                        "saturation_amount": 1.0,
                        "eq_bands": [
                            {'freq': 80, 'gain': 2.0, 'q': 0.8},
                            {'freq': 10000, 'gain': -4.0, 'q': 0.5}
                        ]
                    },
                    "music": {
                        "saturation_amount": 0.8,
                        "width": 0.8
                    }
                }
            }
        }
    
    def process_multiple_configurations(self, base_output_dir: str) -> Dict:
        """Process the mix with all 20 configurations"""
        configs = self._get_mix_configurations()
        results = {}
        
        print(f"🎛️ PROCESSING 20 MIX CONFIGURATIONS")
        print("=" * 50)
        
        for i, (config_name, config_data) in enumerate(configs.items(), 1):
            print(f"\n[{i}/20] Processing: {config_name}")
            print(f"Description: {config_data['description']}")
            
            # Create output directory for this config
            config_dir = os.path.join(base_output_dir, f"{i:02d}_{config_name}")
            
            # Store original settings
            original_settings = self._backup_current_settings()
            
            try:
                # Apply configuration
                self.apply_mix_configuration(config_name)
                
                # Process mix
                result = self.process_mix(config_dir)
                results[config_name] = {
                    'result': result,
                    'directory': config_dir,
                    'description': config_data['description']
                }
                
                print(f"✅ {config_name} complete!")
                
            except Exception as e:
                print(f"❌ Error processing {config_name}: {e}")
                results[config_name] = {
                    'error': str(e),
                    'directory': config_dir,
                    'description': config_data['description']
                }
            
            finally:
                # Restore original settings
                self._restore_settings(original_settings)
        
        print(f"\n🏆 ALL 20 CONFIGURATIONS COMPLETE!")
        print(f"📁 Output directory: {base_output_dir}")
        
        return results
    
    def _backup_current_settings(self) -> Dict:
        """Backup current channel and bus settings"""
        backup = {
            'channels': {},
            'buses': {}
        }
        
        # Backup channel settings
        for ch_id, strip in self.channel_strips.items():
            backup['channels'][ch_id] = {}
            for attr in dir(strip):
                if not attr.startswith('_') and not callable(getattr(strip, attr)):
                    try:
                        backup['channels'][ch_id][attr] = getattr(strip, attr)
                    except:
                        pass
        
        # Backup bus settings  
        for bus_name, bus in self.buses.items():
            backup['buses'][bus_name] = {}
            for attr in dir(bus):
                if not attr.startswith('_') and not callable(getattr(bus, attr)):
                    try:
                        backup['buses'][bus_name][attr] = getattr(bus, attr)
                    except:
                        pass
        
        return backup
    
    def _restore_settings(self, backup: Dict):
        """Restore channel and bus settings from backup"""
        # Restore channel settings
        for ch_id, settings in backup['channels'].items():
            if ch_id in self.channel_strips:
                strip = self.channel_strips[ch_id]
                for attr, value in settings.items():
                    if hasattr(strip, attr):
                        try:
                            setattr(strip, attr, value)
                        except:
                            pass
        
        # Restore bus settings
        for bus_name, settings in backup['buses'].items():
            if bus_name in self.buses:
                bus = self.buses[bus_name]
                for attr, value in settings.items():
                    if hasattr(bus, attr):
                        try:
                            setattr(bus, attr, value)
                        except:
                            pass