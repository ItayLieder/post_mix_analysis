#!/usr/bin/env python3
"""
Professional Reverb and Spatial Processing Engine
Implements studio-quality reverb algorithms and spatial effects
"""

import numpy as np
from scipy import signal
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ReverbParameters:
    """Professional reverb parameters"""
    room_size: float = 0.5      # 0-1
    damping: float = 0.5        # High frequency damping
    pre_delay: float = 20       # ms
    decay_time: float = 2.0     # seconds
    diffusion: float = 0.7      # 0-1
    density: float = 0.8        # 0-1
    early_level: float = 0.3    # Early reflections level
    tail_level: float = 0.7     # Reverb tail level
    width: float = 1.0          # Stereo width
    high_cut: float = 8000      # Hz
    low_cut: float = 100        # Hz


class AllPassFilter:
    """All-pass filter for reverb diffusion"""
    
    def __init__(self, delay_samples: int, feedback: float = 0.7):
        self.delay_line = np.zeros(delay_samples)
        self.feedback = feedback
        self.write_pos = 0
    
    def process(self, input_sample: float) -> float:
        # Read from delay line
        delayed = self.delay_line[self.write_pos]
        
        # Calculate output
        output = -input_sample * self.feedback + delayed
        
        # Write to delay line
        self.delay_line[self.write_pos] = input_sample + delayed * self.feedback
        
        # Update write position
        self.write_pos = (self.write_pos + 1) % len(self.delay_line)
        
        return output


class CombFilter:
    """Comb filter for reverb"""
    
    def __init__(self, delay_samples: int, feedback: float = 0.7, damping: float = 0.2):
        self.delay_line = np.zeros(delay_samples)
        self.feedback = feedback
        self.damping = damping
        self.filter_state = 0.0
        self.write_pos = 0
    
    def process(self, input_sample: float) -> float:
        # Read from delay line
        delayed = self.delay_line[self.write_pos]
        
        # Apply damping filter (one-pole lowpass)
        self.filter_state += (delayed - self.filter_state) * (1 - self.damping)
        
        # Calculate output
        output = input_sample + self.filter_state * self.feedback
        
        # Write to delay line
        self.delay_line[self.write_pos] = output
        
        # Update write position
        self.write_pos = (self.write_pos + 1) % len(self.delay_line)
        
        return delayed


class ReverbTank:
    """Freeverb-style reverb tank"""
    
    def __init__(self, sr: int, params: ReverbParameters):
        self.sr = sr
        self.params = params
        
        # Comb filter delays (in samples)
        comb_delays = [
            int(0.025 * sr * params.room_size),   # 25ms base
            int(0.027 * sr * params.room_size),
            int(0.031 * sr * params.room_size),
            int(0.033 * sr * params.room_size),
            int(0.037 * sr * params.room_size),
            int(0.041 * sr * params.room_size),
            int(0.043 * sr * params.room_size),
            int(0.047 * sr * params.room_size),
        ]
        
        # All-pass delays
        allpass_delays = [
            int(0.005 * sr),    # 5ms
            int(0.017 * sr),    # 17ms
            int(0.03 * sr),     # 30ms
            int(0.061 * sr),    # 61ms
        ]
        
        feedback = 0.28 + (params.decay_time - 1.0) * 0.1
        feedback = np.clip(feedback, 0.1, 0.98)
        
        # Create filters for left channel
        self.left_combs = [
            CombFilter(delay, feedback, params.damping) 
            for delay in comb_delays
        ]
        self.left_allpass = [
            AllPassFilter(delay, 0.5) 
            for delay in allpass_delays
        ]
        
        # Create filters for right channel (slightly different delays for stereo)
        right_comb_delays = [d + 23 for d in comb_delays]  # Offset for stereo
        right_allpass_delays = [d + 7 for d in allpass_delays]
        
        self.right_combs = [
            CombFilter(delay, feedback, params.damping) 
            for delay in right_comb_delays
        ]
        self.right_allpass = [
            AllPassFilter(delay, 0.5) 
            for delay in right_allpass_delays
        ]
        
        # Pre-delay buffer
        predelay_samples = int(params.pre_delay * sr / 1000)
        self.predelay_buffer = np.zeros(max(1, predelay_samples))
        self.predelay_pos = 0
        
    def process_mono(self, input_mono: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Process mono input and return stereo reverb"""
        left_out = np.zeros_like(input_mono)
        right_out = np.zeros_like(input_mono)
        
        for i, sample in enumerate(input_mono):
            # Pre-delay
            delayed = self.predelay_buffer[self.predelay_pos]
            self.predelay_buffer[self.predelay_pos] = sample
            self.predelay_pos = (self.predelay_pos + 1) % len(self.predelay_buffer)
            
            # Left channel processing
            left_comb_sum = 0
            for comb in self.left_combs:
                left_comb_sum += comb.process(delayed * 0.015)
            
            # All-pass chain
            left_processed = left_comb_sum
            for allpass in self.left_allpass:
                left_processed = allpass.process(left_processed)
            
            # Right channel processing
            right_comb_sum = 0
            for comb in self.right_combs:
                right_comb_sum += comb.process(delayed * 0.015)
            
            # All-pass chain
            right_processed = right_comb_sum
            for allpass in self.right_allpass:
                right_processed = allpass.process(right_processed)
            
            left_out[i] = left_processed
            right_out[i] = right_processed
        
        return left_out, right_out


def create_impulse_response(
    sr: int, 
    length_seconds: float = 3.0,
    room_type: str = 'hall'
) -> np.ndarray:
    """Create realistic impulse response for convolution reverb"""
    
    length_samples = int(length_seconds * sr)
    
    if room_type == 'plate':
        # Plate reverb: bright, metallic
        t = np.linspace(0, length_seconds, length_samples)
        decay = np.exp(-t * 3.0)  # Fast decay
        noise = np.random.normal(0, 1, length_samples)
        
        # Emphasize high frequencies
        b, a = signal.butter(2, 2000, 'high', fs=sr)
        noise = signal.filtfilt(b, a, noise)
        
        ir = noise * decay
        
    elif room_type == 'chamber':
        # Chamber: warm, musical
        t = np.linspace(0, length_seconds, length_samples)
        decay = np.exp(-t * 1.5)  # Slow decay
        noise = np.random.normal(0, 1, length_samples)
        
        # Warm tone
        b, a = signal.butter(2, 4000, 'low', fs=sr)
        noise = signal.filtfilt(b, a, noise)
        
        ir = noise * decay
        
    else:  # hall
        # Concert hall: natural, spacious
        t = np.linspace(0, length_seconds, length_samples)
        
        # Early reflections
        early = np.zeros(length_samples)
        early_times = [0.01, 0.023, 0.037, 0.054, 0.079]
        for et in early_times:
            idx = min(int(et * sr), length_samples - 1)
            early[idx] = np.random.uniform(0.1, 0.4)
        
        # Reverb tail
        tail_start = int(0.1 * sr)
        tail = np.random.normal(0, 1, length_samples - tail_start)
        
        # Natural decay
        decay_curve = np.exp(-t[tail_start:] * 0.8)
        tail *= decay_curve
        
        # Combine
        ir = np.concatenate([early[:tail_start], tail])
        
        # Natural frequency response
        b, a = signal.butter(1, 8000, 'low', fs=sr)
        ir = signal.filtfilt(b, a, ir)
    
    # Normalize
    ir = ir / np.max(np.abs(ir))
    
    return ir


class ConvolutionReverb:
    """High-quality convolution reverb"""
    
    def __init__(self, sr: int, impulse_response: np.ndarray, 
                 wet_level: float = 0.3, pre_delay_ms: float = 0):
        self.sr = sr
        self.ir = impulse_response
        self.wet_level = wet_level
        
        # Pre-delay
        predelay_samples = int(pre_delay_ms * sr / 1000)
        self.predelay_buffer = np.zeros(max(1, predelay_samples))
        self.predelay_pos = 0
        
        # For overlap-add convolution
        self.block_size = 2048
        self.hop_size = self.block_size // 2
        self.input_buffer = np.zeros(self.block_size)
        self.output_buffer = np.zeros(self.block_size + len(self.ir) - 1)
        self.position = 0
        
    def process(self, audio: np.ndarray) -> np.ndarray:
        """Process audio through convolution reverb"""
        if audio.ndim == 2:
            # Stereo - process each channel
            left_reverb = self._process_mono(audio[:, 0])
            right_reverb = self._process_mono(audio[:, 1])
            reverb = np.stack([left_reverb, right_reverb], axis=-1)
        else:
            # Mono
            reverb_mono = self._process_mono(audio)
            reverb = np.stack([reverb_mono, reverb_mono], axis=-1)
        
        return reverb * self.wet_level
    
    def _process_mono(self, mono_audio: np.ndarray) -> np.ndarray:
        """Process mono audio"""
        # Simple convolution for now
        reverb = signal.fftconvolve(mono_audio, self.ir, mode='same')
        
        # Apply pre-delay if needed
        if len(self.predelay_buffer) > 1:
            delayed_reverb = np.zeros_like(reverb)
            for i in range(len(reverb)):
                delayed = self.predelay_buffer[self.predelay_pos]
                self.predelay_buffer[self.predelay_pos] = reverb[i]
                self.predelay_pos = (self.predelay_pos + 1) % len(self.predelay_buffer)
                delayed_reverb[i] = delayed
            reverb = delayed_reverb
        
        return reverb


class SpatialProcessor:
    """Advanced spatial processing for mixing"""
    
    def __init__(self, sr: int):
        self.sr = sr
    
    def create_3d_position(self, audio: np.ndarray, 
                          azimuth: float = 0,      # -180 to 180 degrees
                          elevation: float = 0,     # -90 to 90 degrees
                          distance: float = 1.0     # 0.1 to 10
                          ) -> np.ndarray:
        """Position audio in 3D space"""
        
        if audio.ndim == 1:
            audio = np.stack([audio, audio], axis=-1)
        
        # Convert to radians
        az_rad = np.radians(azimuth)
        el_rad = np.radians(elevation)
        
        # Calculate HRTF approximation
        left_gain = np.cos(az_rad - np.pi/4) * np.cos(el_rad)
        right_gain = np.cos(az_rad + np.pi/4) * np.cos(el_rad)
        
        # Distance attenuation
        distance_gain = 1.0 / (1.0 + distance * 0.5)
        
        # Air absorption (high frequency loss with distance)
        if distance > 1.0:
            from dsp_premitives import shelf_filter
            audio = shelf_filter(audio, self.sr, 8000, 
                               -distance * 2, 'high')
        
        # Apply positioning
        positioned = audio.copy()
        positioned[:, 0] *= left_gain * distance_gain
        positioned[:, 1] *= right_gain * distance_gain
        
        return positioned
    
    def add_room_reflection(self, audio: np.ndarray,
                          room_size: float = 0.5,
                          absorption: float = 0.3) -> np.ndarray:
        """Add early reflections for room simulation"""
        
        # Calculate reflection delays based on room size
        base_delay_ms = room_size * 50  # 5-50ms range
        
        reflections = [
            {'delay_ms': base_delay_ms * 0.3, 'gain': 0.6 * (1 - absorption)},
            {'delay_ms': base_delay_ms * 0.7, 'gain': 0.4 * (1 - absorption)},
            {'delay_ms': base_delay_ms * 1.0, 'gain': 0.3 * (1 - absorption)},
            {'delay_ms': base_delay_ms * 1.4, 'gain': 0.2 * (1 - absorption)},
        ]
        
        reflected = audio.copy()
        
        for refl in reflections:
            delay_samples = int(refl['delay_ms'] * self.sr / 1000)
            if delay_samples < len(audio):
                # Create delayed version
                delayed = np.pad(audio, (delay_samples, 0), mode='constant')[:len(audio)]
                
                # Add to mix
                reflected += delayed * refl['gain']
        
        return reflected


# Preset reverbs for easy use
REVERB_PRESETS = {
    'vocal_hall': ReverbParameters(
        room_size=0.8, damping=0.3, pre_delay=30,
        decay_time=2.5, early_level=0.2, tail_level=0.8,
        width=1.2, high_cut=10000, low_cut=80
    ),
    'drum_room': ReverbParameters(
        room_size=0.4, damping=0.7, pre_delay=10,
        decay_time=1.2, early_level=0.4, tail_level=0.6,
        width=1.0, high_cut=8000, low_cut=150
    ),
    'plate': ReverbParameters(
        room_size=0.6, damping=0.1, pre_delay=20,
        decay_time=2.0, early_level=0.1, tail_level=0.9,
        width=1.4, high_cut=15000, low_cut=60
    ),
    'chamber': ReverbParameters(
        room_size=0.5, damping=0.5, pre_delay=25,
        decay_time=1.8, early_level=0.3, tail_level=0.7,
        width=1.1, high_cut=6000, low_cut=100
    )
}


def create_reverb_send(sr: int, reverb_type: str = 'vocal_hall') -> 'ReverbTank':
    """Create a reverb send with preset parameters"""
    params = REVERB_PRESETS.get(reverb_type, REVERB_PRESETS['vocal_hall'])
    return ReverbTank(sr, params)