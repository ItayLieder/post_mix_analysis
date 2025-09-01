#!/usr/bin/env python3
"""
Professional Mixing Engine
Multi-channel mixing system that processes raw tracks into balanced stems
"""

import numpy as np
import soundfile as sf
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import os
from dataclasses import dataclass
from scipy import signal

from utils import ensure_stereo, to_mono, db_to_linear, linear_to_db
from dsp_premitives import (
    peaking_eq as parametric_eq, shelf_filter, highpass_filter, lowpass_filter,
    compressor
)
# from processors import add_saturation, apply_stereo_width  # Not available


@dataclass
class ChannelStrip:
    """Individual channel processing strip"""
    name: str
    category: str
    audio: np.ndarray
    sample_rate: int
    
    # Processing parameters
    gain: float = 1.0
    pan: float = 0.0  # -1 to 1
    mute: bool = False
    solo: bool = False
    phase_invert: bool = False
    
    # EQ settings
    eq_enabled: bool = True
    eq_bands: Dict[str, Dict[str, float]] = None  # freq: {gain, q}
    
    # Dynamics
    gate_enabled: bool = False
    gate_threshold: float = -40.0
    
    comp_enabled: bool = False
    comp_threshold: float = -20.0
    comp_ratio: float = 3.0
    comp_attack: float = 10.0
    comp_release: float = 100.0
    
    # Effects sends (0-1)
    reverb_send: float = 0.0
    delay_send: float = 0.0
    
    def process(self) -> np.ndarray:
        """Process the channel through its strip"""
        if self.mute:
            return np.zeros_like(self.audio)
        
        processed = self.audio.copy()
        
        # Phase inversion
        if self.phase_invert:
            processed *= -1
        
        # Gate
        if self.gate_enabled:
            # Simple gate implementation using threshold
            threshold_linear = db_to_linear(self.gate_threshold)
            mask = np.abs(processed) > threshold_linear
            processed = processed * mask
        
        # EQ
        if self.eq_enabled and self.eq_bands:
            for freq_str, params in self.eq_bands.items():
                # Parse frequency string more robustly
                freq_str_clean = freq_str.lower()
                if 'khz' in freq_str_clean:
                    freq = float(freq_str_clean.replace('khz', '')) * 1000
                elif 'hz' in freq_str_clean:
                    freq = float(freq_str_clean.replace('hz', ''))
                elif 'k' in freq_str_clean:
                    freq = float(freq_str_clean.replace('k', '')) * 1000
                else:
                    try:
                        freq = float(freq_str_clean)
                    except ValueError:
                        print(f"Warning: Could not parse frequency '{freq_str}', skipping")
                        continue
                        
                gain_db = params.get('gain', 0)
                q = params.get('q', 0.7)
                if gain_db != 0:
                    processed = parametric_eq(processed, self.sample_rate, 
                                            freq, gain_db, q)
        
        # Compression
        if self.comp_enabled:
            processed = compressor(
                processed, self.sample_rate,
                threshold_db=self.comp_threshold,
                ratio=self.comp_ratio,
                attack_ms=self.comp_attack,
                release_ms=self.comp_release
            )
        
        # Apply gain
        processed *= self.gain
        
        # Aggressive clipping protection (especially for drums)
        peak = np.max(np.abs(processed))
        if peak > 0.7:  # Much earlier intervention
            # Aggressive soft limiting to prevent any clipping
            reduction_factor = 0.7 / peak  # Scale down to safe level
            processed = np.tanh(processed * reduction_factor) * 0.65  # Heavy limiting
            print(f"⚠️ Applied clipping protection to {self.name} (peak was {linear_to_db(peak):.1f}dBFS, reduced to -3.7dBFS)")
        
        # Apply pan
        stereo = ensure_stereo(processed)
        if self.pan != 0:
            # Equal power panning
            left_gain = np.cos((self.pan + 1) * np.pi / 4)
            right_gain = np.sin((self.pan + 1) * np.pi / 4)
            stereo[:, 0] *= left_gain
            stereo[:, 1] *= right_gain
        
        return stereo


@dataclass
class MixBus:
    """Bus for grouping and processing multiple channels"""
    name: str
    channels: List[ChannelStrip]
    
    # Bus processing
    eq_enabled: bool = False
    comp_enabled: bool = False
    comp_threshold: float = -15.0
    comp_ratio: float = 2.0
    glue_amount: float = 0.0  # 0-1
    saturation: float = 0.0   # 0-1
    width: float = 1.0        # 0-2
    
    def process(self) -> np.ndarray:
        """Sum and process all channels in the bus"""
        if not self.channels:
            return np.array([[0.0, 0.0]])
        
        # Sum all channels
        bus_sum = None
        for channel in self.channels:
            if not channel.mute:
                channel_out = channel.process()
                if bus_sum is None:
                    bus_sum = channel_out
                else:
                    # Ensure same length
                    min_len = min(len(bus_sum), len(channel_out))
                    bus_sum = bus_sum[:min_len] + channel_out[:min_len]
        
        if bus_sum is None:
            return np.array([[0.0, 0.0]])
        
        # Bus compression (glue)
        if self.comp_enabled or self.glue_amount > 0:
            # Gentler settings for bus compression
            bus_sum = compressor(
                bus_sum, self.channels[0].sample_rate,
                threshold_db=self.comp_threshold,
                ratio=1 + (self.comp_ratio - 1) * max(self.glue_amount, 0.5),
                attack_ms=30,
                release_ms=100,
                knee_db=2.0
            )
        
        # Saturation
        if self.saturation > 0:
            # Simple saturation using tanh
            if self.saturation > 0:
                bus_sum = np.tanh(bus_sum * (1 + self.saturation * 2)) / (1 + self.saturation * 0.5)
        
        # Width
        if self.width != 1.0:
            # Simple stereo width using mid-side processing
            if self.width != 1.0 and bus_sum.ndim == 2:
                from dsp_premitives import mid_side_encode, mid_side_decode
                mid, side = mid_side_encode(bus_sum)
                side *= self.width
                bus_sum = mid_side_decode(mid, side)
        
        return bus_sum


class MixingSession:
    """Complete mixing session manager"""
    
    def __init__(self, channels: Dict, template: str = "modern_pop",
                 template_params: Dict = None, sample_rate: int = 44100,
                 bit_depth: int = 24):
        self.raw_channels = channels
        self.template = template
        self.template_params = template_params or {}
        self.sample_rate = sample_rate
        self.bit_depth = bit_depth
        
        self.channel_strips = {}
        self.buses = {}
        self.master_bus = None
        
        # Load and prepare channels
        self._load_channels()
        
    def _load_channels(self):
        """Load audio files and create channel strips"""
        print("📁 Loading audio channels...")
        
        for category, tracks in self.raw_channels.items():
            if not tracks:
                continue
                
            for name, path_or_config in tracks.items():
                # Handle both simple path and advanced config
                if isinstance(path_or_config, dict):
                    path = path_or_config['path']
                    hints = path_or_config.get('hints', {})
                else:
                    path = path_or_config
                    hints = {}
                
                # Load audio
                try:
                    audio, sr = sf.read(path)
                    
                    # Resample if necessary
                    if sr != self.sample_rate:
                        from scipy import signal as sp
                        num_samples = int(len(audio) * self.sample_rate / sr)
                        audio = sp.resample(audio, num_samples)
                    
                    # Create channel strip
                    strip = ChannelStrip(
                        name=name,
                        category=category,
                        audio=audio,
                        sample_rate=self.sample_rate
                    )
                    
                    # Apply template settings
                    self._apply_template_to_channel(strip, hints)
                    
                    # Store
                    channel_id = f"{category}.{name}"
                    self.channel_strips[channel_id] = strip
                    
                    print(f"  ✓ Loaded: {channel_id}")
                    
                except Exception as e:
                    print(f"  ✗ Failed to load {name}: {e}")
    
    def _apply_template_to_channel(self, strip: ChannelStrip, hints: Dict):
        """Apply template-based processing to channel"""
        from mix_templates import get_template
        
        template = get_template(self.template)
        channel_type = self._identify_channel_type(strip.name, strip.category, hints)
        
        # Get template settings for this channel type
        settings = template.get_channel_settings(channel_type, self.template_params)
        
        # Apply settings to strip
        strip.eq_bands = settings.get('eq', {})
        strip.comp_enabled = settings.get('compression', {}).get('enabled', False)
        strip.comp_threshold = settings.get('compression', {}).get('threshold', -20)
        strip.comp_ratio = settings.get('compression', {}).get('ratio', 3)
        strip.gate_enabled = settings.get('gate', False)
        strip.pan = settings.get('pan', 0)
        strip.reverb_send = settings.get('reverb_send', 0)
        strip.delay_send = settings.get('delay_send', 0)
    
    def _identify_channel_type(self, name: str, category: str, hints: Dict) -> str:
        """Identify channel type from name, category, and hints"""
        name_lower = name.lower()
        
        # Check hints first
        if 'subtype' in hints:
            return hints['subtype']
        
        # Common patterns
        patterns = {
            'kick': ['kick', 'bd', 'bassdrum'],
            'snare': ['snare', 'sd', 'snr'],
            'hihat': ['hat', 'hh', 'hihat'],
            'tom': ['tom'],
            'overhead': ['oh', 'overhead'],
            'room': ['room', 'amb'],
            'bass': ['bass', 'sub'],
            'guitar_rhythm': ['rhythm', 'rhy'],
            'guitar_lead': ['lead', 'solo'],
            'piano': ['piano', 'pno'],
            'synth': ['synth', 'syn'],
            'vocal_lead': ['lead', 'main', 'vox'],
            'vocal_harmony': ['harm', 'bgv', 'backup'],
        }
        
        for ch_type, keywords in patterns.items():
            if any(kw in name_lower for kw in keywords):
                return ch_type
        
        # Default by category
        return category
    
    def analyze_all_channels(self) -> Dict:
        """Analyze all loaded channels"""
        analysis = {}
        
        for channel_id, strip in self.channel_strips.items():
            category, name = channel_id.split('.', 1)
            
            if category not in analysis:
                analysis[category] = {}
            
            # Analyze channel
            analysis[category][name] = self._analyze_channel(strip)
        
        return analysis
    
    def _analyze_channel(self, strip: ChannelStrip) -> Dict:
        """Analyze a single channel"""
        mono = to_mono(strip.audio)
        
        # Frequency analysis
        fft = np.fft.rfft(mono[:8192])
        freqs = np.fft.rfftfreq(8192, 1/strip.sample_rate)
        magnitude = np.abs(fft)
        
        # Find dominant frequency range
        peak_idx = np.argmax(magnitude)
        peak_freq = freqs[peak_idx]
        
        if peak_freq < 200:
            freq_range = "Low (Sub/Bass)"
        elif peak_freq < 2000:
            freq_range = "Mid"
        elif peak_freq < 8000:
            freq_range = "High-Mid" 
        else:
            freq_range = "High"
        
        # Dynamic range
        peak = np.max(np.abs(mono))
        rms = np.sqrt(np.mean(mono**2))
        dynamic_range = linear_to_db(peak) - linear_to_db(rms)
        
        # Suggested processing chain
        suggested_chain = []
        
        # Based on frequency content
        if peak_freq < 200 and "kick" in strip.name.lower():
            suggested_chain = ["Gate", "EQ", "Compression", "Saturation"]
        elif "vocal" in strip.category.lower():
            suggested_chain = ["Gate", "EQ", "De-esser", "Compression", "Reverb"]
        elif "guitar" in strip.category.lower():
            suggested_chain = ["EQ", "Compression", "Delay"]
        else:
            suggested_chain = ["EQ", "Compression"]
        
        return {
            "detected_type": self._identify_channel_type(strip.name, strip.category, {}),
            "freq_range": freq_range,
            "peak_freq": f"{peak_freq:.0f} Hz",
            "dynamic_range": dynamic_range,
            "suggested_chain": suggested_chain,
            "peak_level": linear_to_db(peak),
        }
    
    def configure(self, settings: Dict):
        """Apply mix configuration"""
        # Configure buses
        if 'buses' in settings:
            self._configure_buses(settings['buses'])
        
        # Configure sends
        if 'sends' in settings:
            self._configure_sends(settings['sends'])
        
        # Apply mix balance
        if 'mix_balance' in settings:
            self._apply_mix_balance(settings['mix_balance'])
        
        # Apply direct channel overrides (from GUI)
        if 'channel_overrides' in settings:
            self._apply_channel_overrides(settings['channel_overrides'])
        
        # Configure master
        if 'master' in settings:
            self._configure_master(settings['master'])
    
    def _configure_buses(self, bus_config: Dict):
        """Set up mix buses"""
        for bus_name, config in bus_config.items():
            # Find matching channels
            channels = []
            for pattern in config['channels']:
                for ch_id, strip in self.channel_strips.items():
                    if self._match_pattern(ch_id, pattern):
                        channels.append(strip)
            
            # Create bus
            bus = MixBus(
                name=bus_name,
                channels=channels,
                comp_enabled=config.get('compression', 0) > 0,
                glue_amount=config.get('glue', 0),
                saturation=config.get('saturation', 0),
                width=config.get('width', 1.0)
            )
            
            self.buses[bus_name] = bus
    
    def _match_pattern(self, channel_id: str, pattern: str) -> bool:
        """Match channel ID against pattern (supports wildcards)"""
        import fnmatch
        return fnmatch.fnmatch(channel_id, pattern)
    
    def _configure_sends(self, sends_config: Dict):
        """Configure effect sends"""
        for send_name, config in sends_config.items():
            for channel_pattern, amount in config.get('sends_from', {}).items():
                for ch_id, strip in self.channel_strips.items():
                    if self._match_pattern(ch_id, channel_pattern):
                        if 'reverb' in send_name:
                            strip.reverb_send = amount
                        elif 'delay' in send_name:
                            strip.delay_send = amount
    
    def _configure_master(self, master_config: Dict):
        """Configure master bus settings"""
        self.master_settings = master_config
    
    def _apply_mix_balance(self, balance_config: Dict):
        """Apply intelligent mix balance adjustments"""
        print("🎚️ Applying intelligent mix balance...")
        
        # Extract balance parameters (0.0 to 1.0) with safety limits
        vocal_prominence = balance_config.get('vocal_prominence', 0.5)
        drum_punch = min(balance_config.get('drum_punch', 0.5), 0.75)  # Cap at 0.75 to prevent clipping
        bass_foundation = balance_config.get('bass_foundation', 0.5)
        instrument_presence = balance_config.get('instrument_presence', 0.5)
        
        if balance_config.get('drum_punch', 0.5) > 0.75:
            print(f"    ⚠️ Drum punch capped at 75% (was {balance_config.get('drum_punch', 0.5):.1%}) to prevent clipping")
        
        print(f"  • Vocal prominence: {vocal_prominence:.1%}")
        print(f"  • Drum punch: {drum_punch:.1%}")
        print(f"  • Bass foundation: {bass_foundation:.1%}")
        print(f"  • Instrument presence: {instrument_presence:.1%}")
        
        # Apply adjustments to each channel
        for channel_id, strip in self.channel_strips.items():
            category = strip.category.lower()
            
            # Vocal adjustments
            if 'vocal' in category:
                self._apply_vocal_balance(strip, vocal_prominence)
            
            # Drum adjustments  
            elif 'drum' in category:
                self._apply_drum_balance(strip, drum_punch, vocal_prominence)
                
            # Bass adjustments
            elif 'bass' in category:
                self._apply_bass_balance(strip, bass_foundation, drum_punch)
                
            # Instrument adjustments (guitars, keys, synths, etc.)
            else:
                self._apply_instrument_balance(strip, instrument_presence, vocal_prominence)
    
    def _apply_vocal_balance(self, strip: ChannelStrip, prominence: float):
        """Apply intelligent vocal balance adjustments"""
        # Level adjustment (more sophisticated than simple gain)
        level_boost_db = (prominence - 0.5) * 4  # -2dB to +2dB range
        strip.gain *= db_to_linear(level_boost_db)
        
        # Presence adjustment (affects EQ and compression)
        if prominence > 0.6:
            # Forward vocals: boost presence, reduce reverb
            if not strip.eq_bands:
                strip.eq_bands = {}
            strip.eq_bands['3khz'] = {'gain': 1 + (prominence - 0.6) * 5, 'q': 0.8}  # Presence boost
            strip.reverb_send *= 0.8  # Less reverb for forward vocals
            
        elif prominence < 0.4:
            # Background vocals: gentle, more reverb  
            if not strip.eq_bands:
                strip.eq_bands = {}
            strip.eq_bands['200hz'] = {'gain': -1 - (0.4 - prominence) * 2, 'q': 0.7}  # Reduce muddiness
            strip.reverb_send *= 1.3  # More reverb for background vocals
            
        # Compression adjustment for consistency
        if prominence > 0.7:
            # More compressed for prominent vocals
            strip.comp_ratio = min(strip.comp_ratio * 1.2, 8.0)
            strip.comp_threshold -= 2.0
    
    def _apply_drum_balance(self, strip: ChannelStrip, punch: float, vocal_prominence: float):
        """Apply intelligent drum balance adjustments"""
        # Ultra-conservative level adjustment to prevent clipping
        punch_boost_db = (punch - 0.5) * 1.5  # -0.75dB to +0.75dB range (heavily reduced)
        strip.gain *= db_to_linear(punch_boost_db)
        
        # Punch-specific processing (more conservative)
        if punch > 0.6:
            # Punchy drums: more compression, transient enhancement
            strip.comp_ratio = min(strip.comp_ratio * 1.2, 6.0)  # Reduced from 10.0
            strip.comp_attack = max(strip.comp_attack * 0.8, 2.0)  # Less aggressive
            
            # Ultra-conservative EQ for punch (minimal gains to prevent clipping)
            if not strip.eq_bands:
                strip.eq_bands = {}
            
            # Very subtle punch frequencies based on drum type
            if 'kick' in strip.name.lower():
                strip.eq_bands['60hz'] = {'gain': 0.5 + punch * 0.5, 'q': 0.8}  # Max +0.75dB
                strip.eq_bands['4khz'] = {'gain': 0.3 + punch * 0.5, 'q': 0.9}  # Max +0.65dB  
            elif 'snare' in strip.name.lower():
                strip.eq_bands['200hz'] = {'gain': 0.3 + punch * 0.5, 'q': 0.8}  # Max +0.65dB
                strip.eq_bands['5khz'] = {'gain': 0.5 + punch * 0.8, 'q': 0.7}  # Max +1.06dB
        
        # Interaction with vocals (frequency masking consideration)
        if vocal_prominence > 0.6 and punch > 0.5:
            # Carve space for vocals in drum frequencies
            if not strip.eq_bands:
                strip.eq_bands = {}
            if 'snare' in strip.name.lower() or 'tom' in strip.name.lower():
                # Slight cut in vocal frequency range
                strip.eq_bands['1khz'] = {'gain': -0.5 - vocal_prominence, 'q': 0.6}
    
    def _apply_bass_balance(self, strip: ChannelStrip, foundation: float, drum_punch: float):
        """Apply intelligent bass balance adjustments"""
        # Level adjustment
        foundation_boost_db = (foundation - 0.5) * 5  # -2.5dB to +2.5dB
        strip.gain *= db_to_linear(foundation_boost_db)
        
        # Foundation-specific processing
        if foundation > 0.6:
            # Strong foundation: enhance low end, control muddiness
            if not strip.eq_bands:
                strip.eq_bands = {}
            strip.eq_bands['80hz'] = {'gain': 1 + foundation * 3, 'q': 0.7}  # Sub boost
            strip.eq_bands['500hz'] = {'gain': -1 - foundation, 'q': 0.8}   # Cut mud
            
            # More compression for consistency
            strip.comp_ratio = min(strip.comp_ratio * 1.2, 6.0)
        
        # Interaction with drums (rhythm section cohesion)
        if drum_punch > 0.6 and foundation > 0.5:
            # Sync with punchy drums - slight compression adjustment
            strip.comp_attack = max(strip.comp_attack * 0.9, 5.0)
    
    def _apply_instrument_balance(self, strip: ChannelStrip, presence: float, vocal_prominence: float):
        """Apply intelligent instrument balance adjustments"""
        # Level adjustment
        presence_boost_db = (presence - 0.5) * 3  # -1.5dB to +1.5dB
        strip.gain *= db_to_linear(presence_boost_db)
        
        # Presence-specific processing
        if presence > 0.6:
            # Present instruments: clarity and definition
            if not strip.eq_bands:
                strip.eq_bands = {}
            
            # Different treatment for different instruments
            if 'guitar' in strip.category.lower():
                strip.eq_bands['3khz'] = {'gain': 1 + presence * 2, 'q': 0.7}  # Presence
                strip.eq_bands['500hz'] = {'gain': -0.5 - presence, 'q': 0.8}  # Reduce mud
            elif 'keys' in strip.category.lower() or 'synth' in strip.category.lower():
                strip.eq_bands['5khz'] = {'gain': 0.5 + presence * 1.5, 'q': 0.6}  # Air
        
        elif presence < 0.4:
            # Background instruments: sit back in mix
            if not strip.eq_bands:
                strip.eq_bands = {}
            # Gentle high-frequency roll-off
            strip.eq_bands['8khz'] = {'gain': -0.5 - (0.4 - presence) * 2, 'q': 0.5}
            
        # Interaction with vocals (make space)
        if vocal_prominence > 0.6:
            if not strip.eq_bands:
                strip.eq_bands = {}
            # Subtle cut in vocal range for guitars/keys
            if 'guitar' in strip.category.lower() or 'keys' in strip.category.lower():
                strip.eq_bands['2khz'] = {'gain': -0.5 - vocal_prominence * 0.5, 'q': 0.5}

    def _apply_channel_overrides(self, channel_overrides: Dict[str, float]):
        """Apply direct channel-level gain overrides (from GUI)"""
        print("🎚️ Applying GUI channel overrides...")
        
        applied_count = 0
        for channel_id, gain_multiplier in channel_overrides.items():
            if channel_id in self.channel_strips and abs(gain_multiplier - 1.0) > 0.01:
                strip = self.channel_strips[channel_id]
                strip.gain *= gain_multiplier
                
                # Clamp to safe range
                strip.gain = min(max(strip.gain, 0.01), 3.0)
                
                change_pct = (gain_multiplier - 1.0) * 100
                direction = "↑" if change_pct > 0 else "↓"
                print(f"  {direction} {channel_id}: {gain_multiplier:.2f} ({change_pct:+.0f}%)")
                applied_count += 1
        
        print(f"  ✅ Applied {applied_count} channel overrides")

    def override_channel_settings(self, channel_id: str, settings: Dict):
        """Override specific channel settings"""
        if channel_id in self.channel_strips:
            strip = self.channel_strips[channel_id]
            
            # Apply overrides
            if 'eq' in settings:
                strip.eq_bands = settings['eq']
            if 'compression' in settings:
                strip.comp_enabled = True
                strip.comp_ratio = settings['compression'].get('ratio', 3)
                strip.comp_attack = settings['compression'].get('attack', 10)
            if 'gate' in settings:
                strip.gate_enabled = settings['gate']
    
    def process_mix(self, output_dir: str, export_individual_channels: bool = False,
                    export_buses: bool = True, export_stems: bool = True,
                    export_full_mix: bool = True, progress_callback=None) -> Dict:
        """Process the complete mix"""
        import time
        start_time = time.time()
        
        os.makedirs(output_dir, exist_ok=True)
        
        if progress_callback:
            progress_callback("Processing individual channels...")
        
        # Process channels through buses
        bus_outputs = {}
        for bus_name, bus in self.buses.items():
            if progress_callback:
                progress_callback(f"Processing {bus_name}...")
            bus_outputs[bus_name] = bus.process()
        
        # Create stem outputs
        stem_outputs = self._create_stems(bus_outputs)
        
        # Create full mix
        if export_full_mix:
            if progress_callback:
                progress_callback("Creating full mix...")
            full_mix = self._create_full_mix(stem_outputs)
            
            # Apply master processing
            full_mix = self._apply_master_processing(full_mix)
            
            # Export
            mix_path = os.path.join(output_dir, "full_mix.wav")
            sf.write(mix_path, full_mix, self.sample_rate, subtype='PCM_24')
        
        # Export stems
        if export_stems:
            if progress_callback:
                progress_callback("Exporting stems...")
            stem_dir = os.path.join(output_dir, "stems")
            os.makedirs(stem_dir, exist_ok=True)
            
            for stem_name, stem_audio in stem_outputs.items():
                stem_path = os.path.join(stem_dir, f"{stem_name}.wav")
                sf.write(stem_path, stem_audio, self.sample_rate, subtype='PCM_24')
        
        # Analysis
        peak_db = linear_to_db(np.max(np.abs(full_mix)))
        rms_db = linear_to_db(np.sqrt(np.mean(full_mix**2)))
        
        return {
            "peak_db": peak_db,
            "rms_db": rms_db,
            "lufs": rms_db - 0.691,  # Approximate
            "dynamic_range": peak_db - rms_db,
            "time": time.time() - start_time
        }
    
    def _create_stems(self, bus_outputs: Dict) -> Dict:
        """Create stems from bus outputs"""
        stems = {}
        
        # Map buses to stems
        stem_mapping = {
            "drums": ["drum_bus"],
            "bass": ["bass_bus"],
            "vocals": ["vocal_bus"],
            "music": ["instrument_bus"]
        }
        
        for stem_name, bus_names in stem_mapping.items():
            stem_sum = None
            for bus_name in bus_names:
                if bus_name in bus_outputs:
                    if stem_sum is None:
                        stem_sum = bus_outputs[bus_name]
                    else:
                        min_len = min(len(stem_sum), len(bus_outputs[bus_name]))
                        stem_sum = stem_sum[:min_len] + bus_outputs[bus_name][:min_len]
            
            if stem_sum is not None:
                stems[stem_name] = stem_sum
        
        return stems
    
    def _create_full_mix(self, stems: Dict) -> np.ndarray:
        """Sum all stems to create full mix"""
        full_mix = None
        
        for stem_audio in stems.values():
            if full_mix is None:
                full_mix = stem_audio.copy()
            else:
                min_len = min(len(full_mix), len(stem_audio))
                full_mix = full_mix[:min_len] + stem_audio[:min_len]
        
        return full_mix if full_mix is not None else np.array([[0.0, 0.0]])
    
    def _apply_master_processing(self, audio: np.ndarray) -> np.ndarray:
        """Apply master bus processing"""
        processed = audio.copy()
        
        if hasattr(self, 'master_settings'):
            # Master EQ
            if self.master_settings.get('eq_mode') == 'gentle':
                processed = shelf_filter(processed, self.sample_rate, 100, 0.5, 'high')
                processed = shelf_filter(processed, self.sample_rate, 10000, 0.5, 'high')
            
            # Master compression
            if self.master_settings.get('compression', 0) > 0:
                processed = compressor(
                    processed, self.sample_rate,
                    threshold_db=-10,
                    ratio=1.5,
                    attack_ms=30,
                    release_ms=100,
                    knee_db=2
                )
            
            # Limiter
            if self.master_settings.get('limiter'):
                # Simple limiter using compression with high ratio
                processed = compressor(processed, self.sample_rate, threshold_db=-1.0, 
                                     ratio=20.0, attack_ms=0.1, release_ms=10)
        
        return processed
    
    def export_stems(self, output_dir: str, stem_mapping: Dict,
                     config: Dict) -> Dict:
        """Export stems with specific configuration"""
        os.makedirs(output_dir, exist_ok=True)
        exported = {}
        
        # Process buses to stems
        bus_outputs = {}
        for bus_name, bus in self.buses.items():
            bus_outputs[bus_name] = bus.process()
        
        # Create and export stems
        for stem_name, bus_list in stem_mapping.items():
            stem_sum = None
            
            for bus_name in bus_list:
                if bus_name in bus_outputs:
                    if stem_sum is None:
                        stem_sum = bus_outputs[bus_name].copy()
                    else:
                        min_len = min(len(stem_sum), len(bus_outputs[bus_name]))
                        stem_sum = stem_sum[:min_len] + bus_outputs[bus_name][:min_len]
            
            if stem_sum is not None:
                # Apply normalization
                if config.get('normalization') == 'peak':
                    target = db_to_linear(config.get('target_level', -6))
                    current_peak = np.max(np.abs(stem_sum))
                    if current_peak > 0:
                        stem_sum *= (target / current_peak)
                
                # Export
                stem_path = os.path.join(output_dir, f"{stem_name}.wav")
                sf.write(stem_path, stem_sum, self.sample_rate, 
                        subtype=f"PCM_{config.get('bit_depth', 24)}")
                exported[stem_name] = stem_path
        
        return exported
    
    def plot_spectrum(self, ax):
        """Plot frequency spectrum of mix"""
        # Placeholder for visualization
        pass
    
    def plot_stereo_field(self, ax):
        """Plot stereo field visualization"""
        # Placeholder for visualization
        pass
    
    def plot_dynamics(self, ax):
        """Plot dynamic range visualization"""
        # Placeholder for visualization
        pass
    
    def plot_levels(self, ax):
        """Plot level meters"""
        # Placeholder for visualization
        pass


print("🎛️ Mixing Engine loaded!")
print("   • Multi-channel support with intelligent routing")
print("   • Template-based processing")
print("   • Bus architecture with grouping")
print("   • Exports stems for post-mix pipeline")