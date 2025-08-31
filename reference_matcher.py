#!/usr/bin/env python3
"""
Reference Mix Matcher
Takes reference analysis and adjusts our professional mixing to match the characteristics
"""

import numpy as np
from typing import Dict, List, Optional
from reference_analyzer import ReferenceAnalysis, ReferenceMixAnalyzer
from pro_mixing_engine import ProMixingSession


class ReferenceMatcher:
    """Intelligently match our mix to reference characteristics"""
    
    def __init__(self):
        self.analyzer = ReferenceMixAnalyzer()
    
    def analyze_and_match(self, reference_path: str, pro_session: ProMixingSession, 
                         stem_paths: Optional[Dict[str, str]] = None) -> Dict:
        """Analyze reference and adjust our mix to match"""
        print("🎯 REFERENCE MATCHING SYSTEM")
        print("="*50)
        
        # Analyze reference full mix
        reference_analysis = self.analyzer.analyze_reference(reference_path)
        
        # Print reference analysis
        from reference_analyzer import print_reference_analysis
        print_reference_analysis(reference_analysis)
        
        # Analyze stems if provided
        stem_analysis = None
        if stem_paths:
            print(f"\n🎛️ ANALYZING REFERENCE STEMS...")
            stem_analysis = self._analyze_reference_stems(stem_paths)
            self._print_stem_analysis(stem_analysis)
        
        # Generate matching strategy
        matching_strategy = self._create_matching_strategy(reference_analysis, stem_analysis)
        
        # Apply strategy to our mix
        self._apply_matching_strategy(pro_session, matching_strategy, stem_analysis)
        
        return {
            'reference_analysis': reference_analysis,
            'stem_analysis': stem_analysis,
            'matching_strategy': matching_strategy
        }
    
    def _analyze_reference_stems(self, stem_paths: Dict[str, str]) -> Dict:
        """Analyze individual reference stems"""
        import soundfile as sf
        
        stem_analysis = {}
        
        for stem_name, path in stem_paths.items():
            try:
                print(f"  🔍 Analyzing {stem_name} stem...")
                
                # Load stem
                audio, sr = sf.read(path)
                
                # Analyze stem characteristics
                analysis = self.analyzer.analyze_reference(path)
                
                # Calculate relative level (compared to other stems)
                peak_level = np.max(np.abs(audio))
                rms_level = np.sqrt(np.mean(audio**2))
                
                stem_analysis[stem_name] = {
                    'analysis': analysis,
                    'peak_level': peak_level,
                    'rms_level': rms_level,
                    'level_db': 20 * np.log10(rms_level) if rms_level > 0 else -60,
                    'audio': audio,
                    'sr': sr
                }
                
            except Exception as e:
                print(f"    ⚠️ Could not analyze {stem_name}: {e}")
                continue
        
        # Calculate relative balance between stems
        if len(stem_analysis) > 1:
            levels = [data['level_db'] for data in stem_analysis.values()]
            max_level = max(levels)
            
            for stem_name, data in stem_analysis.items():
                data['relative_level_db'] = data['level_db'] - max_level
                
        print(f"    ✅ Analyzed {len(stem_analysis)} stems")
        return stem_analysis
    
    def _print_stem_analysis(self, stem_analysis: Dict):
        """Print stem analysis results"""
        print(f"\n📊 REFERENCE STEM ANALYSIS:")
        print("-" * 40)
        
        # Show relative levels
        print("🔊 STEM LEVELS (relative):")
        for stem_name, data in stem_analysis.items():
            rel_level = data.get('relative_level_db', 0)
            print(f"  • {stem_name:10}: {rel_level:+5.1f} dB")
        
        # Show frequency characteristics
        print(f"\n🎵 FREQUENCY CHARACTERISTICS:")
        for stem_name, data in stem_analysis.items():
            analysis = data['analysis']
            balance = analysis.frequency_balance
            dominant = f"{balance.get('bass', -60):+4.0f} / {balance.get('mids', -60):+4.0f} / {balance.get('presence', -60):+4.0f}"
            print(f"  • {stem_name:10}: {dominant} dB (B/M/H)")
        
        # Show dynamics
        print(f"\n⚡ DYNAMICS:")
        for stem_name, data in stem_analysis.items():
            analysis = data['analysis']
            print(f"  • {stem_name:10}: {analysis.dynamic_range:4.1f}dB range, {analysis.punch_factor:.2f} punch")

    def _create_matching_strategy(self, ref_analysis: ReferenceAnalysis, stem_analysis: Optional[Dict] = None) -> Dict:
        """Create strategy to match reference characteristics"""
        print("\n🎯 CREATING MATCHING STRATEGY...")
        
        strategy = {
            'compression_strategy': self._plan_compression_matching(ref_analysis, stem_analysis),
            'eq_strategy': self._plan_eq_matching(ref_analysis, stem_analysis),
            'stereo_strategy': self._plan_stereo_matching(ref_analysis, stem_analysis),
            'saturation_strategy': self._plan_saturation_matching(ref_analysis, stem_analysis),
            'transient_strategy': self._plan_transient_matching(ref_analysis, stem_analysis),
            'master_strategy': self._plan_master_matching(ref_analysis, stem_analysis),
            'balance_strategy': self._plan_balance_matching(stem_analysis) if stem_analysis else None
        }
        
        return strategy
    
    def _plan_balance_matching(self, stem_analysis: Dict) -> Dict:
        """Plan balance adjustments based on reference stems"""
        if not stem_analysis:
            return {}
            
        print(f"  🎚️  Balance: Matching reference stem levels")
        
        balance_strategy = {}
        
        # Calculate target gains for each stem based on reference levels
        for stem_name, data in stem_analysis.items():
            rel_level = data.get('relative_level_db', 0)
            
            # Convert relative level to gain multiplier
            if stem_name == 'drums':
                balance_strategy['drums_target_gain'] = rel_level
            elif stem_name == 'bass':
                balance_strategy['bass_target_gain'] = rel_level
            elif stem_name == 'vocals':
                balance_strategy['vocals_target_gain'] = rel_level
            elif stem_name == 'music':
                balance_strategy['music_target_gain'] = rel_level
        
        return balance_strategy

    def _plan_compression_matching(self, ref_analysis: ReferenceAnalysis, stem_analysis: Optional[Dict] = None) -> Dict:
        """Plan compression settings to match reference dynamics"""
        strategy = {}
        
        # Determine compression approach based on reference dynamic range
        if ref_analysis.dynamic_range > 12:
            # Very dynamic reference - use minimal compression
            strategy['approach'] = 'minimal'
            strategy['channel_ratios'] = {'kick': 2.0, 'snare': 2.5, 'vocal': 2.0}
            strategy['bus_ratios'] = {'drums': 1.8, 'vocals': 1.5}
            strategy['parallel_blend'] = 0.2
            
        elif ref_analysis.dynamic_range > 8:
            # Moderately dynamic - balanced compression
            strategy['approach'] = 'balanced' 
            strategy['channel_ratios'] = {'kick': 3.0, 'snare': 3.5, 'vocal': 3.0}
            strategy['bus_ratios'] = {'drums': 2.5, 'vocals': 2.0}
            strategy['parallel_blend'] = 0.35
            
        else:
            # Heavily compressed reference - match the energy
            strategy['approach'] = 'aggressive'
            strategy['channel_ratios'] = {'kick': 4.0, 'snare': 4.5, 'vocal': 4.0}
            strategy['bus_ratios'] = {'drums': 3.5, 'vocals': 3.0}
            strategy['parallel_blend'] = 0.5
        
        # Adjust for punch factor
        if ref_analysis.punch_factor > 0.2:
            # Punchy reference - preserve transients
            strategy['attack_times'] = {'kick': 5, 'snare': 3, 'vocal': 8}
            strategy['transient_preservation'] = True
        else:
            # Smooth reference - can use faster attacks
            strategy['attack_times'] = {'kick': 2, 'snare': 1, 'vocal': 3}
            strategy['transient_preservation'] = False
        
        print(f"  🗜️  Compression: {strategy['approach']} approach")
        return strategy
    
    def _plan_eq_matching(self, ref_analysis: ReferenceAnalysis, stem_analysis: Optional[Dict] = None) -> Dict:
        """Plan EQ to match reference frequency balance"""
        strategy = {}
        
        # Target frequency balance from reference
        target_balance = ref_analysis.frequency_balance
        
        # Determine EQ approach
        bass_level = target_balance.get('bass', -50)
        mid_level = target_balance.get('mids', -50) 
        high_level = target_balance.get('presence', -50)
        
        # Bass strategy
        if bass_level > -35:
            strategy['bass_approach'] = 'boost'
            strategy['bass_boost'] = min(4, (bass_level + 40) / 5)  # Scale to reasonable boost
        else:
            strategy['bass_approach'] = 'control'
            strategy['bass_cut'] = min(3, (-30 - bass_level) / 10)
        
        # Mid strategy  
        if mid_level > -45:
            strategy['mid_approach'] = 'boost'
            strategy['mid_boost'] = min(3, (mid_level + 50) / 10)
        else:
            strategy['mid_approach'] = 'cut'
            strategy['mid_cut'] = min(4, (-40 - mid_level) / 10)
        
        # High strategy
        if high_level > -55:
            strategy['high_approach'] = 'boost'
            strategy['high_boost'] = min(4, (high_level + 60) / 10)
        else:
            strategy['high_approach'] = 'control'
            strategy['high_cut'] = min(2, (-50 - high_level) / 15)
        
        # Dominant frequency matching
        strategy['dominant_frequencies'] = ref_analysis.dominant_frequencies
        
        print(f"  🎛️  EQ: Bass {strategy.get('bass_approach', 'control')}, "
              f"Mid {strategy.get('mid_approach', 'control')}, "
              f"High {strategy.get('high_approach', 'control')}")
        
        return strategy
    
    def _plan_stereo_matching(self, ref_analysis: ReferenceAnalysis, stem_analysis: Optional[Dict] = None) -> Dict:
        """Plan stereo processing to match reference imaging"""
        strategy = {}
        
        target_width = ref_analysis.stereo_width
        target_correlation = ref_analysis.phase_correlation
        
        if target_width > 1.2:
            strategy['width_approach'] = 'widen'
            strategy['target_width'] = min(1.8, target_width)
        elif target_width < 0.3:
            strategy['width_approach'] = 'narrow'
            strategy['target_width'] = max(0.2, target_width)
        else:
            strategy['width_approach'] = 'natural'
            strategy['target_width'] = target_width
        
        # Phase correlation guidance
        if target_correlation < 0.5:
            strategy['phase_approach'] = 'decorrelated'
        elif target_correlation > 0.9:
            strategy['phase_approach'] = 'mono_like'
        else:
            strategy['phase_approach'] = 'balanced'
        
        print(f"  🔊 Stereo: {strategy['width_approach']} width ({target_width:.2f})")
        return strategy
    
    def _plan_saturation_matching(self, ref_analysis: ReferenceAnalysis, stem_analysis: Optional[Dict] = None) -> Dict:
        """Plan saturation to match reference character"""
        strategy = {}
        
        # Estimate saturation level from compression and clarity
        compression_level = ref_analysis.compression_estimate
        clarity = ref_analysis.clarity_score
        
        if compression_level > 0.6 and clarity < 0.7:
            # Heavy, warm sound
            strategy['approach'] = 'warm'
            strategy['type'] = 'tape'
            strategy['amount'] = 0.4
        elif clarity > 1.2:
            # Very clean, bright sound
            strategy['approach'] = 'clean'
            strategy['type'] = 'console'
            strategy['amount'] = 0.1
        else:
            # Balanced character
            strategy['approach'] = 'balanced'
            strategy['type'] = 'console'
            strategy['amount'] = 0.25
        
        print(f"  🔥 Saturation: {strategy['approach']} ({strategy['type']})")
        return strategy
    
    def _plan_transient_matching(self, ref_analysis: ReferenceAnalysis, stem_analysis: Optional[Dict] = None) -> Dict:
        """Plan transient processing to match reference punch"""
        strategy = {}
        
        punch_factor = ref_analysis.punch_factor
        attack_strength = ref_analysis.attack_characteristics['strength']
        
        if punch_factor > 0.3:
            # Very punchy reference
            strategy['approach'] = 'enhance'
            strategy['attack_enhancement'] = min(0.8, punch_factor * 2)
            strategy['sustain_control'] = -0.3
        elif punch_factor > 0.1:
            # Moderately punchy
            strategy['approach'] = 'moderate'
            strategy['attack_enhancement'] = min(0.5, punch_factor * 3)
            strategy['sustain_control'] = -0.1
        else:
            # Smooth reference
            strategy['approach'] = 'gentle'
            strategy['attack_enhancement'] = 0.1
            strategy['sustain_control'] = 0.0
        
        print(f"  ⚡ Transients: {strategy['approach']} (punch: {punch_factor:.2f})")
        return strategy
    
    def _plan_master_matching(self, ref_analysis: ReferenceAnalysis, stem_analysis: Optional[Dict] = None) -> Dict:
        """Plan master chain to match reference loudness and character"""
        strategy = {}
        
        target_lufs = ref_analysis.loudness_lufs
        target_peak = ref_analysis.peak_db
        dynamic_range = ref_analysis.dynamic_range
        
        # Master compression
        if dynamic_range < 8:
            strategy['master_comp_ratio'] = 1.8
            strategy['master_comp_threshold'] = -15
        else:
            strategy['master_comp_ratio'] = 1.3
            strategy['master_comp_threshold'] = -20
        
        # Limiting strategy
        if target_peak > -0.5:
            strategy['limiter_ceiling'] = -0.3
            strategy['limiter_ratio'] = 20
        else:
            strategy['limiter_ceiling'] = -1.0
            strategy['limiter_ratio'] = 10
        
        # Loudness target
        strategy['target_lufs'] = target_lufs
        
        print(f"  🎚️  Master: Target {target_lufs:.1f} LUFS, {dynamic_range:.1f}dB range")
        return strategy
    
    def _apply_matching_strategy(self, session: ProMixingSession, strategy: Dict, stem_analysis: Optional[Dict] = None):
        """Apply the matching strategy to our professional mix"""
        print(f"\n🔧 APPLYING MATCHING STRATEGY...")
        
        # Apply compression strategy
        self._apply_compression_strategy(session, strategy['compression_strategy'])
        
        # Apply EQ strategy
        self._apply_eq_strategy(session, strategy['eq_strategy'])
        
        # Apply stereo strategy
        self._apply_stereo_strategy(session, strategy['stereo_strategy'])
        
        # Apply saturation strategy
        self._apply_saturation_strategy(session, strategy['saturation_strategy'])
        
        # Apply transient strategy
        self._apply_transient_strategy(session, strategy['transient_strategy'])
        
        # Apply balance strategy if available
        if strategy.get('balance_strategy'):
            self._apply_balance_strategy(session, strategy['balance_strategy'])
        
        print("  ✅ All strategies applied!")
    
    def _apply_compression_strategy(self, session: ProMixingSession, strategy: Dict):
        """Apply compression matching strategy"""
        # Channel-level compression
        for ch_id, strip in session.channel_strips.items():
            if 'kick' in ch_id.lower() and 'kick' in strategy['channel_ratios']:
                strip.comp_ratio = strategy['channel_ratios']['kick']
                strip.comp_attack = strategy['attack_times']['kick']
                
            elif 'snare' in ch_id.lower() and 'snare' in strategy['channel_ratios']:
                strip.comp_ratio = strategy['channel_ratios']['snare']
                strip.comp_attack = strategy['attack_times']['snare']
                
            elif 'vocal' in ch_id.lower() and 'vocal' in strategy['channel_ratios']:
                strip.comp_ratio = strategy['channel_ratios']['vocal']
                strip.comp_attack = strategy['attack_times']['vocal']
        
        # Bus-level compression (if buses exist)
        if hasattr(session, 'buses'):
            for bus_name, bus in session.buses.items():
                if bus_name in strategy['bus_ratios']:
                    bus.comp_ratio = strategy['bus_ratios'][bus_name]
                
            if bus_name == 'drums' and hasattr(bus, 'parallel_comp_mix'):
                bus.parallel_comp_mix = strategy['parallel_blend']
        
        print("    ✓ Compression settings applied")
    
    def _apply_eq_strategy(self, session: ProMixingSession, strategy: Dict):
        """Apply EQ matching strategy"""
        # Apply to drum bus (if buses exist)
        if hasattr(session, 'buses') and 'drums' in session.buses:
            drum_bus = session.buses['drums']
            
            # Modify existing EQ bands based on strategy
            new_eq_bands = []
            
            # Bass adjustment
            if strategy.get('bass_approach') == 'boost':
                new_eq_bands.append({'freq': 80, 'gain': strategy.get('bass_boost', 2), 'q': 0.7})
            elif strategy.get('bass_approach') == 'control':
                new_eq_bands.append({'freq': 120, 'gain': -strategy.get('bass_cut', 1), 'q': 0.8})
            
            # Mid adjustment  
            if strategy.get('mid_approach') == 'boost':
                new_eq_bands.append({'freq': 1000, 'gain': strategy.get('mid_boost', 1.5), 'q': 0.6})
            elif strategy.get('mid_approach') == 'cut':
                new_eq_bands.append({'freq': 400, 'gain': -strategy.get('mid_cut', 2), 'q': 0.7})
            
            # High adjustment
            if strategy.get('high_approach') == 'boost':
                new_eq_bands.append({'freq': 8000, 'gain': strategy.get('high_boost', 2), 'q': 0.5})
            
            # Replace EQ bands
            drum_bus.eq_bands = new_eq_bands
        
        print("    ✓ EQ settings applied")
    
    def _apply_stereo_strategy(self, session: ProMixingSession, strategy: Dict):
        """Apply stereo matching strategy"""
        target_width = strategy.get('target_width', 1.0)
        
        # Apply to buses that support stereo processing (if buses exist)
        if hasattr(session, 'buses'):
            for bus_name, bus in session.buses.items():
                if hasattr(bus, 'width'):
                    if strategy.get('width_approach') == 'widen':
                        bus.width = min(2.0, target_width * 1.2)
                    bus.stereo_spread_enabled = True
                elif strategy.get('width_approach') == 'narrow':
                    bus.width = max(0.5, target_width)
                    bus.stereo_spread_enabled = False
                else:
                    bus.width = target_width
        
        print("    ✓ Stereo settings applied")
    
    def _apply_saturation_strategy(self, session: ProMixingSession, strategy: Dict):
        """Apply saturation matching strategy"""
        saturation_type = strategy.get('type', 'console')
        saturation_amount = strategy.get('amount', 0.2)
        
        # Apply to channel strips
        for ch_id, strip in session.channel_strips.items():
            if strip.saturation_enabled:
                strip.saturation_type = saturation_type
                strip.saturation_drive = saturation_amount
        
        # Apply to buses (if buses exist)
        if hasattr(session, 'buses'):
            for bus_name, bus in session.buses.items():
                if hasattr(bus, 'saturation_enabled') and bus.saturation_enabled:
                    bus.saturation_amount = saturation_amount * 0.8  # Slightly less on buses
        
        print("    ✓ Saturation settings applied")
    
    def _apply_transient_strategy(self, session: ProMixingSession, strategy: Dict):
        """Apply transient matching strategy"""
        attack_enhancement = strategy.get('attack_enhancement', 0.3)
        sustain_control = strategy.get('sustain_control', -0.1)
        
        # Apply to drum channels
        for ch_id, strip in session.channel_strips.items():
            if 'drum' in ch_id.lower() and hasattr(strip, 'transient_enabled'):
                if strategy.get('approach') == 'enhance':
                    strip.transient_enabled = True
                    strip.transient_attack = attack_enhancement
                    strip.transient_sustain = sustain_control
                elif strategy.get('approach') == 'gentle':
                    strip.transient_enabled = True
                    strip.transient_attack = attack_enhancement * 0.5
                    strip.transient_sustain = sustain_control * 0.5
        
        print("    ✓ Transient settings applied")
    
    def _apply_balance_strategy(self, session: ProMixingSession, strategy: Dict):
        """Apply balance strategy based on reference stems"""
        print("    🎚️ Applying reference stem balance...")
        
        # Apply bus-level adjustments based on stem analysis (if buses exist)
        if hasattr(session, 'buses'):
            for bus_name, bus in session.buses.items():
                target_key = f"{bus_name}_target_gain"
                if target_key in strategy:
                    target_gain_db = strategy[target_key]
                gain_multiplier = 10 ** (target_gain_db / 20)
                
                # Apply to all channels feeding this bus
                bus_channels = [ch_id for ch_id in session.channel_strips.keys() 
                               if bus_name in ch_id.lower()]
                
                for ch_id in bus_channels:
                    session.channel_strips[ch_id].gain *= gain_multiplier
                
                print(f"      • {bus_name}: {target_gain_db:+.1f} dB ({len(bus_channels)} channels)")
        
        print("    ✓ Balance settings applied")