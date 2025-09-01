#!/usr/bin/env python3
"""
Intelligent Mix Analysis and Auto-Adjustment
AI-powered mixing decisions based on professional standards
"""

import numpy as np
from scipy import signal, stats
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json

from dsp_premitives import measure_peak, measure_rms, lufs_integrated_approx
# from advanced_dsp import frequency_slot_eq


@dataclass
class MixAnalysis:
    """Complete mix analysis results"""
    frequency_balance: Dict[str, float]
    dynamic_range: float
    stereo_width: float
    loudness_lufs: float
    peak_db: float
    rms_db: float
    issues: List[str]
    recommendations: List[Dict]
    masking_analysis: Dict
    phase_correlation: float


class MixAnalyzer:
    """Professional mix analyzer"""
    
    def __init__(self, sr: int = 44100):
        self.sr = sr
        
        # Define frequency bands for analysis
        self.frequency_bands = {
            'sub_bass': (20, 60),
            'bass': (60, 250), 
            'low_mids': (250, 500),
            'mids': (500, 2000),
            'upper_mids': (2000, 4000),
            'presence': (4000, 8000),
            'air': (8000, 20000)
        }
        
        # Professional mix standards
        self.standards = {
            'target_lufs': -14.0,
            'max_peak_db': -1.0,
            'min_dynamic_range': 8.0,
            'optimal_stereo_width': 1.2,
            'frequency_balance_tolerance': 3.0  # dB
        }
    
    def analyze_full_mix(self, audio: np.ndarray) -> MixAnalysis:
        """Complete analysis of a stereo mix"""
        
        # Basic measurements
        peak_db = 20 * np.log10(np.max(np.abs(audio)))
        rms_db = 20 * np.log10(np.sqrt(np.mean(audio**2)))
        dynamic_range = peak_db - rms_db
        
        # LUFS loudness
        lufs = lufs_integrated_approx(audio, self.sr)
        
        # Frequency analysis
        freq_balance = self._analyze_frequency_balance(audio)
        
        # Stereo analysis
        stereo_width, phase_corr = self._analyze_stereo(audio)
        
        # Masking analysis
        masking = self._analyze_masking(audio)
        
        # Identify issues
        issues = self._identify_issues(
            freq_balance, dynamic_range, stereo_width, 
            lufs, peak_db, phase_corr
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(issues, freq_balance)
        
        return MixAnalysis(
            frequency_balance=freq_balance,
            dynamic_range=dynamic_range,
            stereo_width=stereo_width,
            loudness_lufs=lufs,
            peak_db=peak_db,
            rms_db=rms_db,
            issues=issues,
            recommendations=recommendations,
            masking_analysis=masking,
            phase_correlation=phase_corr
        )
    
    def analyze_individual_channel(self, audio: np.ndarray, 
                                 channel_name: str) -> Dict:
        """Analyze individual channel for optimal processing"""
        
        # Basic measurements
        peak_db = 20 * np.log10(np.max(np.abs(audio)) + 1e-10)
        rms_db = 20 * np.log10(np.sqrt(np.mean(audio**2)) + 1e-10)
        
        # Frequency content analysis
        mono = audio if audio.ndim == 1 else np.mean(audio, axis=1)
        freqs, psd = signal.welch(mono, self.sr, nperseg=4096)
        
        # Find dominant frequencies
        dominant_freqs = []
        for i, power in enumerate(psd):
            if power > np.mean(psd) * 3:  # 3x above average
                dominant_freqs.append(freqs[i])
        
        # Transient analysis
        transient_ratio = self._analyze_transients(mono)
        
        # Recommended processing
        processing_rec = self._recommend_channel_processing(
            channel_name, peak_db, rms_db, dominant_freqs, transient_ratio
        )
        
        return {
            'peak_db': peak_db,
            'rms_db': rms_db,
            'dynamic_range': peak_db - rms_db,
            'dominant_frequencies': dominant_freqs,
            'transient_ratio': transient_ratio,
            'processing_recommendations': processing_rec
        }
    
    def auto_balance_mix(self, channels: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Automatically balance channel levels"""
        
        # Analyze each channel
        channel_analysis = {}
        for name, audio in channels.items():
            channel_analysis[name] = self.analyze_individual_channel(audio, name)
        
        # Calculate relative importance of each instrument
        importance_weights = self._calculate_importance_weights(channel_analysis)
        
        # Calculate optimal gains
        optimal_gains = {}
        
        # Reference level: drums should be prominent but not dominating
        ref_rms = -18.0  # Target RMS level
        
        for name, analysis in channel_analysis.items():
            current_rms = analysis['rms_db']
            importance = importance_weights[name]
            
            # Calculate target level based on importance
            target_rms = ref_rms + (importance - 0.5) * 6  # ±3dB range
            
            # Calculate required gain
            gain_db = target_rms - current_rms
            optimal_gains[name] = 10 ** (gain_db / 20)  # Convert to linear
        
        return optimal_gains
    
    def detect_frequency_conflicts(self, channels: Dict[str, np.ndarray]) -> Dict:
        """Detect frequency masking between instruments"""
        
        conflicts = {}
        
        # Analyze frequency content of each channel
        channel_spectra = {}
        for name, audio in channels.items():
            mono = audio if audio.ndim == 1 else np.mean(audio, axis=1)
            freqs, psd = signal.welch(mono, self.sr, nperseg=4096)
            channel_spectra[name] = {'freqs': freqs, 'psd': psd}
        
        # Find conflicts
        channel_names = list(channels.keys())
        
        for i, name1 in enumerate(channel_names):
            for name2 in channel_names[i+1:]:
                # Calculate spectral overlap
                overlap = self._calculate_spectral_overlap(
                    channel_spectra[name1], channel_spectra[name2]
                )
                
                if overlap > 0.7:  # High overlap
                    conflict_freqs = self._find_conflict_frequencies(
                        channel_spectra[name1], channel_spectra[name2]
                    )
                    
                    conflicts[f"{name1}_vs_{name2}"] = {
                        'overlap_ratio': overlap,
                        'conflict_frequencies': conflict_freqs,
                        'severity': 'high' if overlap > 0.8 else 'medium'
                    }
        
        return conflicts
    
    def _analyze_frequency_balance(self, audio: np.ndarray) -> Dict[str, float]:
        """Analyze frequency balance across bands"""
        
        mono = audio if audio.ndim == 1 else np.mean(audio, axis=1)
        freqs, psd = signal.welch(mono, self.sr, nperseg=4096)
        
        balance = {}
        for band_name, (low_freq, high_freq) in self.frequency_bands.items():
            # Find frequency indices
            band_indices = (freqs >= low_freq) & (freqs <= high_freq)
            band_power = np.mean(psd[band_indices]) if np.any(band_indices) else 0
            
            # Convert to dB
            balance[band_name] = 10 * np.log10(band_power + 1e-10)
        
        return balance
    
    def _analyze_stereo(self, audio: np.ndarray) -> Tuple[float, float]:
        """Analyze stereo width and phase correlation"""
        
        if audio.ndim == 1:
            return 1.0, 1.0
        
        left = audio[:, 0]
        right = audio[:, 1]
        
        # Stereo width calculation
        mid = (left + right) / 2
        side = (left - right) / 2
        
        mid_rms = np.sqrt(np.mean(mid**2))
        side_rms = np.sqrt(np.mean(side**2))
        
        if mid_rms > 0:
            width = side_rms / mid_rms
        else:
            width = 0
        
        # Phase correlation
        correlation = np.corrcoef(left, right)[0, 1] if len(left) > 1 else 1.0
        
        return width, correlation
    
    def _analyze_masking(self, audio: np.ndarray) -> Dict:
        """Analyze frequency masking"""
        
        # Simple masking analysis - critical bands
        mono = audio if audio.ndim == 1 else np.mean(audio, axis=1)
        freqs, psd = signal.welch(mono, self.sr, nperseg=4096)
        
        # Calculate masking threshold
        masking_threshold = np.zeros_like(psd)
        
        # Simplified masking model
        for i, freq in enumerate(freqs):
            # Each frequency masks nearby frequencies
            mask_bandwidth = freq * 0.24  # Critical bandwidth approximation
            
            # Find frequencies within masking range
            mask_indices = (freqs >= freq - mask_bandwidth/2) & \
                          (freqs <= freq + mask_bandwidth/2)
            
            if np.any(mask_indices):
                masking_power = np.max(psd[mask_indices])
                masking_threshold[i] = masking_power * 0.1  # -10dB masking
        
        return {
            'frequencies': freqs.tolist(),
            'power_spectrum': psd.tolist(),
            'masking_threshold': masking_threshold.tolist()
        }
    
    def _analyze_transients(self, audio: np.ndarray) -> float:
        """Analyze transient content"""
        
        # Calculate envelope
        envelope = np.abs(audio)
        envelope = signal.filtfilt(
            np.ones(100) / 100, 1, envelope
        )
        
        # Find peaks
        peaks, _ = signal.find_peaks(envelope, height=np.max(envelope) * 0.1)
        
        # Calculate transient ratio
        if len(peaks) > 0:
            peak_energy = np.sum(envelope[peaks])
            total_energy = np.sum(envelope)
            transient_ratio = peak_energy / total_energy if total_energy > 0 else 0
        else:
            transient_ratio = 0
        
        return transient_ratio
    
    def _identify_issues(self, freq_balance: Dict, dynamic_range: float,
                        stereo_width: float, lufs: float, peak_db: float,
                        phase_corr: float) -> List[str]:
        """Identify mix issues"""
        
        issues = []
        
        # Loudness issues
        if lufs > -10:
            issues.append("Mix is too loud - may cause distortion")
        elif lufs < -20:
            issues.append("Mix is too quiet - lacks impact")
        
        if peak_db > -0.1:
            issues.append("Mix is clipping - reduce levels")
        
        # Dynamic range issues
        if dynamic_range < self.standards['min_dynamic_range']:
            issues.append("Mix is over-compressed - lacks dynamics")
        
        # Frequency balance issues
        bass_level = freq_balance.get('bass', -60)
        treble_level = freq_balance.get('air', -60)
        
        if bass_level - treble_level > 10:
            issues.append("Mix is too bass-heavy")
        elif treble_level - bass_level > 10:
            issues.append("Mix is too bright")
        
        low_mid_level = freq_balance.get('low_mids', -60)
        mid_level = freq_balance.get('mids', -60)
        
        if low_mid_level > mid_level + 5:
            issues.append("Mix is muddy - too much low-mid energy")
        
        # Stereo issues
        if stereo_width < 0.5:
            issues.append("Mix lacks stereo width")
        elif stereo_width > 2.0:
            issues.append("Mix may be too wide - mono compatibility issues")
        
        if phase_corr < 0.5:
            issues.append("Phase correlation issues detected")
        
        return issues
    
    def _generate_recommendations(self, issues: List[str], 
                                freq_balance: Dict) -> List[Dict]:
        """Generate processing recommendations"""
        
        recommendations = []
        
        for issue in issues:
            if "too loud" in issue:
                recommendations.append({
                    'type': 'master_gain',
                    'parameter': 'output_level',
                    'adjustment': -3.0,
                    'description': 'Reduce master output level'
                })
            
            elif "too quiet" in issue:
                recommendations.append({
                    'type': 'master_gain',
                    'parameter': 'output_level', 
                    'adjustment': 2.0,
                    'description': 'Increase master output level'
                })
            
            elif "clipping" in issue:
                recommendations.append({
                    'type': 'limiter',
                    'parameter': 'ceiling',
                    'adjustment': -0.5,
                    'description': 'Lower limiter ceiling'
                })
            
            elif "over-compressed" in issue:
                recommendations.append({
                    'type': 'compression',
                    'parameter': 'reduce_ratio',
                    'adjustment': 0.5,
                    'description': 'Reduce compression ratios across mix'
                })
            
            elif "bass-heavy" in issue:
                recommendations.append({
                    'type': 'eq',
                    'parameter': 'low_shelf',
                    'frequency': 100,
                    'adjustment': -2.0,
                    'description': 'Reduce low frequencies'
                })
            
            elif "too bright" in issue:
                recommendations.append({
                    'type': 'eq',
                    'parameter': 'high_shelf',
                    'frequency': 8000,
                    'adjustment': -1.5,
                    'description': 'Reduce high frequencies'
                })
            
            elif "muddy" in issue:
                recommendations.append({
                    'type': 'eq',
                    'parameter': 'bell',
                    'frequency': 300,
                    'adjustment': -3.0,
                    'q': 1.0,
                    'description': 'Cut muddy low-mid frequencies'
                })
        
        return recommendations
    
    def _calculate_importance_weights(self, channel_analysis: Dict) -> Dict[str, float]:
        """Calculate importance weights for auto-balance"""
        
        weights = {}
        
        for name, analysis in channel_analysis.items():
            name_lower = name.lower()
            
            # Base importance by instrument type
            if 'kick' in name_lower:
                base_weight = 0.8  # Very important
            elif 'snare' in name_lower:
                base_weight = 0.7
            elif 'vocal' in name_lower and 'lead' in name_lower:
                base_weight = 0.9  # Most important
            elif 'vocal' in name_lower:
                base_weight = 0.6
            elif 'bass' in name_lower:
                base_weight = 0.7
            elif 'guitar' in name_lower:
                base_weight = 0.5
            else:
                base_weight = 0.4  # Supporting elements
            
            # Adjust based on transient content
            transient_boost = analysis['transient_ratio'] * 0.2
            
            weights[name] = base_weight + transient_boost
        
        return weights
    
    def _calculate_spectral_overlap(self, spectrum1: Dict, spectrum2: Dict) -> float:
        """Calculate spectral overlap between two signals"""
        
        psd1 = np.array(spectrum1['psd'])
        psd2 = np.array(spectrum2['psd'])
        
        # Normalize PSDs
        psd1_norm = psd1 / np.sum(psd1)
        psd2_norm = psd2 / np.sum(psd2)
        
        # Calculate overlap using minimum of both spectra
        overlap = np.sum(np.minimum(psd1_norm, psd2_norm))
        
        return overlap
    
    def _find_conflict_frequencies(self, spectrum1: Dict, spectrum2: Dict) -> List[float]:
        """Find specific frequencies where conflicts occur"""
        
        freqs = np.array(spectrum1['freqs'])
        psd1 = np.array(spectrum1['psd'])
        psd2 = np.array(spectrum2['psd'])
        
        # Find frequencies where both signals have significant energy
        threshold = np.maximum(np.mean(psd1), np.mean(psd2)) * 2
        
        conflict_indices = (psd1 > threshold) & (psd2 > threshold)
        conflict_freqs = freqs[conflict_indices].tolist()
        
        return conflict_freqs
    
    def _recommend_channel_processing(self, channel_name: str, peak_db: float,
                                    rms_db: float, dominant_freqs: List[float],
                                    transient_ratio: float) -> Dict:
        """Recommend processing for individual channel"""
        
        name_lower = channel_name.lower()
        recommendations = {
            'eq_bands': [],
            'compression': {},
            'effects': []
        }
        
        # EQ recommendations based on instrument type
        if 'kick' in name_lower:
            recommendations['eq_bands'] = [
                {'freq': 60, 'gain': 2, 'q': 0.7, 'description': 'Sub punch'},
                {'freq': 300, 'gain': -2, 'q': 0.8, 'description': 'Remove mud'},
                {'freq': 3500, 'gain': 2, 'q': 0.8, 'description': 'Attack click'}
            ]
            recommendations['compression'] = {
                'threshold': -15, 'ratio': 4, 'attack': 5, 'release': 50
            }
            
        elif 'snare' in name_lower:
            recommendations['eq_bands'] = [
                {'freq': 200, 'gain': 1.5, 'q': 0.8, 'description': 'Body'},
                {'freq': 5000, 'gain': 3, 'q': 0.7, 'description': 'Crack'}
            ]
            recommendations['compression'] = {
                'threshold': -12, 'ratio': 3, 'attack': 3, 'release': 80
            }
            recommendations['effects'] = ['gate', 'reverb']
            
        elif 'vocal' in name_lower:
            recommendations['eq_bands'] = [
                {'freq': 100, 'gain': -6, 'q': 0.7, 'description': 'HPF'},
                {'freq': 2500, 'gain': 2, 'q': 0.8, 'description': 'Presence'},
                {'freq': 10000, 'gain': 1.5, 'q': 0.5, 'description': 'Air'}
            ]
            recommendations['compression'] = {
                'threshold': -18, 'ratio': 3, 'attack': 10, 'release': 100
            }
            recommendations['effects'] = ['de-esser', 'reverb']
            
        elif 'bass' in name_lower:
            recommendations['eq_bands'] = [
                {'freq': 80, 'gain': 2, 'q': 0.7, 'description': 'Fundamental'},
                {'freq': 250, 'gain': -1, 'q': 0.8, 'description': 'Clean up mud'},
                {'freq': 800, 'gain': 1, 'q': 0.8, 'description': 'Definition'}
            ]
            recommendations['compression'] = {
                'threshold': -15, 'ratio': 4, 'attack': 10, 'release': 100
            }
            recommendations['effects'] = ['saturation']
        
        # Adjust based on analysis
        dynamic_range = peak_db - rms_db
        
        if dynamic_range > 20:
            # Very dynamic - may need compression
            if 'compression' in recommendations and 'ratio' in recommendations['compression']:
                recommendations['compression']['ratio'] *= 1.2
        elif dynamic_range < 6:
            # Already compressed - be gentle
            if 'compression' in recommendations and 'ratio' in recommendations['compression']:
                recommendations['compression']['ratio'] *= 0.8
        
        # Transient-based recommendations
        if transient_ratio > 0.3:
            recommendations['effects'].append('transient_shaper')
        
        return recommendations


class AutoMixer:
    """AI-powered automatic mixing"""
    
    def __init__(self, sr: int = 44100):
        self.sr = sr
        self.analyzer = MixAnalyzer(sr)
    
    def auto_mix(self, channels: Dict[str, np.ndarray]) -> Dict:
        """Automatically create a professional mix"""
        
        print("🤖 AI Auto-Mixing in progress...")
        
        # Step 1: Analyze all channels
        print("  Analyzing channels...")
        channel_analyses = {}
        for name, audio in channels.items():
            channel_analyses[name] = self.analyzer.analyze_individual_channel(audio, name)
        
        # Step 2: Detect conflicts
        print("  Detecting frequency conflicts...")
        conflicts = self.analyzer.detect_frequency_conflicts(channels)
        
        # Step 3: Auto-balance levels
        print("  Calculating optimal balance...")
        optimal_gains = self.analyzer.auto_balance_mix(channels)
        
        # Step 4: Generate processing chain
        print("  Generating processing recommendations...")
        processing_chain = {}
        for name, analysis in channel_analyses.items():
            processing_chain[name] = analysis['processing_recommendations']
        
        # Step 5: Resolve conflicts with EQ
        print("  Resolving frequency conflicts...")
        eq_solutions = self._resolve_frequency_conflicts(conflicts)
        
        return {
            'optimal_gains': optimal_gains,
            'processing_chain': processing_chain,
            'eq_solutions': eq_solutions,
            'conflicts_detected': conflicts,
            'channel_analyses': channel_analyses
        }
    
    def _resolve_frequency_conflicts(self, conflicts: Dict) -> Dict:
        """Generate EQ solutions for frequency conflicts"""
        
        solutions = {}
        
        for conflict_name, conflict_data in conflicts.items():
            channels = conflict_name.split('_vs_')
            conflict_freqs = conflict_data['conflict_frequencies']
            
            if len(conflict_freqs) > 0:
                # Simple solution: cut one instrument, boost the other
                primary_channel = channels[0]
                secondary_channel = channels[1]
                
                # Determine which should be primary (keep energy)
                if 'vocal' in primary_channel.lower():
                    # Vocals win
                    pass
                elif 'kick' in primary_channel.lower() and 'bass' in secondary_channel.lower():
                    # Kick wins over bass
                    pass
                else:
                    # Swap
                    primary_channel, secondary_channel = secondary_channel, primary_channel
                
                # Generate EQ moves
                for freq in conflict_freqs[:3]:  # Limit to 3 main conflicts
                    solutions[f"{primary_channel}_boost_{freq:.0f}"] = {
                        'channel': primary_channel,
                        'freq': freq,
                        'gain': 1.5,
                        'q': 1.0,
                        'description': f'Boost {primary_channel} at {freq:.0f}Hz'
                    }
                    
                    solutions[f"{secondary_channel}_cut_{freq:.0f}"] = {
                        'channel': secondary_channel,
                        'freq': freq,
                        'gain': -2.0,
                        'q': 1.2,
                        'description': f'Cut {secondary_channel} at {freq:.0f}Hz to make room'
                    }
        
        return solutions