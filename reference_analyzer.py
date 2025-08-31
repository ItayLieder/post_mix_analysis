#!/usr/bin/env python3
"""
Reference Mix Analysis System
Analyzes a reference mix to understand why it works and guide our own mixing decisions
"""

import numpy as np
import soundfile as sf
from scipy import signal
from scipy.fft import fft, fftfreq
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
from pathlib import Path

@dataclass
class ReferenceAnalysis:
    """Complete analysis of a reference mix"""
    # Basic metrics
    peak_db: float
    rms_db: float  
    loudness_lufs: float
    dynamic_range: float
    
    # Frequency analysis
    frequency_spectrum: np.ndarray
    frequency_bins: np.ndarray
    frequency_balance: Dict[str, float]
    dominant_frequencies: List[Tuple[float, float]]
    
    # Temporal analysis
    tempo_estimate: float
    transient_profile: np.ndarray
    attack_characteristics: Dict[str, float]
    
    # Stereo analysis
    stereo_width: float
    phase_correlation: float
    stereo_balance: float
    
    # Dynamic analysis
    compression_estimate: float
    punch_factor: float
    energy_distribution: Dict[str, float]
    
    # Mix characteristics
    clarity_score: float
    separation_score: float
    fullness_score: float
    
    # Analysis results (will be populated after creation)
    issues: Optional[List[str]] = None
    strengths: Optional[List[str]] = None  
    recommendations: Optional[List[str]] = None


class ReferenceMixAnalyzer:
    """Comprehensive analysis of reference mixes"""
    
    def __init__(self, sample_rate: int = 44100):
        self.sr = sample_rate
        self.frequency_bands = {
            'sub_bass': (20, 60),
            'bass': (60, 250), 
            'low_mids': (250, 500),
            'mids': (500, 2000),
            'upper_mids': (2000, 4000),
            'presence': (4000, 8000),
            'air': (8000, 20000)
        }
    
    def analyze_reference(self, audio_path: str) -> ReferenceAnalysis:
        """Complete analysis of reference mix"""
        print(f"🔍 Analyzing reference mix: {Path(audio_path).name}")
        
        # Load audio
        audio, sr = sf.read(audio_path)
        if sr != self.sr:
            from scipy import signal as sp
            num_samples = int(len(audio) * self.sr / sr)
            audio = sp.resample(audio, num_samples)
        
        # Ensure stereo
        if audio.ndim == 1:
            audio = np.stack([audio, audio], axis=-1)
        
        print("  📊 Running comprehensive analysis...")
        
        # Run all analysis components
        basic_metrics = self._analyze_basic_metrics(audio)
        frequency_analysis = self._analyze_frequency_spectrum(audio)
        temporal_analysis = self._analyze_temporal_characteristics(audio)
        stereo_analysis = self._analyze_stereo_characteristics(audio)
        dynamic_analysis = self._analyze_dynamic_characteristics(audio)
        mix_analysis = self._analyze_mix_characteristics(audio)
        
        # Combine into full analysis
        analysis = ReferenceAnalysis(
            **basic_metrics,
            **frequency_analysis,
            **temporal_analysis,
            **stereo_analysis,
            **dynamic_analysis,
            **mix_analysis
        )
        
        # Generate insights
        analysis.issues = self._identify_issues(analysis)
        analysis.strengths = self._identify_strengths(analysis)
        analysis.recommendations = self._generate_recommendations(analysis)
        
        print("  ✅ Reference analysis complete!")
        return analysis
    
    def _analyze_basic_metrics(self, audio: np.ndarray) -> Dict:
        """Basic loudness and dynamic metrics"""
        # Peak and RMS
        peak = np.max(np.abs(audio))
        peak_db = 20 * np.log10(peak) if peak > 0 else -60
        
        rms = np.sqrt(np.mean(audio**2))
        rms_db = 20 * np.log10(rms) if rms > 0 else -60
        
        # Approximate LUFS (simplified)
        # Real LUFS requires K-weighting and gating
        mono = np.mean(audio, axis=1) if audio.ndim == 2 else audio
        loudness_lufs = rms_db - 23  # Rough approximation
        
        # Dynamic range 
        dynamic_range = peak_db - rms_db
        
        return {
            'peak_db': peak_db,
            'rms_db': rms_db,
            'loudness_lufs': loudness_lufs,
            'dynamic_range': dynamic_range
        }
    
    def _analyze_frequency_spectrum(self, audio: np.ndarray) -> Dict:
        """Comprehensive frequency analysis"""
        # Get mono for frequency analysis
        mono = np.mean(audio, axis=1) if audio.ndim == 2 else audio
        
        # FFT analysis
        n_fft = 8192
        freqs = fftfreq(n_fft, 1/self.sr)[:n_fft//2]
        spectrum = np.abs(fft(mono, n_fft))[:n_fft//2]
        
        # Convert to dB
        spectrum_db = 20 * np.log10(spectrum + 1e-10)
        
        # Frequency band analysis
        frequency_balance = {}
        for band_name, (low, high) in self.frequency_bands.items():
            mask = (freqs >= low) & (freqs <= high)
            if np.any(mask):
                band_energy = np.mean(spectrum_db[mask])
                frequency_balance[band_name] = band_energy
        
        # Find dominant frequencies
        peaks, _ = signal.find_peaks(spectrum_db, height=-40, distance=20)
        dominant_frequencies = []
        for peak in peaks[:10]:  # Top 10 peaks
            freq = freqs[peak]
            magnitude = spectrum_db[peak]
            if freq > 20 and freq < 20000:  # Audible range
                dominant_frequencies.append((freq, magnitude))
        
        # Sort by magnitude
        dominant_frequencies.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'frequency_spectrum': spectrum_db,
            'frequency_bins': freqs,
            'frequency_balance': frequency_balance,
            'dominant_frequencies': dominant_frequencies[:5]  # Top 5
        }
    
    def _analyze_temporal_characteristics(self, audio: np.ndarray) -> Dict:
        """Tempo, transients, and timing analysis"""
        mono = np.mean(audio, axis=1) if audio.ndim == 2 else audio
        
        # Tempo estimation (basic onset detection)
        # Differentiate signal to find transients
        diff = np.diff(np.abs(mono))
        
        # Find peaks (potential onsets)
        threshold = np.std(diff) * 2
        onsets = np.where(diff > threshold)[0]
        
        # Estimate tempo from onset intervals
        if len(onsets) > 1:
            intervals = np.diff(onsets) / self.sr  # Convert to seconds
            # Filter reasonable intervals (0.2 to 2 seconds)
            intervals = intervals[(intervals > 0.2) & (intervals < 2.0)]
            if len(intervals) > 0:
                avg_interval = np.median(intervals)
                tempo_estimate = 60 / avg_interval  # Convert to BPM
            else:
                tempo_estimate = 120  # Default
        else:
            tempo_estimate = 120
        
        # Transient profile
        envelope = np.abs(mono)
        # Smooth envelope
        envelope_smooth = signal.filtfilt(
            np.ones(100) / 100, 1, envelope
        )
        
        # Transient detection
        transient_profile = envelope - envelope_smooth
        transient_profile = np.maximum(0, transient_profile)
        
        # Attack characteristics
        attack_strength = np.mean(transient_profile)
        attack_consistency = 1 / (np.std(transient_profile) + 1e-10)
        
        attack_characteristics = {
            'strength': attack_strength,
            'consistency': attack_consistency
        }
        
        return {
            'tempo_estimate': tempo_estimate,
            'transient_profile': transient_profile,
            'attack_characteristics': attack_characteristics
        }
    
    def _analyze_stereo_characteristics(self, audio: np.ndarray) -> Dict:
        """Stereo width, imaging, and phase analysis"""
        if audio.ndim == 1:
            return {
                'stereo_width': 0.0,
                'phase_correlation': 1.0,
                'stereo_balance': 0.0
            }
        
        left = audio[:, 0]
        right = audio[:, 1]
        
        # Mid-side analysis
        mid = (left + right) / 2
        side = (left - right) / 2
        
        # Stereo width
        mid_energy = np.mean(mid**2)
        side_energy = np.mean(side**2)
        
        if mid_energy > 0:
            stereo_width = side_energy / mid_energy
        else:
            stereo_width = 0
        
        # Phase correlation
        correlation = np.corrcoef(left, right)[0, 1]
        phase_correlation = correlation if not np.isnan(correlation) else 1.0
        
        # Stereo balance (L/R energy difference)
        left_energy = np.mean(left**2)
        right_energy = np.mean(right**2)
        
        if left_energy + right_energy > 0:
            stereo_balance = (right_energy - left_energy) / (left_energy + right_energy)
        else:
            stereo_balance = 0.0
        
        return {
            'stereo_width': stereo_width,
            'phase_correlation': phase_correlation,
            'stereo_balance': stereo_balance
        }
    
    def _analyze_dynamic_characteristics(self, audio: np.ndarray) -> Dict:
        """Compression, punch, and energy analysis"""
        mono = np.mean(audio, axis=1) if audio.ndim == 2 else audio
        
        # Estimate compression amount
        # Heavily compressed audio has less dynamic range
        envelope = np.abs(mono)
        envelope_smooth = signal.filtfilt(np.ones(1000) / 1000, 1, envelope)
        
        # Compression estimate based on envelope consistency
        envelope_variation = np.std(envelope_smooth) / (np.mean(envelope_smooth) + 1e-10)
        compression_estimate = max(0, 1 - envelope_variation * 10)  # 0 = no compression, 1 = heavy
        
        # Punch factor (transient to sustain ratio)
        # High punch = strong transients relative to sustain
        fast_envelope = signal.filtfilt(np.ones(10) / 10, 1, envelope)
        slow_envelope = signal.filtfilt(np.ones(1000) / 1000, 1, envelope)
        
        transient_energy = np.mean((fast_envelope - slow_envelope)**2)
        sustain_energy = np.mean(slow_envelope**2)
        
        if sustain_energy > 0:
            punch_factor = transient_energy / sustain_energy
        else:
            punch_factor = 0
        
        # Energy distribution over time
        chunk_size = self.sr // 4  # 250ms chunks
        energy_chunks = []
        
        for i in range(0, len(mono) - chunk_size, chunk_size):
            chunk = mono[i:i + chunk_size]
            energy = np.mean(chunk**2)
            energy_chunks.append(energy)
        
        energy_chunks = np.array(energy_chunks)
        
        energy_distribution = {
            'mean': np.mean(energy_chunks),
            'std': np.std(energy_chunks),
            'peak_to_avg': np.max(energy_chunks) / (np.mean(energy_chunks) + 1e-10)
        }
        
        return {
            'compression_estimate': compression_estimate,
            'punch_factor': punch_factor,
            'energy_distribution': energy_distribution
        }
    
    def _analyze_mix_characteristics(self, audio: np.ndarray) -> Dict:
        """High-level mix quality characteristics"""
        mono = np.mean(audio, axis=1) if audio.ndim == 2 else audio
        
        # Clarity score (high frequency content relative to mids)
        spectrum = np.abs(fft(mono))
        freqs = fftfreq(len(spectrum), 1/self.sr)
        
        # Get frequency ranges
        mid_mask = (freqs >= 500) & (freqs <= 2000)
        high_mask = (freqs >= 4000) & (freqs <= 8000)
        
        mid_energy = np.mean(spectrum[mid_mask]**2) if np.any(mid_mask) else 1
        high_energy = np.mean(spectrum[high_mask]**2) if np.any(high_mask) else 1
        
        clarity_score = high_energy / (mid_energy + 1e-10)
        
        # Separation score (stereo width and phase correlation balance)
        if audio.ndim == 2:
            correlation = np.corrcoef(audio[:, 0], audio[:, 1])[0, 1]
            if not np.isnan(correlation):
                # Good separation = some decorrelation but not too much
                separation_score = 1 - abs(correlation - 0.7)  # Optimal around 0.7
            else:
                separation_score = 0.5
        else:
            separation_score = 0.0  # Mono
        
        # Fullness score (frequency spectrum spread)
        spectrum_db = 20 * np.log10(spectrum + 1e-10)
        # Check if all frequency bands have reasonable energy
        band_scores = []
        for band_name, (low, high) in self.frequency_bands.items():
            mask = (freqs >= low) & (freqs <= high)
            if np.any(mask):
                band_energy = np.mean(spectrum_db[mask])
                # Score based on how much energy is in this band
                band_score = max(0, min(1, (band_energy + 80) / 40))  # Normalize roughly
                band_scores.append(band_score)
        
        fullness_score = np.mean(band_scores) if band_scores else 0
        
        return {
            'clarity_score': clarity_score,
            'separation_score': separation_score,
            'fullness_score': fullness_score
        }
    
    def _identify_issues(self, analysis: ReferenceAnalysis) -> List[str]:
        """Identify potential issues with the reference"""
        issues = []
        
        if analysis.peak_db > -0.1:
            issues.append("May be clipping")
        if analysis.dynamic_range < 6:
            issues.append("Heavily compressed")
        if analysis.phase_correlation < 0.5:
            issues.append("Phase issues")
        if analysis.stereo_width < 0.1:
            issues.append("Too narrow")
        if analysis.stereo_width > 2.0:
            issues.append("Too wide")
        
        return issues
    
    def _identify_strengths(self, analysis: ReferenceAnalysis) -> List[str]:
        """Identify what makes this reference good"""
        strengths = []
        
        if 6 <= analysis.dynamic_range <= 15:
            strengths.append("Good dynamic range")
        if 0.6 <= analysis.phase_correlation <= 0.9:
            strengths.append("Good phase correlation")
        if analysis.punch_factor > 0.1:
            strengths.append("Good punch and transients")
        if analysis.clarity_score > 0.8:
            strengths.append("Clear high frequencies")
        if analysis.separation_score > 0.7:
            strengths.append("Good stereo separation")
        if analysis.fullness_score > 0.6:
            strengths.append("Full frequency spectrum")
        
        return strengths
    
    def _generate_recommendations(self, analysis: ReferenceAnalysis) -> List[str]:
        """Generate mixing recommendations based on reference"""
        recommendations = []
        
        # Dynamic range recommendations
        if analysis.dynamic_range < 10:
            recommendations.append("Use gentle compression to preserve dynamics")
        else:
            recommendations.append("Can use more aggressive compression")
        
        # Frequency balance recommendations
        bass_energy = analysis.frequency_balance.get('bass', -60)
        mid_energy = analysis.frequency_balance.get('mids', -60)
        
        if bass_energy - mid_energy > 10:
            recommendations.append("Reference is bass-heavy - boost low end")
        elif mid_energy - bass_energy > 10:
            recommendations.append("Reference is mid-heavy - boost mids")
        
        # Stereo recommendations
        if analysis.stereo_width > 1.0:
            recommendations.append("Use stereo widening techniques")
        else:
            recommendations.append("Keep stereo width controlled")
        
        # Punch recommendations
        if analysis.punch_factor > 0.2:
            recommendations.append("Preserve transients - avoid over-compression")
        
        return recommendations


def print_reference_analysis(analysis: ReferenceAnalysis):
    """Print comprehensive analysis results"""
    print("\n" + "="*60)
    print("🎯 REFERENCE MIX ANALYSIS")
    print("="*60)
    
    print(f"\n📊 BASIC METRICS:")
    print(f"  Peak Level:     {analysis.peak_db:6.1f} dBFS")
    print(f"  RMS Level:      {analysis.rms_db:6.1f} dBFS") 
    print(f"  LUFS Loudness:  {analysis.loudness_lufs:6.1f} LUFS")
    print(f"  Dynamic Range:  {analysis.dynamic_range:6.1f} dB")
    
    print(f"\n🎵 FREQUENCY BALANCE:")
    for band, level in analysis.frequency_balance.items():
        print(f"  {band:12}: {level:6.1f} dB")
    
    print(f"\n🎶 TEMPORAL CHARACTERISTICS:")
    print(f"  Tempo Estimate: {analysis.tempo_estimate:6.1f} BPM")
    print(f"  Attack Strength:{analysis.attack_characteristics['strength']:6.3f}")
    print(f"  Punch Factor:   {analysis.punch_factor:6.3f}")
    
    print(f"\n🔊 STEREO CHARACTERISTICS:")
    print(f"  Stereo Width:   {analysis.stereo_width:6.2f}")
    print(f"  Phase Correl:   {analysis.phase_correlation:6.3f}")
    print(f"  Balance (L/R):  {analysis.stereo_balance:+6.3f}")
    
    print(f"\n🎛️ MIX CHARACTERISTICS:")
    print(f"  Clarity Score:  {analysis.clarity_score:6.2f}")
    print(f"  Separation:     {analysis.separation_score:6.2f}")
    print(f"  Fullness:       {analysis.fullness_score:6.2f}")
    print(f"  Compression:    {analysis.compression_estimate:6.2f}")
    
    if analysis.dominant_frequencies:
        print(f"\n🎯 DOMINANT FREQUENCIES:")
        for freq, mag in analysis.dominant_frequencies:
            print(f"  {freq:8.0f} Hz: {mag:6.1f} dB")
    
    if analysis.strengths:
        print(f"\n✅ STRENGTHS:")
        for strength in analysis.strengths:
            print(f"  • {strength}")
    
    if analysis.issues:
        print(f"\n⚠️  POTENTIAL ISSUES:")
        for issue in analysis.issues:
            print(f"  • {issue}")
    
    if analysis.recommendations:
        print(f"\n💡 MIXING RECOMMENDATIONS:")
        for rec in analysis.recommendations:
            print(f"  • {rec}")
    
    print("="*60)


# Example usage
if __name__ == "__main__":
    # Test with a reference file
    reference_path = "/path/to/reference/mix.wav"
    
    analyzer = ReferenceMixAnalyzer()
    analysis = analyzer.analyze_reference(reference_path)
    print_reference_analysis(analysis)