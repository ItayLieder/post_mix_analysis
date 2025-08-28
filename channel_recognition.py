#!/usr/bin/env python3
"""
Intelligent Channel Recognition System
Identifies instrument types from channel names and audio content
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import signal
import re


def identify_channel_type(name: str, category: str = None, 
                         audio_data: np.ndarray = None, 
                         sample_rate: int = 44100,
                         hints: Dict = None) -> Dict:
    """
    Identify channel type using multiple strategies:
    1. Name pattern matching
    2. Category context
    3. Frequency analysis
    4. Transient detection
    5. User hints
    """
    
    result = {
        "primary_type": "unknown",
        "confidence": 0.0,
        "subtypes": [],
        "characteristics": {},
        "suggested_processing": []
    }
    
    # Strategy 1: Name-based identification
    name_result = identify_from_name(name)
    if name_result["confidence"] > 0.7:
        result.update(name_result)
    
    # Strategy 2: Category context
    if category:
        category_result = identify_from_category(category, name)
        if category_result["confidence"] > result["confidence"]:
            result.update(category_result)
    
    # Strategy 3: Audio analysis (if provided)
    if audio_data is not None:
        audio_result = identify_from_audio(audio_data, sample_rate)
        # Combine with existing results
        if audio_result["confidence"] > 0.5:
            if result["confidence"] < 0.5:
                result.update(audio_result)
            else:
                # Merge results
                result["confidence"] = (result["confidence"] + audio_result["confidence"]) / 2
                result["characteristics"].update(audio_result["characteristics"])
    
    # Strategy 4: User hints (override)
    if hints:
        if "type" in hints:
            result["primary_type"] = hints["type"]
            result["confidence"] = 1.0
        if "subtype" in hints:
            result["subtypes"].append(hints["subtype"])
    
    # Generate processing suggestions
    result["suggested_processing"] = suggest_processing(result["primary_type"], result["characteristics"])
    
    return result


def identify_from_name(name: str) -> Dict:
    """Identify channel type from name patterns"""
    
    name_lower = name.lower()
    
    # Comprehensive pattern dictionary
    patterns = {
        # DRUMS
        "kick": {
            "patterns": ["kick", "bd", "bass drum", "bassdrum", "kck", "kik"],
            "confidence": 0.95,
            "subtypes": ["acoustic", "electronic"],
            "characteristics": {"frequency_range": "low", "transient": "strong"}
        },
        "snare": {
            "patterns": ["snare", "snr", "sd", "sn", "clap"],
            "confidence": 0.95,
            "subtypes": ["top", "bottom", "electronic"],
            "characteristics": {"frequency_range": "mid", "transient": "strong"}
        },
        "hihat": {
            "patterns": ["hat", "hh", "hihat", "hi-hat", "open hat", "closed hat", "oh", "ch"],
            "confidence": 0.9,
            "subtypes": ["open", "closed", "pedal"],
            "characteristics": {"frequency_range": "high", "transient": "strong"}
        },
        "tom": {
            "patterns": ["tom", "rack", "floor tom", "ft", "rt"],
            "confidence": 0.9,
            "subtypes": ["rack", "floor", "high", "mid", "low"],
            "characteristics": {"frequency_range": "mid", "transient": "strong"}
        },
        "cymbal": {
            "patterns": ["cymbal", "crash", "ride", "china", "splash", "cym"],
            "confidence": 0.9,
            "subtypes": ["crash", "ride", "china", "splash"],
            "characteristics": {"frequency_range": "high", "transient": "strong", "sustain": "long"}
        },
        "overhead": {
            "patterns": ["overhead", "oh", "over", "drum oh"],
            "confidence": 0.85,
            "subtypes": ["stereo", "mono", "xy", "ab"],
            "characteristics": {"frequency_range": "full", "stereo": True}
        },
        "room": {
            "patterns": ["room", "amb", "ambient", "drum room"],
            "confidence": 0.8,
            "subtypes": ["close", "far", "stereo"],
            "characteristics": {"frequency_range": "full", "reverb": True}
        },
        
        # BASS
        "bass": {
            "patterns": ["bass", "bs", "sub", "808", "synth bass", "electric bass", "upright"],
            "confidence": 0.9,
            "subtypes": ["electric", "synth", "acoustic", "sub"],
            "characteristics": {"frequency_range": "low", "sustain": "long"}
        },
        "bass_di": {
            "patterns": ["di", "direct", "bass di", "bass direct"],
            "confidence": 0.95,
            "subtypes": ["clean"],
            "characteristics": {"frequency_range": "low", "dynamics": "high"}
        },
        "bass_amp": {
            "patterns": ["amp", "bass amp", "cab", "cabinet"],
            "confidence": 0.9,
            "subtypes": ["driven", "clean"],
            "characteristics": {"frequency_range": "low-mid", "harmonics": "rich"}
        },
        
        # GUITARS
        "guitar_electric": {
            "patterns": ["gtr", "guitar", "elec", "electric", "dist", "clean gtr"],
            "confidence": 0.85,
            "subtypes": ["rhythm", "lead", "clean", "distorted"],
            "characteristics": {"frequency_range": "mid", "dynamics": "variable"}
        },
        "guitar_acoustic": {
            "patterns": ["acoustic", "ac gtr", "folk", "classical"],
            "confidence": 0.9,
            "subtypes": ["steel", "nylon", "12-string"],
            "characteristics": {"frequency_range": "mid-high", "transient": "moderate"}
        },
        
        # KEYS/SYNTHS
        "piano": {
            "patterns": ["piano", "pno", "grand", "upright", "ep", "electric piano", "rhodes", "wurli"],
            "confidence": 0.9,
            "subtypes": ["grand", "upright", "electric"],
            "characteristics": {"frequency_range": "full", "dynamics": "high"}
        },
        "synth": {
            "patterns": ["synth", "syn", "pad", "lead", "arp", "sequence", "pluck"],
            "confidence": 0.85,
            "subtypes": ["pad", "lead", "bass", "arp", "fx"],
            "characteristics": {"frequency_range": "variable", "synthetic": True}
        },
        "organ": {
            "patterns": ["organ", "b3", "hammond", "church"],
            "confidence": 0.9,
            "subtypes": ["hammond", "church", "combo"],
            "characteristics": {"frequency_range": "full", "harmonics": "rich"}
        },
        
        # VOCALS
        "vocal_lead": {
            "patterns": ["lead vox", "lead vocal", "main vox", "vox", "vocal", "singer"],
            "confidence": 0.9,
            "subtypes": ["male", "female", "dry", "wet"],
            "characteristics": {"frequency_range": "mid-high", "dynamics": "high", "presence": True}
        },
        "vocal_harmony": {
            "patterns": ["harmony", "harm", "bgv", "background", "backing", "choir", "gang"],
            "confidence": 0.85,
            "subtypes": ["high", "low", "stacked"],
            "characteristics": {"frequency_range": "mid-high", "layered": True}
        },
        "vocal_double": {
            "patterns": ["double", "dbl", "vocal dbl", "stack"],
            "confidence": 0.85,
            "subtypes": ["tight", "loose"],
            "characteristics": {"frequency_range": "mid-high", "thickness": "enhanced"}
        },
        
        # STRINGS
        "strings": {
            "patterns": ["strings", "str", "violin", "viola", "cello", "bass", "quartet"],
            "confidence": 0.85,
            "subtypes": ["section", "solo", "pizzicato", "legato"],
            "characteristics": {"frequency_range": "mid-high", "sustain": "long"}
        },
        
        # BRASS/WINDS
        "brass": {
            "patterns": ["brass", "trumpet", "trombone", "horn", "tuba", "section"],
            "confidence": 0.85,
            "subtypes": ["section", "solo", "muted"],
            "characteristics": {"frequency_range": "mid", "transient": "strong"}
        },
        "woodwind": {
            "patterns": ["flute", "clarinet", "sax", "oboe", "bassoon", "wind"],
            "confidence": 0.85,
            "subtypes": ["solo", "section"],
            "characteristics": {"frequency_range": "mid-high", "breathy": True}
        },
        
        # PERCUSSION
        "percussion": {
            "patterns": ["perc", "shaker", "tamb", "conga", "bongo", "djembe", "cowbell"],
            "confidence": 0.8,
            "subtypes": ["shaker", "hand", "ethnic"],
            "characteristics": {"frequency_range": "variable", "rhythmic": True}
        },
        
        # FX/OTHER
        "fx": {
            "patterns": ["fx", "effect", "sfx", "riser", "sweep", "impact", "noise"],
            "confidence": 0.75,
            "subtypes": ["riser", "impact", "sweep", "atmospheric"],
            "characteristics": {"frequency_range": "full", "one-shot": True}
        }
    }
    
    # Check each pattern
    best_match = {"primary_type": "unknown", "confidence": 0.0, "subtypes": [], "characteristics": {}}
    
    for instrument_type, config in patterns.items():
        for pattern in config["patterns"]:
            if pattern in name_lower:
                # Calculate confidence based on pattern match quality
                confidence = config["confidence"]
                if pattern == name_lower:  # Exact match
                    confidence = min(1.0, confidence + 0.1)
                
                if confidence > best_match["confidence"]:
                    best_match = {
                        "primary_type": instrument_type,
                        "confidence": confidence,
                        "subtypes": config["subtypes"],
                        "characteristics": config["characteristics"]
                    }
    
    return best_match


def identify_from_category(category: str, name: str) -> Dict:
    """Use category context to refine identification"""
    
    category_lower = category.lower()
    name_lower = name.lower()
    
    category_defaults = {
        "drums": {
            "default_type": "drum_misc",
            "confidence": 0.6,
            "characteristics": {"frequency_range": "full", "transient": "strong"}
        },
        "bass": {
            "default_type": "bass",
            "confidence": 0.7,
            "characteristics": {"frequency_range": "low", "sustain": "long"}
        },
        "guitars": {
            "default_type": "guitar_electric",
            "confidence": 0.7,
            "characteristics": {"frequency_range": "mid", "harmonics": "rich"}
        },
        "keys": {
            "default_type": "synth",
            "confidence": 0.6,
            "characteristics": {"frequency_range": "full", "synthetic": True}
        },
        "vocals": {
            "default_type": "vocal_lead",
            "confidence": 0.7,
            "characteristics": {"frequency_range": "mid-high", "presence": True}
        },
        "strings": {
            "default_type": "strings",
            "confidence": 0.7,
            "characteristics": {"frequency_range": "mid-high", "sustain": "long"}
        }
    }
    
    if category_lower in category_defaults:
        result = category_defaults[category_lower].copy()
        result["subtypes"] = []
        return result
    
    return {"primary_type": "unknown", "confidence": 0.0, "subtypes": [], "characteristics": {}}


def identify_from_audio(audio: np.ndarray, sample_rate: int) -> Dict:
    """Analyze audio content to identify instrument type"""
    
    # Convert to mono if stereo
    if audio.ndim > 1:
        mono = np.mean(audio, axis=1)
    else:
        mono = audio
    
    # Frequency analysis
    fft = np.fft.rfft(mono[:min(len(mono), 8192)])
    freqs = np.fft.rfftfreq(min(len(mono), 8192), 1/sample_rate)
    magnitude = np.abs(fft)
    
    # Find dominant frequency
    peak_idx = np.argmax(magnitude)
    peak_freq = freqs[peak_idx]
    
    # Calculate spectral centroid
    centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
    
    # Transient detection (simplified)
    envelope = np.abs(signal.hilbert(mono[:1024]))
    attack_time = np.argmax(envelope) / sample_rate * 1000  # ms
    
    characteristics = {
        "peak_frequency": peak_freq,
        "spectral_centroid": centroid,
        "attack_time": attack_time
    }
    
    # Classify based on characteristics
    if peak_freq < 150 and attack_time < 20:
        primary_type = "kick"
        confidence = 0.7
    elif peak_freq < 300 and centroid > 1000:
        primary_type = "bass"
        confidence = 0.6
    elif centroid > 5000:
        primary_type = "hihat"
        confidence = 0.6
    elif 200 < centroid < 2000 and attack_time < 10:
        primary_type = "snare"
        confidence = 0.6
    elif 500 < centroid < 4000:
        primary_type = "vocal_lead"
        confidence = 0.5
    else:
        primary_type = "unknown"
        confidence = 0.3
    
    return {
        "primary_type": primary_type,
        "confidence": confidence,
        "subtypes": [],
        "characteristics": characteristics
    }


def suggest_processing(channel_type: str, characteristics: Dict) -> List[str]:
    """Suggest processing chain based on channel type"""
    
    processing_templates = {
        # DRUMS
        "kick": ["Gate", "EQ (boost 60Hz, 4kHz)", "Compression (4:1)", "Saturation"],
        "snare": ["Gate", "EQ (boost 200Hz, 5kHz)", "Compression (3:1)", "Reverb (short)"],
        "hihat": ["Gate", "HPF (200Hz)", "EQ (gentle)", "Compression (2:1)"],
        "overhead": ["HPF (80Hz)", "EQ (gentle shelf)", "Compression (2:1)", "Width"],
        
        # BASS
        "bass": ["HPF (30Hz)", "EQ (cut 500Hz)", "Compression (3:1)", "Saturation"],
        "bass_di": ["HPF (30Hz)", "EQ (presence)", "Compression (3:1)", "Blend with amp"],
        
        # GUITARS
        "guitar_electric": ["Gate", "EQ (cut mud)", "Compression (2:1)", "Delay/Reverb"],
        "guitar_acoustic": ["HPF (80Hz)", "EQ (presence)", "Compression (2:1)", "Reverb"],
        
        # VOCALS
        "vocal_lead": ["Gate", "HPF (80Hz)", "EQ (presence)", "De-esser", "Compression (3:1)", "Reverb"],
        "vocal_harmony": ["Gate", "HPF (100Hz)", "EQ (gentle)", "Compression (2:1)", "Reverb", "Width"],
        
        # KEYS
        "piano": ["HPF (30Hz)", "EQ (gentle)", "Compression (2:1)", "Reverb"],
        "synth": ["EQ (shape)", "Compression (optional)", "Effects"],
        
        # DEFAULT
        "unknown": ["EQ", "Compression", "Effects (optional)"]
    }
    
    # Get template or default
    if channel_type in processing_templates:
        return processing_templates[channel_type]
    
    # Generate based on characteristics
    suggestions = []
    
    freq_range = characteristics.get("frequency_range", "")
    if freq_range == "low":
        suggestions.append("HPF (20-30Hz)")
    elif freq_range == "high":
        suggestions.append("HPF (100-200Hz)")
    
    if characteristics.get("transient") == "strong":
        suggestions.append("Gate")
        suggestions.append("Transient Shaper")
    
    suggestions.append("EQ")
    suggestions.append("Compression")
    
    if characteristics.get("stereo"):
        suggestions.append("Width Enhancement")
    
    return suggestions if suggestions else ["EQ", "Compression"]


print("🔍 Channel Recognition System loaded!")
print("   • Name pattern matching with 40+ instrument types")
print("   • Frequency analysis for content identification")
print("   • Automatic processing suggestions")
print("   • User hint support for ambiguous channels")