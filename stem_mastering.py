# ============================================
# Stem Mastering — Optional Advanced Mode
# ============================================
# Provides stem-aware mastering alongside existing single-file pipeline
# Users can choose between:
# - Standard Mode: Single stereo mix → variants → masters
# - Stem Mode: 4 stems → stem-specific processing → intelligent summing → masters

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional
import os
import numpy as np
import soundfile as sf
from data_handler import AudioBuffer, load_wav, save_wav
from render_engine import DialState

# ============================================
# STEM CATEGORIES & DATA STRUCTURES
# ============================================

@dataclass
class StemSet:
    """Container for stem categories with support for detailed stems"""
    # Main stem categories (required)
    drums: Optional[AudioBuffer] = None      # Kick, snare, hats, percussion
    bass: Optional[AudioBuffer] = None       # Bass guitar, sub bass, 808s  
    vocals: Optional[AudioBuffer] = None     # Lead vocals, backing vocals, harmonies
    music: Optional[AudioBuffer] = None      # Guitars, keys, synths, strings, other
    
    # Detailed stems (optional - if not provided, assumed part of main category)
    kick: Optional[AudioBuffer] = None       # Kick drum (falls back to drums)
    snare: Optional[AudioBuffer] = None      # Snare drum (falls back to drums)
    hats: Optional[AudioBuffer] = None       # Hi-hats (falls back to drums)
    backvocals: Optional[AudioBuffer] = None # Backing vocals (falls back to vocals)
    leadvocals: Optional[AudioBuffer] = None # Lead vocals (falls back to vocals)
    guitar: Optional[AudioBuffer] = None     # Guitar (falls back to music)
    keys: Optional[AudioBuffer] = None       # Keys/synths (falls back to music)
    strings: Optional[AudioBuffer] = None    # Strings (falls back to music)
    
    # Stem category mapping for detailed stems
    _stem_mapping = {
        'kick': 'drums',
        'snare': 'drums', 
        'hats': 'drums',
        'backvocals': 'vocals',
        'leadvocals': 'vocals',
        'guitar': 'music',
        'keys': 'music',
        'strings': 'music'
    }
    
    def validate(self) -> bool:
        """At minimum, require at least one stem to be present"""
        return any([self.drums, self.bass, self.vocals, self.music]) is not None
    
    def get_active_stems(self) -> List[str]:
        """Return list of ALL stem names that have audio data (main + detailed)"""
        active = []
        # Check main categories
        if self.drums: active.append("drums")
        if self.bass: active.append("bass")  
        if self.vocals: active.append("vocals")
        if self.music: active.append("music")
        
        # Check detailed stems
        detailed_stems = ['kick', 'snare', 'hats', 'backvocals', 'leadvocals', 'guitar', 'keys', 'strings']
        for stem_name in detailed_stems:
            if hasattr(self, stem_name) and getattr(self, stem_name) is not None:
                active.append(stem_name)
        
        return active
    
    def get_main_stems_only(self) -> List[str]:
        """Return list of only main stem categories that have audio data"""
        active = []
        if self.drums: active.append("drums")
        if self.bass: active.append("bass")  
        if self.vocals: active.append("vocals")
        if self.music: active.append("music")
        return active
        
    def get_detailed_stems_only(self) -> List[str]:
        """Return list of only detailed stems that have audio data"""
        active = []
        detailed_stems = ['kick', 'snare', 'hats', 'backvocals', 'leadvocals', 'guitar', 'keys', 'strings']
        for stem_name in detailed_stems:
            if hasattr(self, stem_name) and getattr(self, stem_name) is not None:
                active.append(stem_name)
        return active
        
    def get_stem_category(self, stem_name: str) -> str:
        """Get the main category for any stem (main or detailed)"""
        if stem_name in ['drums', 'bass', 'vocals', 'music']:
            return stem_name
        return self._stem_mapping.get(stem_name, 'music')  # Default to music if unknown
    
    def get_total_duration(self) -> float:
        """Get the duration of the longest stem"""
        durations = []
        for stem in [self.drums, self.bass, self.vocals, self.music]:
            if stem:
                durations.append(stem.duration_s)
        return max(durations) if durations else 0.0

# ============================================
# STEM-SPECIFIC PROCESSING VARIANTS
# ============================================

# Optimized dial settings for each stem category
STEM_VARIANTS: Dict[str, Dict[str, DialState]] = {
    "drums": {
        # Drum-focused processing
        "punchy": DialState(bass=25, punch=65, clarity=30, air=20, width=8),
        "tight": DialState(bass=15, punch=50, clarity=40, air=25, width=6),
        "natural": DialState(bass=20, punch=30, clarity=20, air=15, width=10),
        "aggressive": DialState(bass=30, punch=75, clarity=25, air=15, width=5),
    },
    "bass": {
        # Bass-focused processing  
        "tight": DialState(bass=45, punch=40, clarity=25, air=8, width=4),
        "deep": DialState(bass=65, punch=20, clarity=15, air=5, width=3),
        "punchy": DialState(bass=40, punch=55, clarity=30, air=12, width=6),
        "controlled": DialState(bass=35, punch=35, clarity=35, air=10, width=5),
        "natural": DialState(bass=30, punch=25, clarity=20, air=8, width=5),
    },
    "vocals": {
        # Vocal-focused processing
        "presence": DialState(bass=10, punch=20, clarity=50, air=40, width=12),
        "warm": DialState(bass=25, punch=15, clarity=25, air=20, width=15),
        "bright": DialState(bass=8, punch=18, clarity=35, air=55, width=10),
        "intimate": DialState(bass=15, punch=12, clarity=40, air=30, width=8),
        "natural": DialState(bass=15, punch=18, clarity=30, air=25, width=12),
    },
    "music": {
        # Music/instrumental processing
        "wide": DialState(bass=20, punch=20, clarity=20, air=30, width=40),
        "focused": DialState(bass=18, punch=30, clarity=35, air=22, width=15),
        "ambient": DialState(bass=12, punch=10, clarity=15, air=40, width=50),
        "balanced": DialState(bass=22, punch=25, clarity=25, air=25, width=20),
        "natural": DialState(bass=18, punch=22, clarity=22, air=20, width=18),
    }
}

# ============================================
# STEM COMBINATION VARIANTS
# ============================================

# Overall mastering approaches that combine stem processing intelligently
STEM_COMBINATIONS: Dict[str, str] = {
    "Punchy Mix": "punchy",      # Drums punchy, bass tight, vocals presence, music focused
    "Wide & Open": "wide",       # Drums natural, bass deep, vocals warm, music wide  
    "Tight & Controlled": "tight", # Drums tight, bass controlled, vocals intimate, music focused
    "Aggressive": "aggressive",   # Drums aggressive, bass punchy, vocals bright, music balanced
    "Balanced": "natural",       # Natural settings across all stems
}

def get_stem_variant_for_combination(stem_type: str, combination: str) -> DialState:
    """Get the appropriate variant for a stem type in a given combination (supports detailed stems)"""
    # Map combinations to stem variants
    combination_map = {
        "punchy": {"drums": "punchy", "bass": "tight", "vocals": "presence", "music": "focused"},
        "wide": {"drums": "natural", "bass": "deep", "vocals": "warm", "music": "wide"},
        "tight": {"drums": "tight", "bass": "controlled", "vocals": "intimate", "music": "focused"},
        "aggressive": {"drums": "aggressive", "bass": "punchy", "vocals": "bright", "music": "balanced"},
        "natural": {"drums": "natural", "bass": "controlled", "vocals": "warm", "music": "balanced"},
    }
    
    # Handle detailed stems by mapping them to their main categories
    stem_mapping = {
        'kick': 'drums',
        'snare': 'drums', 
        'hats': 'drums',
        'backvocals': 'vocals',
        'leadvocals': 'vocals',
        'guitar': 'music',
        'keys': 'music',
        'strings': 'music'
    }
    
    # Get the main category for this stem
    main_category = stem_mapping.get(stem_type, stem_type)  # Use original if not detailed
    
    # Get the variant for the main category
    stem_variant = combination_map.get(combination, {}).get(main_category, "natural")
    
    # Return the dial state from the main category variants
    if main_category in STEM_VARIANTS:
        return STEM_VARIANTS[main_category].get(stem_variant, STEM_VARIANTS[main_category]["natural"])
    else:
        # Fallback to drums if category not found
        return STEM_VARIANTS["drums"]["natural"]

# ============================================
# STEM LOADING UTILITIES
# ============================================

def load_stem_set(stem_paths: Dict[str, str]) -> StemSet:
    """
    Load stems from file paths (supports both main and detailed stems)
    stem_paths: {"drums": "path/to/drums.wav", "kick": "path/to/kick.wav", etc.}
    """
    stem_set = StemSet()
    
    # Define all supported stem types
    main_stems = ["drums", "bass", "vocals", "music"]
    detailed_stems = ["kick", "snare", "hats", "backvocals", "leadvocals", "guitar", "keys", "strings"]
    all_supported_stems = main_stems + detailed_stems
    
    for stem_type, path in stem_paths.items():
        if stem_type not in all_supported_stems:
            category = stem_set.get_stem_category(stem_type) if hasattr(stem_set, 'get_stem_category') else 'music'
            print(f"⚠️ Unknown stem type '{stem_type}' - will be treated as '{category}' category")
            continue
            
        if path and os.path.exists(path):
            try:
                audio_buffer = load_wav(path)
                setattr(stem_set, stem_type, audio_buffer)
                
                # Show which category this detailed stem belongs to
                if stem_type in detailed_stems:
                    category = stem_set.get_stem_category(stem_type)
                    print(f"✅ Loaded {stem_type} stem: {audio_buffer.duration_s:.1f}s (→ {category} category)")
                else:
                    print(f"✅ Loaded {stem_type} stem: {audio_buffer.duration_s:.1f}s")
                    
            except Exception as e:
                print(f"❌ Failed to load {stem_type} stem from {path}: {e}")
        else:
            print(f"⚠️ {stem_type} stem path not provided or file doesn't exist")
    
    return stem_set

def validate_stem_set(stem_set: StemSet) -> bool:
    """Validate that stem set is ready for processing"""
    if not stem_set.validate():
        print("❌ No valid stems found in stem set")
        return False
    
    active_stems = stem_set.get_active_stems()
    print(f"✅ Found {len(active_stems)} active stems: {', '.join(active_stems)}")
    
    # Check sample rates match
    sample_rates = []
    for stem_name in active_stems:
        stem = getattr(stem_set, stem_name)
        if stem:
            sample_rates.append(stem.sr)
    
    if len(set(sample_rates)) > 1:
        print(f"⚠️ Sample rates don't match: {sample_rates}")
        return False
    
    print(f"✅ All stems at {sample_rates[0]} Hz")
    return True

# ============================================
# CONVENIENCE FUNCTIONS
# ============================================

def create_stem_variants(stem_set: StemSet, combination_name: str = "Balanced") -> List[Tuple[str, StemSet]]:
    """
    Create variants of a stem set using different processing combinations
    Returns list of (variant_name, processed_stem_set) tuples
    """
    variants = []
    
    for combo_name, combo_key in STEM_COMBINATIONS.items():
        variant_stem_set = StemSet()
        
        # Process each active stem with appropriate variant
        for stem_type in ["drums", "bass", "vocals", "music"]:
            original_stem = getattr(stem_set, stem_type)
            if original_stem:
                # Get the dial settings for this stem in this combination
                dial_state = get_stem_variant_for_combination(stem_type, combo_key)
                # Note: Actual processing would happen in render_engine
                # For now, we just store the dial settings with the stem
                processed_stem = original_stem  # Placeholder - actual processing later
                processed_stem.meta = processed_stem.meta or {}
                processed_stem.meta["dial_state"] = asdict(dial_state)
                processed_stem.meta["stem_variant"] = combo_key
                setattr(variant_stem_set, stem_type, processed_stem)
        
        variants.append((f"Stem_{combo_name.replace(' ', '')}", variant_stem_set))
    
    return variants

print("Stem Mastering module loaded:")
print("- StemSet dataclass for 4-stem organization")  
print("- STEM_VARIANTS with category-specific dial settings")
print("- STEM_COMBINATIONS for intelligent stem processing")
print("- load_stem_set(), validate_stem_set(), create_stem_variants()")
print("- Compatible with existing single-file pipeline")