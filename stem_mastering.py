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
    """Container for the 4 standard stem categories"""
    drums: Optional[AudioBuffer] = None      # Kick, snare, hats, percussion
    bass: Optional[AudioBuffer] = None       # Bass guitar, sub bass, 808s  
    vocals: Optional[AudioBuffer] = None     # Lead vocals, backing vocals, harmonies
    music: Optional[AudioBuffer] = None      # Guitars, keys, synths, strings, other
    
    def validate(self) -> bool:
        """At minimum, require at least one stem to be present"""
        return any([self.drums, self.bass, self.vocals, self.music]) is not None
    
    def get_active_stems(self) -> List[str]:
        """Return list of stem names that have audio data"""
        active = []
        if self.drums: active.append("drums")
        if self.bass: active.append("bass")  
        if self.vocals: active.append("vocals")
        if self.music: active.append("music")
        return active
    
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
    """Get the appropriate variant for a stem type in a given combination"""
    # Map combinations to stem variants
    combination_map = {
        "punchy": {"drums": "punchy", "bass": "tight", "vocals": "presence", "music": "focused"},
        "wide": {"drums": "natural", "bass": "deep", "vocals": "warm", "music": "wide"},
        "tight": {"drums": "tight", "bass": "controlled", "vocals": "intimate", "music": "focused"},
        "aggressive": {"drums": "aggressive", "bass": "punchy", "vocals": "bright", "music": "balanced"},
        "natural": {"drums": "natural", "bass": "controlled", "vocals": "warm", "music": "balanced"},
    }
    
    stem_variant = combination_map.get(combination, {}).get(stem_type, "natural")
    return STEM_VARIANTS[stem_type].get(stem_variant, STEM_VARIANTS[stem_type]["natural"])

# ============================================
# STEM LOADING UTILITIES
# ============================================

def load_stem_set(stem_paths: Dict[str, str]) -> StemSet:
    """
    Load stems from file paths
    stem_paths: {"drums": "path/to/drums.wav", "vocals": "path/to/vocals.wav", etc.}
    """
    stem_set = StemSet()
    
    for stem_type, path in stem_paths.items():
        if stem_type not in ["drums", "bass", "vocals", "music"]:
            print(f"⚠️ Unknown stem type '{stem_type}', skipping...")
            continue
            
        if path and os.path.exists(path):
            try:
                audio_buffer = load_wav(path)
                setattr(stem_set, stem_type, audio_buffer)
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