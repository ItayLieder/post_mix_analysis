#!/usr/bin/env python3
"""
Example: Using Detailed Stems in Post-Mix Analysis

This demonstrates how to use the new detailed stem feature that supports
specific stems like "kick", "backvocals", etc. with fallback to main categories.
"""

# Example usage in your notebook configuration cell:

# ============================================
# DETAILED STEM PATHS CONFIGURATION
# ============================================

# Option 1: Mix of main stems and detailed stems
STEM_PATHS = {
    # Main stem categories (if you have full stems)
    "bass": "/path/to/bass.wav",
    "vocals": "/path/to/main_vocals.wav",  # This will be used if no detailed vocals
    
    # Detailed drum stems (instead of one drums.wav)
    "kick": "/path/to/kick.wav",
    "snare": "/path/to/snare.wav", 
    "hats": "/path/to/hats.wav",
    
    # Detailed vocal stems (will override main vocals if present)
    "leadvocals": "/path/to/lead_vocals.wav",
    "backvocals": "/path/to/backing_vocals.wav",
    
    # Detailed music stems
    "guitar": "/path/to/guitars.wav",
    "keys": "/path/to/keys.wav",
    "strings": "/path/to/strings.wav",
}

# Option 2: Only detailed stems (no main categories)
STEM_PATHS_DETAILED_ONLY = {
    "kick": "/path/to/kick.wav",
    "snare": "/path/to/snare.wav",
    "bass": "/path/to/bass.wav", 
    "leadvocals": "/path/to/lead_vox.wav",
    "backvocals": "/path/to/bg_vox.wav",
    "guitar": "/path/to/guitar.wav",
    "keys": "/path/to/synths.wav"
}

# ============================================
# DETAILED STEM BALANCE CONTROL
# ============================================

from stem_balance_helper import set_stem_balance

# Method 1: Set specific detailed stems
set_stem_balance(
    kick=0.90,          # Kick drum loud and punchy
    snare=0.75,         # Snare controlled
    hats=0.60,          # Hi-hats tamed
    leadvocals=0.95,    # Lead vocals prominent
    backvocals=0.65,    # Backing vocals supporting
    guitar=0.80,        # Guitar present but not dominating
    strings=0.85        # Strings with good presence
)

# Method 2: Mix main categories with detailed overrides
set_stem_balance(
    drums=0.70,         # General drum level
    vocals=0.80,        # General vocal level
    kick=0.85,          # But kick specifically louder
    backvocals=0.60     # But backing vocals specifically quieter
)

# Method 3: Environment variables for detailed stems
"""
export STEM_GAIN_KICK=0.90
export STEM_GAIN_BACKVOCALS=0.65
export STEM_GAIN_STRINGS=0.85
"""

# ============================================
# PROCESSING BEHAVIOR
# ============================================

"""
How the system handles detailed stems:

1. If you provide "kick.wav", it gets processed as a "kick" stem
   - Uses kick-specific gain (0.80 by default)
   - Gets kick-specific DSP processing (drums category variants)
   - Falls back to drums category if kick gain not specified

2. If you provide "drums.wav" (no kick.wav), it gets processed as "drums"
   - Uses drums gain (0.75 by default)
   - Gets general drums DSP processing

3. Mixing is intelligent:
   - kick + snare + hats = summed with individual gains
   - OR drums = processed as one stem
   - System prevents double-processing the same content

4. Categories for detailed stems:
   kick, snare, hats → drums category
   leadvocals, backvocals → vocals category  
   guitar, keys, strings → music category
   bass → bass category (no subcategories yet)
"""

print("📝 Detailed stems example loaded!")
print("Supported detailed stems:")
print("  🥁 Drums: kick, snare, hats")
print("  🎤 Vocals: leadvocals, backvocals") 
print("  🎵 Music: guitar, keys, strings")
print("  🎸 Bass: bass (no subcategories)")