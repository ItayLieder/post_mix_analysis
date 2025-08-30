#!/usr/bin/env python3
"""
Example: Using Advanced Stem Processing Features

This demonstrates the new creative stem processing capabilities that make
stem pre-masters SIGNIFICANTLY better than single-file processing.
"""

# ============================================
# CONFIGURATION WITH ADVANCED PROCESSING
# ============================================

# In your notebook configuration cell:

# Option 1: Use only basic combinations (original behavior)
CONFIG.pipeline.stem_combinations = [
    ("Stem_PunchyMix", "punchy"),
    ("Stem_WideAndOpen", "wide"),
    ("Stem_TightAndControlled", "tight"),
]

# Option 2: Use only advanced combinations
CONFIG.pipeline.stem_combinations = [
    ("Stem_RadioReady", "advanced:RadioReady"),       # Clear separation for broadcast
    ("Stem_LiveBand", "advanced:LiveBand"),           # Natural band positioning
    ("Stem_EDM_Club", "advanced:EDM_Club"),           # Wide, pumping, aggressive
    ("Stem_ModernPop", "advanced:Modern_Pop"),        # Polished pop production
]

# Option 3: Mix both (default - gives you everything!)
CONFIG.pipeline.stem_combinations = [
    # Basic combinations (dial-based processing)
    ("Stem_PunchyMix", "punchy"),
    ("Stem_Balanced", "natural"),
    
    # Advanced combinations (multi-dimensional processing)
    ("Stem_RadioReady", "advanced:RadioReady"),
    ("Stem_LiveBand", "advanced:LiveBand"),
    ("Stem_EDM_Club", "advanced:EDM_Club"),
    ("Stem_Intimate", "advanced:Intimate_Acoustic"),
    ("Stem_VintageSoul", "advanced:Vintage_Soul"),
]

# Enable/disable advanced processing
CONFIG.pipeline.use_advanced_stem_processing = True  # Set to False to disable


# ============================================
# WHAT EACH ADVANCED VARIANT DOES
# ============================================

"""
RadioReady:
- Smart Panning: Focused center positioning
- Frequency: Surgical carving to prevent masking  
- Dynamics: Subtle ducking for clarity
- Spatial: Natural ambience
- Harmonics: Crispy excitation for presence
→ Perfect for radio/streaming where clarity matters

LiveBand:
- Smart Panning: Orchestral stage positioning
- Frequency: Vintage console EQ curves
- Dynamics: Musical groove-based ducking
- Spatial: Stadium reverb for live feel
- Harmonics: Warm tube saturation
→ Makes it sound like a real band playing live

EDM_Club:
- Smart Panning: Extreme width for club systems
- Frequency: Modern hyped curves (scooped mids, big lows)
- Dynamics: Heavy pumping sidechain
- Spatial: Big reverberant space
- Harmonics: Aggressive distortion
→ Designed for club sound systems

Intimate_Acoustic:
- Smart Panning: Natural, close positioning
- Frequency: Gentle, clean EQ
- Dynamics: Very subtle interaction
- Spatial: Intimate room ambience
- Harmonics: Warm, gentle saturation
→ Like the band is in your living room

Experimental:
- Smart Panning: Asymmetric creative positioning
- Frequency: Extreme surgical cuts
- Dynamics: Heavy ducking effects
- Spatial: Psychedelic modulated reverbs
- Harmonics: Heavy distortion/fuzz
→ For creative, unusual productions

Vintage_Soul:
- Smart Panning: Classic L-C-R positioning
- Frequency: Vintage console curves
- Dynamics: Musical groove interaction
- Spatial: Natural room sound
- Harmonics: Tape saturation
→ 60s/70s soul sound

Modern_Pop:
- Smart Panning: Wide, commercial spread
- Frequency: Hyped modern curves
- Dynamics: Subtle, polished
- Spatial: Clean, controlled space
- Harmonics: Bright excitation
→ Radio-ready pop production

Heavy_Rock:
- Smart Panning: Wide guitars, centered rhythm
- Frequency: Surgical separation, scooped
- Dynamics: Groove-based interaction
- Spatial: Big stadium sound
- Harmonics: Aggressive saturation
→ Modern rock/metal production
"""


# ============================================
# UNIQUE ADVANTAGES OVER SINGLE-FILE
# ============================================

"""
Why Stem Pre-Masters Are Superior:

1. SMART PANNING:
   - Kick stays centered while hi-hats pan left
   - Backing vocals spread wide while lead stays center
   - Guitar and keys positioned to avoid masking
   → IMPOSSIBLE with a stereo mix!

2. FREQUENCY SLOTTING:
   - Kick gets 80Hz boost, bass gets 60Hz boost without conflict
   - Vocals get presence at 3kHz while guitars cut at 3kHz
   - Each element gets its own frequency space
   → Can't do this with mixed content!

3. DYNAMIC INTERACTION:
   - Kick triggers bass ducking (sidechain compression)
   - Lead vocals duck the music automatically
   - Snare can pump the guitars for groove
   → No way to achieve this post-mix!

4. SPATIAL DEPTH:
   - Drums get small room reverb
   - Vocals get plate reverb  
   - Guitars get hall reverb
   → Single reverb on mix sounds flat!

5. HARMONIC ENHANCEMENT:
   - Bass gets warm tube saturation
   - Drums get transistor crunch
   - Vocals get exciter brightness
   → One saturation type ruins the mix!

When you send these to LANDR or any mastering service,
they're getting a MUCH better source than a flat stereo mix!
"""


# ============================================
# HOW TO USE IN YOUR NOTEBOOK
# ============================================

"""
1. Import advanced processing:
   from advanced_stem_processing import ADVANCED_VARIANTS
   
2. Choose your variants in config:
   CONFIG.pipeline.stem_combinations = [
       ("MyCustomMix", "advanced:RadioReady"),
   ]

3. Run stem processing as normal - it automatically detects advanced variants!

4. The output folders will contain stems processed with:
   - Smart panning positions
   - Frequency carving for separation  
   - Dynamic interaction between elements
   - Individual spatial treatment
   - Targeted harmonic enhancement

5. Each variant folder contains:
   - Individual processed stems (kick_processed.wav, etc.)
   - Final mixed result with all processing
   - Ready for mastering with LANDR or local processing
"""

print("✨ Advanced Stem Processing Examples loaded!")
print("🎯 Your stem pre-masters now have capabilities impossible with single files!")
print("🎧 Send these to LANDR for superior results compared to stereo mixes!")