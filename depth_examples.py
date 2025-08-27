#!/usr/bin/env python3
"""
Depth Processing Examples - Transform flat mixes into dimensional soundscapes

Perfect for mixes that sound flat, lifeless, or lack spatial dimension.
"""

# ============================================
# THE PROBLEM: FLAT MIXES
# ============================================

"""
🎧 COMMON MIXING PROBLEM: Everything sounds flat and in-your-face

Symptoms:
• All instruments seem at the same distance
• No sense of space or dimension
• Mix sounds cramped and 2D
• Elements compete for the same spatial position
• Lacks the depth of professional recordings

Traditional solutions are limited:
• Reverb on the whole mix makes everything muddy
• EQ can't position individual elements
• Pan only moves things left-right, not front-back
• Single stereo file prevents individual positioning

SOLUTION: Stem-based depth processing!
"""

# ============================================
# DEPTH VARIANTS AVAILABLE
# ============================================

"""
🏞️ DEPTH PROCESSING VARIANTS:

1. Depth_Natural
   • Realistic front-to-back positioning
   • Vocals forward, drums close, music mid-distance, strings back
   • Subtle early reflections and natural HF rolloff
   → Most mixes benefit from this

2. Depth_Dramatic  
   • Exaggerated depth for dramatic effect
   • Vocals very close, strings very distant
   • Strong distance cues and atmospheric reverb
   → Great for cinematic or ambient music

3. Depth_Intimate
   • Everything relatively close but with subtle layering
   • Creates cozy, personal listening experience
   • Gentle depth cues without pushing anything too far
   → Perfect for acoustic, folk, or intimate productions

4. Depth_Stadium
   • Large space feeling with distant elements
   • Simulates live venue acoustics
   • Everything pushed back except vocals
   → Great for live albums or stadium rock feel

5. Depth_VocalFocus
   • Vocals way forward, everything else pushed back
   • Maximum clarity for vocal-driven tracks
   • Creates strong foreground/background separation
   → Perfect for pop, R&B, or podcast/spoken word
"""

# ============================================
# HOW IT WORKS (THE SCIENCE)
# ============================================

"""
🧠 DEPTH PERCEPTION CUES:

1. VOLUME ATTENUATION
   • Distant sounds are quieter (inverse square law)
   • But musical scaling - we don't make things disappear!
   
2. HIGH FREQUENCY ROLLOFF  
   • Air absorbs high frequencies over distance
   • Distant elements lose sparkle and brightness
   • Close elements can get presence boost

3. EARLY REFLECTIONS
   • Room reflections arrive after direct sound
   • More reflections = more distance perception
   • Different patterns for different instruments

4. REVERB/AMBIENCE
   • Distant sources have more reverb
   • Simulates room acoustics naturally
   • Tailored to instrument characteristics

5. SPECTRAL TILT
   • Distance = less high frequencies
   • Closeness = more presence/brightness
   • Creates natural depth gradient

6. STEREO WIDTH EFFECTS
   • Mid-side processing for depth
   • Haas effect (precedence) for positioning
   • Different delays create front-back illusion
"""

# ============================================
# CONFIGURATION EXAMPLES
# ============================================

# Example 1: Just add natural depth (recommended starting point)
CONFIG.pipeline.stem_combinations = [
    ("Stem_Original", "natural"),           # Reference
    ("Stem_WithDepth", "depth:natural"),    # Natural depth added
]

# Example 2: Compare different depth styles
CONFIG.pipeline.stem_combinations = [
    ("Stem_Natural", "depth:natural"),      # Realistic
    ("Stem_Dramatic", "depth:dramatic"),    # Exaggerated  
    ("Stem_Intimate", "depth:intimate"),    # Close and cozy
    ("Stem_Stadium", "depth:stadium"),      # Big space
    ("Stem_VocalFocus", "depth:focused"),   # Vocals forward
]

# Example 3: Combine with other processing
CONFIG.pipeline.stem_combinations = [
    # Basic with depth
    ("Stem_PunchyDepth", "depth:dramatic"),
    
    # Advanced with depth (best of both worlds!)
    ("Stem_RadioDepth", "advanced:RadioReady"), 
    ("Stem_RadioDepth_Natural", "depth:natural"),
    
    # For A/B comparison
    ("Stem_Before", "natural"),             # Flat version
    ("Stem_After", "depth:natural"),        # Depth version
]

# Enable/disable depth processing
CONFIG.pipeline.use_depth_processing = True

# ============================================
# WHAT GETS POSITIONED WHERE
# ============================================

"""
📍 TYPICAL DEPTH POSITIONING:

NATURAL variant distances:
• Bass: 0.7x (very close for impact)
• Lead Vocals: 0.6x (forward and intimate) 
• Kick: 0.8x (close but not too close)
• Snare: 0.9x (slightly behind kick)
• Guitar: 1.2x (mid-distance)
• Keys: 1.4x (slightly back)
• Backing Vocals: 1.5x (back for contrast)
• Strings: 1.8x (far back, atmospheric)

DRAMATIC variant:
• Bass: 0.4x (extremely close)
• Lead Vocals: 0.3x (right in your face)
• Backing Vocals: 2.5x (very distant)
• Strings: 3.0x (very far atmospheric)

INTIMATE variant:
• Everything closer together
• More subtle depth differences
• Creates cozy listening experience

STADIUM variant:
• Everything pushed back
• Simulates large venue acoustics
• Vocals still forward for clarity

VOCAL_FOCUS variant:
• Vocals: 0.3x (way forward)
• Everything else: 1.8x+ (way back)
• Maximum vocal clarity
"""

# ============================================
# BEFORE/AFTER COMPARISON
# ============================================

"""
🎧 WHAT YOU'LL HEAR:

BEFORE (flat mix):
• All elements at same perceived distance
• Sounds like everything recorded in same room
• Lacks professional polish and dimension
• Elements fight for same spatial position
• Mix feels cramped and 2D

AFTER (depth processed):
• Clear front-to-back layering
• Vocals float in front of the mix
• Drums feel close and impactful
• Music elements create spatial backdrop
• Mix has professional 3D dimension
• Each element has its own space
• Sounds like expensive studio production

TECHNICAL IMPROVEMENTS:
• Better separation between elements
• Reduced frequency masking
• Enhanced stereo image
• More engaging listening experience
• Professional spatial hierarchy
"""

# ============================================
# BEST PRACTICES
# ============================================

"""
💡 RECOMMENDATIONS:

For Most Mixes:
• Start with "depth:natural" 
• Compare with original to hear difference
• Try "depth:dramatic" if you want more effect

For Specific Genres:
• Pop/R&B: "depth:focused" (vocal emphasis)
• Rock/Metal: "depth:stadium" (big space feel)
• Acoustic/Folk: "depth:intimate" (cozy feel)
• Electronic: "depth:dramatic" (creative effects)
• Cinematic: "depth:dramatic" or "depth:stadium"

For LANDR/External Mastering:
• Use "depth:natural" or "depth:intimate"
• Avoid overly dramatic effects
• Depth processing improves source material significantly

Performance Notes:
• Depth processing is light on CPU
• Much faster than extreme variants
• Can be used on all mixes safely
• No strange artifacts or weird effects
"""

# ============================================
# USAGE IN NOTEBOOK
# ============================================

"""
🚀 HOW TO USE:

1. Add depth variants to your stem combinations:
   CONFIG.pipeline.stem_combinations.append(
       ("My_Mix_WithDepth", "depth:natural")
   )

2. Run your stem processing as normal - depth variants auto-detected!

3. Compare results:
   • Original mix (flat)
   • Depth processed (dimensional)
   
4. Send depth-processed versions to LANDR for superior mastering!

The depth processing gives LANDR source material with:
• Professional spatial positioning
• Natural front-to-back layering  
• Enhanced stereo information
• Reduced element conflicts
• Better separation for mastering algorithms
"""

print("🏞️ Depth Processing Examples loaded!")
print("✨ Transform flat mixes into dimensional soundscapes!")
print("")
print("🎯 Perfect for:")
print("   • Mixes that sound flat and lifeless")
print("   • Home studio recordings lacking dimension") 
print("   • Tracks that need professional spatial polish")
print("   • Preparing superior source material for LANDR")
print("")
print("📍 5 positioning styles:")
print("   • Natural: Realistic depth (recommended)")
print("   • Dramatic: Exaggerated for effect")
print("   • Intimate: Close and cozy")
print("   • Stadium: Big venue feel")
print("   • VocalFocus: Vocals forward, rest back")
print("")
print("🎧 Your mixes will never sound flat again!")