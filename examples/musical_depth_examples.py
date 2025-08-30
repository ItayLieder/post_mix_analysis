#!/usr/bin/env python3
"""
Musical Depth Examples - Professional depth that respects your mix balance

Perfect for "RadioReady with depth" - subtle, musical, professional
"""

# ============================================
# THE SOLUTION: MUSICAL DEPTH
# ============================================

"""
🎵 MUSICAL DEPTH PROCESSING

What you asked for: "RadioReady with depth" - professional enhancement that:
• Respects your carefully chosen stem balance
• Adds dimension without going overboard  
• Sounds natural and musical (not processed)
• Maintains commercial compatibility
• Perfect for LANDR and streaming

Think: "Professional mixing engineer added subtle depth"
Not: "Crazy experimental processing"
"""

# ============================================
# MUSICAL DEPTH VARIANTS
# ============================================

"""
🎯 5 SUBTLE, PROFESSIONAL VARIANTS:

1. Musical_Balanced (RECOMMENDED)
   • Gentle depth maintaining your exact balance
   • Vocals: 25% closer, backing vocals: 20% further
   • Max EQ change: ±1dB (barely noticeable)
   • Perfect for most tracks

2. Musical_VocalForward  
   • Subtle vocal emphasis with natural backing
   • Lead vocals: 35% closer, music: 15% further
   • Great for vocal-driven tracks
   • Still maintains balance

3. Musical_Warm
   • Cozy, intimate depth feeling
   • Everything closer together but layered
   • Perfect for acoustic, folk, R&B
   • Comfortable listening experience

4. Musical_Clear
   • Clarity-focused with subtle separation
   • Each element gets its own space
   • Great for busy arrangements
   • Maintains punch and energy

5. Musical_Polished
   • Professional studio dimension
   • Subtle but present depth cues
   • Commercial-ready enhancement
   • Perfect for streaming/mastering
"""

# ============================================
# HOW IT'S DIFFERENT FROM REGULAR DEPTH
# ============================================

"""
🎵 MUSICAL vs REGULAR DEPTH:

REGULAR DEPTH (too much):
• Distance differences: 0.3x to 3.0x (10x range!)
• EQ changes: up to -5dB (very noticeable)
• Reflections: up to 25% wet (obvious)
• Can sound processed/artificial

MUSICAL DEPTH (just right):
• Distance differences: 0.65x to 1.40x (2x range)
• EQ changes: max ±1dB (subtle)
• Reflections: max 10% wet (barely noticeable)
• Sounds like natural studio acoustics

RESULT: Professional depth without weirdness!
"""

# ============================================
# CONFIGURATION EXAMPLES
# ============================================

# Example 1: Just add musical depth to your current setup
CONFIG.pipeline.stem_combinations = [
    # Your current favorites
    ("Original", "natural"),
    ("RadioReady", "advanced:RadioReady"),
    
    # Add musical depth versions
    ("RadioReady_WithDepth", "musical:balanced"),    # THIS IS WHAT YOU WANT!
    ("RadioReady_VocalFocus", "musical:vocal_forward"),
]

# Example 2: Compare subtle vs dramatic depth
CONFIG.pipeline.stem_combinations = [
    ("Flat_Original", "natural"),           # No depth
    ("Musical_Depth", "musical:balanced"),  # Subtle depth (recommended)  
    ("Dramatic_Depth", "depth:dramatic"),   # Too much (for comparison)
]

# Example 3: Multiple musical depth styles
CONFIG.pipeline.stem_combinations = [
    ("Musical_Balanced", "musical:balanced"),    # General purpose
    ("Musical_Clear", "musical:clear"),          # For busy mixes
    ("Musical_Warm", "musical:warm"),            # For intimate tracks
    ("Musical_Polished", "musical:polished"),    # For commercial release
]

# Enable musical depth processing
CONFIG.pipeline.use_musical_depth_processing = True

# ============================================
# WHAT YOU'LL HEAR
# ============================================

"""
🎧 BEFORE vs AFTER (Musical Depth):

BEFORE (flat):
• All elements at same perceived distance
• Mix sounds good but lacks dimension
• Professional but somewhat flat

AFTER (musical depth):
• Vocals gently forward (not in-your-face)
• Music elements subtly layered behind
• Still sounds like YOUR mix, just with more space
• Professional dimension like expensive studio
• Maintains all your balance choices
• Ready for LANDR with better source material

TECHNICAL IMPROVEMENTS:
• Better front-to-back separation
• Enhanced stereo image (subtle)
• Reduced frequency conflicts
• More engaging listening experience
• Commercial-ready enhancement
"""

# ============================================
# PERFECT FOR LANDR USERS
# ============================================

"""
🎯 WHY MUSICAL DEPTH + LANDR = PERFECT:

Your Current Workflow:
Stems → Balance → LANDR → Final Master

New Improved Workflow:  
Stems → Balance → Musical Depth → LANDR → Superior Master

What LANDR Gets:
• Your carefully balanced mix (unchanged)
• PLUS professional spatial positioning
• PLUS subtle dimension enhancement
• PLUS better stereo information
• = Superior source material for mastering

Result: LANDR produces better masters because it starts with better source!

The musical depth processing is like having a professional mixing engineer 
add subtle depth to your mix before sending to mastering.
"""

# ============================================
# PERFORMANCE & COMPATIBILITY
# ============================================

"""
⚡ PERFORMANCE:
• Very light CPU usage (faster than advanced variants)
• No weird artifacts or glitches
• Safe for all mix types
• No extreme processing

🎯 COMPATIBILITY:
• Perfect for streaming platforms
• Great for radio/broadcast
• Ideal for LANDR and external mastering
• Commercial-ready enhancement
• Maintains mix translation

💡 RECOMMENDATION:
Start with "musical:balanced" - it's designed to enhance any mix 
without changing its character. Perfect "RadioReady with depth" sound!
"""

print("🎵 Musical Depth Examples loaded!")
print("✨ Professional depth that respects your balance!")
print("")
print("🎯 Perfect for:")
print("   • 'RadioReady with depth' sound")
print("   • Professional enhancement without weirdness")
print("   • Preparing superior source for LANDR")  
print("   • Commercial releases needing subtle dimension")
print("")
print("📊 5 musical styles:")
print("   • Balanced: General purpose (RECOMMENDED)")
print("   • VocalForward: Subtle vocal emphasis")
print("   • Warm: Cozy intimate feel")
print("   • Clear: Clarity-focused separation")
print("   • Polished: Professional commercial ready")
print("")
print("🎧 Sounds like: 'Professional studio depth' not 'digital effects'!")