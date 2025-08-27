#!/usr/bin/env python3
"""
Stem Balance Helper - Easy configuration for stem balancing
"""

from config import CONFIG
import json

def print_current_balance():
    """Display current stem balance settings"""
    print("🎚️  Current Stem Balance Settings:")
    print("="*40)
    for stem, gain in CONFIG.pipeline.stem_gains.items():
        db = 20 * (gain ** 0.5)  # Approximate dB representation
        bar = "█" * int(gain * 10)
        print(f"{stem:8s}: {gain:.2f} ({db:+.1f}dB) {bar}")
    print("="*40)
    print(f"Auto-gain compensation: {CONFIG.pipeline.auto_gain_compensation}")
    print(f"Target peak: {CONFIG.pipeline.stem_sum_target_peak} ({20*np.log10(CONFIG.pipeline.stem_sum_target_peak):.1f} dBFS)")

def set_stem_balance(drums=None, bass=None, vocals=None, music=None, **detailed_stems):
    """
    Adjust stem balance gains for main categories and detailed stems.
    
    Args:
        drums: Gain for drums (0.1 to 1.0, default: 0.75)
        bass: Gain for bass (0.1 to 1.0, default: 0.65)
        vocals: Gain for vocals (0.1 to 1.0, default: 0.85)
        music: Gain for music (0.1 to 1.0, default: 0.80)
        **detailed_stems: Any detailed stem (kick=0.8, backvocals=0.7, etc.)
    
    Examples:
        # Main categories only
        set_stem_balance(drums=0.65, music=0.90)
        
        # Mix of main and detailed stems
        set_stem_balance(music=0.80, kick=0.85, backvocals=0.65, guitar=0.90)
        
        # Detailed stems only
        set_stem_balance(kick=0.9, snare=0.7, leadvocals=0.95, strings=0.85)
    """
    # Handle main categories
    if drums is not None:
        CONFIG.pipeline.stem_gains["drums"] = max(0.1, min(1.0, drums))
    if bass is not None:
        CONFIG.pipeline.stem_gains["bass"] = max(0.1, min(1.0, bass))
    if vocals is not None:
        CONFIG.pipeline.stem_gains["vocals"] = max(0.1, min(1.0, vocals))
    if music is not None:
        CONFIG.pipeline.stem_gains["music"] = max(0.1, min(1.0, music))
    
    # Handle detailed stems
    valid_detailed_stems = ['kick', 'snare', 'hats', 'backvocals', 'leadvocals', 'guitar', 'keys', 'strings']
    for stem_name, gain_value in detailed_stems.items():
        if stem_name in valid_detailed_stems:
            CONFIG.pipeline.stem_gains[stem_name] = max(0.1, min(1.0, gain_value))
            print(f"🎯 Set {stem_name}: {gain_value:.2f}")
        else:
            print(f"⚠️ Unknown detailed stem: {stem_name} (ignoring)")
    
    print("✅ Updated stem balance:")
    print_current_balance()

def use_preset_balance(preset):
    """
    Use a preset stem balance configuration.
    
    Available presets:
    - 'default': Balanced mix with audible music
    - 'vocal_forward': Emphasize vocals
    - 'instrumental': Emphasize music/instruments
    - 'drum_heavy': Emphasize drums and bass
    - 'flat': Equal gain for all stems
    """
    presets = {
        'default': {
            "drums": 0.75,
            "bass": 0.65,
            "vocals": 0.85,
            "music": 0.80
        },
        'vocal_forward': {
            "drums": 0.65,
            "bass": 0.60,
            "vocals": 0.95,
            "music": 0.70
        },
        'instrumental': {
            "drums": 0.70,
            "bass": 0.65,
            "vocals": 0.70,
            "music": 0.95
        },
        'drum_heavy': {
            "drums": 0.90,
            "bass": 0.80,
            "vocals": 0.75,
            "music": 0.65
        },
        'flat': {
            "drums": 0.70,
            "bass": 0.70,
            "vocals": 0.70,
            "music": 0.70
        }
    }
    
    if preset not in presets:
        print(f"❌ Unknown preset '{preset}'. Available: {', '.join(presets.keys())}")
        return
    
    CONFIG.pipeline.stem_gains = presets[preset]
    print(f"✅ Applied '{preset}' balance preset:")
    print_current_balance()

def save_custom_balance(name, description=""):
    """Save current balance as a custom preset to a file"""
    import os
    
    preset_file = "custom_stem_balances.json"
    
    # Load existing presets if file exists
    if os.path.exists(preset_file):
        with open(preset_file, 'r') as f:
            custom_presets = json.load(f)
    else:
        custom_presets = {}
    
    # Add current balance
    custom_presets[name] = {
        "gains": dict(CONFIG.pipeline.stem_gains),
        "description": description,
        "auto_gain": CONFIG.pipeline.auto_gain_compensation
    }
    
    # Save to file
    with open(preset_file, 'w') as f:
        json.dump(custom_presets, f, indent=2)
    
    print(f"✅ Saved balance preset '{name}' to {preset_file}")

def load_custom_balance(name):
    """Load a previously saved custom balance"""
    import os
    
    preset_file = "custom_stem_balances.json"
    
    if not os.path.exists(preset_file):
        print(f"❌ No custom presets found. Create one with save_custom_balance()")
        return
    
    with open(preset_file, 'r') as f:
        custom_presets = json.load(f)
    
    if name not in custom_presets:
        print(f"❌ Preset '{name}' not found. Available: {', '.join(custom_presets.keys())}")
        return
    
    preset = custom_presets[name]
    CONFIG.pipeline.stem_gains = preset["gains"]
    CONFIG.pipeline.auto_gain_compensation = preset.get("auto_gain", True)
    
    print(f"✅ Loaded custom balance '{name}':")
    if preset.get("description"):
        print(f"   Description: {preset['description']}")
    print_current_balance()

# Import numpy for the dB calculation
import numpy as np

if __name__ == "__main__":
    print("🎛️  Stem Balance Helper")
    print("="*50)
    print_current_balance()
    print("\nUsage examples:")
    print("  set_stem_balance(music=0.90)  # Make music louder")
    print("  use_preset_balance('instrumental')  # Use preset")
    print("  save_custom_balance('my_mix', 'Custom for track X')")