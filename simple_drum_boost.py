#!/usr/bin/env python3
"""
Simple Drum Boost - Direct drum boost without GUI
"""

def create_extreme_drum_boost():
    """Create extreme drum boost settings for testing"""
    
    print("🥁 EXTREME DRUM BOOST TEST")
    print("=" * 40)
    print("This will create VERY LOUD drums so you can definitely hear them")
    print()
    
    # Extreme settings to make sure we hear the difference
    extreme_settings = {
        "mix_balance": {
            "vocal_prominence": 0.1,    # Almost mute vocals  
            "drum_punch": 0.9,          # Max drum punch
            "bass_foundation": 0.3,     # Reduce bass
            "instrument_presence": 0.2, # Reduce instruments
        },
        "channel_overrides": {
            "drums.kick": 3.0,      # 300% volume
            "drums.snare": 3.0,     # 300% volume  
            "drums.hihat": 2.0,     # 200% volume
            "drums.tom": 2.0,       # 200% volume
            "drums.cymbal": 2.0,    # 200% volume
            # Mute vocals to isolate drums
            "vocals.lead_vocal1": 0.1,
            "vocals.lead_vocal2": 0.1, 
            "vocals.lead_vocal3": 0.1,
            # Reduce everything else
            "bass.bass_guitar5": 0.3,
            "bass.bass1": 0.3,
        }
    }
    
    print("🎚️ Extreme Test Settings Generated:")
    print(f"  • Drums boosted to 200-300%")
    print(f"  • Vocals reduced to 10%") 
    print(f"  • Bass reduced to 30%")
    print(f"  • If you still can't hear drums, there's a bigger issue!")
    
    return extreme_settings

def apply_simple_drum_boost(channels):
    """Apply simple drum boost directly"""
    
    settings = create_extreme_drum_boost()
    
    print("\n📋 COPY THIS INTO YOUR NOTEBOOK:")
    print("="*50)
    print()
    print("# Replace your GUI cell with this:")
    print("selected_balance = {")
    for k, v in settings["mix_balance"].items():
        print(f'    "{k}": {v},')
    print("}")
    print()
    print("channel_overrides = {")
    for k, v in settings["channel_overrides"].items():
        print(f'    "{k}": {v},')
    print("}")
    print()
    print("print('✅ Extreme drum boost applied - drums should be VERY loud!')")
    
    return settings

if __name__ == "__main__":
    settings = create_extreme_drum_boost()
    print(f"\nSettings created: {settings}")