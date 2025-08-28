#!/usr/bin/env python3
"""
Test Mixing Debug - Check if the mixing engine is actually applying values
"""

def debug_mixing_engine_application(channels):
    """Debug what's happening in the mixing engine"""
    
    print("🔍 DEBUGGING MIXING ENGINE")
    print("=" * 50)
    
    # Create extreme test values
    test_overrides = {
        "drums.kick": 3.0,
        "drums.snare": 2.5,
        "vocals.lead_vocal1": 0.1,
    }
    
    print("🎚️ Testing with extreme values:")
    for ch, val in test_overrides.items():
        print(f"  {ch}: {val}")
    
    try:
        from mixing_engine import MixingSession
        
        # Create session
        session = MixingSession(
            channels=channels,
            template="modern_pop",
            sample_rate=44100,
            bit_depth=24
        )
        
        print(f"\n📊 Available channel strips: {list(session.channel_strips.keys())[:10]}...")
        
        # Test the _apply_channel_overrides method directly
        print("\n🔧 Testing _apply_channel_overrides directly...")
        
        # Call the method directly to see what happens
        session._apply_channel_overrides(test_overrides)
        
        # Check the results
        print("\n📊 Checking results after override application:")
        for channel_id, expected_multiplier in test_overrides.items():
            if channel_id in session.channel_strips:
                strip = session.channel_strips[channel_id]
                actual_gain = strip.gain
                print(f"  {channel_id}:")
                print(f"    Expected multiplier: {expected_multiplier}")
                print(f"    Actual gain: {actual_gain}")
                print(f"    Original gain was probably: {actual_gain / expected_multiplier if expected_multiplier != 0 else 'N/A'}")
                
                if abs(actual_gain - expected_multiplier) < 0.1:
                    print(f"    ✅ Gain matches multiplier")
                else:
                    print(f"    ❌ Gain doesn't match - something went wrong")
            else:
                print(f"  ❌ {channel_id} not found in channel strips")
                print(f"      Available: {list(session.channel_strips.keys())}")
        
        return session
        
    except Exception as e:
        print(f"❌ Error in debug: {e}")
        import traceback
        traceback.print_exc()
        return None

def quick_channel_list_check(channels):
    """Quick check of channel naming"""
    print("\n📋 CHANNEL NAME CHECK")
    print("=" * 30)
    print("Your input channels:")
    for category, tracks in channels.items():
        for track_name in tracks.keys():
            channel_id = f"{category}.{track_name}"
            print(f"  {channel_id}")
    
    print("\nTesting with these specific names in overrides...")

if __name__ == "__main__":
    print("🔍 Mixing Engine Debug Tool")
    print("Run this to see exactly what's happening with balance application")