#!/usr/bin/env python3
"""
Debug Balance - Simple test to verify balance values are applied
"""

def test_balance_integration(channels):
    """Test if balance values actually get applied to the mixing engine"""
    
    print("🔍 DEBUGGING BALANCE INTEGRATION")
    print("=" * 50)
    
    # Create simple test balance settings
    test_balance = {
        "mix_balance": {
            "vocal_prominence": 0.2,
            "drum_punch": 0.8,
            "bass_foundation": 0.6,
            "instrument_presence": 0.5,
        },
        "channel_overrides": {
            "drums.kick": 2.0,    # Max boost
            "drums.snare": 1.5,   # 50% boost
            "vocals.lead_vocal1": 0.3,  # 70% reduction
        }
    }
    
    print("🎚️ Test Balance Settings:")
    print("  Mix Balance:", test_balance["mix_balance"])
    print("  Channel Overrides:", test_balance["channel_overrides"])
    
    # Test with mixing engine
    try:
        from mixing_engine import MixingSession
        
        print("\n🎛️ Testing with mixing engine...")
        
        # Create session
        session = MixingSession(
            channels=channels,
            template="modern_pop",
            sample_rate=44100,
            bit_depth=24
        )
        
        # Apply test settings
        mix_settings = {
            "buses": {
                "drum_bus": {"channels": ["drums.*"], "compression": 0.5},
                "vocal_bus": {"channels": ["vocals.*", "backvocals.*"], "compression": 0.3},
            },
            "master": {"compression": 0.1, "limiter": True},
            "mix_balance": test_balance["mix_balance"],
            "channel_overrides": test_balance["channel_overrides"]
        }
        
        session.configure(mix_settings)
        
        # Check if values were applied
        print("\n📊 Checking if values were applied...")
        
        # Check channel strips
        for channel_id, expected_value in test_balance["channel_overrides"].items():
            if channel_id in session.channel_strips:
                strip = session.channel_strips[channel_id]
                actual_gain = strip.gain
                print(f"  {channel_id}: Expected ~{expected_value:.2f}, Got {actual_gain:.2f}")
                
                if abs(actual_gain - expected_value) < 0.1:
                    print(f"    ✅ Values match!")
                else:
                    print(f"    ❌ Values don't match - balance not applied correctly")
            else:
                print(f"  ❌ Channel {channel_id} not found in session")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing mixing engine: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_simple_balance_test():
    """Create simple manual balance test"""
    print("\n" + "="*60)
    print("🧪 SIMPLE BALANCE TEST")  
    print("="*60)
    print("Let's test balance manually without GUI:")
    print()
    
    # Manual balance settings - extreme values to make sure we hear differences
    manual_balance = {
        "mix_balance": {
            "vocal_prominence": 0.1,   # Very low vocals
            "drum_punch": 0.9,         # Very high drums
            "bass_foundation": 0.5,
            "instrument_presence": 0.3,
        },
        "channel_overrides": {
            # Extreme drum boost
            "drums.kick": 3.0,
            "drums.snare": 2.5, 
            "drums.hihat": 2.0,
            # Extreme vocal reduction
            "vocals.lead_vocal1": 0.2,
            "vocals.lead_vocal2": 0.2,
            "vocals.lead_vocal3": 0.2,
        }
    }
    
    print("📋 Copy this into your notebook for manual testing:")
    print()
    print("# MANUAL BALANCE TEST")
    print("selected_balance = {")
    for k, v in manual_balance["mix_balance"].items():
        print(f'    "{k}": {v},')
    print("}")
    print()
    print("channel_overrides = {")
    for k, v in manual_balance["channel_overrides"].items():
        print(f'    "{k}": {v},')
    print("}")
    print()
    print("# Then configure your session with both:")
    print("mix_settings['mix_balance'] = selected_balance")  
    print("mix_settings['channel_overrides'] = channel_overrides")
    
    return manual_balance

if __name__ == "__main__":
    print("🔍 Balance Debug Tool")
    print("Use this to test if balance values are being applied correctly")
    
    create_simple_balance_test()