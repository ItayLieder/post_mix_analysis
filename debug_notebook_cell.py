# 🔍 DEBUG MIXING ENGINE - ADD THIS CELL TO YOUR NOTEBOOK

print("🔍 DEBUGGING MIXING ENGINE APPLICATION")
print("=" * 50)

# Check if session exists and has channel strips
if 'session' in locals():
    print(f"✅ Session exists")
    print(f"📊 Channel strips: {len(session.channel_strips)} loaded")
    
    # Show first few channel strip names
    strip_names = list(session.channel_strips.keys())[:10]
    print(f"📋 First 10 strips: {strip_names}")
    
    # Test extreme override directly on the existing session
    extreme_overrides = {
        "drums.kick": 5.0,
        "drums.snare": 5.0,
        "vocals.lead_vocal1": 0.01,
    }
    
    print("\n🧪 Testing extreme overrides directly on existing session...")
    
    # Show current gain values BEFORE override
    print("\n📊 BEFORE override:")
    for channel_id in extreme_overrides.keys():
        if channel_id in session.channel_strips:
            strip = session.channel_strips[channel_id]
            print(f"  {channel_id}: gain = {strip.gain:.3f}")
        else:
            print(f"  ❌ {channel_id} NOT FOUND in strips")
    
    # Apply overrides directly
    print("\n🎚️ Applying extreme overrides...")
    session._apply_channel_overrides(extreme_overrides)
    
    # Show gain values AFTER override  
    print("\n📊 AFTER override:")
    for channel_id in extreme_overrides.keys():
        if channel_id in session.channel_strips:
            strip = session.channel_strips[channel_id]
            print(f"  {channel_id}: gain = {strip.gain:.3f}")
            
            # Check if it actually changed
            expected = extreme_overrides[channel_id]
            if abs(strip.gain - expected) < 0.1:
                print(f"    ✅ Successfully applied! ({expected})")
            else:
                print(f"    ❌ Not applied correctly! Expected {expected}")
        else:
            print(f"  ❌ {channel_id} NOT FOUND in strips")
    
    # Now test a quick re-render to see if changes stick
    print("\n🎛️ Re-processing mix with new values...")
    
    # Create a small test output to see if drums are actually louder
    import os
    test_output_dir = "/tmp/debug_mix_test"
    os.makedirs(test_output_dir, exist_ok=True)
    
    try:
        # Process just a short segment for testing
        debug_results = session.process_mix(
            output_dir=test_output_dir,
            export_full_mix=True,
            export_individual_channels=False,
            export_buses=False,
            export_stems=False
        )
        
        test_file = os.path.join(test_output_dir, "full_mix.wav")
        if os.path.exists(test_file):
            size = os.path.getsize(test_file) / (1024 * 1024)
            print(f"✅ Debug mix created: {size:.1f} MB")
            print(f"📁 Listen to: {test_file}")
            print("🎧 Drums should be EXTREMELY loud if overrides worked")
        else:
            print("❌ Debug mix not created")
            
    except Exception as e:
        print(f"❌ Error creating debug mix: {e}")
        
else:
    print("❌ No session found - run the previous cells first")