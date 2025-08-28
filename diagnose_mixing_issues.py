# 🔍 DIAGNOSE MIXING ENGINE ISSUES

print("🔍 DIAGNOSING MIXING ENGINE PROBLEMS")
print("=" * 60)

if 'session' in locals():
    import numpy as np
    
    print("🎛️ MIXING ENGINE ANALYSIS")
    print("=" * 40)
    
    # 1. Check if GUI changes are actually applied
    print("\n1️⃣ GUI INTEGRATION CHECK:")
    if 'channel_overrides' in locals() and channel_overrides:
        print(f"   ✅ GUI provided {len(channel_overrides)} channel overrides")
        for ch, val in list(channel_overrides.items())[:3]:
            actual_gain = session.channel_strips[ch].gain if ch in session.channel_strips else "NOT FOUND"
            print(f"   • {ch}: GUI={val:.2f}, Engine={actual_gain}")
    else:
        print("   ❌ No GUI channel overrides found")
    
    # 2. Check bus processing effects
    print("\n2️⃣ BUS PROCESSING CHECK:")
    for bus_name, bus_obj in session.buses.items():
        print(f"   📊 {bus_name}:")
        print(f"      Channels: {len(getattr(bus_obj, 'input_channels', []))}")
        print(f"      Compression: {getattr(bus_obj, 'compression', 'N/A')}")
        
        # Check if bus is crushing dynamics
        if hasattr(bus_obj, 'processed_audio') and bus_obj.processed_audio is not None:
            bus_peak = np.max(np.abs(bus_obj.processed_audio))
            print(f"      Output Peak: {bus_peak:.3f}")
            if bus_peak < 0.1:
                print("      ⚠️ Bus output very quiet - compression too aggressive?")
    
    # 3. Check master processing
    print("\n3️⃣ MASTER PROCESSING CHECK:")
    if hasattr(session, 'master_bus'):
        master = session.master_bus
        print(f"   Master Compression: {getattr(master, 'compression', 'N/A')}")
        print(f"   Master Limiting: {getattr(master, 'limiter', 'N/A')}")
        print(f"   Target LUFS: {getattr(master, 'target_lufs', 'N/A')}")
        
        # Check if master is crushing everything
        if hasattr(master, 'processed_audio') and master.processed_audio is not None:
            master_peak = np.max(np.abs(master.processed_audio))
            master_rms = np.sqrt(np.mean(master.processed_audio**2))
            print(f"   Master Output Peak: {master_peak:.3f}")
            print(f"   Master Output RMS: {master_rms:.3f}")
            
            if master_peak > 0.95:
                print("   ⚠️ Master hitting hard limiter - may be crushing dynamics")
    
    # 4. Compare individual vs bus levels
    print("\n4️⃣ LEVEL COMPARISON (Individual vs Bus):")
    
    # Check drum levels specifically
    drum_channels = [ch for ch in session.channel_strips.keys() if 'drums.' in ch]
    total_drum_level = 0
    
    for ch in drum_channels[:3]:  # Check first 3 drums
        if ch in session.channel_strips:
            strip = session.channel_strips[ch]
            individual_peak = np.max(np.abs(strip.audio * strip.gain)) if strip.audio is not None else 0
            total_drum_level += individual_peak
            print(f"   🥁 {ch}: individual peak = {individual_peak:.3f}")
    
    print(f"   Total drum contribution: {total_drum_level:.3f}")
    
    # 5. Check for common mixing problems
    print("\n5️⃣ COMMON MIXING PROBLEMS:")
    
    problems_found = []
    
    # Too much compression
    drum_compression = session.mix_settings.get('buses', {}).get('drum_bus', {}).get('compression', 0)
    if drum_compression > 0.7:
        problems_found.append(f"Drum compression too high: {drum_compression}")
    
    # Master limiting too aggressive  
    master_settings = session.mix_settings.get('master', {})
    if master_settings.get('target_lufs', -14) > -12:
        problems_found.append(f"Target LUFS too loud: {master_settings.get('target_lufs')}")
    
    # Check for phase issues
    all_audio = []
    for strip in session.channel_strips.values():
        if strip.audio is not None:
            all_audio.append(strip.audio * strip.gain)
    
    if all_audio:
        mixed_sum = sum(all_audio)
        individual_sum = sum(np.abs(audio) for audio in all_audio)
        if len(mixed_sum) > 0 and len(individual_sum) > 0:
            phase_ratio = np.max(np.abs(mixed_sum)) / np.max(individual_sum)
            if phase_ratio < 0.5:
                problems_found.append(f"Possible phase cancellation: {phase_ratio:.2f}")
    
    if problems_found:
        print("   ❌ PROBLEMS FOUND:")
        for problem in problems_found:
            print(f"      • {problem}")
    else:
        print("   ✅ No obvious problems detected")
    
    # 6. Recommendations
    print("\n6️⃣ RECOMMENDATIONS:")
    
    if total_drum_level < 0.3:
        print("   🥁 DRUM POWER ISSUE:")
        print("      • Reduce drum bus compression from 0.8 to 0.3")
        print("      • Increase individual drum gains to 3.0+ in GUI") 
        print("      • Remove drum bus entirely (bypass)")
        
    print("\n   🎚️ MIX QUALITY ISSUE:")
    print("      • Reduce master compression from 0.2 to 0.05")
    print("      • Change target LUFS from -14 to -16") 
    print("      • Reduce all bus compression by 50%")
    print("      • Try processing individual channels only (no buses)")

else:
    print("❌ No session found - run mixing cells first")