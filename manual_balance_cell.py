# 🎚️ MANUAL BALANCE CONTROL FOR PROFESSIONAL MIXING
# Add this cell after Step 3 and before Step 5 in the professional notebook

if 'pro_session' in locals():
    print("🎚️ MANUAL BALANCE CONTROL")
    print("=" * 40)
    
    # Manual channel overrides (same format as before)
    manual_channel_overrides = {
        # DRUMS - Adjust these values as needed
        'drums.kick': 7.5,        # Very loud kick
        'drums.snare': 7.5,       # Very loud snare  
        'drums.hihat': 3.0,       # Moderate hi-hat
        'drums.tom': 4.5,         # Moderate tom
        'drums.cymbal': 3.0,      # Moderate cymbal
        
        # VOCALS - Adjust these values
        'vocals.lead_vocal1': 0.8,   # Lead vocal level
        'vocals.lead_vocal2': 0.8,
        'vocals.lead_vocal3': 0.8,
        
        # BACKING VOCALS
        'backvocals.backing_vocal': 0.3,
        'backvocals.lead_vocal1': 0.3,
        'backvocals.lead_vocal2': 0.3, 
        'backvocals.lead_vocal3': 0.3,
        'backvocals.lead_vocal4': 2.1,
        
        # BASS
        'bass.bass_guitar5': 0.6,
        'bass.bass1': 0.6,
        'bass.bass_guitar3': 0.6,
        'bass.bass_synth2': 4.2,     # Powerful bass synth
        'bass.bass_synth4': 0.6,
        
        # GUITARS
        'guitars.electric_guitar2': 0.4,
        'guitars.electric_guitar3': 0.3,
        'guitars.electric_guitar4': 0.4,
        'guitars.electric_guitar5': 0.4,
        'guitars.electric_guitar6': 0.4,
        'guitars.acoustic_guitar1': 0.4,
        
        # KEYS
        'keys.bell3': 0.8,
        'keys.clavinet1': 0.6,
        'keys.piano2': 0.8,
        'keys.piano4': 0.8,
        
        # SYNTHS
        'synths.pad2': 0.8,
        'synths.pad3': 0.5,
        'synths.rythmic_synth1': 0.6,
        
        # FX
        'fx.perc6': 1.2,
        'fx.fx1': 0.8,
        'fx.fx2': 0.9,
        'fx.fx3': 0.7,
        'fx.fx4': 0.8,
        'fx.fx5': 0.6,
    }
    
    # Apply manual overrides
    applied_count = 0
    
    for ch_id, strip in pro_session.channel_strips.items():
        if ch_id in manual_channel_overrides:
            # Override the current gain with manual setting
            manual_gain = manual_channel_overrides[ch_id]
            strip.gain = manual_gain
            
            gain_db = 20 * np.log10(manual_gain) if manual_gain > 0 else -60
            print(f"  ✓ {ch_id}: Manual gain set to {manual_gain:.2f} ({gain_db:+.1f} dB)")
            applied_count += 1
    
    print(f"\n✅ Applied {applied_count} manual balance overrides")
    print("🎛️ Professional processing with your manual balance ready!")
    
    # Helper function to quickly adjust groups
    def adjust_group(group_name, multiplier):
        """Quickly adjust a group of instruments"""
        adjusted = 0
        for ch_id, strip in pro_session.channel_strips.items():
            if group_name in ch_id.lower():
                strip.gain *= multiplier
                adjusted += 1
        print(f"  • {group_name}: {adjusted} channels adjusted by {multiplier:.2f}x")
        return adjusted
    
    # Quick group adjustments (uncomment and modify as needed)
    print("\n🎚️ QUICK GROUP ADJUSTMENTS:")
    print("Uncomment and modify these lines to make quick adjustments:")
    print("# adjust_group('vocals', 1.2)    # Boost all vocals by 20%")
    print("# adjust_group('drums', 0.8)     # Reduce all drums by 20%") 
    print("# adjust_group('guitars', 1.5)   # Boost all guitars by 50%")
    print("# adjust_group('bass', 0.9)      # Slightly reduce bass")
    
    # Example: Uncomment these if needed
    # adjust_group('vocals', 1.2)
    # adjust_group('guitars', 0.7)
    
else:
    print("❌ No professional session available")