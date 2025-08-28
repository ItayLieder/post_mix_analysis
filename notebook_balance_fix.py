#!/usr/bin/env python3
"""
Notebook Balance Fix - Replace broken GUI with working interactive balance
"""

def create_working_balance_interface(channels):
    """Create working balance interface for the notebook"""
    
    print("🎚️ BALANCE CONTROL INTERFACE")
    print("=" * 50)
    print("The GUI sliders don't work in your environment.")
    print("Instead, use these simple commands to adjust balance:\n")
    
    # Create a simple balance manager
    balance_values = {}
    
    # Initialize all channels to neutral
    for category, tracks in channels.items():
        for track_name in tracks.keys():
            channel_id = f"{category}.{track_name}"
            balance_values[channel_id] = 1.0
    
    print("📋 AVAILABLE CHANNELS:")
    for category, tracks in channels.items():
        print(f"\n  {category.upper()}:")
        for track_name in tracks.keys():
            channel_id = f"{category}.{track_name}"
            print(f"    • {channel_id}")
    
    print("\n🎚️ BALANCE COMMANDS:")
    print("  # Adjust individual channels:")
    print("  balance_values['bass.bass_guitar5'] = 1.3    # Boost by 30%")
    print("  balance_values['vocals.lead_vocal1'] = 0.7   # Reduce by 30%")
    print("")
    print("  # Adjust groups (copy/paste these):")
    print("  # Boost all drums by 20%")
    print("  for ch in balance_values:")
    print("      if 'drums.' in ch:")
    print("          balance_values[ch] = 1.2")
    print("")
    print("  # Reduce all vocals by 20%") 
    print("  for ch in balance_values:")
    print("      if 'vocals.' in ch or 'backvocals.' in ch:")
    print("          balance_values[ch] = 0.8")
    
    print("\n💡 VALUES: 0.5 = half volume, 1.0 = neutral, 1.5 = boost 50%")
    print("🎯 After making changes, run: apply_balance_to_mixing(balance_values)")
    
    return balance_values

def apply_balance_to_mixing(balance_values):
    """Apply balance settings to the mixing engine"""
    
    # Convert to channel overrides format
    channel_overrides = {k: v for k, v in balance_values.items() if abs(v - 1.0) > 0.01}
    
    print("🎚️ APPLYING BALANCE SETTINGS")
    print("=" * 40)
    
    if channel_overrides:
        print(f"📊 Applying {len(channel_overrides)} balance changes:")
        for channel_id, value in sorted(channel_overrides.items()):
            change_pct = (value - 1.0) * 100
            direction = "↑" if change_pct > 0 else "↓"
            print(f"  {direction} {channel_id:30} = {value:.2f} ({change_pct:+.0f}%)")
        
        # Create the selected_balance with overrides
        selected_balance = {
            "vocal_prominence": 0.3,
            "drum_punch": 0.65,
            "bass_foundation": 0.6,
            "instrument_presence": 0.5,
            "channel_overrides": channel_overrides  # Add individual channel control
        }
        
        print(f"\n✅ Balance settings ready for mixing engine")
        print("📋 Copy this to your notebook:")
        print(f"selected_balance = {selected_balance}")
        
        return selected_balance
    else:
        print("📊 No balance changes detected (all channels at 1.0)")
        return {
            "vocal_prominence": 0.3,
            "drum_punch": 0.65,
            "bass_foundation": 0.6,
            "instrument_presence": 0.5,
        }

def show_balance_status(balance_values):
    """Show current balance status"""
    print("\n🎚️ CURRENT BALANCE STATUS")
    print("=" * 40)
    
    changed = {k: v for k, v in balance_values.items() if abs(v - 1.0) > 0.01}
    
    if changed:
        # Group by category
        categories = {}
        for channel_id, value in changed.items():
            category = channel_id.split('.')[0]
            if category not in categories:
                categories[category] = []
            categories[category].append((channel_id, value))
        
        for category, channels in categories.items():
            print(f"\n📁 {category.upper()}:")
            for channel_id, value in channels:
                track_name = channel_id.split('.')[1]
                change_pct = (value - 1.0) * 100
                direction = "↑" if change_pct > 0 else "↓"
                print(f"  {direction} {track_name:20} = {value:.2f} ({change_pct:+.0f}%)")
    else:
        print("📊 All channels at neutral balance (1.0)")
        print("💡 Use the commands above to adjust balance")

# Quick preset functions
def preset_boost_drums(balance_values, amount=0.2):
    """Boost all drums"""
    for ch in balance_values:
        if 'drums.' in ch:
            balance_values[ch] = min(2.0, 1.0 + amount)
    print(f"✅ All drums boosted by {amount*100:.0f}%")

def preset_reduce_vocals(balance_values, amount=0.2):
    """Reduce all vocals"""
    for ch in balance_values:
        if 'vocals.' in ch or 'backvocals.' in ch:
            balance_values[ch] = max(0.0, 1.0 - amount)
    print(f"✅ All vocals reduced by {amount*100:.0f}%")

def preset_boost_bass(balance_values, amount=0.15):
    """Boost all bass"""
    for ch in balance_values:
        if 'bass.' in ch:
            balance_values[ch] = min(2.0, 1.0 + amount)
    print(f"✅ All bass boosted by {amount*100:.0f}%")

def reset_balance(balance_values):
    """Reset all to neutral"""
    for ch in balance_values:
        balance_values[ch] = 1.0
    print("✅ All channels reset to neutral (1.0)")


if __name__ == "__main__":
    print("🎚️ Balance Control System")
    print("Use this to replace the broken GUI in your notebook")