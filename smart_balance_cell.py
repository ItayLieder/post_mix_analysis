# 🎚️ ADD THIS TO YOUR NOTEBOOK - SMART DYNAMIC BALANCE CONTROL

from dynamic_balance_control import generate_balance_controls, create_command_interface

if 'channels' in locals() and channels:
    print("🎚️ SMART BALANCE CONTROL - Works with ANY channels!")
    print("=" * 70)
    
    # Count what we have
    total_channels = sum(len(tracks) for tracks in channels.values())
    print(f"📊 Found {total_channels} channels in {len(channels)} categories")
    print(f"📁 Categories: {', '.join(channels.keys())}")
    
    print("\n" + "=" * 70)
    print("OPTION 1: FUNCTION-BASED CONTROL (Recommended)")
    print("=" * 70)
    
    # Initialize
    print("""
# Run this code:
channel_overrides = {}

# PRESET FUNCTIONS - Use these to quickly adjust groups:
def boost_drums(amount=3.0):
    for track in channels.get('drums', {}).keys():
        channel_overrides[f'drums.{track}'] = amount
    print(f"✅ Set {len(channels.get('drums', {}))} drum channels to {amount}x")

def reduce_vocals(amount=0.5):
    for track in channels.get('vocals', {}).keys():
        channel_overrides[f'vocals.{track}'] = amount
    print(f"✅ Set {len(channels.get('vocals', {}))} vocal channels to {amount}x")

def boost_bass(amount=1.5):
    for track in channels.get('bass', {}).keys():
        channel_overrides[f'bass.{track}'] = amount
    print(f"✅ Set {len(channels.get('bass', {}))} bass channels to {amount}x")

def set_category(category, amount):
    if category in channels:
        for track in channels[category].keys():
            channel_overrides[f'{category}.{track}'] = amount
        print(f"✅ Set {len(channels[category])} {category} channels to {amount}x")

def set_channel(channel_id, value):
    channel_overrides[channel_id] = value
    print(f"✅ Set {channel_id} to {value}x")

def reset_all():
    global channel_overrides
    channel_overrides = {}
    print("🔄 Reset all overrides")

def show_overrides():
    if channel_overrides:
        print(f"📊 Current overrides ({len(channel_overrides)} channels):")
        for ch, val in sorted(channel_overrides.items()):
            print(f"  {ch}: {val}x")
    else:
        print("📊 No overrides set (all channels at 1.0x)")

# USAGE EXAMPLES:
boost_drums(4.0)        # Make drums 4x louder
reduce_vocals(0.3)      # Make vocals 30% volume
boost_bass(2.0)         # Make bass 2x louder
set_category('guitars', 0.8)  # Make all guitars 80% volume
set_channel('keys.piano4', 1.5)  # Boost specific piano
show_overrides()        # See what's changed
""")
    
    print("\n" + "=" * 70)
    print("OPTION 2: DIRECT DICTIONARY EDIT")
    print("=" * 70)
    
    # Generate the editable code
    generated = generate_balance_controls(channels)
    
    print("\n" + "=" * 70)
    print("📋 AVAILABLE CHANNELS IN YOUR PROJECT:")
    for category, tracks in channels.items():
        track_list = ', '.join(tracks.keys())
        print(f"  {category}: {track_list}")
    
    print("\n🎯 HOW TO USE:")
    print("1. Choose Option 1 (functions) or Option 2 (direct edit)")
    print("2. Copy the code to a new cell")
    print("3. Run the functions or edit values")
    print("4. Run your mixing cells (12 & 14)")
    
else:
    print("❌ No channels loaded! Run the channel loading cell first.")