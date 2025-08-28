# 🎚️ ADD THIS CELL TO YOUR NOTEBOOK - WORKING TKINTER SLIDERS

from simple_slider_gui import create_balance_sliders

# Create GUI and get balance settings
if 'channels' in locals() and channels:
    print("🎚️ Opening Balance Control GUI...")
    print("📋 A window will open with sliders for all channels")
    
    # This will open a GUI window
    balance_result = create_balance_sliders(channels)
    
    if balance_result:
        # Extract the channel overrides
        channel_overrides = balance_result['channel_overrides']
        balance_settings = balance_result
        
        print("✅ Balance settings applied!")
        print(f"📊 Modified {len(channel_overrides)} channels")
        
        # Set variables for the mixing engine
        globals()['channel_overrides'] = channel_overrides
        globals()['balance_settings'] = balance_settings
        
        print("🎯 Ready for mixing! Variables set:")
        print(f"  • channel_overrides: {len(channel_overrides)} channels")
        print("  • balance_settings: Ready")
        
    else:
        print("❌ No changes applied (cancelled or closed)")
        
else:
    print("❌ No channels found!")
    print("📝 Please run the channel loading cell first")