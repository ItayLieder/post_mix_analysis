# Add This to Your Notebook

## Replace cell 10 (or add after it) with:

```python
# 🎚️ CLEAN BALANCE CONTROL SYSTEM

from controls.balance_control import create_balance_controls, quick_balance

# Quick preset option (uncomment one):
# channel_overrides = quick_balance(channels, drums=4.0, vocals=0.3, bass=1.5)  # Drum power
# channel_overrides = quick_balance(channels, vocals=2.0, drums=0.5)  # Vocal focus

# Interactive control system
controls = create_balance_controls(channels)

# Apply your balance adjustments
controls.boost_drums(4.0)      # Make drums 4x louder
controls.reduce_vocals(0.3)    # Make vocals 30% volume  
controls.boost_bass(1.5)       # Make bass 1.5x louder

# Get the overrides for mixing engine
channel_overrides = controls.get_overrides()

# Show what we've set
controls.show()

print(f"\n✅ Balance ready: {len(channel_overrides)} channels modified")
```

## Available Commands:

```python
# Individual commands you can use:
controls.boost_drums(3.0)                    # All drums 3x
controls.reduce_vocals(0.5)                  # All vocals 50%
controls.boost_bass(2.0)                     # All bass 2x
controls.set_category('guitars', 0.8)        # All guitars 80%
controls.set_channel('keys.piano4', 1.5)     # Specific channel
controls.apply_preset('drum_power')          # Use preset
controls.reset()                              # Reset all
controls.show()                               # Display current

# Get final overrides for mixing:
channel_overrides = controls.get_overrides()
```

## Available Presets:

- `'drum_power'` - Drums 4x, vocals 0.4x, bass 1.5x
- `'vocal_focus'` - Vocals 1.5x, drums 0.5x, bass 0.8x  
- `'balanced'` - Everything at 1.0x
- `'instrumental'` - Reduced vocals, boosted instruments

## That's it! 

No GUI needed. Works with any channels. Dynamic discovery.