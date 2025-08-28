# 🎚️ SIMPLE BALANCE FIX FOR YOUR NOTEBOOK
# Replace the broken GUI cell (cell-10) with this code

print("🎚️ SIMPLE BALANCE CONTROL")
print("=" * 40)

# Your channels are already loaded. Let's create simple balance control.
print("📊 YOUR CHANNELS:")
for category, tracks in channels.items():
    print(f"  {category}: {list(tracks.keys())}")

print("\n" + "="*50)
print("🎯 QUICK FIX FOR YOUR VOCALS/DRUMS ISSUE:")
print("="*50)

# Simple presets - just copy and paste these:
print("\n# Option 1: Use this preset (copy/paste into next cell):")
print("selected_balance = {")
print('    "vocal_prominence": 0.2,     # Reduce loud vocals')
print('    "drum_punch": 0.7,           # Boost weak drums') 
print('    "bass_foundation": 0.6,')
print('    "instrument_presence": 0.5,')
print("}")

print("\n# Option 2: Individual channel control (copy/paste if you want specific control):")
print("selected_balance = {")
print('    "vocal_prominence": 0.3,')
print('    "drum_punch": 0.65,')
print('    "bass_foundation": 0.6,')
print('    "instrument_presence": 0.5,')
print('    "channel_overrides": {')
print('        # Reduce specific loud vocals:')
print('        "vocals.lead_vocal1": 0.7,')
print('        "vocals.lead_vocal2": 0.7,') 
print('        "vocals.lead_vocal3": 0.7,')
print('        # Boost specific weak drums:')
print('        "drums.kick": 1.3,')
print('        "drums.snare": 1.2,')
print('    }')
print("}")

print("\n✅ INSTRUCTIONS:")
print("1. Copy one of the options above")
print("2. Paste it into a new cell") 
print("3. Run that cell")
print("4. Continue with your existing mixing process")
print("\nThis will fix your vocals/drums balance issue!")