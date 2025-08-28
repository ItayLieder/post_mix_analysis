# 🎚️ DIRECT SLIDER VALUES - NO GUI NEEDED
# Copy this code directly into a notebook cell

print("🎚️ BALANCE CONTROL - DIRECT VALUE ENTRY")
print("=" * 60)

# Create the balance dictionary with sliders represented as values
# Adjust these values directly (0.1 to 5.0, where 1.0 is neutral)
channel_overrides = {
    # 🥁 DRUMS - Increase these for more drum power
    'drums.kick': 4.0,      # ← Adjust this value (1.0 = neutral, 4.0 = 400% boost)
    'drums.snare': 4.0,     # ← Adjust this value
    'drums.hihat': 3.0,     # ← Adjust this value
    'drums.tom': 3.0,       # ← Adjust this value
    'drums.cymbal': 3.0,    # ← Adjust this value
    
    # 🎤 VOCALS - Decrease these to reduce vocal volume
    'vocals.lead_vocal1': 0.4,  # ← Adjust this value (0.4 = 60% reduction)
    'vocals.lead_vocal2': 0.4,  # ← Adjust this value
    'vocals.lead_vocal3': 0.4,  # ← Adjust this value
    
    # 🎸 BASS - Adjust bass levels
    'bass.bass_guitar5': 1.5,   # ← Adjust this value
    'bass.bass1': 1.5,          # ← Adjust this value
    'bass.bass_guitar3': 1.0,   # ← Adjust this value
    'bass.bass_synth2': 1.0,    # ← Adjust this value
    'bass.bass_synth4': 1.0,    # ← Adjust this value
    
    # 🎹 GUITARS - Leave at 1.0 or adjust as needed
    'guitars.electric_guitar4': 1.0,
    'guitars.electric_guitar5': 1.0,
    'guitars.electric_guitar6': 1.0,
    'guitars.electric_guitar2': 1.0,
    'guitars.acoustic_guitar1': 1.0,
    'guitars.electric_guitar3': 1.0,
    
    # 🎵 KEYS
    'keys.bell3': 1.0,
    'keys.clavinet1': 1.0,
    'keys.piano4': 1.0,
    'keys.piano2': 1.0,
    
    # 🎤 BACKING VOCALS
    'backvocals.lead_vocal3': 0.6,
    'backvocals.lead_vocal2': 0.6,
    'backvocals.backing_vocal': 0.6,
    'backvocals.lead_vocal1': 0.6,
    'backvocals.lead_vocal4': 0.6,
    
    # 🎹 SYNTHS
    'synths.rythmic_synth1': 1.0,
    'synths.pad3': 0.8,
    'synths.pad2': 0.8,
}

# Remove channels that are at neutral (1.0) to simplify
channel_overrides = {k: v for k, v in channel_overrides.items() if abs(v - 1.0) > 0.01}

# Display what's changed
print(f"✅ Balance settings configured: {len(channel_overrides)} channels modified")
print("\n📊 Current settings:")

# Group by category for display
categories = {}
for ch, val in channel_overrides.items():
    cat = ch.split('.')[0]
    if cat not in categories:
        categories[cat] = []
    categories[cat].append((ch, val))

for cat, channels in categories.items():
    print(f"\n{cat.upper()}:")
    for ch, val in channels:
        change_pct = (val - 1.0) * 100
        bar = '█' * int(val * 2) + '░' * (10 - int(val * 2))
        direction = "↑" if change_pct > 0 else "↓"
        print(f"  {direction} {ch:30} [{bar}] {val:.1f} ({change_pct:+.0f}%)")

print("\n" + "=" * 60)
print("🎯 INSTRUCTIONS:")
print("1. Edit the values above directly in the code")
print("2. Values: 0.1 = 90% reduction, 1.0 = neutral, 5.0 = 500% boost")
print("3. Re-run this cell after making changes")
print("4. Then run your mixing cells (12 and 14)")
print("\n✅ channel_overrides variable is ready for mixing engine!")