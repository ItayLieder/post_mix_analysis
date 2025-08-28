# 🎚️ WORKING BALANCE CONTROL - REPLACE YOUR BROKEN CELL WITH THIS

from working_balance_control import create_working_balance_gui

print("🎚️ WORKING BALANCE CONTROL")
print("=" * 50)
print("✅ Real GUI using tkinter - works everywhere!")
print("🎛️ Group and individual channel controls")
print("⚡ Quick preset buttons")
print()

# Launch the working GUI
print("🚀 Opening balance control window...")
selected_balance = create_working_balance_gui(channels)

print("\n✅ Balance settings ready for mixing engine!")
print("🎯 Continue with your notebook - the 'selected_balance' variable is set!")

# Show what was applied
if "channel_overrides" in selected_balance:
    overrides = selected_balance["channel_overrides"]
    print(f"\n📊 Applied {len(overrides)} individual adjustments:")
    for channel_id, value in sorted(overrides.items()):
        change_pct = (value - 1.0) * 100
        direction = "↑" if change_pct > 0 else "↓"
        print(f"  {direction} {channel_id}: {value:.2f} ({change_pct:+.0f}%)")
else:
    print("\n📊 Using standard balance preset")