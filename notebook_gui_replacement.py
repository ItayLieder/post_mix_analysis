# 🎚️ WORKING GUI REPLACEMENT FOR YOUR NOTEBOOK
# Replace the broken ipywidgets cell with this code

from working_balance_control import create_working_balance_gui

print("🎚️ WORKING BALANCE CONTROL")
print("=" * 50)
print("✅ This uses tkinter GUI that actually works in PyCharm!")
print("🎛️ Real sliders with group and individual controls")
print("⚡ Quick preset buttons for common adjustments")
print("📊 Export functionality for mixing engine")
print()

# Create and run the working GUI
print("🚀 Starting GUI... (window will open)")
selected_balance = create_working_balance_gui(channels)

print("\n" + "="*50)
print("🎯 BALANCE SETTINGS READY FOR MIXING:")
print("="*50)
print("The 'selected_balance' variable is now ready to use!")
print("Continue with your existing notebook cells.")

# Show the final balance
if "channel_overrides" in selected_balance:
    print(f"\n📊 Applied individual channel adjustments:")
    for ch, val in selected_balance["channel_overrides"].items():
        change_pct = (val - 1.0) * 100
        direction = "↑" if change_pct > 0 else "↓"
        print(f"  {direction} {ch}: {change_pct:+.0f}%")

print(f"\n✅ Balance configuration complete!")