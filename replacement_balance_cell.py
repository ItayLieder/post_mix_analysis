# 🎚️ WORKING BALANCE CONTROL (REPLACES BROKEN GUI)

from notebook_balance_fix import (
    create_working_balance_interface, 
    apply_balance_to_mixing,
    show_balance_status,
    preset_boost_drums,
    preset_reduce_vocals, 
    preset_boost_bass,
    reset_balance
)

# Create the working balance interface
print("🔧 Creating working balance control to replace broken GUI...")
balance_values = create_working_balance_interface(channels)

print("\n" + "="*60)
print("🎯 QUICK EXAMPLES - COPY AND PASTE THESE:")
print("="*60)

print("\n# Example 1: Fix your vocals/drums issue")
print("preset_reduce_vocals(balance_values, 0.3)  # Reduce vocals 30%")
print("preset_boost_drums(balance_values, 0.2)    # Boost drums 20%")

print("\n# Example 2: Individual channel control")
print("balance_values['bass.bass_guitar5'] = 1.4   # Boost specific bass 40%")
print("balance_values['vocals.lead_vocal1'] = 0.6  # Reduce specific vocal 40%")

print("\n# Example 3: Category adjustments")
print("# Boost all bass by 15%:")
print("for ch in balance_values:")
print("    if 'bass.' in ch:")
print("        balance_values[ch] = 1.15")

print("\n# Check current status:")
print("show_balance_status(balance_values)")

print("\n# Apply to mixing engine:")
print("selected_balance = apply_balance_to_mixing(balance_values)")

print("\n" + "="*60)
print("💡 TIP: Run the examples above, then check status, then apply!")
print("="*60)