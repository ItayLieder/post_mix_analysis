# 🎚️ DYNAMIC BALANCE CONTROL - WORKS WITH ANY CHANNELS

def generate_balance_controls(channels_dict):
    """
    Dynamically generates balance controls for any set of channels
    No hardcoding - works with whatever channels are loaded
    """
    print("🎚️ DYNAMIC BALANCE CONTROL")
    print("=" * 70)
    
    # Dynamically create the control code
    code_lines = []
    code_lines.append("# 🎚️ AUTO-GENERATED BALANCE CONTROLS FOR YOUR CHANNELS")
    code_lines.append("# Edit these values: 0.1 = 10% volume, 1.0 = normal, 5.0 = 500% boost\n")
    code_lines.append("channel_overrides = {")
    
    # Dynamically add every channel found in the project
    for category, tracks in channels_dict.items():
        code_lines.append(f"    # {category.upper()}")
        for track_name in tracks.keys():
            channel_id = f"{category}.{track_name}"
            # Start everything at 1.0 (neutral)
            code_lines.append(f"    '{channel_id}': 1.0,  # ← Adjust this")
        code_lines.append("")
    
    code_lines.append("}")
    code_lines.append("")
    
    # Generate the full code string
    generated_code = "\n".join(code_lines)
    
    # Save to a file that can be edited
    with open("balance_controls_generated.py", "w") as f:
        f.write(generated_code)
    
    print(f"📋 Generated controls for {sum(len(t) for t in channels_dict.values())} channels")
    print("\n" + "─" * 70)
    print(generated_code)
    print("─" * 70)
    
    print("\n🎯 INSTRUCTIONS:")
    print("1. Copy the code above into a new cell")
    print("2. Edit the values for channels you want to adjust")
    print("3. Delete or leave at 1.0 any channels you don't want to change")
    print("4. Run that cell, then run your mixing cells")
    
    return generated_code

# Alternative: Create an interactive text-based menu
def create_text_menu_controls(channels_dict):
    """
    Creates an interactive text-based control system
    """
    import json
    
    print("🎚️ INTERACTIVE BALANCE CONTROL")
    print("=" * 70)
    
    # Initialize all channels at 1.0
    balance_dict = {}
    for category, tracks in channels_dict.items():
        for track_name in tracks.keys():
            channel_id = f"{category}.{track_name}"
            balance_dict[channel_id] = 1.0
    
    # Create quick presets
    print("\n📋 QUICK PRESETS (or edit individually below):")
    print("─" * 40)
    
    preset_code = f"""
# Quick group adjustments - uncomment and modify as needed:

# Boost all drums by 3x
for ch in {[k for k in balance_dict.keys() if 'drums.' in k]}:
    channel_overrides[ch] = 3.0

# Reduce all vocals by 50%  
for ch in {[k for k in balance_dict.keys() if 'vocals.' in k]}:
    channel_overrides[ch] = 0.5

# Boost all bass by 1.5x
for ch in {[k for k in balance_dict.keys() if 'bass.' in k]}:
    channel_overrides[ch] = 1.5
    
# Or set individual channels:
channel_overrides = {{
    {chr(10).join(f"    '{ch}': 1.0,  # Adjust" for ch in list(balance_dict.keys())[:5])}
    # ... add more as needed
}}
"""
    
    print(preset_code)
    
    return balance_dict

# Function to create a simple command-based interface
def create_command_interface(channels_dict):
    """
    Creates a command-based interface for balance control
    """
    print("🎚️ COMMAND-BASED BALANCE CONTROL")
    print("=" * 70)
    
    commands = f"""
# Initialize empty overrides
channel_overrides = {{}}

# Available channels in your project:
{chr(10).join(f"# - {cat}.{track}" for cat, tracks in channels_dict.items() for track in tracks.keys())}

# COMMANDS TO USE:

# 1. Boost all drums:
def boost_drums(amount=3.0):
    for cat, tracks in channels.items():
        if cat == 'drums':
            for track in tracks.keys():
                channel_overrides[f'{{cat}}.{{track}}'] = amount

# 2. Reduce all vocals:
def reduce_vocals(amount=0.5):
    for cat, tracks in channels.items():
        if cat == 'vocals':
            for track in tracks.keys():
                channel_overrides[f'{{cat}}.{{track}}'] = amount

# 3. Set specific channel:
def set_channel(channel_id, value):
    channel_overrides[channel_id] = value

# 4. Set all in category:
def set_category(category, value):
    if category in channels:
        for track in channels[category].keys():
            channel_overrides[f'{{category}}.{{track}}'] = value

# EXAMPLES:
boost_drums(4.0)           # Boost drums 4x
reduce_vocals(0.3)         # Reduce vocals to 30%
set_channel('bass.bass1', 2.0)  # Boost specific bass track
set_category('guitars', 0.8)    # Reduce all guitars to 80%

# Show what's set:
print(f"Set {{len(channel_overrides)}} overrides:")
for ch, val in channel_overrides.items():
    print(f"  {{ch}}: {{val}}")
"""
    
    print(commands)
    return commands

if __name__ == "__main__":
    print("🎚️ Dynamic Balance Control Module Ready!")
    print("Functions available:")
    print("  • generate_balance_controls(channels) - Generate editable controls")
    print("  • create_text_menu_controls(channels) - Create preset commands")
    print("  • create_command_interface(channels) - Create function-based controls")