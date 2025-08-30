"""
Balance Control System for Mixing Engine
Single source of truth for channel balance adjustments
"""

class BalanceController:
    """
    Clean, simple balance control for any set of channels.
    No GUI, just pure functionality that works.
    """
    
    def __init__(self, channels_dict):
        """
        Initialize with the channels dictionary from your session.
        
        Args:
            channels_dict: Dictionary like {'drums': {'kick': path, 'snare': path}, ...}
        """
        self.channels = channels_dict
        self.overrides = {}
        
        # Build flat list of all channel IDs
        self.channel_ids = []
        for category, tracks in channels_dict.items():
            for track_name in tracks.keys():
                self.channel_ids.append(f"{category}.{track_name}")
    
    def set_channel(self, channel_id, value):
        """Set a specific channel's balance."""
        if value == 1.0:
            self.overrides.pop(channel_id, None)
        else:
            self.overrides[channel_id] = value
        return self
    
    def set_category(self, category, value):
        """Set all channels in a category to the same value."""
        if category in self.channels:
            for track_name in self.channels[category].keys():
                channel_id = f"{category}.{track_name}"
                self.set_channel(channel_id, value)
        return self
    
    def boost_drums(self, amount=3.0):
        """Boost all drum channels."""
        return self.set_category('drums', amount)
    
    def reduce_vocals(self, amount=0.5):
        """Reduce all vocal channels."""
        self.set_category('vocals', amount)
        self.set_category('backvocals', amount)
        return self
    
    def boost_bass(self, amount=1.5):
        """Boost all bass channels."""
        return self.set_category('bass', amount)
    
    def reset(self):
        """Reset all overrides to neutral."""
        self.overrides = {}
        return self
    
    def get_overrides(self):
        """Get the channel_overrides dictionary for the mixing engine."""
        return self.overrides.copy()
    
    def apply_preset(self, preset_name):
        """Apply a predefined preset."""
        presets = {
            'drum_power': {
                'drums': 4.0,
                'vocals': 0.4,
                'bass': 1.5
            },
            'vocal_focus': {
                'drums': 0.5,
                'vocals': 1.5,
                'bass': 0.8
            },
            'balanced': {
                'drums': 1.0,
                'vocals': 1.0,
                'bass': 1.0
            },
            'instrumental': {
                'drums': 1.2,
                'vocals': 0.2,
                'bass': 1.3,
                'guitars': 1.5,
                'keys': 1.4
            }
        }
        
        if preset_name in presets:
            self.reset()
            for category, value in presets[preset_name].items():
                self.set_category(category, value)
        return self
    
    def show_status(self):
        """Display current balance settings."""
        if not self.overrides:
            print("📊 No balance overrides (all channels at 1.0x)")
            return
        
        print(f"📊 Balance Overrides ({len(self.overrides)} channels):")
        
        # Group by category for display
        by_category = {}
        for channel_id, value in self.overrides.items():
            category = channel_id.split('.')[0]
            if category not in by_category:
                by_category[category] = []
            by_category[category].append((channel_id, value))
        
        for category, channels in sorted(by_category.items()):
            print(f"\n{category.upper()}:")
            for channel_id, value in sorted(channels):
                change_pct = (value - 1.0) * 100
                direction = "↑" if change_pct > 0 else "↓"
                track = channel_id.split('.')[1]
                print(f"  {direction} {track:20} = {value:.1f}x ({change_pct:+.0f}%)")
    
    def generate_code(self):
        """Generate Python code for current settings."""
        if not self.overrides:
            return "channel_overrides = {}"
        
        lines = ["channel_overrides = {"]
        for channel_id, value in sorted(self.overrides.items()):
            lines.append(f"    '{channel_id}': {value},")
        lines.append("}")
        return "\n".join(lines)


def quick_balance(channels, **kwargs):
    """
    Quick one-liner balance setup.
    
    Example:
        channel_overrides = quick_balance(channels, drums=4.0, vocals=0.3, bass=1.5)
    """
    controller = BalanceController(channels)
    
    for category, value in kwargs.items():
        controller.set_category(category, value)
    
    return controller.get_overrides()


# Notebook-friendly interface
def create_balance_controls(channels):
    """
    Create balance controls for notebook use.
    Returns both controller and ready-to-use functions.
    """
    controller = BalanceController(channels)
    
    # Create convenience functions that capture the controller
    def boost_drums(amount=3.0):
        controller.boost_drums(amount)
        controller.show_status()
        return controller.get_overrides()
    
    def reduce_vocals(amount=0.5):
        controller.reduce_vocals(amount)
        controller.show_status()
        return controller.get_overrides()
    
    def boost_bass(amount=1.5):
        controller.boost_bass(amount)
        controller.show_status()
        return controller.get_overrides()
    
    def set_category(category, amount):
        controller.set_category(category, amount)
        controller.show_status()
        return controller.get_overrides()
    
    def set_channel(channel_id, value):
        controller.set_channel(channel_id, value)
        controller.show_status()
        return controller.get_overrides()
    
    def apply_preset(preset_name):
        controller.apply_preset(preset_name)
        controller.show_status()
        return controller.get_overrides()
    
    def reset():
        controller.reset()
        print("🔄 All overrides reset")
        return controller.get_overrides()
    
    def show():
        controller.show_status()
    
    def get_overrides():
        return controller.get_overrides()
    
    # Return namespace with all functions
    class Controls:
        pass
    
    controls = Controls()
    controls.boost_drums = boost_drums
    controls.reduce_vocals = reduce_vocals
    controls.boost_bass = boost_bass
    controls.set_category = set_category
    controls.set_channel = set_channel
    controls.apply_preset = apply_preset
    controls.reset = reset
    controls.show = show
    controls.get_overrides = get_overrides
    controls.controller = controller
    
    return controls