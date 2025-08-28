#!/usr/bin/env python3
"""
Simple Balance Control - Command line interface for balance control
Works without Jupyter widgets - just Python dictionaries and functions
"""

from typing import Dict, Any
import json

class StemBalancer:
    """Simple balance control without GUI widgets"""
    
    def __init__(self, channels: Dict):
        self.channels = channels
        self.balance_values = {}
        self.groups = {}
        
        # Initialize all channels to 1.0 (neutral)
        for category, tracks in channels.items():
            for track_name in tracks.keys():
                channel_id = f"{category}.{track_name}"
                self.balance_values[channel_id] = 1.0
        
        # Create smart groups
        self._create_groups()
    
    def _create_groups(self):
        """Create channel groups for batch control"""
        all_channels = list(self.balance_values.keys())
        
        # Category groups
        for category in self.channels.keys():
            self.groups[category] = [ch for ch in all_channels if ch.startswith(f"{category}.")]
        
        # Smart groups based on names
        self.groups["All Bass Elements"] = [ch for ch in all_channels if 'bass' in ch.lower()]
        self.groups["All Drums"] = [ch for ch in all_channels if any(word in ch.lower() for word in ['drum', 'kick', 'snare', 'hat', 'crash', 'tom'])]
        self.groups["All Vocals"] = [ch for ch in all_channels if any(word in ch.lower() for word in ['vocal', 'vox', 'harmony', 'chorus'])]
        self.groups["All Synths"] = [ch for ch in all_channels if 'synt' in ch.lower()]
    
    def show_channels(self):
        """Display all channels and their current values"""
        print("\n🎚️ CURRENT BALANCE VALUES")
        print("=" * 50)
        
        for category, tracks in self.channels.items():
            print(f"\n📁 {category.upper()}:")
            for track_name in tracks.keys():
                channel_id = f"{category}.{track_name}"
                value = self.balance_values[channel_id]
                change_pct = (value - 1.0) * 100
                direction = "↑" if change_pct > 0 else "↓" if change_pct < 0 else "="
                print(f"  {direction} {track_name:20} = {value:.2f} ({change_pct:+.0f}%)")
    
    def show_groups(self):
        """Display available groups"""
        print("\n🏷️ AVAILABLE GROUPS")
        print("=" * 30)
        
        for group_name, channels in self.groups.items():
            if channels:  # Only show groups with channels
                print(f"{group_name:20} ({len(channels)} channels)")
                for ch in channels[:3]:  # Show first 3 channels
                    print(f"  • {ch.split('.')[-1]}")
                if len(channels) > 3:
                    print(f"  • ... and {len(channels) - 3} more")
                print()
    
    def set_channel(self, channel_id: str, value: float):
        """Set individual channel balance"""
        if channel_id in self.balance_values:
            value = max(0.0, min(2.0, value))  # Clamp between 0.0 and 2.0
            self.balance_values[channel_id] = value
            change_pct = (value - 1.0) * 100
            print(f"✅ {channel_id} = {value:.2f} ({change_pct:+.0f}%)")
        else:
            print(f"❌ Channel not found: {channel_id}")
    
    def set_group(self, group_name: str, value: float):
        """Set balance for all channels in a group"""
        if group_name in self.groups:
            value = max(0.0, min(2.0, value))  # Clamp between 0.0 and 2.0
            channels = self.groups[group_name]
            
            for channel_id in channels:
                if channel_id in self.balance_values:
                    self.balance_values[channel_id] = value
            
            change_pct = (value - 1.0) * 100
            print(f"✅ {group_name} ({len(channels)} channels) = {value:.2f} ({change_pct:+.0f}%)")
        else:
            print(f"❌ Group not found: {group_name}")
    
    def preset_boost_drums(self, amount: float = 0.2):
        """Preset: Boost drums by amount (default 20%)"""
        if "drums" in self.groups:
            new_value = min(2.0, 1.0 + amount)
            self.set_group("drums", new_value)
        else:
            print("❌ No drums group found")
    
    def preset_reduce_vocals(self, amount: float = 0.2):
        """Preset: Reduce vocals by amount (default 20%)"""
        if "vocals" in self.groups:
            new_value = max(0.0, 1.0 - amount)
            self.set_group("vocals", new_value)
        else:
            print("❌ No vocals group found")
    
    def preset_balance_bass(self):
        """Preset: Balance bass elements with slight variations"""
        bass_channels = [ch for ch in self.balance_values.keys() if 'bass' in ch.lower()]
        for i, channel_id in enumerate(bass_channels):
            # Vary bass levels: 0.8, 0.9, 1.0, 1.1, etc.
            level = 0.8 + (i * 0.1)
            self.set_channel(channel_id, min(level, 2.0))
    
    def reset_all(self):
        """Reset all channels to neutral (1.0)"""
        for channel_id in self.balance_values.keys():
            self.balance_values[channel_id] = 1.0
        print("✅ All channels reset to neutral balance")
    
    def get_balance_dict(self) -> Dict[str, float]:
        """Get current balance values as dictionary"""
        return self.balance_values.copy()
    
    def get_changed_values(self) -> Dict[str, float]:
        """Get only channels that have been changed from neutral"""
        return {k: v for k, v in self.balance_values.items() if abs(v - 1.0) > 0.01}
    
    def save_preset(self, name: str):
        """Save current settings as a preset"""
        preset = {
            "name": name,
            "balance_values": self.get_changed_values()
        }
        
        filename = f"balance_preset_{name.lower().replace(' ', '_')}.json"
        with open(filename, 'w') as f:
            json.dump(preset, f, indent=2)
        
        print(f"✅ Preset saved: {filename}")
    
    def load_preset(self, filename: str):
        """Load balance settings from preset file"""
        try:
            with open(filename, 'r') as f:
                preset = json.load(f)
            
            # Reset first
            self.reset_all()
            
            # Apply preset values
            balance_values = preset.get("balance_values", {})
            for channel_id, value in balance_values.items():
                if channel_id in self.balance_values:
                    self.balance_values[channel_id] = value
            
            print(f"✅ Preset loaded: {filename}")
            print(f"📊 Applied {len(balance_values)} balance changes")
            
        except FileNotFoundError:
            print(f"❌ Preset file not found: {filename}")
        except Exception as e:
            print(f"❌ Error loading preset: {e}")


def create_stem_balancer(channels):
    """Create stem balancer and show usage instructions"""
    balancer = StemBalancer(channels)
    
    print("🎚️ STEM BALANCE CONTROL")
    print("=" * 50)
    print("Available commands:")
    print("  balancer.show_channels()              - Show all channels")
    print("  balancer.show_groups()                - Show available groups")
    print("  balancer.set_channel('bass.bass_synt', 1.5)  - Set individual channel")
    print("  balancer.set_group('drums', 1.2)      - Set group balance")
    print("  balancer.preset_boost_drums()         - Quick drums boost")
    print("  balancer.preset_reduce_vocals()       - Quick vocals reduction")
    print("  balancer.preset_balance_bass()        - Balance bass elements")
    print("  balancer.reset_all()                  - Reset to neutral")
    print("  balancer.get_changed_values()         - Get current changes")
    print("  balancer.save_preset('my_mix')        - Save current settings")
    print("\n💡 Values: 0.0 = muted, 1.0 = neutral, 2.0 = doubled")
    
    return balancer


# Example usage
if __name__ == "__main__":
    # Example channels
    example_channels = {
        "drums": {
            "kick": "/path/to/kick.wav",
            "snare": "/path/to/snare.wav", 
            "hats": "/path/to/hats.wav",
        },
        "bass": {
            "bass_synt": "/path/to/bass_synt.wav",
            "bass_synt2": "/path/to/bass_synt2.wav",
        },
        "vocals": {
            "main_verse": "/path/to/main.wav",
            "harmony": "/path/to/harmony.wav",
        }
    }
    
    balancer = create_stem_balancer(example_channels)
    balancer.show_channels()
    balancer.show_groups()