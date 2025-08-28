#!/usr/bin/env python3
"""
Interactive Balance Control - Menu-driven interface that works in PyCharm
"""

from typing import Dict
import os

class InteractiveBalancer:
    """Interactive balance control with menu interface"""
    
    def __init__(self, channels: Dict):
        self.channels = channels
        self.balance_values = {}
        self.groups = {}
        
        # Initialize all channels to 1.0 (neutral)
        for category, tracks in channels.items():
            for track_name in tracks.keys():
                channel_id = f"{category}.{track_name}"
                self.balance_values[channel_id] = 1.0
        
        self._create_groups()
    
    def _create_groups(self):
        """Create channel groups"""
        all_channels = list(self.balance_values.keys())
        
        # Category groups
        for category in self.channels.keys():
            self.groups[category] = [ch for ch in all_channels if ch.startswith(f"{category}.")]
        
        # Smart groups
        self.groups["All Bass Elements"] = [ch for ch in all_channels if 'bass' in ch.lower()]
        self.groups["All Drums"] = [ch for ch in all_channels if any(word in ch.lower() for word in ['drum', 'kick', 'snare', 'hat', 'crash', 'tom'])]
        self.groups["All Vocals"] = [ch for ch in all_channels if any(word in ch.lower() for word in ['vocal', 'vox', 'harmony', 'chorus'])]
        self.groups["All Synths"] = [ch for ch in all_channels if 'synt' in ch.lower()]
    
    def clear_screen(self):
        """Clear screen for better UX"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def show_current_state(self):
        """Show current balance state"""
        print("\n🎚️ CURRENT BALANCE")
        print("=" * 40)
        
        changed_any = False
        for category, tracks in self.channels.items():
            category_changed = False
            category_display = []
            
            for track_name in tracks.keys():
                channel_id = f"{category}.{track_name}"
                value = self.balance_values[channel_id]
                change_pct = (value - 1.0) * 100
                
                if abs(change_pct) > 0.1:
                    changed_any = True
                    category_changed = True
                
                direction = "↑" if change_pct > 0 else "↓" if change_pct < 0 else "="
                category_display.append(f"  {direction} {track_name:15} = {value:.2f} ({change_pct:+.0f}%)")
            
            if category_changed or not changed_any:
                print(f"\n📁 {category.upper()}:")
                for line in category_display:
                    print(line)
        
        if not changed_any:
            print("\n💡 All channels at neutral balance (1.0)")
    
    def main_menu(self):
        """Main interactive menu"""
        while True:
            self.clear_screen()
            print("🎚️ INTERACTIVE BALANCE CONTROL")
            print("=" * 50)
            
            self.show_current_state()
            
            print("\n🎯 BALANCE CONTROL OPTIONS:")
            print("1. Adjust Individual Channel")
            print("2. Adjust Group") 
            print("3. Quick Presets")
            print("4. Reset All to Neutral")
            print("5. Save Current Settings")
            print("6. Load Settings")
            print("7. Export Balance Dictionary")
            print("0. Exit")
            
            choice = input("\nSelect option (0-7): ").strip()
            
            if choice == '0':
                print("\n✅ Balance control complete!")
                return self.balance_values
            elif choice == '1':
                self.adjust_individual_channel()
            elif choice == '2':
                self.adjust_group()
            elif choice == '3':
                self.quick_presets_menu()
            elif choice == '4':
                self.reset_all()
            elif choice == '5':
                self.save_settings()
            elif choice == '6':
                self.load_settings()
            elif choice == '7':
                self.export_balance_dict()
            else:
                print("❌ Invalid option. Press Enter to continue...")
                input()
    
    def adjust_individual_channel(self):
        """Menu for adjusting individual channels"""
        self.clear_screen()
        print("🎛️ ADJUST INDIVIDUAL CHANNEL")
        print("=" * 40)
        
        # Show channels with numbers
        channel_list = []
        idx = 1
        for category, tracks in self.channels.items():
            print(f"\n📁 {category.upper()}:")
            for track_name in tracks.keys():
                channel_id = f"{category}.{track_name}"
                value = self.balance_values[channel_id]
                change_pct = (value - 1.0) * 100
                direction = "↑" if change_pct > 0 else "↓" if change_pct < 0 else "="
                
                print(f"  {idx:2d}. {direction} {track_name:15} = {value:.2f} ({change_pct:+.0f}%)")
                channel_list.append((channel_id, track_name))
                idx += 1
        
        try:
            choice = input(f"\nSelect channel (1-{len(channel_list)}) or 0 to go back: ").strip()
            
            if choice == '0':
                return
            
            channel_idx = int(choice) - 1
            if 0 <= channel_idx < len(channel_list):
                channel_id, track_name = channel_list[channel_idx]
                current_value = self.balance_values[channel_id]
                
                print(f"\n🎚️ Adjusting: {track_name}")
                print(f"Current value: {current_value:.2f}")
                print("💡 Enter new value (0.0 = muted, 1.0 = neutral, 2.0 = doubled)")
                
                new_value = input("New value: ").strip()
                try:
                    new_val = float(new_value)
                    new_val = max(0.0, min(2.0, new_val))
                    self.balance_values[channel_id] = new_val
                    
                    change_pct = (new_val - 1.0) * 100
                    print(f"✅ {track_name} set to {new_val:.2f} ({change_pct:+.0f}%)")
                    input("Press Enter to continue...")
                    
                except ValueError:
                    print("❌ Invalid value. Press Enter to continue...")
                    input()
            else:
                print("❌ Invalid selection. Press Enter to continue...")
                input()
                
        except ValueError:
            print("❌ Invalid input. Press Enter to continue...")
            input()
    
    def adjust_group(self):
        """Menu for adjusting groups"""
        self.clear_screen()
        print("🏷️ ADJUST GROUP")
        print("=" * 30)
        
        # Show groups with numbers
        group_list = []
        idx = 1
        for group_name, channels in self.groups.items():
            if channels:
                print(f"{idx:2d}. {group_name:20} ({len(channels)} channels)")
                group_list.append(group_name)
                idx += 1
        
        try:
            choice = input(f"\nSelect group (1-{len(group_list)}) or 0 to go back: ").strip()
            
            if choice == '0':
                return
            
            group_idx = int(choice) - 1
            if 0 <= group_idx < len(group_list):
                group_name = group_list[group_idx]
                channels = self.groups[group_name]
                
                print(f"\n🏷️ Adjusting group: {group_name}")
                print(f"Affects {len(channels)} channels:")
                for ch in channels[:5]:
                    print(f"  • {ch.split('.')[-1]}")
                if len(channels) > 5:
                    print(f"  • ... and {len(channels) - 5} more")
                
                print("\n💡 Enter new value for all channels in group")
                print("(0.0 = muted, 1.0 = neutral, 2.0 = doubled)")
                
                new_value = input("New value: ").strip()
                try:
                    new_val = float(new_value)
                    new_val = max(0.0, min(2.0, new_val))
                    
                    for channel_id in channels:
                        if channel_id in self.balance_values:
                            self.balance_values[channel_id] = new_val
                    
                    change_pct = (new_val - 1.0) * 100
                    print(f"✅ {group_name} set to {new_val:.2f} ({change_pct:+.0f}%)")
                    input("Press Enter to continue...")
                    
                except ValueError:
                    print("❌ Invalid value. Press Enter to continue...")
                    input()
            else:
                print("❌ Invalid selection. Press Enter to continue...")
                input()
                
        except ValueError:
            print("❌ Invalid input. Press Enter to continue...")
            input()
    
    def quick_presets_menu(self):
        """Quick presets menu"""
        self.clear_screen()
        print("⚡ QUICK PRESETS")
        print("=" * 30)
        print("1. Boost Drums (+20%)")
        print("2. Reduce Vocals (-20%)")
        print("3. Balance Bass Elements")
        print("4. Boost All Bass (+15%)")
        print("5. Vocal Prominence (-10% drums, +10% vocals)")
        print("0. Back to main menu")
        
        choice = input("\nSelect preset (0-5): ").strip()
        
        if choice == '1':
            self.apply_preset_boost_drums()
        elif choice == '2':
            self.apply_preset_reduce_vocals()
        elif choice == '3':
            self.apply_preset_balance_bass()
        elif choice == '4':
            self.apply_preset_boost_bass()
        elif choice == '5':
            self.apply_preset_vocal_prominence()
        elif choice != '0':
            print("❌ Invalid option. Press Enter to continue...")
            input()
    
    def apply_preset_boost_drums(self):
        """Boost all drums by 20%"""
        drums = [ch for ch in self.balance_values.keys() if any(word in ch.lower() for word in ['drum', 'kick', 'snare', 'hat'])]
        for channel_id in drums:
            self.balance_values[channel_id] = min(2.0, self.balance_values[channel_id] * 1.2)
        print("✅ Drums boosted by 20%")
        input("Press Enter to continue...")
    
    def apply_preset_reduce_vocals(self):
        """Reduce all vocals by 20%"""
        vocals = [ch for ch in self.balance_values.keys() if any(word in ch.lower() for word in ['vocal', 'vox', 'harmony'])]
        for channel_id in vocals:
            self.balance_values[channel_id] = max(0.0, self.balance_values[channel_id] * 0.8)
        print("✅ Vocals reduced by 20%")
        input("Press Enter to continue...")
    
    def apply_preset_balance_bass(self):
        """Balance bass elements with variations"""
        bass_channels = [ch for ch in self.balance_values.keys() if 'bass' in ch.lower()]
        for i, channel_id in enumerate(bass_channels):
            level = 0.8 + (i * 0.1)
            self.balance_values[channel_id] = min(level, 2.0)
        print("✅ Bass elements balanced with variations")
        input("Press Enter to continue...")
    
    def apply_preset_boost_bass(self):
        """Boost all bass by 15%"""
        bass_channels = [ch for ch in self.balance_values.keys() if 'bass' in ch.lower()]
        for channel_id in bass_channels:
            self.balance_values[channel_id] = min(2.0, self.balance_values[channel_id] * 1.15)
        print("✅ All bass boosted by 15%")
        input("Press Enter to continue...")
    
    def apply_preset_vocal_prominence(self):
        """Make vocals more prominent"""
        # Reduce drums slightly
        drums = [ch for ch in self.balance_values.keys() if any(word in ch.lower() for word in ['drum', 'kick', 'snare', 'hat'])]
        for channel_id in drums:
            self.balance_values[channel_id] = max(0.0, self.balance_values[channel_id] * 0.9)
        
        # Boost vocals slightly  
        vocals = [ch for ch in self.balance_values.keys() if any(word in ch.lower() for word in ['vocal', 'vox', 'harmony'])]
        for channel_id in vocals:
            self.balance_values[channel_id] = min(2.0, self.balance_values[channel_id] * 1.1)
        
        print("✅ Vocal prominence applied (-10% drums, +10% vocals)")
        input("Press Enter to continue...")
    
    def reset_all(self):
        """Reset all channels to neutral"""
        confirm = input("Reset all channels to neutral (1.0)? (y/N): ").strip().lower()
        if confirm == 'y':
            for channel_id in self.balance_values.keys():
                self.balance_values[channel_id] = 1.0
            print("✅ All channels reset to neutral")
        input("Press Enter to continue...")
    
    def save_settings(self):
        """Save current settings"""
        name = input("Enter preset name: ").strip()
        if name:
            import json
            filename = f"balance_preset_{name.replace(' ', '_')}.json"
            with open(filename, 'w') as f:
                json.dump({
                    "name": name,
                    "balance_values": {k: v for k, v in self.balance_values.items() if abs(v - 1.0) > 0.01}
                }, f, indent=2)
            print(f"✅ Settings saved as {filename}")
        input("Press Enter to continue...")
    
    def load_settings(self):
        """Load settings"""
        filename = input("Enter preset filename: ").strip()
        if filename:
            try:
                import json
                with open(filename, 'r') as f:
                    data = json.load(f)
                
                # Reset first
                for channel_id in self.balance_values.keys():
                    self.balance_values[channel_id] = 1.0
                
                # Apply loaded values
                balance_values = data.get("balance_values", {})
                for channel_id, value in balance_values.items():
                    if channel_id in self.balance_values:
                        self.balance_values[channel_id] = value
                
                print(f"✅ Settings loaded from {filename}")
            except Exception as e:
                print(f"❌ Error loading settings: {e}")
        input("Press Enter to continue...")
    
    def export_balance_dict(self):
        """Export balance dictionary for mixing engine"""
        changed = {k: v for k, v in self.balance_values.items() if abs(v - 1.0) > 0.01}
        
        self.clear_screen()
        print("📊 BALANCE DICTIONARY FOR MIXING ENGINE")
        print("=" * 50)
        
        if changed:
            print("Copy this dictionary to use with your mixing engine:\n")
            print("channel_balance = {")
            for channel_id, value in changed.items():
                print(f'    "{channel_id}": {value:.2f},')
            print("}")
            
            print(f"\n📈 Total modified channels: {len(changed)}")
        else:
            print("channel_balance = {}  # All channels at neutral")
        
        input("\nPress Enter to continue...")


def create_interactive_balancer(channels):
    """Create and start interactive balance control"""
    balancer = InteractiveBalancer(channels)
    print("🎚️ Starting Interactive Balance Control...")
    print("💡 This will open a full-screen menu interface")
    input("Press Enter to continue...")
    
    balance_result = balancer.main_menu()
    
    # Return the changed values for integration
    changed = {k: v for k, v in balance_result.items() if abs(v - 1.0) > 0.01}
    return changed


# Example usage
if __name__ == "__main__":
    example_channels = {
        "drums": {"kick": "path1", "snare": "path2", "hats": "path3"},
        "bass": {"bass_synt": "path4", "bass_synt2": "path5"},
        "vocals": {"main_verse": "path6", "harmony": "path7"}
    }
    
    result = create_interactive_balancer(example_channels)