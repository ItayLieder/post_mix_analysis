#!/usr/bin/env python3
"""
Balance Control Demo - Shows how to use the balance system
"""

from simple_balance_control import create_stem_balancer

def demo_balance_control():
    """Demo the balance control system"""
    
    # Example channels (replace with your actual channels)
    channels = {
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
    
    # Create the balancer
    print("🎚️ Creating balance control system...")
    balancer = create_stem_balancer(channels)
    
    print("\n" + "="*60)
    print("🎯 DEMO: Balance Control Examples")
    print("="*60)
    
    # Show initial state
    print("\n1️⃣ Initial state (all neutral):")
    balancer.show_channels()
    
    # Example 1: Individual channel control
    print("\n2️⃣ Individual channel adjustments:")
    balancer.set_channel('bass.bass_synt', 1.3)  # Boost bass_synt by 30%
    balancer.set_channel('bass.bass_synt2', 0.8)  # Reduce bass_synt2 by 20%
    balancer.set_channel('drums.kick', 1.1)      # Slightly boost kick
    
    # Example 2: Group control
    print("\n3️⃣ Group adjustments:")
    balancer.set_group('vocals', 0.7)            # Reduce all vocals by 30%
    
    # Show current state
    print("\n4️⃣ Current balance after adjustments:")
    balancer.show_channels()
    
    # Show only changed values
    print("\n5️⃣ Changed values only:")
    changed = balancer.get_changed_values()
    for channel_id, value in changed.items():
        change_pct = (value - 1.0) * 100
        direction = "↑" if change_pct > 0 else "↓"
        print(f"  {direction} {channel_id}: {value:.2f} ({change_pct:+.0f}%)")
    
    # Example 3: Presets
    print("\n6️⃣ Using presets:")
    balancer.preset_boost_drums(0.3)  # 30% drums boost
    balancer.preset_balance_bass()    # Smart bass balancing
    
    print("\n7️⃣ Final balance:")
    final_balance = balancer.get_changed_values()
    for channel_id, value in final_balance.items():
        change_pct = (value - 1.0) * 100
        direction = "↑" if change_pct > 0 else "↓"
        print(f"  {direction} {channel_id}: {value:.2f} ({change_pct:+.0f}%)")
    
    print("\n✅ Demo complete! This dictionary can be used with your mixing engine:")
    print(f"channel_balance = {final_balance}")
    
    return balancer, final_balance

if __name__ == "__main__":
    balancer, balance = demo_balance_control()