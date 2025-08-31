#!/usr/bin/env python3
"""Test to understand the gain calculations"""

from audio_utils import db_to_linear

# Simulate what happens to drums.kick with the impressive session values
print("GAIN CALCULATION TEST")
print("="*50)

# Initial gain
initial_gain = 1.0
print(f"Initial gain: {initial_gain}")

# From _apply_drum_balance with drum_punch=0.6, vocal_prominence=0.5
drum_punch = 0.6
vocal_prominence = 0.5
punch_boost_db = (drum_punch - 0.5) * 3  # 0.3 dB
gain_after_balance = initial_gain * db_to_linear(punch_boost_db)
print(f"\nAfter _apply_drum_balance (drum_punch={drum_punch}):")
print(f"  punch_boost_db = {punch_boost_db:.2f} dB")
print(f"  multiplier = {db_to_linear(punch_boost_db):.4f}")
print(f"  gain = {gain_after_balance:.4f}")

# From _apply_channel_overrides with drums.kick=7.5
channel_override = 7.5
final_gain = gain_after_balance * channel_override
print(f"\nAfter _apply_channel_overrides (override={channel_override}):")
print(f"  final gain = {final_gain:.4f}")
print(f"  final gain in dB = {20 * (final_gain**0.5):.1f} dB")

print("\n" + "="*50)
print("RESULT: The impressive session had:")
print(f"  - drum_punch = 0.6 in mix_balance")
print(f"  - drums.kick = 7.5 in channel_overrides")
print(f"  - Final kick gain = {final_gain:.2f}x (very loud!)")
print("\nThis explains why the mix was so punchy!")