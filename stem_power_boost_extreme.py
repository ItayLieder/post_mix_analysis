"""
EXTREME stem power boost for super weak drums and bass.
This bypasses all high-pass filtering and applies targeted frequency boosts.
"""
import os
import numpy as np
from config import CONFIG

def apply_extreme_stem_power_boost():
    """Apply EXTREME power boosts to fix super weak drums and bass"""
    
    print("🚀 APPLYING EXTREME STEM POWER BOOST!")
    print("⚡ BYPASSING ALL HIGH-PASS FILTERS!")
    print("🎯 TARGETED FREQUENCY DOMAIN BOOSTS!")
    print("=" * 60)
    
    # 1. COMPLETELY DISABLE ALL HIGH-PASS FILTERING
    # Override the prep_hpf_hz to effectively disable it
    CONFIG.audio.prep_hpf_hz = 5.0  # Set to 5Hz (effectively disabled)
    os.environ['PREP_HPF_HZ'] = '5.0'  # Environment override too
    
    # 2. EXTREMELY AGGRESSIVE STEM GAINS
    os.environ['STEM_GAIN_DRUMS'] = '2.5'    # EXTREME drums boost
    os.environ['STEM_GAIN_BASS'] = '2.2'     # EXTREME bass boost  
    os.environ['STEM_GAIN_VOCALS'] = '3.5'   # EXTREME vocals boost
    os.environ['STEM_GAIN_MUSIC'] = '1.5'    # Strong music boost
    
    # 3. DETAILED STEM EXTREME BOOSTS
    os.environ['STEM_GAIN_KICK'] = '3.0'        # Kick EXTREMELY loud
    os.environ['STEM_GAIN_SNARE'] = '2.5'       # Snare very loud
    os.environ['STEM_GAIN_LEADVOCALS'] = '4.0'  # Lead vocals EXTREMELY loud
    
    # 4. DISABLE AUTO-GAIN COMPENSATION ENTIRELY
    CONFIG.pipeline.auto_gain_compensation = False
    os.environ['AUTO_GAIN_COMPENSATION'] = 'false'
    
    # 5. SET EXTREME TARGET PEAK
    CONFIG.pipeline.stem_sum_target_peak = 0.98  # Almost full scale
    os.environ['STEM_SUM_TARGET_PEAK'] = '0.98'
    
    # 6. FORCE EXTREME MAKEUP GAIN SETTINGS
    os.environ['EXTREME_MAKEUP_GAIN'] = 'true'
    os.environ['MAX_MAKEUP_GAIN'] = '5.0'  # Allow 5x gain
    
    print("💥 EXTREME POWER SETTINGS APPLIED!")
    print("=" * 50)
    print("🚫 High-pass filters: DISABLED (5Hz cutoff)")
    print("🥁 Drums gain: 2.5x (kick: 3.0x, snare: 2.5x)")  
    print("🎸 Bass gain: 2.2x")
    print("🎤 Vocals gain: 3.5x (lead: 4.0x)")
    print("🎵 Music gain: 1.5x")
    print("⚙️ Auto-gain compensation: DISABLED")
    print("🎚️ Target peak: 98% (-0.17dB)")
    print("💪 Max makeup gain: 5.0x allowed")
    print("=" * 50)
    print("🔥 This should create MASSIVELY more powerful drums and bass!")
    print("If this works, we know the exact source of the weakness")
    
    return True

def restore_normal_settings():
    """Restore normal settings after testing"""
    print("🔄 Restoring normal stem processing settings...")
    
    # Remove all environment overrides
    env_vars_to_remove = [
        'STEM_GAIN_DRUMS', 'STEM_GAIN_BASS', 'STEM_GAIN_VOCALS', 'STEM_GAIN_MUSIC',
        'STEM_GAIN_KICK', 'STEM_GAIN_SNARE', 'STEM_GAIN_LEADVOCALS',
        'PREP_HPF_HZ', 'AUTO_GAIN_COMPENSATION', 'STEM_SUM_TARGET_PEAK',
        'EXTREME_MAKEUP_GAIN', 'MAX_MAKEUP_GAIN'
    ]
    
    for var in env_vars_to_remove:
        if var in os.environ:
            del os.environ[var]
    
    # Restore config defaults
    CONFIG.audio.prep_hpf_hz = 20.0
    CONFIG.pipeline.auto_gain_compensation = False  # Keep disabled as per previous fix
    CONFIG.pipeline.stem_sum_target_peak = 0.95
    
    print("✅ Normal settings restored")

if __name__ == "__main__":
    apply_extreme_stem_power_boost()