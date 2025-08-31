"""
Temporary power boost for weak stem processing.
Use this instead of the normal stem balance to force much louder levels.
"""
import os
from config import CONFIG

def boost_stem_levels():
    """Apply aggressive stem level boosts to fix weak processing"""
    
    # Force environment variables to override config
    os.environ['STEM_GAIN_DRUMS'] = '1.8'    # Much louder
    os.environ['STEM_GAIN_BASS'] = '1.6'     # Much louder  
    os.environ['STEM_GAIN_VOCALS'] = '3.0'   # Even MORE vocals
    os.environ['STEM_GAIN_MUSIC'] = '1.2'    # Louder music
    
    # Also boost detailed stem levels
    os.environ['STEM_GAIN_LEADVOCALS'] = '3.2'  # Lead vocals very loud
    os.environ['STEM_GAIN_KICK'] = '2.0'        # Kick much louder
    os.environ['STEM_GAIN_SNARE'] = '1.8'       # Snare louder
    
    print("🚀 AGGRESSIVE STEM POWER BOOST APPLIED!")
    print("=" * 50)
    print("🎤 Vocals: 3.0 (was 2.0)")
    print("🥁 Drums: 1.8 (was 0.9)")  
    print("🎸 Bass: 1.6 (was 0.8)")
    print("🎵 Music: 1.2 (was 0.55)")
    print("=" * 50)
    print("💪 This should create MUCH more powerful mixes!")
    print("If this works, we'll know the issue is in the gain staging")
    
    return True

if __name__ == "__main__":
    boost_stem_levels()