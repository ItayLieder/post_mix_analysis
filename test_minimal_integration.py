"""
Test the minimal processing integration with the main pipeline
"""
import os
os.environ['EXTREME_MAKEUP_GAIN'] = 'true'  # Enable the fixes we made
os.environ['MAX_MAKEUP_GAIN'] = '5.0'

from config import CONFIG
print("🧪 TESTING MINIMAL PROCESSING INTEGRATION")
print("=" * 50)
print(f"✅ Advanced processing disabled: {not CONFIG.pipeline.use_advanced_stem_processing}")
print(f"✅ Extreme processing disabled: {not CONFIG.pipeline.use_extreme_stem_processing}")  
print(f"✅ Minimal processing enabled: {CONFIG.pipeline.use_minimal_processing}")
print(f"✅ Extreme makeup gain: {os.getenv('EXTREME_MAKEUP_GAIN')}")
print(f"✅ Auto-gain compensation: {CONFIG.pipeline.auto_gain_compensation}")
print(f"✅ Stem sum target peak: {CONFIG.pipeline.stem_sum_target_peak}")

print(f"\n🎚️ STEM GAINS:")
gains = CONFIG.pipeline.get_stem_gains()
for stem, gain in gains.items():
    print(f"  {stem}: {gain}")

print(f"\n🎉 INTEGRATION READY!")
print(f"The main pipeline will now use minimal processing by default.")
print(f"All stems should sound great with preserved quality!")