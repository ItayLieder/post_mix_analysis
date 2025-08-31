"""
Debug why stem gains aren't being applied correctly
"""
import os
import sys
sys.path.append('/Users/itay/Documents/GitHub/post_mix_analysis')

from config import CONFIG

print("🔍 DEBUG: STEM GAINS INVESTIGATION")
print("=" * 50)

print("1. CONFIG.pipeline.stem_gains (direct access):")
for stem, gain in CONFIG.pipeline.stem_gains.items():
    print(f"   {stem:12s}: {gain}")

print(f"\n2. CONFIG.pipeline.get_stem_gains() (method call):")
gains = CONFIG.pipeline.get_stem_gains()
for stem, gain in gains.items():
    print(f"   {stem:12s}: {gain}")

print(f"\n3. Environment variables check:")
env_vars = ['STEM_GAIN_DRUMS', 'STEM_GAIN_BASS', 'STEM_GAIN_VOCALS', 'STEM_GAIN_MUSIC']
for var in env_vars:
    value = os.getenv(var, 'Not set')
    print(f"   {var}: {value}")

print(f"\n4. Auto-gain compensation status:")
print(f"   Enabled: {CONFIG.pipeline.auto_gain_compensation}")

print(f"\n5. Expected behavior for main stems:")
print(f"   drums should be: 2.5x")
print(f"   bass should be: 2.2x") 
print(f"   vocals should be: 3.5x")
print(f"   music should be: 1.5x")

print(f"\n🎯 ISSUE ANALYSIS:")
if all(gain > 1.0 for stem, gain in gains.items() if stem in ['drums', 'bass', 'vocals', 'music']):
    print("✅ CONFIG has powerful gains correctly")
else:
    print("❌ CONFIG gains are not powerful")
    
print(f"\nThe issue must be in the pipeline processing or summing logic.")

# Test what the render engine would see
print(f"\n6. Testing render engine CONFIG import:")
try:
    from render_engine import StemRenderEngine
    print("✅ Render engine imports CONFIG correctly")
    
    # Create a fake stem set to test
    class FakeStemSet:
        def get_active_stems(self):
            return ['drums', 'bass', 'vocals', 'music']
    
    fake_stem_set = FakeStemSet()
    
    # Test what the render engine would see
    print("7. Testing what render engine sees:")
    test_gains = CONFIG.pipeline.get_stem_gains()
    print("   Render engine would see these gains:")
    for stem in ['drums', 'bass', 'vocals', 'music']:
        print(f"     {stem}: {test_gains.get(stem, 'NOT FOUND')}")
        
except ImportError as e:
    print(f"❌ Issue importing render engine: {e}")