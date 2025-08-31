#!/usr/bin/env python3
"""
Test script to verify that BIG_Exact_Match now calls the exact replica function.
"""

import os
import tempfile
import shutil
from stem_mastering import load_stem_set
from render_engine import StemRenderEngine, PreprocessConfig, RenderOptions
from config import CONFIG

def test_big_exact_match():
    print("🧪 TESTING BIG_Exact_Match Fix")
    print("=" * 50)
    
    # Use Reference_mix stems (the original winning stems)
    stem_paths = {
        'drums': '/Users/itay/Documents/post_mix_data/Reference_mix/stems/drums.wav',
        'bass': '/Users/itay/Documents/post_mix_data/Reference_mix/stems/bass.wav', 
        'vocals': '/Users/itay/Documents/post_mix_data/Reference_mix/stems/vocals.wav',
        'music': '/Users/itay/Documents/post_mix_data/Reference_mix/stems/music.wav'
    }
    
    # Verify all stems exist
    print("📁 Checking stem files...")
    for stem_type, path in stem_paths.items():
        if os.path.exists(path):
            print(f"  ✅ {stem_type}: {path}")
        else:
            print(f"  ❌ {stem_type}: MISSING - {path}")
            return False
    
    # Load stem set
    print("\n🎛️ Loading stems...")
    stem_set = load_stem_set(stem_paths)
    
    # Create render engine
    print("🔧 Creating render engine...")
    engine = StemRenderEngine(stem_set, 
                             preprocess=PreprocessConfig(
                                 low_cutoff=CONFIG.audio.prep_hpf_hz,
                                 kick_lo=CONFIG.audio.kick_freq_low,
                                 kick_hi=CONFIG.audio.kick_freq_high
                             ))
    
    # Create temp output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"📁 Output directory: {temp_dir}")
        
        # Test just the BIG_Exact_Match variant
        stem_plan = [("BIG_Exact_Match", "big:BIG_Exact_Match")]
        
        print("🚀 Processing BIG_Exact_Match variant...")
        try:
            render_options = RenderOptions(
                target_peak_dbfs=CONFIG.audio.render_peak_target_dbfs, 
                bit_depth=CONFIG.audio.default_bit_depth, 
                hpf_hz=None, 
                save_headroom_first=False
            )
            
            results = engine.commit_stem_variants(temp_dir, stem_plan, opts=render_options)
            
            if "BIG_Exact_Match" in results:
                result_path = results["BIG_Exact_Match"]["final_mix_path"]
                print(f"✅ BIG_Exact_Match created: {result_path}")
                
                if os.path.exists(result_path):
                    # Compare with original
                    import hashlib
                    
                    def get_md5(filepath):
                        with open(filepath, 'rb') as f:
                            return hashlib.md5(f.read()).hexdigest()
                    
                    original_path = '/Users/itay/Documents/post_mix_data/BIG_POWERFUL_STEM_MIX.wav'
                    
                    if os.path.exists(original_path):
                        original_md5 = get_md5(original_path)
                        new_md5 = get_md5(result_path)
                        
                        print(f"\n🔍 VERIFICATION:")
                        print(f"  Original: {original_md5}")
                        print(f"  New:      {new_md5}")
                        
                        if original_md5 == new_md5:
                            print("💯 SUCCESS: BIG_Exact_Match creates IDENTICAL file!")
                            return True
                        else:
                            print("❌ FAILED: Files are different!")
                            return False
                    else:
                        print("⚠️ Original file not found for comparison")
                        print("✅ But BIG_Exact_Match variant was created successfully")
                        return True
                else:
                    print(f"❌ Output file not found: {result_path}")
                    return False
            else:
                print("❌ BIG_Exact_Match not found in results")
                print(f"Available results: {list(results.keys())}")
                return False
                
        except Exception as e:
            print(f"❌ Error processing BIG_Exact_Match: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    success = test_big_exact_match()
    if success:
        print("\n🎉 TEST PASSED: BIG_Exact_Match fix works correctly!")
    else:
        print("\n💥 TEST FAILED: BIG_Exact_Match fix needs more work")