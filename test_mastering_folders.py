#!/usr/bin/env python3
"""
Test script to verify the mastering folder structure changes.
Creates a dummy audio file and runs the mastering process to check output structure.
"""

import os
import numpy as np
import soundfile as sf
import tempfile
from mastering_orchestrator import MasteringOrchestrator, LocalMasterProvider
from data_handler import make_workspace, Manifest

def test_mastering_folder_structure():
    """Test that mastering creates folders with 8 files (4 styles × 2 versions)."""
    
    print("🧪 Testing mastering folder structure (8 files per master)...")
    
    # Create temporary workspace
    workspace = make_workspace(project="test_mastering_folders")
    manifest = Manifest(project="test_mastering_folders", workspace=workspace)
    
    # Create a simple test audio file
    duration_s = 5
    sr = 44100
    t = np.linspace(0, duration_s, int(sr * duration_s))
    # Simple sine wave
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    audio = np.column_stack([audio, audio])  # Make stereo
    
    # Save test audio
    test_file = os.path.join(workspace.inputs, "test_audio.wav")
    sf.write(test_file, audio, sr, subtype="PCM_24")
    print(f"✅ Created test audio file: {test_file}")
    
    # Create orchestrator and provider
    orchestrator = MasteringOrchestrator(workspace, manifest)
    provider = LocalMasterProvider(bit_depth="PCM_24")
    
    # Test 1: Single file mastering with default 4 styles (should create 8 files)
    print("\n📁 Test 1: Single file mastering (expecting 8 files)...")
    results = orchestrator.run(
        premaster_path=test_file,
        providers=[provider],
        styles=None,  # Will use default 4 styles
        out_tag="test_masters"
    )
    
    # Check results - expecting 8 files total
    expected_styles = ["loud", "neutral", "warm", "bright"]
    output_dir = os.path.join(workspace.outputs, "test_masters", "test_audio")
    
    print(f"\n📂 Checking output directory: {output_dir}")
    if os.path.exists(output_dir):
        files = os.listdir(output_dir)
        print(f"   Found {len(files)} files: {sorted(files)}")
        
        # Check for regular versions
        print("\n   Regular versions:")
        for style in expected_styles:
            if f"{style}.wav" in files:
                print(f"   ✅ {style}.wav found")
            else:
                print(f"   ❌ {style}.wav missing!")
        
        # Check for -14 LUFS versions
        print("\n   -14 LUFS versions:")
        for style in expected_styles:
            if f"{style}_-14LUFS.wav" in files:
                print(f"   ✅ {style}_-14LUFS.wav found")
            else:
                print(f"   ❌ {style}_-14LUFS.wav missing!")
        
        if len(files) == 8:
            print(f"\n   ✅ SUCCESS: Found expected 8 files (4 styles × 2 versions)")
        else:
            print(f"\n   ❌ ERROR: Expected 8 files, but found {len(files)}")
    else:
        print(f"   ❌ Output directory not created!")
    
    # Test 2: Multiple variants (each should have 8 files)
    print("\n📁 Test 2: Multiple variants mastering (expecting 8 files per variant)...")
    
    # Create some dummy variant metadata
    variant1_path = os.path.join(workspace.outputs, "variant1.wav")
    variant2_path = os.path.join(workspace.outputs, "variant2.wav")
    sf.write(variant1_path, audio, sr, subtype="PCM_24")
    sf.write(variant2_path, audio * 0.8, sr, subtype="PCM_24")  # Slightly quieter
    
    variant_metadata = [
        {"out_path": variant1_path},
        {"out_path": variant2_path}
    ]
    
    all_results = orchestrator.run_all_variants(
        variant_metadata=variant_metadata,
        providers=[provider],
        styles=None,  # Will use default 4 styles
        out_tag="test_all_masters"
    )
    
    # Check structure for each variant
    for variant_name in ["variant1", "variant2"]:
        variant_dir = os.path.join(workspace.outputs, "test_all_masters", variant_name)
        print(f"\n📂 Checking variant directory: {variant_dir}")
        
        if os.path.exists(variant_dir):
            files = os.listdir(variant_dir)
            print(f"   Found {len(files)} files: {sorted(files)}")
            
            # Check for regular versions
            print("\n   Regular versions:")
            for style in expected_styles:
                if f"{style}.wav" in files:
                    print(f"   ✅ {style}.wav found")
                else:
                    print(f"   ❌ {style}.wav missing!")
            
            # Check for -14 LUFS versions
            print("\n   -14 LUFS versions:")
            for style in expected_styles:
                if f"{style}_-14LUFS.wav" in files:
                    print(f"   ✅ {style}_-14LUFS.wav found")
                else:
                    print(f"   ❌ {style}_-14LUFS.wav missing!")
            
            if len(files) == 8:
                print(f"\n   ✅ SUCCESS: Found expected 8 files for {variant_name}")
            else:
                print(f"\n   ❌ ERROR: Expected 8 files for {variant_name}, but found {len(files)}")
        else:
            print(f"   ❌ Variant directory not created!")
    
    print("\n✅ Test complete!")
    print(f"📁 Results saved in: {workspace.root}")
    
    return workspace.root

if __name__ == "__main__":
    result_path = test_mastering_folder_structure()
    print(f"\n🎉 Test finished! Check results at:\n   {result_path}")