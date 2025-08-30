#!/usr/bin/env python3
"""
Final test: Each master should be a FOLDER with 8 files inside.
"""

import os
import numpy as np
import soundfile as sf
from mastering_orchestrator import MasteringOrchestrator, LocalMasterProvider
from data_handler import make_workspace, Manifest

def test_final_folder_structure():
    """Test that each master is a folder with 8 files."""
    
    print("🎯 FINAL TEST: Master folders with 8 files each")
    print("=" * 60)
    
    # Create workspace
    workspace = make_workspace(project="test_final_structure")
    manifest = Manifest(project="test_final_structure", workspace=workspace)
    
    # Create one test file
    duration_s = 5
    sr = 44100
    t = np.linspace(0, duration_s, int(sr * duration_s))
    audio = 0.3 * np.sin(2 * np.pi * 440 * t)
    audio = np.column_stack([audio, audio])  # Make stereo
    
    test_path = os.path.join(workspace.inputs, "TestMix.wav")
    sf.write(test_path, audio, sr, subtype="PCM_24")
    print(f"✅ Created: TestMix.wav")
    
    # Create orchestrator and process
    orchestrator = MasteringOrchestrator(workspace, manifest)
    provider = LocalMasterProvider(bit_depth="PCM_24")
    
    print(f"\n🎭 Processing with MasteringOrchestrator...")
    results = orchestrator.run(
        premaster_path=test_path,
        providers=[provider],
        out_tag="masters"
    )
    
    # Check the structure
    master_folder = os.path.join(workspace.outputs, "masters", "TestMix")
    print(f"\n📁 Checking: {master_folder}")
    
    if not os.path.exists(master_folder):
        print("❌ FAIL: Master folder not created!")
        return False
        
    files = sorted(os.listdir(master_folder))
    print(f"📂 Found {len(files)} files in TestMix/ folder:")
    
    for f in files:
        size_kb = os.path.getsize(os.path.join(master_folder, f)) / 1024
        print(f"   • {f} ({size_kb:.1f} KB)")
    
    # Expected files
    expected = [
        "loud.wav", "loud_-14LUFS.wav",
        "neutral.wav", "neutral_-14LUFS.wav", 
        "warm.wav", "warm_-14LUFS.wav",
        "bright.wav", "bright_-14LUFS.wav"
    ]
    
    print(f"\n🔍 Checking for expected 8 files:")
    all_good = True
    for exp in expected:
        if exp in files:
            print(f"   ✅ {exp}")
        else:
            print(f"   ❌ {exp} - MISSING!")
            all_good = False
    
    if all_good and len(files) == 8:
        print(f"\n🎉 SUCCESS! TestMix/ folder contains exactly 8 files!")
        print(f"✅ This is the correct structure you wanted.")
        return True
    else:
        print(f"\n❌ FAIL: Expected 8 files, found {len(files)}")
        return False

if __name__ == "__main__":
    success = test_final_folder_structure()
    if success:
        print(f"\n🎊 Perfect! The mastering orchestrator now creates:")
        print(f"   masters/TestMix/ (folder) containing 8 audio files")
    else:
        print(f"\n💥 Still not working correctly.")