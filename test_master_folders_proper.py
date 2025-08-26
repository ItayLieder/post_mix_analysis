#!/usr/bin/env python3
"""
Test the proper folder structure for mastering:
Each master should have its own folder with 8 files (4 styles × 2 versions).
"""

import os
import sys
import numpy as np
import soundfile as sf
from mastering_orchestrator import MasteringOrchestrator, LocalMasterProvider
from data_handler import make_workspace, Manifest

def test_proper_mastering_structure():
    """Test that mastering creates proper folder structure with 8 files each."""
    
    print("🧪 Testing proper mastering folder structure...")
    print("=" * 60)
    
    # Create workspace
    workspace = make_workspace(project="test_master_folders_proper")
    manifest = Manifest(project="test_master_folders_proper", workspace=workspace)
    
    print(f"📁 Workspace created: {workspace.root}")
    
    # Create test audio files
    duration_s = 10
    sr = 44100
    t = np.linspace(0, duration_s, int(sr * duration_s))
    
    # Create a richer test signal
    audio = 0.3 * np.sin(2 * np.pi * 440 * t)  # A440
    audio += 0.2 * np.sin(2 * np.pi * 880 * t)  # A880
    audio += 0.1 * np.sin(2 * np.pi * 220 * t)  # A220
    audio = np.column_stack([audio, audio])  # Make stereo
    
    # Save multiple test files to simulate different premasters
    test_files = []
    test_names = ["Mix_V1", "Mix_V2_Bright", "Mix_V3_Punchy"]
    
    for i, name in enumerate(test_names):
        # Vary the audio slightly for each version
        variant_audio = audio * (0.8 + i * 0.1)  # Different levels
        test_path = os.path.join(workspace.inputs, f"{name}.wav")
        sf.write(test_path, variant_audio, sr, subtype="PCM_24")
        test_files.append((name, test_path))
        print(f"✅ Created test file: {name}")
    
    print("\n" + "=" * 60)
    
    # Create orchestrator with local provider
    orchestrator = MasteringOrchestrator(workspace, manifest)
    provider = LocalMasterProvider(bit_depth="PCM_24")
    
    # Test each file to verify folder structure
    for name, test_path in test_files:
        print(f"\n🎭 Processing: {name}")
        print("-" * 40)
        
        # Run mastering with default settings (should create 8 files)
        results = orchestrator.run(
            premaster_path=test_path,
            providers=[provider],
            styles=None,  # Uses default 4 styles
            out_tag="masters",
            level_match_preview_lufs=-14.0  # Creates -14 LUFS versions
        )
        
        # Check the NEW folder structure - 4 style folders, each with 2 files
        masters_base = os.path.join(workspace.outputs, "masters")
        expected_styles = ["loud", "neutral", "warm", "bright"]
        
        print(f"📂 Checking style folders in: {masters_base}")
        
        all_correct = True
        total_files = 0
        
        for style in expected_styles:
            style_folder = os.path.join(masters_base, style)
            print(f"\n   📁 {style}/ folder:")
            
            if not os.path.exists(style_folder):
                print(f"     ❌ Folder missing!")
                all_correct = False
                continue
                
            # List files in this style folder
            files = sorted(os.listdir(style_folder))
            total_files += len(files)
            print(f"     Found {len(files)} files:")
            
            for f in files:
                size_kb = os.path.getsize(os.path.join(style_folder, f)) / 1024
                print(f"       • {f} ({size_kb:.1f} KB)")
            
            # Check for expected files (regular + -14LUFS version)
            expected_regular = f"{name}.wav"
            expected_lufs = f"{name}_-14LUFS.wav"
            
            if expected_regular in files:
                print(f"     ✅ {expected_regular}")
            else:
                print(f"     ❌ {expected_regular} - MISSING!")
                all_correct = False
                
            if expected_lufs in files:
                print(f"     ✅ {expected_lufs}")
            else:
                print(f"     ❌ {expected_lufs} - MISSING!")
                all_correct = False
        
        if all_correct and total_files == 8:  # 4 styles × 2 files each
            print(f"\n   ✅ SUCCESS: Correct structure for {name} (4 style folders, 8 total files)")
        else:
            print(f"\n   ❌ FAIL: Expected 4 style folders with 8 total files, found {total_files} files")
    
    print("\n" + "=" * 60)
    print("📊 FINAL SUMMARY")
    print("=" * 60)
    
    # Check overall structure - should be 4 style folders
    masters_dir = os.path.join(workspace.outputs, "masters")
    if os.path.exists(masters_dir):
        style_folders = [f for f in os.listdir(masters_dir) if os.path.isdir(os.path.join(masters_dir, f))]
        print(f"📁 Style folders created: {len(style_folders)}")
        
        total_files = 0
        for folder in style_folders:
            folder_path = os.path.join(masters_dir, folder)
            file_count = len(os.listdir(folder_path))
            total_files += file_count
            print(f"   • {folder}/  ({file_count} files)")
            
        print(f"\n📊 Structure Summary:")
        print(f"   Style folders: {len(style_folders)} (expected: 4)")
        print(f"   Total audio files: {total_files} (expected: 3 tests × 2 files per style = 6 per folder)")
        
        if len(style_folders) == 4:
            print("   ✅ Correct number of style folders!")
        else:
            print("   ❌ Wrong number of style folders!")
    
    print(f"\n📍 Full results at: {workspace.root}")
    print("✅ Test complete!")
    
    return workspace.root

if __name__ == "__main__":
    try:
        result_path = test_proper_mastering_structure()
        print(f"\n🎉 Success! Check the results at:\n   {result_path}")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)