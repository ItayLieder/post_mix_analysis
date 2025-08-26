#!/usr/bin/env python3
"""
Quick test cell for your notebook to verify the orchestrator works.
Add this as a new cell and run it to test.
"""

# Quick test to verify MasteringOrchestrator creates folders
print("🧪 QUICK TEST: Testing MasteringOrchestrator folder creation...")

# Create a simple test file
import os
import numpy as np
import soundfile as sf
from mastering_orchestrator import MasteringOrchestrator, LocalMasterProvider
from data_handler import make_workspace, Manifest

# Create test workspace
test_workspace = make_workspace(project="notebook_test")
test_manifest = Manifest(project="notebook_test", workspace=test_workspace)

# Create simple test audio
duration_s = 3
sr = 44100
t = np.linspace(0, duration_s, int(sr * duration_s))
test_audio = 0.3 * np.sin(2 * np.pi * 440 * t)
test_audio = np.column_stack([test_audio, test_audio])

test_path = os.path.join(test_workspace.inputs, "NotebookTest.wav")
sf.write(test_path, test_audio, sr, subtype="PCM_24")

# Test the orchestrator
orchestrator = MasteringOrchestrator(test_workspace, test_manifest)
provider = LocalMasterProvider(bit_depth="PCM_24")

print("🎭 Running orchestrator test...")
results = orchestrator.run(
    premaster_path=test_path,
    providers=[provider],
    out_tag="test_masters"
)

# Check results
test_folder = os.path.join(test_workspace.outputs, "test_masters", "NotebookTest")
if os.path.exists(test_folder):
    files = [f for f in os.listdir(test_folder) if f.endswith('.wav')]
    print(f"✅ SUCCESS! Created folder: NotebookTest/ with {len(files)} files")
    for f in sorted(files):
        print(f"   • {f}")
    
    if len(files) == 8:
        print("🎉 PERFECT! 8 files created as expected!")
        print("🎯 Your notebook should use MasteringOrchestrator.run() like this test!")
    else:
        print(f"⚠️  Expected 8 files, got {len(files)}")
else:
    print("❌ FAIL: No folder created!")

print(f"\nTest workspace: {test_workspace.root}")