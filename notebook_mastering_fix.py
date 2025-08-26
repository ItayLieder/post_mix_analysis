#!/usr/bin/env python3
"""
CORRECTED mastering code for your notebook.
Replace the mastering section in cell 6 with this code.
"""

# ============================================
# MASTERING STEP - USING ORCHESTRATOR (CORRECTED)
# ============================================

try:
    print("\n🎭 Running mastering orchestration with PROPER folder structure...")
    
    # Create mastering orchestrator
    orchestrator = MasteringOrchestrator(workspace_paths, manifest)
    provider = LocalMasterProvider(bit_depth="PCM_24")
    
    # Process all premasters using the orchestrator
    master_results = []
    all_master_paths = []
    
    print(f"📂 Found {len(variant_metadata)} premaster files to master")
    
    for variant_meta in variant_metadata:
        premaster_path = variant_meta["out_path"]
        
        if not os.path.exists(premaster_path):
            print(f"⚠️ Skipping missing file: {premaster_path}")
            continue
            
        print(f"\n🎭 Mastering: {os.path.basename(premaster_path)}")
        
        # Use the orchestrator to create proper folder structure
        results = orchestrator.run(
            premaster_path=premaster_path,
            providers=[provider],
            styles=None,  # Uses default 4 styles
            out_tag="masters",
            level_match_preview_lufs=-14.0
        )
        
        # Collect all result paths
        for result in results:
            master_results.append(result)
            all_master_paths.append(result.out_path)
            
            # Also add the -14 LUFS versions (they're created automatically)
            base_dir = os.path.dirname(result.out_path)
            lufs_file = os.path.join(base_dir, f"{result.style}_-14LUFS.wav")
            if os.path.exists(lufs_file):
                all_master_paths.append(lufs_file)
    
    print(f"\n✅ Mastering completed successfully!")
    print(f"📊 Results:")
    print(f"   🎯 Master folders: {len(variant_metadata)}")
    print(f"   📁 Total audio files: {len(all_master_paths)}")
    
    # Show folder structure
    masters_dir = os.path.join(workspace_paths.outputs, "masters")
    if os.path.exists(masters_dir):
        folders = [f for f in os.listdir(masters_dir) if os.path.isdir(os.path.join(masters_dir, f))]
        print(f"\n📁 Master folders created:")
        for folder in sorted(folders):
            folder_path = os.path.join(masters_dir, folder)
            file_count = len([f for f in os.listdir(folder_path) if f.endswith('.wav')])
            print(f"   • {folder}/ ({file_count} files)")
    
    # Update master_paths for later use
    master_paths = all_master_paths

except Exception as e:
    print(f"❌ Mastering orchestration failed: {e}")
    raise

# The rest of your notebook should work the same way...
print(f"\n🎊 MASTERS ARE NOW FOLDERS!")
print(f"Each master is a folder containing 8 audio files (4 styles × 2 versions)")