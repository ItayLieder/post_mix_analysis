#!/usr/bin/env python3
"""
Load Mixed Stems - Helper script to load the latest mixed stems into post-mix pipeline
"""

import os
import glob
from pathlib import Path

def get_latest_mixed_stems():
    """Find the latest mixed stems from mixing sessions"""
    
    # Find all mixing session directories
    mixing_sessions_dir = "/Users/itay/Documents/post_mix_data/mixing_sessions"
    
    if not os.path.exists(mixing_sessions_dir):
        print(f"❌ Mixing sessions directory not found: {mixing_sessions_dir}")
        return None
    
    # Get all session directories, sorted by creation time (newest first)
    session_dirs = []
    for item in os.listdir(mixing_sessions_dir):
        session_path = os.path.join(mixing_sessions_dir, item)
        if os.path.isdir(session_path) and item.startswith("session_"):
            session_dirs.append(session_path)
    
    # Sort by modification time, newest first
    session_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    if not session_dirs:
        print("❌ No mixing sessions found")
        return None
    
    # Check the newest session for stems
    latest_session = session_dirs[0]
    stems_dir = os.path.join(latest_session, "stems")
    
    print(f"🔍 Checking latest session: {os.path.basename(latest_session)}")
    
    if not os.path.exists(stems_dir):
        print(f"❌ No stems directory found in: {stems_dir}")
        return None
    
    # Look for the 4 required stems
    required_stems = ["drums.wav", "bass.wav", "vocals.wav", "music.wav"]
    stem_paths = {}
    
    for stem_name in required_stems:
        stem_path = os.path.join(stems_dir, stem_name)
        if os.path.exists(stem_path):
            stem_size_mb = os.path.getsize(stem_path) / (1024 * 1024)
            stem_paths[stem_name.replace('.wav', '')] = stem_path
            print(f"  ✅ Found {stem_name}: {stem_size_mb:.1f} MB")
        else:
            print(f"  ❌ Missing {stem_name}")
            return None
    
    return stem_paths

def setup_for_postmix():
    """Set up stems for post-mix processing"""
    
    stem_paths = get_latest_mixed_stems()
    
    if not stem_paths:
        print("\n❌ Could not find valid mixed stems")
        print("\n📝 To fix this:")
        print("  1. Make sure you've run the mixing notebook successfully")
        print("  2. Check that stems were exported (look for 'Stems exported' message)")
        print("  3. Verify stem files exist in the mixing session directory")
        return None
    
    print(f"\n✅ Found {len(stem_paths)} stems ready for post-mix processing")
    
    # Format for post-mix system (match expected variable names)
    formatted_paths = {
        'drums_path': stem_paths.get('drums'),
        'bass_path': stem_paths.get('bass'),
        'vocals_path': stem_paths.get('vocals'),
        'music_path': stem_paths.get('music'),
    }
    
    print("\n📁 Formatted paths for post-mix:")
    for name, path in formatted_paths.items():
        if path:
            print(f"  {name}: {os.path.basename(path)}")
        else:
            print(f"  {name}: ❌ Not found")
    
    return formatted_paths

if __name__ == "__main__":
    stem_paths = setup_for_postmix()
    
    if stem_paths:
        print(f"\n🎯 Ready! Use these paths in your post-mix notebook:")
        print(f"stem_paths = {stem_paths}")
    else:
        print(f"\n❌ Setup failed - no stems available")