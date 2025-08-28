# 🔍 DIAGNOSE ALL SOURCE AUDIO LEVELS

print("🔍 DIAGNOSING ALL SOURCE AUDIO LEVELS")
print("=" * 60)

if 'session' in locals():
    import numpy as np
    
    print("📊 Analyzing all loaded audio files...")
    print("\nFormat: Channel → Raw Peak | Raw RMS | Quality")
    print("-" * 60)
    
    # Categories to organize results
    categories = {}
    
    for channel_id, strip in session.channel_strips.items():
        category = channel_id.split('.')[0]
        track_name = channel_id.split('.')[1]
        
        if category not in categories:
            categories[category] = []
        
        # Analyze audio
        if strip.audio is not None and len(strip.audio) > 0:
            peak = np.max(np.abs(strip.audio))
            rms = np.sqrt(np.mean(strip.audio**2))
            
            # Quality assessment
            if peak > 0.7:
                quality = "✅ GOOD"
            elif peak > 0.3:
                quality = "⚠️ QUIET"
            elif peak > 0.1:
                quality = "❌ VERY QUIET"
            else:
                quality = "🔇 NEARLY SILENT"
            
            categories[category].append({
                'name': track_name,
                'channel_id': channel_id,
                'peak': peak,
                'rms': rms,
                'quality': quality,
                'gain': strip.gain
            })
        else:
            categories[category].append({
                'name': track_name,
                'channel_id': channel_id,
                'peak': 0,
                'rms': 0,
                'quality': "💀 NO AUDIO",
                'gain': strip.gain
            })
    
    # Display results by category
    problem_files = []
    good_files = []
    
    for category, tracks in categories.items():
        print(f"\n📁 {category.upper()}:")
        
        for track in tracks:
            peak_str = f"{track['peak']:.3f}"
            rms_str = f"{track['rms']:.3f}"
            gain_str = f"(gain:{track['gain']:.2f})"
            
            print(f"  {track['name']:20} → {peak_str:>6} | {rms_str:>6} | {track['quality']} {gain_str}")
            
            # Categorize problems
            if track['peak'] < 0.3:
                problem_files.append({
                    'channel': track['channel_id'],
                    'peak': track['peak'],
                    'issue': 'TOO QUIET' if track['peak'] > 0.05 else 'NEARLY SILENT'
                })
            else:
                good_files.append(track['channel_id'])
    
    # Summary
    print("\n" + "="*60)
    print("📊 SUMMARY:")
    print("="*60)
    
    if problem_files:
        print(f"\n❌ PROBLEM FILES ({len(problem_files)} channels):")
        print("These files are too quiet and need fixing:")
        
        for prob in sorted(problem_files, key=lambda x: x['peak']):
            print(f"  • {prob['channel']:30} - {prob['issue']} (peak: {prob['peak']:.3f})")
        
        print(f"\n💡 RECOMMENDATIONS:")
        print(f"  1. Normalize these {len(problem_files)} files to -6dB peak")
        print(f"  2. Or boost them 3-10x in the mixing engine")
        print(f"  3. Check if they're actually empty/corrupted")
    
    if good_files:
        print(f"\n✅ GOOD FILES ({len(good_files)} channels):")
        print("These files have proper levels:")
        for good in good_files[:10]:  # Show first 10
            print(f"  • {good}")
        if len(good_files) > 10:
            print(f"  • ... and {len(good_files) - 10} more")
    
    # Overall assessment
    total_files = len(problem_files) + len(good_files)
    problem_pct = (len(problem_files) / total_files * 100) if total_files > 0 else 0
    
    print(f"\n🎯 OVERALL ASSESSMENT:")
    if problem_pct > 50:
        print(f"  🚨 MAJOR ISSUE: {problem_pct:.0f}% of files are too quiet")
        print(f"     Your source material needs significant level correction!")
    elif problem_pct > 25:
        print(f"  ⚠️ MODERATE ISSUE: {problem_pct:.0f}% of files are too quiet")
        print(f"     Some source files need level correction")
    else:
        print(f"  ✅ GOOD: Only {problem_pct:.0f}% of files have level issues")

else:
    print("❌ No session found - run the previous cells first")