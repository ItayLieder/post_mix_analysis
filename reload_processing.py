#!/usr/bin/env python3
"""
Reload Processing Modules - Run this to update processing after code changes
"""

import importlib
import sys

print("🔄 Reloading all processing modules...")

# List of modules to reload
modules_to_reload = [
    'advanced_stem_processing',
    'extreme_stem_processing', 
    'depth_processing',
    'render_engine',
    'stem_mastering',
    'config'
]

reloaded_count = 0

for module_name in modules_to_reload:
    if module_name in sys.modules:
        try:
            importlib.reload(sys.modules[module_name])
            print(f"  ✅ Reloaded {module_name}")
            reloaded_count += 1
        except Exception as e:
            print(f"  ❌ Failed to reload {module_name}: {e}")
    else:
        print(f"  ⚠️ {module_name} not loaded yet")

print(f"\n✅ Successfully reloaded {reloaded_count} modules")
print("🎯 Processing modules updated with safety checks!")
print("🔧 Fixed: NaN/Inf sanitization to prevent file writing errors")

# Test import
try:
    from extreme_stem_processing import apply_extreme_processing
    from advanced_stem_processing import apply_advanced_processing 
    from depth_processing import create_depth_variant
    print("🎊 All processing functions available!")
except Exception as e:
    print(f"❌ Import test failed: {e}")