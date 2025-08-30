#!/usr/bin/env python3
"""
Reload stem mastering modules after updates
"""

import importlib
import sys

# Force reload stem mastering modules
modules_to_reload = [
    'stem_mastering',
    'stem_balance_helper', 
    'render_engine'
]

print("🔄 Reloading stem mastering modules...")

for module_name in modules_to_reload:
    if module_name in sys.modules:
        importlib.reload(sys.modules[module_name])
        print(f"  ✅ Reloaded {module_name}")
    else:
        print(f"  ⚠️ {module_name} not loaded yet")

print("🎯 Modules reloaded! You can now use detailed stems.")

# Test the fixed function
try:
    from stem_mastering import get_stem_variant_for_combination
    
    # Test with detailed stems
    kick_variant = get_stem_variant_for_combination("kick", "punchy")
    guitar_variant = get_stem_variant_for_combination("guitar", "wide") 
    
    print(f"\n✅ Test successful:")
    print(f"  kick (punchy): {kick_variant}")
    print(f"  guitar (wide): {guitar_variant}")
    
except Exception as e:
    print(f"❌ Still has issues: {e}")