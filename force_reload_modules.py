#!/usr/bin/env python3
"""
Force reload the mastering orchestrator module to get the latest changes.
Run this cell in your notebook if the mastering orchestrator is still using old code.
"""

import sys
import importlib

# Force reload the mastering orchestrator module
if 'mastering_orchestrator' in sys.modules:
    print("🔄 Reloading mastering_orchestrator module...")
    importlib.reload(sys.modules['mastering_orchestrator'])
    print("✅ Module reloaded!")
else:
    print("ℹ️  mastering_orchestrator not yet loaded")

# Import fresh
from mastering_orchestrator import MasteringOrchestrator, LocalMasterProvider

print("🎯 Testing if the updated code is active...")

# Quick test to see if we have the updated print statement
import inspect
source = inspect.getsource(MasteringOrchestrator.run)
if "Creating mastered folder:" in source:
    print("✅ SUCCESS: Updated code is active!")
    print("   The mastering orchestrator will now create folders (not files)")
else:
    print("❌ FAIL: Still using old code")
    print("   You may need to restart your Jupyter kernel completely")

print("\n📝 To use in your notebook, run:")
print("   1. Restart your Jupyter kernel (Kernel -> Restart)")  
print("   2. Re-run all your import cells")
print("   3. The mastering orchestrator should now create folders")