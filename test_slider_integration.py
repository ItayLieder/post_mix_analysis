# 🔍 TEST SLIDER INTEGRATION - WHY SLIDERS NOT WORKING

print("🔍 TESTING SLIDER INTEGRATION")
print("=" * 50)

# Test 1: Check if GUI returns any values at all
try:
    from working_balance_control import WorkingBalanceGUI
    
    # Create test channels
    test_channels = {
        'drums': {'kick': 'test1', 'snare': 'test2'},
        'vocals': {'lead': 'test3'}
    }
    
    # Create GUI object (don't show window)
    gui = WorkingBalanceGUI(test_channels)
    
    # Manually set some slider values (simulate user input)
    gui.balance_values['drums.kick'] = 3.0
    gui.balance_values['drums.snare'] = 2.5
    gui.balance_values['vocals.lead'] = 0.5
    
    # Test get_balance_dict method
    result = gui.get_balance_dict()
    
    print("✅ GUI object creation: SUCCESS")
    print("✅ Manual value setting: SUCCESS")
    print(f"📊 GUI result structure: {result}")
    
    if "channel_overrides" in result:
        overrides = result["channel_overrides"]
        print(f"✅ Channel overrides found: {len(overrides)} channels")
        for ch, val in overrides.items():
            print(f"   • {ch}: {val}")
    else:
        print("❌ No channel_overrides in result - GUI broken!")
    
except Exception as e:
    print(f"❌ GUI test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*50)

# Test 2: Check what the notebook GUI actually returned
print("🔍 CHECKING NOTEBOOK GUI RESULTS")

if 'balance_settings' in globals():
    print("✅ balance_settings variable exists")
    print(f"📊 Content: {balance_settings}")
    
    if isinstance(balance_settings, dict):
        if "channel_overrides" in balance_settings:
            overrides = balance_settings["channel_overrides"]
            print(f"✅ Found {len(overrides)} channel overrides")
            for ch, val in overrides.items():
                print(f"   • {ch}: {val}")
        else:
            print("❌ No channel_overrides key in balance_settings")
            print(f"   Available keys: {list(balance_settings.keys())}")
    else:
        print(f"❌ balance_settings is not a dict: {type(balance_settings)}")
else:
    print("❌ balance_settings variable not found")

if 'channel_overrides' in globals():
    print(f"✅ channel_overrides variable exists: {channel_overrides}")
else:
    print("❌ channel_overrides variable not found")

print("\n" + "="*50)
print("🎯 DIAGNOSIS:")

# The problem is likely:
print("Most likely issues:")
print("1. GUI window closed without clicking 'Apply & Close'")
print("2. GUI returned None or empty dict")
print("3. Notebook variables not properly set")
print("4. GUI integration broken in working_balance_control.py")

print("\n💡 SIMPLE FIX:")
print("Replace GUI with direct variable assignment:")
print()
print("# MANUAL SLIDER VALUES - REPLACE GUI CELL WITH THIS:")
print("channel_overrides = {")
print("    'drums.kick': 3.0,     # 300% boost")
print("    'drums.snare': 3.0,    # 300% boost") 
print("    'drums.hihat': 2.5,    # 250% boost")
print("    'drums.tom': 2.5,      # 250% boost")
print("    'drums.cymbal': 2.0,   # 200% boost")
print("    'vocals.lead_vocal1': 0.5,  # 50% reduction")
print("    'vocals.lead_vocal2': 0.5,  # 50% reduction")
print("}")
print()
print("print(f'✅ Manual overrides set: {len(channel_overrides)} channels')")
print("for ch, val in channel_overrides.items():")
print("    print(f'  • {ch}: {val}')")

print("\nThis bypasses the broken GUI completely!")