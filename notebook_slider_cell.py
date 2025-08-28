# 🎚️ ADD THIS CELL TO YOUR NOTEBOOK - WORKING SLIDERS

from fixed_balance_gui import create_balance_gui

# Create the working slider GUI
if 'channels' in locals() and channels:
    print("🎚️ Creating Working Balance Sliders...")
    balance_gui = create_balance_gui(channels)
    
    print("\n📋 Instructions:")
    print("1. Use GROUP sliders to adjust entire categories (drums, vocals, etc)")
    print("2. Use INDIVIDUAL sliders for precise control of specific tracks")  
    print("3. Click '✅ Apply Balance' when you're happy with the settings")
    print("4. Then run the mixing cells (cell-12, cell-14)")
    
else:
    print("❌ No channels loaded!")
    print("📝 Please run the channel loading cell first")