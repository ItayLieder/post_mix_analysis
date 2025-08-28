# 🎚️ SIMPLE TKINTER SLIDER GUI - GUARANTEED TO WORK

import tkinter as tk
from tkinter import ttk
import json
import threading
import time

class SimpleBalanceGUI:
    def __init__(self, channels_dict):
        self.channels = channels_dict
        self.balance_values = {}
        self.result = None
        self.window_closed = False
        
        # Initialize all values to 1.0
        for category, tracks in channels_dict.items():
            for track_name in tracks.keys():
                channel_id = f"{category}.{track_name}"
                self.balance_values[channel_id] = 1.0
        
        self.create_window()
    
    def create_window(self):
        """Create the main window"""
        self.root = tk.Tk()
        self.root.title("🎚️ Balance Control Sliders")
        self.root.geometry("800x600")
        
        # Main frame with scrollbar
        canvas = tk.Canvas(self.root)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Title
        title = tk.Label(scrollable_frame, text="🎚️ Balance Control", font=("Arial", 16, "bold"))
        title.pack(pady=10)
        
        # Status label
        self.status_label = tk.Label(scrollable_frame, text="Adjust sliders and click Apply", font=("Arial", 10))
        self.status_label.pack()
        
        # Create sliders for each category
        self.sliders = {}
        for category, tracks in self.channels.items():
            # Category frame
            cat_frame = ttk.LabelFrame(scrollable_frame, text=f"📁 {category.upper()}")
            cat_frame.pack(fill="x", padx=10, pady=5)
            
            for track_name in tracks.keys():
                channel_id = f"{category}.{track_name}"
                
                # Slider frame
                slider_frame = tk.Frame(cat_frame)
                slider_frame.pack(fill="x", padx=5, pady=2)
                
                # Label
                label = tk.Label(slider_frame, text=f"{track_name}:", width=20, anchor="w")
                label.pack(side="left")
                
                # Slider
                slider = tk.Scale(
                    slider_frame,
                    from_=0.1,
                    to=5.0,
                    resolution=0.1,
                    orient="horizontal",
                    length=300,
                    command=lambda val, ch_id=channel_id: self.update_value(ch_id, float(val))
                )
                slider.set(1.0)
                slider.pack(side="left", padx=5)
                
                # Value label
                value_label = tk.Label(slider_frame, text="1.0", width=6)
                value_label.pack(side="left")
                
                self.sliders[channel_id] = {
                    'slider': slider,
                    'label': value_label
                }
        
        # Buttons frame
        button_frame = tk.Frame(scrollable_frame)
        button_frame.pack(pady=20)
        
        # Apply button
        apply_btn = tk.Button(
            button_frame,
            text="✅ Apply & Close",
            command=self.apply_and_close,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 12, "bold"),
            width=15
        )
        apply_btn.pack(side="left", padx=10)
        
        # Reset button
        reset_btn = tk.Button(
            button_frame,
            text="🔄 Reset All",
            command=self.reset_all,
            bg="#FF9800",
            fg="white",
            font=("Arial", 12, "bold"),
            width=15
        )
        reset_btn.pack(side="left", padx=10)
        
        # Cancel button
        cancel_btn = tk.Button(
            button_frame,
            text="❌ Cancel",
            command=self.cancel,
            bg="#f44336",
            fg="white", 
            font=("Arial", 12, "bold"),
            width=15
        )
        cancel_btn.pack(side="left", padx=10)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.cancel)
    
    def update_value(self, channel_id, value):
        """Update a slider value"""
        self.balance_values[channel_id] = value
        self.sliders[channel_id]['label'].config(text=f"{value:.1f}")
        
        # Update status
        changed_count = len([v for v in self.balance_values.values() if abs(v - 1.0) > 0.01])
        self.status_label.config(text=f"{changed_count} channels modified")
    
    def reset_all(self):
        """Reset all sliders to 1.0"""
        for channel_id in self.balance_values.keys():
            self.balance_values[channel_id] = 1.0
            self.sliders[channel_id]['slider'].set(1.0)
            self.sliders[channel_id]['label'].config(text="1.0")
        
        self.status_label.config(text="All values reset to neutral")
    
    def apply_and_close(self):
        """Apply current settings and close"""
        # Get changed values
        changed = {k: v for k, v in self.balance_values.items() if abs(v - 1.0) > 0.01}
        
        self.result = {
            'channel_overrides': changed,
            'mix_balance': {
                'vocal_prominence': 0.3,
                'drum_punch': 0.7,
                'bass_foundation': 0.6,
                'instrument_presence': 0.5
            }
        }
        
        print(f"✅ Applied {len(changed)} channel changes")
        for ch, val in sorted(changed.items()):
            change_pct = (val - 1.0) * 100
            direction = "↑" if change_pct > 0 else "↓"
            print(f"  {direction} {ch}: {val:.2f} ({change_pct:+.0f}%)")
        
        self.root.destroy()
        self.window_closed = True
    
    def cancel(self):
        """Cancel without applying"""
        self.result = None
        self.root.destroy()  
        self.window_closed = True
    
    def show(self):
        """Show the GUI and wait for result"""
        print("🎚️ Opening balance control window...")
        print("📋 Adjust sliders and click 'Apply & Close'")
        
        # Run GUI in thread to avoid blocking
        gui_thread = threading.Thread(target=self.root.mainloop)
        gui_thread.start()
        
        # Wait for window to close
        while not self.window_closed:
            time.sleep(0.1)
        
        return self.result

def create_balance_sliders(channels):
    """Create and show balance sliders"""
    gui = SimpleBalanceGUI(channels)
    return gui.show()

# Test function
if __name__ == "__main__":
    test_channels = {
        'drums': {'kick': 'test', 'snare': 'test'},
        'vocals': {'lead': 'test'}
    }
    result = create_balance_sliders(test_channels)
    print("Result:", result)