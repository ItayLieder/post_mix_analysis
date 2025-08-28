#!/usr/bin/env python3
"""
Working Balance Control - Real-time interactive balance that actually works
Uses tkinter GUI which works everywhere including PyCharm
"""

import tkinter as tk
from tkinter import ttk, messagebox
import json
from typing import Dict, Any

class WorkingBalanceGUI:
    """Working GUI using tkinter - works in PyCharm, Jupyter, everywhere"""
    
    def __init__(self, channels: Dict):
        self.channels = channels
        self.balance_values = {}
        self.sliders = {}
        self.groups = {}
        
        # Initialize all channels to neutral (1.0)
        for category, tracks in channels.items():
            for track_name in tracks.keys():
                channel_id = f"{category}.{track_name}"
                self.balance_values[channel_id] = 1.0
        
        # Create smart groups
        self._create_groups()
        
        # Create the GUI
        self.root = tk.Tk()
        self.root.title("🎚️ Balance Control")
        self.root.geometry("800x600")
        self._create_interface()
    
    def _create_groups(self):
        """Create channel groups for batch control"""
        all_channels = list(self.balance_values.keys())
        
        # Category groups
        for category in self.channels.keys():
            self.groups[category] = [ch for ch in all_channels if ch.startswith(f"{category}.")]
        
        # Smart groups
        self.groups["All Bass"] = [ch for ch in all_channels if 'bass' in ch.lower()]
        self.groups["All Drums"] = [ch for ch in all_channels if any(word in ch.lower() for word in ['drum', 'kick', 'snare', 'hat', 'crash', 'tom'])]
        self.groups["All Vocals"] = [ch for ch in all_channels if any(word in ch.lower() for word in ['vocal', 'vox', 'harmony', 'chorus'])]
        self.groups["All Synths"] = [ch for ch in all_channels if 'synt' in ch.lower()]
    
    def _create_interface(self):
        """Create the tkinter interface"""
        
        # Main container with scrollbar
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Canvas for scrolling
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Header
        header = ttk.Label(scrollable_frame, text="🎚️ Balance Control", font=('Arial', 16, 'bold'))
        header.pack(pady=(0, 20))
        
        # Group Controls Section
        self._create_group_section(scrollable_frame)
        
        # Individual Channel Controls
        self._create_channel_section(scrollable_frame)
        
        # Control buttons
        self._create_buttons(scrollable_frame)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
    
    def _create_group_section(self, parent):
        """Create group control sliders"""
        group_frame = ttk.LabelFrame(parent, text="🏷️ Group Controls", padding=10)
        group_frame.pack(fill="x", pady=(0, 10))
        
        for group_name, channel_list in self.groups.items():
            if not channel_list:
                continue
            
            # Group frame
            frame = ttk.Frame(group_frame)
            frame.pack(fill="x", pady=2)
            
            # Label
            label = ttk.Label(frame, text=f"{group_name} ({len(channel_list)})", width=25)
            label.pack(side="left")
            
            # Slider
            var = tk.DoubleVar(value=1.0)
            slider = ttk.Scale(frame, from_=0.0, to=2.0, variable=var, length=300, 
                             command=lambda val, group=group_name: self._on_group_change(group, float(val)))
            slider.pack(side="left", padx=5)
            
            # Value label
            value_label = ttk.Label(frame, text="1.00", width=6)
            value_label.pack(side="left", padx=5)
            
            # Update value label when slider changes
            def update_value_label(val, label_ref=value_label):
                label_ref.config(text=f"{float(val):.2f}")
            
            slider.config(command=lambda val, group=group_name, label_ref=value_label: (
                self._on_group_change(group, float(val)),
                update_value_label(val, label_ref)
            ))
            
            self.sliders[f"group_{group_name}"] = (var, slider, value_label)
    
    def _create_channel_section(self, parent):
        """Create individual channel sliders"""
        channel_frame = ttk.LabelFrame(parent, text="🎛️ Individual Channels", padding=10)
        channel_frame.pack(fill="x", pady=(0, 10))
        
        for category, tracks in self.channels.items():
            # Category header
            cat_label = ttk.Label(channel_frame, text=f"📁 {category.upper()}", font=('Arial', 10, 'bold'))
            cat_label.pack(anchor="w", pady=(10, 5))
            
            for track_name in tracks.keys():
                channel_id = f"{category}.{track_name}"
                
                # Channel frame
                frame = ttk.Frame(channel_frame)
                frame.pack(fill="x", pady=1)
                
                # Label
                label = ttk.Label(frame, text=track_name, width=25)
                label.pack(side="left")
                
                # Slider
                var = tk.DoubleVar(value=1.0)
                slider = ttk.Scale(frame, from_=0.0, to=2.0, variable=var, length=300,
                                 command=lambda val, ch=channel_id: self._on_channel_change(ch, float(val)))
                slider.pack(side="left", padx=5)
                
                # Value label
                value_label = ttk.Label(frame, text="1.00", width=6)
                value_label.pack(side="left", padx=5)
                
                # Reset button
                reset_btn = ttk.Button(frame, text="↻", width=3,
                                     command=lambda ch=channel_id: self._reset_channel(ch))
                reset_btn.pack(side="left", padx=2)
                
                # Update value label when slider changes
                def update_value_label(val, label_ref=value_label):
                    label_ref.config(text=f"{float(val):.2f}")
                
                slider.config(command=lambda val, ch=channel_id, label_ref=value_label: (
                    self._on_channel_change(ch, float(val)),
                    update_value_label(val, label_ref)
                ))
                
                self.sliders[channel_id] = (var, slider, value_label)
    
    def _create_buttons(self, parent):
        """Create control buttons"""
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill="x", pady=10)
        
        # Quick presets
        presets_frame = ttk.LabelFrame(button_frame, text="⚡ Quick Presets", padding=5)
        presets_frame.pack(fill="x", pady=(0, 5))
        
        preset_buttons = [
            ("Boost Drums +20%", lambda: self._preset_boost_drums()),
            ("Reduce Vocals -20%", lambda: self._preset_reduce_vocals()),
            ("Balance Bass", lambda: self._preset_balance_bass()),
            ("Fix Vocals/Drums", lambda: self._preset_fix_vocals_drums()),
        ]
        
        for i, (text, command) in enumerate(preset_buttons):
            btn = ttk.Button(presets_frame, text=text, command=command)
            btn.grid(row=i//2, column=i%2, padx=5, pady=2, sticky="ew")
        
        presets_frame.grid_columnconfigure(0, weight=1)
        presets_frame.grid_columnconfigure(1, weight=1)
        
        # Control buttons
        controls_frame = ttk.Frame(button_frame)
        controls_frame.pack(fill="x", pady=5)
        
        ttk.Button(controls_frame, text="Reset All", command=self._reset_all).pack(side="left", padx=5)
        ttk.Button(controls_frame, text="Show Summary", command=self._show_summary).pack(side="left", padx=5)
        ttk.Button(controls_frame, text="Export Settings", command=self._export_settings).pack(side="left", padx=5)
        ttk.Button(controls_frame, text="Apply & Close", command=self._apply_and_close).pack(side="right", padx=5)
    
    def _on_group_change(self, group_name, value):
        """Handle group slider changes"""
        if group_name in self.groups:
            for channel_id in self.groups[group_name]:
                if channel_id in self.balance_values:
                    self.balance_values[channel_id] = value
                    # Update individual slider
                    if channel_id in self.sliders:
                        var, slider, label = self.sliders[channel_id]
                        var.set(value)
                        label.config(text=f"{value:.2f}")
    
    def _on_channel_change(self, channel_id, value):
        """Handle individual channel changes"""
        self.balance_values[channel_id] = value
    
    def _reset_channel(self, channel_id):
        """Reset individual channel to neutral"""
        if channel_id in self.sliders:
            var, slider, label = self.sliders[channel_id]
            var.set(1.0)
            label.config(text="1.00")
            self.balance_values[channel_id] = 1.0
    
    def _preset_boost_drums(self):
        """Preset: Boost all drums by 20%"""
        drums = [ch for ch in self.balance_values.keys() if any(word in ch.lower() for word in ['drum', 'kick', 'snare', 'hat'])]
        for channel_id in drums:
            self._set_channel_value(channel_id, 1.2)
    
    def _preset_reduce_vocals(self):
        """Preset: Reduce all vocals by 20%"""
        vocals = [ch for ch in self.balance_values.keys() if any(word in ch.lower() for word in ['vocal', 'vox', 'harmony'])]
        for channel_id in vocals:
            self._set_channel_value(channel_id, 0.8)
    
    def _preset_balance_bass(self):
        """Preset: Balance bass elements"""
        bass_channels = [ch for ch in self.balance_values.keys() if 'bass' in ch.lower()]
        for i, channel_id in enumerate(bass_channels):
            level = 0.8 + (i * 0.1)
            self._set_channel_value(channel_id, min(level, 2.0))
    
    def _preset_fix_vocals_drums(self):
        """Preset: Fix vocals/drums issue"""
        # Reduce vocals
        vocals = [ch for ch in self.balance_values.keys() if any(word in ch.lower() for word in ['vocal', 'vox', 'harmony'])]
        for channel_id in vocals:
            self._set_channel_value(channel_id, 0.7)
        
        # Boost drums
        drums = [ch for ch in self.balance_values.keys() if any(word in ch.lower() for word in ['drum', 'kick', 'snare', 'hat'])]
        for channel_id in drums:
            self._set_channel_value(channel_id, 1.2)
        
        messagebox.showinfo("Preset Applied", "✅ Vocals reduced 30%, drums boosted 20%")
    
    def _set_channel_value(self, channel_id, value):
        """Set channel value and update slider"""
        value = max(0.0, min(2.0, value))
        self.balance_values[channel_id] = value
        if channel_id in self.sliders:
            var, slider, label = self.sliders[channel_id]
            var.set(value)
            label.config(text=f"{value:.2f}")
    
    def _reset_all(self):
        """Reset all channels to neutral"""
        for channel_id in self.balance_values.keys():
            self._set_channel_value(channel_id, 1.0)
        
        # Reset group sliders
        for key, (var, slider, label) in self.sliders.items():
            if key.startswith("group_"):
                var.set(1.0)
                label.config(text="1.00")
    
    def _show_summary(self):
        """Show balance summary"""
        changed = {k: v for k, v in self.balance_values.items() if abs(v - 1.0) > 0.01}
        
        if changed:
            summary = "🎚️ BALANCE CHANGES:\n\n"
            for channel_id, value in sorted(changed.items()):
                change_pct = (value - 1.0) * 100
                direction = "↑" if change_pct > 0 else "↓"
                summary += f"{direction} {channel_id}: {value:.2f} ({change_pct:+.0f}%)\n"
            
            summary += f"\nTotal modified: {len(changed)} channels"
        else:
            summary = "📊 All channels at neutral balance (1.0)"
        
        messagebox.showinfo("Balance Summary", summary)
    
    def _export_settings(self):
        """Export current settings"""
        changed = {k: v for k, v in self.balance_values.items() if abs(v - 1.0) > 0.01}
        
        # Show export code
        if changed:
            code = "# Balance settings for mixing engine\n"
            code += "selected_balance = {\n"
            code += '    "vocal_prominence": 0.3,\n'
            code += '    "drum_punch": 0.65,\n'
            code += '    "bass_foundation": 0.6,\n'
            code += '    "instrument_presence": 0.5,\n'
            code += '    "channel_overrides": {\n'
            for channel_id, value in sorted(changed.items()):
                code += f'        "{channel_id}": {value:.2f},\n'
            code += '    }\n'
            code += '}'
        else:
            code = "# No balance changes\nselected_balance = {}"
        
        # Create popup window with code
        popup = tk.Toplevel(self.root)
        popup.title("Export Settings")
        popup.geometry("600x400")
        
        text_widget = tk.Text(popup, wrap=tk.WORD)
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        text_widget.insert(tk.END, code)
        
        copy_btn = ttk.Button(popup, text="Copy to Clipboard", 
                            command=lambda: popup.clipboard_clear() or popup.clipboard_append(code))
        copy_btn.pack(pady=5)
    
    def _apply_and_close(self):
        """Apply settings and close GUI"""
        self.result = self.get_balance_dict()
        try:
            self.root.quit()
            self.root.destroy()
        except:
            pass  # Ignore close errors
    
    def get_balance_dict(self):
        """Get current balance as dictionary in format expected by mixing engine"""
        changed = {k: v for k, v in self.balance_values.items() if abs(v - 1.0) > 0.01}
        
        # Return format expected by mixing engine
        result = {
            "mix_balance": {
                "vocal_prominence": 0.3,
                "drum_punch": 0.65,
                "bass_foundation": 0.6,
                "instrument_presence": 0.5,
            }
        }
        
        # Add channel overrides as separate top-level key if we have changes
        if changed:
            result["channel_overrides"] = changed
            
        return result
    
    def run(self):
        """Run the GUI"""
        self.result = None
        print("🎚️ Opening Balance Control GUI...")
        print("💡 Use the sliders to adjust balance, then click 'Apply & Close'")
        self.root.mainloop()
        return self.result


def create_working_balance_gui(channels):
    """Create and run the working balance GUI"""
    gui = WorkingBalanceGUI(channels)
    balance_result = gui.run()
    
    if balance_result:
        print("✅ Balance settings applied!")
        
        # Show what was changed
        if "channel_overrides" in balance_result:
            changed = balance_result["channel_overrides"]
            print(f"📊 Modified {len(changed)} channels:")
            for channel_id, value in sorted(changed.items()):
                change_pct = (value - 1.0) * 100
                direction = "↑" if change_pct > 0 else "↓"
                print(f"  {direction} {channel_id}: {value:.2f} ({change_pct:+.0f}%)")
        else:
            print("📊 Using preset balance values only")
        
        return balance_result
    else:
        print("⚠️ GUI cancelled - using default balance")
        return {
            "mix_balance": {
                "vocal_prominence": 0.3,
                "drum_punch": 0.65,
                "bass_foundation": 0.6,
                "instrument_presence": 0.5,
            }
        }


# Example usage
if __name__ == "__main__":
    # Example channels
    example_channels = {
        "drums": {"kick": "path1", "snare": "path2", "hats": "path3"},
        "bass": {"bass_guitar5": "path4", "bass_synth2": "path5"},
        "vocals": {"lead_vocal1": "path6", "lead_vocal2": "path7"}
    }
    
    result = create_working_balance_gui(example_channels)
    print(f"Result: {result}")