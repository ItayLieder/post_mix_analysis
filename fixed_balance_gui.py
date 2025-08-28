# 🎚️ WORKING BALANCE SLIDER GUI - FINAL VERSION

import ipywidgets as widgets
from IPython.display import display, clear_output
import json

class WorkingBalanceSliders:
    def __init__(self, channels_dict):
        """Create working sliders for all channels"""
        self.channels = channels_dict
        self.sliders = {}
        self.group_sliders = {}
        self.balance_values = {}
        
        # Initialize all values to 1.0 (neutral)
        for category, tracks in channels_dict.items():
            for track_name in tracks.keys():
                channel_id = f"{category}.{track_name}"
                self.balance_values[channel_id] = 1.0
        
        self.create_gui()
    
    def create_gui(self):
        """Create the complete GUI"""
        print("🎚️ Creating Balance Control Sliders")
        
        # Create group controls first
        group_controls = []
        for category in self.channels.keys():
            group_slider = widgets.FloatSlider(
                value=1.0,
                min=0.1,
                max=3.0,
                step=0.1,
                description=f"{category.upper()}:",
                style={'description_width': '100px'},
                layout=widgets.Layout(width='400px')
            )
            
            # Link to update all channels in this group
            group_slider.observe(lambda change, cat=category: self.update_group(cat, change['new']), names='value')
            self.group_sliders[category] = group_slider
            group_controls.append(group_slider)
        
        # Create individual channel sliders
        individual_controls = []
        for category, tracks in self.channels.items():
            # Category header
            category_label = widgets.HTML(f"<h4>📁 {category.upper()} Individual Channels:</h4>")
            individual_controls.append(category_label)
            
            # Individual sliders for this category
            for track_name in tracks.keys():
                channel_id = f"{category}.{track_name}"
                
                slider = widgets.FloatSlider(
                    value=1.0,
                    min=0.1,
                    max=5.0,
                    step=0.1,
                    description=f"{track_name}:",
                    style={'description_width': '150px'},
                    layout=widgets.Layout(width='500px')
                )
                
                slider.observe(lambda change, ch_id=channel_id: self.update_channel(ch_id, change['new']), names='value')
                self.sliders[channel_id] = slider
                individual_controls.append(slider)
            
            individual_controls.append(widgets.HTML("<br>"))
        
        # Apply button
        apply_btn = widgets.Button(
            description="✅ Apply Balance",
            button_style='success',
            layout=widgets.Layout(width='200px', height='40px')
        )
        apply_btn.on_click(self.apply_balance)
        
        # Reset button  
        reset_btn = widgets.Button(
            description="🔄 Reset All",
            button_style='warning',
            layout=widgets.Layout(width='200px', height='40px')
        )
        reset_btn.on_click(self.reset_all)
        
        # Status output
        self.status_output = widgets.Output()
        
        # Layout everything
        group_box = widgets.VBox([
            widgets.HTML("<h3>🎚️ Group Controls (affects all channels in group):</h3>")
        ] + group_controls)
        
        individual_box = widgets.VBox([
            widgets.HTML("<h3>🎛️ Individual Channel Controls:</h3>")
        ] + individual_controls)
        
        button_box = widgets.HBox([apply_btn, reset_btn])
        
        self.gui = widgets.VBox([
            group_box,
            individual_box,  
            button_box,
            self.status_output
        ])
        
        display(self.gui)
        
        with self.status_output:
            print("✅ Balance sliders ready!")
            print("📊 Use group sliders for quick adjustments")
            print("🎚️ Use individual sliders for fine control")
            print("💾 Click 'Apply Balance' when done")
    
    def update_group(self, category, value):
        """Update all channels in a group"""
        for track_name in self.channels[category].keys():
            channel_id = f"{category}.{track_name}"
            self.balance_values[channel_id] = value
            if channel_id in self.sliders:
                self.sliders[channel_id].value = value
    
    def update_channel(self, channel_id, value):
        """Update individual channel"""
        self.balance_values[channel_id] = value
    
    def reset_all(self, btn):
        """Reset all sliders to 1.0"""
        for channel_id in self.balance_values.keys():
            self.balance_values[channel_id] = 1.0
            if channel_id in self.sliders:
                self.sliders[channel_id].value = 1.0
        
        for group_slider in self.group_sliders.values():
            group_slider.value = 1.0
        
        with self.status_output:
            clear_output()
            print("🔄 All sliders reset to neutral (1.0)")
    
    def apply_balance(self, btn):
        """Apply current balance and make it available to notebook"""
        with self.status_output:
            clear_output()
            
            # Get changed values
            changed = {k: v for k, v in self.balance_values.items() if abs(v - 1.0) > 0.01}
            
            print("✅ Balance Applied!")
            print(f"📊 {len(changed)} channels modified:")
            
            for ch, val in sorted(changed.items()):
                change_pct = (val - 1.0) * 100
                direction = "↑" if change_pct > 0 else "↓"
                print(f"  {direction} {ch}: {val:.2f} ({change_pct:+.0f}%)")
            
            # Make the values available globally
            globals()['channel_overrides'] = changed
            globals()['balance_settings'] = {
                'channel_overrides': changed,
                'mix_balance': {
                    'vocal_prominence': 0.3,
                    'drum_punch': 0.7,
                    'bass_foundation': 0.6,
                    'instrument_presence': 0.5
                }
            }
            
            print("\n💾 Variables set:")
            print(f"  • channel_overrides: {len(changed)} channels")
            print("  • balance_settings: Ready for mixing engine")
            print("\n🎯 Now run your mixing cells!")

def create_balance_gui(channels):
    """Create the balance GUI"""
    return WorkingBalanceSliders(channels)

# Test with current channels if available
if __name__ == "__main__":
    print("🎚️ Fixed Balance GUI Module Ready!")
    print("Usage: gui = create_balance_gui(channels)")