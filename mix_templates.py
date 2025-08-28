#!/usr/bin/env python3
"""
Mix Templates - Genre-specific mixing configurations
Professional mixing templates for different musical styles
"""

from typing import Dict, Any, Optional


class MixTemplate:
    """Base class for mix templates"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.channel_settings = {}
        self.bus_settings = {}
        self.master_settings = {}
        self.spatial_map = {}
        
    def get_channel_settings(self, channel_type: str, custom_params: Dict = None) -> Dict:
        """Get processing settings for a channel type"""
        
        # Get base settings
        if channel_type in self.channel_settings:
            settings = self.channel_settings[channel_type].copy()
        else:
            # Use default settings
            settings = self.get_default_settings(channel_type)
        
        # Apply custom parameters
        if custom_params:
            settings = self.apply_custom_params(settings, custom_params)
        
        return settings
    
    def get_default_settings(self, channel_type: str) -> Dict:
        """Get default settings for unknown channel types"""
        return {
            "eq": {},
            "compression": {"enabled": False},
            "gate": False,
            "pan": 0,
            "reverb_send": 0,
            "delay_send": 0
        }
    
    def apply_custom_params(self, settings: Dict, params: Dict) -> Dict:
        """Apply custom parameter adjustments"""
        
        # Brightness adjustment
        if "brightness" in params:
            brightness = params["brightness"]
            if "eq" not in settings:
                settings["eq"] = {}
            # Adjust high frequency
            settings["eq"]["10khz"] = {"gain": (brightness - 0.5) * 4, "q": 0.7}
        
        # Width adjustment
        if "width" in params and "pan" in settings:
            settings["pan"] *= params["width"]
        
        # Aggression adjustment
        if "aggression" in params and "compression" in settings:
            aggression = params["aggression"]
            if settings["compression"].get("enabled"):
                settings["compression"]["ratio"] *= (1 + aggression * 0.5)
        
        # Vintage adjustment
        if "vintage" in params:
            vintage = params["vintage"]
            settings["saturation"] = vintage * 0.3
        
        return settings


class ModernPopTemplate(MixTemplate):
    """Modern Pop mixing template - bright, wide, polished"""
    
    def __init__(self):
        super().__init__("modern_pop", "Bright, wide, and polished pop sound")
        
        # Channel-specific settings
        self.channel_settings = {
            # DRUMS
            "kick": {
                "eq": {"60hz": {"gain": 3, "q": 0.7}, "4khz": {"gain": 2, "q": 0.9}},
                "compression": {"enabled": True, "threshold": -15, "ratio": 4, "attack": 10},
                "gate": True,
                "pan": 0
            },
            "snare": {
                "eq": {"200hz": {"gain": 2, "q": 0.8}, "5khz": {"gain": 3, "q": 0.7}},
                "compression": {"enabled": True, "threshold": -12, "ratio": 3, "attack": 5},
                "gate": True,
                "pan": 0,
                "reverb_send": 0.1
            },
            "hihat": {
                "eq": {"200hz": {"gain": -6, "q": 0.7}, "8khz": {"gain": 2, "q": 0.6}},
                "compression": {"enabled": False},
                "pan": 0.15
            },
            
            # BASS
            "bass": {
                "eq": {"80hz": {"gain": 2, "q": 0.7}, "800hz": {"gain": -2, "q": 0.8}},
                "compression": {"enabled": True, "threshold": -18, "ratio": 3, "attack": 30},
                "pan": 0
            },
            
            # VOCALS
            "vocal_lead": {
                "eq": {"200hz": {"gain": -2, "q": 0.7}, "3khz": {"gain": 3, "q": 0.8}, "10khz": {"gain": 2, "q": 0.5}},
                "compression": {"enabled": True, "threshold": -15, "ratio": 3, "attack": 5},
                "pan": 0,
                "reverb_send": 0.15,
                "delay_send": 0.08
            },
            "vocal_harmony": {
                "eq": {"300hz": {"gain": -3, "q": 0.7}, "8khz": {"gain": 1, "q": 0.5}},
                "compression": {"enabled": True, "threshold": -18, "ratio": 2.5, "attack": 10},
                "pan": 0.3,  # Will alternate L/R
                "reverb_send": 0.2
            },
            
            # INSTRUMENTS
            "guitar_electric": {
                "eq": {"500hz": {"gain": -3, "q": 0.8}, "3khz": {"gain": 1, "q": 0.7}},
                "compression": {"enabled": True, "threshold": -20, "ratio": 2, "attack": 15},
                "pan": 0.4
            },
            "synth": {
                "eq": {"1khz": {"gain": -2, "q": 0.7}},
                "compression": {"enabled": False},
                "pan": 0.5,
                "reverb_send": 0.1
            },
            "piano": {
                "eq": {"400hz": {"gain": -2, "q": 0.8}, "5khz": {"gain": 1.5, "q": 0.6}},
                "compression": {"enabled": True, "threshold": -22, "ratio": 2, "attack": 20},
                "pan": -0.2
            }
        }
        
        # Bus settings
        self.bus_settings = {
            "drum_bus": {"compression": 0.3, "glue": 0.4, "saturation": 0.1},
            "vocal_bus": {"compression": 0.2, "presence": 0.3, "width": 0.8},
            "instrument_bus": {"width": 1.3, "glue": 0.2}
        }


class RockTemplate(MixTemplate):
    """Rock mixing template - punchy drums, present guitars"""
    
    def __init__(self):
        super().__init__("rock", "Punchy drums and present guitars")
        
        self.channel_settings = {
            "kick": {
                "eq": {"60hz": {"gain": 4, "q": 0.8}, "2khz": {"gain": 3, "q": 0.9}},
                "compression": {"enabled": True, "threshold": -12, "ratio": 5, "attack": 5},
                "gate": True,
                "pan": 0
            },
            "snare": {
                "eq": {"250hz": {"gain": 3, "q": 0.8}, "4khz": {"gain": 4, "q": 0.7}},
                "compression": {"enabled": True, "threshold": -10, "ratio": 4, "attack": 3},
                "gate": True,
                "pan": 0,
                "reverb_send": 0.08
            },
            "guitar_electric": {
                "eq": {"100hz": {"gain": -3, "q": 0.7}, "3khz": {"gain": 2, "q": 0.8}, "6khz": {"gain": 1, "q": 0.6}},
                "compression": {"enabled": True, "threshold": -18, "ratio": 2.5, "attack": 10},
                "pan": 0.5  # Double tracked wide
            },
            "bass": {
                "eq": {"80hz": {"gain": 3, "q": 0.8}, "1khz": {"gain": 1, "q": 0.7}},
                "compression": {"enabled": True, "threshold": -15, "ratio": 4, "attack": 20},
                "saturation": 0.2,
                "pan": 0
            },
            "vocal_lead": {
                "eq": {"300hz": {"gain": -2, "q": 0.7}, "2khz": {"gain": 2, "q": 0.8}, "8khz": {"gain": 1, "q": 0.5}},
                "compression": {"enabled": True, "threshold": -12, "ratio": 4, "attack": 3},
                "pan": 0,
                "reverb_send": 0.1,
                "delay_send": 0.05
            }
        }


class EDMTemplate(MixTemplate):
    """EDM mixing template - huge bass, side-chained, wide"""
    
    def __init__(self):
        super().__init__("edm", "Huge bass, side-chained pumping, wide stereo")
        
        self.channel_settings = {
            "kick": {
                "eq": {"50hz": {"gain": 5, "q": 0.9}, "100hz": {"gain": -3, "q": 0.8}, "5khz": {"gain": 3, "q": 0.8}},
                "compression": {"enabled": True, "threshold": -8, "ratio": 6, "attack": 2},
                "gate": True,
                "pan": 0
            },
            "bass": {
                "eq": {"40hz": {"gain": 6, "q": 0.8}, "200hz": {"gain": -4, "q": 0.7}},
                "compression": {"enabled": True, "threshold": -12, "ratio": 4, "attack": 15},
                "sidechain": "kick",  # Special EDM feature
                "pan": 0
            },
            "synth": {
                "eq": {"500hz": {"gain": -4, "q": 0.7}, "8khz": {"gain": 3, "q": 0.5}},
                "compression": {"enabled": False},
                "sidechain": "kick",
                "pan": 0.7,  # Wide synths
                "reverb_send": 0.15
            },
            "vocal_lead": {
                "eq": {"400hz": {"gain": -3, "q": 0.7}, "4khz": {"gain": 4, "q": 0.8}, "12khz": {"gain": 3, "q": 0.5}},
                "compression": {"enabled": True, "threshold": -12, "ratio": 3.5, "attack": 3},
                "pan": 0,
                "reverb_send": 0.2,
                "delay_send": 0.12
            }
        }


class HipHopTemplate(MixTemplate):
    """Hip-Hop mixing template - knock, 808s, crispy highs"""
    
    def __init__(self):
        super().__init__("hip_hop", "Hard-hitting drums, prominent 808s, crispy highs")
        
        self.channel_settings = {
            "kick": {
                "eq": {"60hz": {"gain": 6, "q": 0.9}, "4khz": {"gain": 4, "q": 0.9}},
                "compression": {"enabled": True, "threshold": -10, "ratio": 8, "attack": 1},
                "gate": True,
                "pan": 0
            },
            "bass": {  # 808
                "eq": {"30hz": {"gain": 8, "q": 0.8}, "80hz": {"gain": 3, "q": 0.7}},
                "compression": {"enabled": True, "threshold": -15, "ratio": 3, "attack": 30},
                "saturation": 0.3,
                "pan": 0
            },
            "snare": {
                "eq": {"200hz": {"gain": 4, "q": 0.8}, "8khz": {"gain": 5, "q": 0.6}},
                "compression": {"enabled": True, "threshold": -8, "ratio": 5, "attack": 2},
                "pan": 0,
                "reverb_send": 0.05
            },
            "hihat": {
                "eq": {"500hz": {"gain": -6, "q": 0.7}, "10khz": {"gain": 4, "q": 0.5}},
                "pan": 0.2,
                "panning_automation": "trap"  # Special pattern
            },
            "vocal_lead": {
                "eq": {"200hz": {"gain": -3, "q": 0.7}, "3khz": {"gain": 3, "q": 0.8}, "10khz": {"gain": 2, "q": 0.5}},
                "compression": {"enabled": True, "threshold": -10, "ratio": 4, "attack": 2},
                "autotune": {"enabled": True, "strength": 0.7},
                "pan": 0,
                "reverb_send": 0.08,
                "delay_send": 0.1
            }
        }


class JazzTemplate(MixTemplate):
    """Jazz mixing template - natural, dynamic, spatial"""
    
    def __init__(self):
        super().__init__("jazz", "Natural dynamics with realistic spatial positioning")
        
        self.channel_settings = {
            "kick": {
                "eq": {"80hz": {"gain": 2, "q": 0.6}, "2khz": {"gain": 1, "q": 0.7}},
                "compression": {"enabled": True, "threshold": -25, "ratio": 2, "attack": 20},
                "pan": 0
            },
            "bass": {
                "eq": {"100hz": {"gain": 1, "q": 0.6}, "800hz": {"gain": 0.5, "q": 0.7}},
                "compression": {"enabled": False},  # Natural dynamics
                "pan": 0.1
            },
            "piano": {
                "eq": {"500hz": {"gain": -1, "q": 0.6}},
                "compression": {"enabled": True, "threshold": -30, "ratio": 1.5, "attack": 40},
                "pan": -0.3,
                "reverb_send": 0.12
            },
            "vocal_lead": {
                "eq": {"300hz": {"gain": -1, "q": 0.6}, "3khz": {"gain": 1, "q": 0.7}},
                "compression": {"enabled": True, "threshold": -22, "ratio": 2, "attack": 15},
                "pan": 0,
                "reverb_send": 0.15
            }
        }


# Template registry
TEMPLATES = {
    "modern_pop": ModernPopTemplate,
    "rock": RockTemplate,
    "edm": EDMTemplate,
    "hip_hop": HipHopTemplate,
    "jazz": JazzTemplate,
}


def get_template(name: str) -> MixTemplate:
    """Get a mix template by name"""
    if name in TEMPLATES:
        return TEMPLATES[name]()
    else:
        # Return generic template
        return MixTemplate("custom", "Custom mixing template")


def list_templates() -> Dict[str, str]:
    """List all available templates"""
    return {
        name: template().description 
        for name, template in TEMPLATES.items()
    }


print("🎨 Mix Templates loaded!")
print("   • 5 genre-specific templates (Pop, Rock, EDM, Hip-Hop, Jazz)")
print("   • Channel-specific EQ, compression, and effects")
print("   • Intelligent spatial positioning")
print("   • Customizable parameters for fine-tuning")