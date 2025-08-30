# 🍝 CONSOLIDATED IMPORTS for post_mix_cleaned.ipynb
# This replaces 16+ import lines with a single import
# Making the notebook much cleaner and faster to load

# Core Python modules (keep in notebook)
# import os, json, numpy as np, soundfile as sf, pandas as pd, matplotlib.pyplot as plt
# from dataclasses import asdict

# Project modules - consolidated here
from config import CONFIG, CFG
from audio_utils import validate_audio
from utils import *
from data_handler import *
from analysis import *
from dsp_premitives import *
from processors import *
from render_engine import *
from pre_master_prep import *
from streaming_normalization_simulator import *
from comparison_reporting import *
from presets_recommendations import *
from logging_versioning import *
from mastering_orchestrator import *
from stem_mastering import *
from stem_balance_helper import set_stem_balance

# Specialized imports already included via mastering_orchestrator import *

print("🍝 Consolidated post-mix imports loaded!")
print("   ✅ All 16+ modules imported successfully")
print("   ✅ Notebook imports now simplified to single line")
print("   ✅ Faster loading and cleaner code")