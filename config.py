"""
Centralized configuration system to replace hardcoded values throughout the codebase.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple
import os


@dataclass
class AudioConfig:
    """Audio processing configuration."""
    
    # Format settings
    default_bit_depth: str = "PCM_24"
    default_sample_rate: int = 48000
    
    # Safety limits
    max_clip_range: float = 4.0
    safe_clip_range: float = 1.0
    
    # Peak/loudness targets  
    prep_hpf_hz: float = 5.0  # Very low to preserve bass content
    prep_peak_target_dbfs: float = -6.0
    render_peak_target_dbfs: float = -1.2
    true_peak_ceiling_db: float = -1.2
    
    # Analysis settings
    lufs_bs1770_offset: float = -0.691
    oversample_factor: int = 4
    
    # Frequency band definitions
    bass_freq_low: float = 20.0
    bass_freq_high: float = 120.0
    air_freq_low: float = 8000.0
    kick_freq_low: float = 40.0
    kick_freq_high: float = 110.0
    
    # Sub-bass monitoring
    subsonic_cutoff: float = 30.0


@dataclass  
class ProcessingConfig:
    """DSP processing configuration."""
    
    # Filter defaults
    default_q_factor: float = 0.7
    shelf_s_factor: float = 0.5
    
    # Dynamics
    compressor_attack_ms: float = 10.0
    compressor_release_ms: float = 100.0
    limiter_attack_ms: float = 1.0
    limiter_release_ms: float = 50.0
    limiter_lookahead_ms: float = 2.0
    limiter_knee_db: float = 1.5
    
    # Envelope detection
    envelope_smoothing_ms: float = 10.0
    
    # Fade parameters
    fade_curve_power: float = 2.0


@dataclass
class MasteringConfig:
    """Mastering-specific configuration."""
    
    # Style strength limits
    min_strength: float = 0.0
    max_strength: float = 1.0
    
    # EQ frequency points by style
    neutral_high_shelf_hz: float = 9000.0
    neutral_low_shelf_hz: float = 90.0
    warm_low_shelf_hz: float = 120.0
    warm_cut_freq_hz: float = 3500.0
    bright_high_shelf_hz: float = 8500.0
    bright_cut_freq_hz: float = 220.0
    
    # Glue compression amounts by style
    neutral_glue_base: float = 0.10
    neutral_glue_scale: float = 0.10
    warm_glue_base: float = 0.12
    warm_glue_scale: float = 0.12
    bright_glue_base: float = 0.10
    bright_glue_scale: float = 0.12
    loud_glue_base: float = 0.16
    loud_glue_scale: float = 0.18
    
    # Job timeouts
    async_job_timeout_seconds: int = 300
    async_poll_interval_seconds: float = 1.0


@dataclass
class AnalysisConfig:
    """Analysis configuration."""
    
    # FFT settings
    default_nfft: int = 1 << 16
    spectrum_nfft: int = 1 << 16
    flatness_nfft: int = 1 << 14
    
    # Windowing
    window_type: str = "hanning"
    
    # Short-term analysis
    short_term_window_s: float = 3.0
    short_term_hop_s: float = 0.5
    momentary_window_s: float = 0.4
    
    # K-weighting filter parameters
    k_weight_hp_freq: float = 38.0
    k_weight_shelf_freq: float = 1681.974450955533
    k_weight_shelf_q: float = 0.7071752369554196
    k_weight_shelf_gain_db: float = 3.99984385397
    
    # Percentiles for DR calculation
    dr_high_percentile: float = 95.0
    dr_low_percentile: float = 10.0


@dataclass
class StreamingConfig:
    """Streaming platform simulation configuration."""
    
    # Platform normalization targets
    spotify_lufs: float = -14.0
    apple_music_lufs: float = -16.0
    youtube_lufs: float = -13.0
    tidal_lufs: float = -14.0
    
    # Platform-specific processing
    youtube_compressor_enabled: bool = True
    youtube_compressor_ratio: float = 2.0
    
    def get_platform_profiles(self) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Get all streaming platform profiles."""
        return [
            ("Spotify", self.spotify_lufs, {}),
            ("Apple Music", self.apple_music_lufs, {}),
            ("YouTube", self.youtube_lufs, {
                "compressor_enabled": self.youtube_compressor_enabled,
                "compressor_ratio": self.youtube_compressor_ratio
            }),
            ("Tidal", self.tidal_lufs, {})
        ]


@dataclass
class ReportingConfig:
    """Reporting and visualization configuration."""
    
    # Preview settings
    preview_duration_seconds: int = 60
    
    # Plot settings
    plot_width: float = 10.0
    plot_height: float = 6.0
    plot_dpi: int = 100
    max_frequency_plot: float = 20000.0
    
    # File naming
    spectrum_plot_name: str = "spectrum_overlay.png"
    loudness_plot_name: str = "loudness_overlay.png"
    report_name: str = "comparison_report.html"
    
    # HTML report styling
    report_title: str = "Post-Mix Analysis Report"
    font_family: str = "system-ui, Arial, sans-serif"
    table_border_color: str = "#ddd"


@dataclass
class WorkspaceConfig:
    """Workspace and file organization."""
    
    # Base workspace directory - outside the Git repo
    workspace_root: str = "/Users/itay/Documents/post_mix_data/PostMixRuns"
    
    # Directory structure (relative to workspace_root)
    inputs_dir: str = "inputs"
    outputs_dir: str = "outputs"
    reports_dir: str = "reports"
    assets_dir: str = "assets"
    bundles_dir: str = "bundles"
    premaster_dir: str = "premaster"
    master_dir: str = "master"
    stream_preview_dir: str = "stream_previews"
    
    # File naming patterns
    workspace_timestamp_format: str = "%Y%m%d-%H%M%S"
    run_id_prefix: str = "run"
    
    # Manifest settings
    manifest_filename: str = "manifest.json"
    run_log_filename: str = "run_log.json"
    
    def get_workspace_root(self) -> str:
        """Get the configured workspace root, with environment variable override support."""
        return os.getenv('POST_MIX_WORKSPACE_ROOT', self.workspace_root)


@dataclass
class PipelineConfig:
    """Processing pipeline mode configuration."""
    
    # Processing mode options
    SINGLE_FILE = "single_file"
    STEM_MASTERING = "stem_mastering"
    
    # Default processing mode
    default_mode: str = SINGLE_FILE
    
    # Stem mastering settings
    stem_required_categories: list = field(default_factory=lambda: ["music"])  # At least music stem required
    stem_optional_categories: list = field(default_factory=lambda: ["drums", "bass", "vocals"])
    
    # Stem combination variants to generate - BIG VARIANTS SYSTEM!
    stem_combinations: list = field(default_factory=lambda: [
        # BIG VARIANTS - The amazing processing that sounds INCREDIBLE!
        ("BIG_Exact_Match", "big:BIG_Exact_Match"),  # EXACT replica of the amazing BIG_POWERFUL_STEM_MIX.wav - FIRST!
        ("BIG_Amazing", "big:BIG_Amazing"),
        ("BIG_Massive_Drums", "big:BIG_Massive_Drums"), 
        ("BIG_Foundation_Bass", "big:BIG_Foundation_Bass"),
        ("BIG_Vocal_Domination", "big:BIG_Vocal_Domination"),
        ("BIG_Cinematic_Wide", "big:BIG_Cinematic_Wide"),
        ("BIG_Radio_Power", "big:BIG_Radio_Power"),
        ("BIG_Club_Energy", "big:BIG_Club_Energy"),
        ("BIG_Modern_Pop", "big:BIG_Modern_Pop"),
        ("BIG_Rock_Power", "big:BIG_Rock_Power"),
        ("BIG_Intimate_Powerful", "big:BIG_Intimate_Powerful"),
        ("BIG_Maximum_Impact", "big:BIG_Maximum_Impact"),
        
        # Original basic combinations (kept for compatibility)
        ("Stem_PunchyMix", "punchy"),
        ("Stem_WideAndOpen", "wide"), 
        ("Stem_TightAndControlled", "tight"),
        ("Stem_Aggressive", "aggressive"),
        ("Stem_Balanced", "natural"),
    ])
    
    # Enable/disable advanced stem processing (DISABLED - causes quality issues)
    use_advanced_stem_processing: bool = False
    
    # Enable/disable extreme stem processing (DISABLED - causes quality issues)
    use_extreme_stem_processing: bool = False
    
    # Enable/disable depth processing (DISABLED - causes quality issues)
    use_depth_processing: bool = False
    
    # Enable/disable musical depth processing (DISABLED - causes quality issues)
    use_musical_depth_processing: bool = False
    
    # Enable/disable hybrid processing (DISABLED - causes quality issues)
    use_hybrid_processing: bool = False
    
    # Use BIG impressive processing (DEFAULT - sounds AMAZING and MUCH BIGGER!)
    use_big_impressive_processing: bool = True
    
    # BPM for tempo-synced effects
    default_bpm: float = 120.0
    
    # Stem balancing configuration - EXACT MATCH TO BIG_POWERFUL_STEM_MIX.wav!
    # These gains EXACTLY match what created the amazing BIG_POWERFUL_STEM_MIX.wav file
    # DO NOT CHANGE - these are the exact values that sound amazing
    stem_gains: dict = field(default_factory=lambda: {
        # Main stem categories - EXACT VALUES FROM AMAZING FILE
        "drums": 3.0,     # HUGE drums (exact match to amazing file)
        "bass": 2.8,      # MASSIVE bass (exact match to amazing file)
        "vocals": 4.0,    # COMMANDING vocals (exact match to amazing file) 
        "music": 2.0,     # BIG musical content (exact match to amazing file)
        
        # Detailed stems (exact match to amazing file proportions)
        "kick": 3.0,         # KICK EXTREMELY LOUD - exact match
        "snare": 2.5,        # SNARE VERY LOUD - exact match  
        "hats": 1.8,         # Hi-hats - exact match
        "backvocals": 2.5,   # Backing vocals - exact match
        "leadvocals": 4.0,   # Lead vocals - exact match
        "guitar": 2.0,       # Guitar - proportional to music
        "keys": 2.0,         # Keys/synths - proportional to music
        "strings": 2.0,      # Strings - proportional to music
    })
    
    # Style-specific stem balance adjustments (optional)
    # These multiply the base stem_gains for each combination style
    stem_balance_styles: dict = field(default_factory=lambda: {
        "punchy": {"drums": 1.1, "bass": 1.0, "vocals": 0.95, "music": 0.9},
        "wide": {"drums": 0.9, "bass": 0.95, "vocals": 1.0, "music": 1.1},
        "tight": {"drums": 1.0, "bass": 1.05, "vocals": 1.0, "music": 0.95},
        "aggressive": {"drums": 1.15, "bass": 1.1, "vocals": 1.05, "music": 0.85},
        "natural": {"drums": 1.0, "bass": 1.0, "vocals": 1.0, "music": 1.0}
    })
    
    # Auto-gain compensation based on number of stems
    # DISABLED - was causing weak mixes by reducing levels too aggressively
    auto_gain_compensation: bool = False
    
    # Target peak after stem summing (before mastering) - BOOSTED FOR POWER
    stem_sum_target_peak: float = 0.98  # -0.17 dBFS (EXTREMELY LOUD!)
    
    # Output control - whether to create individual processed stem files
    # If False, only creates the main mix file (e.g., "BIG_Amazing.wav")
    # If True, also creates individual stem files (e.g., "bass_processed.wav", "drums_processed.wav")
    create_individual_stem_files: bool = True
    
    def get_stem_gains(self) -> dict:
        """Get stem gains with environment variable override support."""
        gains = self.stem_gains.copy()
        
        # Allow environment variable overrides
        if os.getenv('STEM_GAIN_DRUMS'):
            gains['drums'] = float(os.getenv('STEM_GAIN_DRUMS'))
        if os.getenv('STEM_GAIN_BASS'):
            gains['bass'] = float(os.getenv('STEM_GAIN_BASS'))
        if os.getenv('STEM_GAIN_VOCALS'):
            gains['vocals'] = float(os.getenv('STEM_GAIN_VOCALS'))
        if os.getenv('STEM_GAIN_MUSIC'):
            gains['music'] = float(os.getenv('STEM_GAIN_MUSIC'))
            
        return gains
    
    def get_processing_mode(self) -> str:
        """Get processing mode with environment variable override."""
        return os.getenv('POST_MIX_PROCESSING_MODE', self.default_mode)
    
    def is_stem_mode(self) -> bool:
        """Check if stem mastering mode is enabled."""
        return self.get_processing_mode() == self.STEM_MASTERING


from dataclasses import dataclass, field

@dataclass
class GlobalConfig:
    """Main configuration container."""
    
    audio: AudioConfig = field(default_factory=AudioConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    mastering: MasteringConfig = field(default_factory=MasteringConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    streaming: StreamingConfig = field(default_factory=StreamingConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)
    workspace: WorkspaceConfig = field(default_factory=WorkspaceConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    
    @classmethod
    def load_from_env(cls) -> 'GlobalConfig':
        """Load configuration from environment variables."""
        config = cls()
        
        # Override with environment variables if present
        if os.getenv("PREP_PEAK_TARGET"):
            config.audio.prep_peak_target_dbfs = float(os.getenv("PREP_PEAK_TARGET"))
        
        if os.getenv("RENDER_PEAK_TARGET"):
            config.audio.render_peak_target_dbfs = float(os.getenv("RENDER_PEAK_TARGET"))
            
        if os.getenv("DEFAULT_NFFT"):
            config.analysis.default_nfft = int(os.getenv("DEFAULT_NFFT"))
            
        if os.getenv("PREVIEW_DURATION"):
            config.reporting.preview_duration_seconds = int(os.getenv("PREVIEW_DURATION"))
        
        # Workspace root override
        if os.getenv("POST_MIX_WORKSPACE_ROOT"):
            config.workspace.workspace_root = os.getenv("POST_MIX_WORKSPACE_ROOT")
            
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        return {
            "audio": self.audio.__dict__,
            "processing": self.processing.__dict__,
            "mastering": self.mastering.__dict__,
            "analysis": self.analysis.__dict__,
            "streaming": self.streaming.__dict__,
            "reporting": self.reporting.__dict__,
            "workspace": self.workspace.__dict__,
        }


# Global configuration instance
CONFIG = GlobalConfig.load_from_env()


# Legacy compatibility - these can be removed once all files are updated
CFG = type('CFG', (), {
    'default_bit_depth': CONFIG.audio.default_bit_depth,
    'prep_hpf_hz': CONFIG.audio.prep_hpf_hz,
    'prep_peak_target_dbfs': CONFIG.audio.prep_peak_target_dbfs,
    'render_peak_target_dbfs': CONFIG.audio.render_peak_target_dbfs,
    'tp_ceiling_db': CONFIG.audio.true_peak_ceiling_db,
    'preview_seconds': CONFIG.reporting.preview_duration_seconds,
    'nfft': CONFIG.analysis.default_nfft,
})()