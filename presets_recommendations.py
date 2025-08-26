


# ============================================
# Presets & Recommendations — Notebook Layer
# ============================================
# Provides:
# - DialState presets (named, ready to render)
# - Rule-based recommender from analysis metrics → dial suggestions
# - Plan builders for batch rendering with RenderEngine / Orchestrator
#
# Nothing executes on import; you’ll call funcs later.

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Tuple, Optional
import math

# Reuse the DialState from Render Engine if already defined; else define a tiny fallback.
try:
    DialState
except NameError:
    @dataclass
    class DialState:
        bass: float = 0.0     # 0..100
        punch: float = 0.0    # 0..100
        clarity: float = 0.0  # 0..100
        air: float = 0.0      # 0..100
        width: float = 0.0    # 0..100

# ---------- 1) Preset Library (you can tweak these) ----------

PRESETS: Dict[str, DialState] = {
    # Minimal changes (QA/reference)
    "Transparent":       DialState(bass=0,  punch=0,  clarity=0,  air=0,  width=0),
    # Gentle, safe polish
    "Balanced Gentle":   DialState(bass=15, punch=15, clarity=10, air=10, width=5),
    # Bass-focused variants
    "Bassy":            DialState(bass=45, punch=12, clarity=8,  air=6,  width=5),
    "Deep Bass":        DialState(bass=55, punch=8,  clarity=5,  air=5,  width=4),
    "Modern Low-End":   DialState(bass=38, punch=28, clarity=8,  air=8,  width=8),
    # Punch-focused variants
    "Punchy":           DialState(bass=18, punch=48, clarity=15, air=12, width=8),
    "Tight & Punchy":   DialState(bass=28, punch=42, clarity=12, air=10, width=6),
    "Aggressive Punch": DialState(bass=22, punch=58, clarity=18, air=10, width=6),
    # Clarity-focused variants
    "Clear":            DialState(bass=8,  punch=15, clarity=38, air=15, width=8),
    "Clarity & Air":    DialState(bass=10, punch=12, clarity=28, air=28, width=6),
    "Crystal Clear":    DialState(bass=5,  punch=18, clarity=45, air=22, width=10),
    # Air/brightness variants
    "Airy":             DialState(bass=12, punch=15, clarity=18, air=35, width=8),
    "Bright":           DialState(bass=8,  punch=12, clarity=22, air=42, width=6),
    "Sparkle":          DialState(bass=6,  punch=10, clarity=25, air=48, width=8),
    # Width/stereo variants
    "Wide":             DialState(bass=15, punch=12, clarity=12, air=15, width=28),
    "Wide Pop":         DialState(bass=18, punch=14, clarity=10, air=16, width=18),
    "Ultra Wide":       DialState(bass=12, punch=10, clarity=8,  air=12, width=38),
    # Combination variants
    "Warm & Wide":      DialState(bass=32, punch=15, clarity=8,  air=10, width=22),
    "Punchy & Bright":  DialState(bass=15, punch=35, clarity=20, air=32, width=8),
    "Bass & Punch":     DialState(bass=42, punch=38, clarity=10, air=8,  width=6),
    "Clear & Wide":     DialState(bass=10, punch=12, clarity=32, air=18, width=25),
    # Extreme variants
    "Maximum Bass":     DialState(bass=70, punch=5,  clarity=2,  air=2,  width=3),
    "Maximum Punch":    DialState(bass=12, punch=70, clarity=25, air=15, width=10),
    "Maximum Clarity":  DialState(bass=3,  punch=20, clarity=70, air=35, width=12),
    "Maximum Air":      DialState(bass=5,  punch=8,  clarity=30, air=70, width=10),
    "Maximum Width":    DialState(bass=8,  punch=8,  clarity=8,  air=12, width=70),
    # Genre-specific variants
    "Hip Hop":          DialState(bass=48, punch=35, clarity=10, air=8,  width=5),
    "Rock":             DialState(bass=25, punch=40, clarity=22, air=18, width=12),
    "Electronic":       DialState(bass=35, punch=45, clarity=15, air=20, width=15),
    "Jazz":             DialState(bass=20, punch=15, clarity=25, air=25, width=20),
    "Classical":        DialState(bass=10, punch=8,  clarity=20, air=30, width=25),
    "Vocal":            DialState(bass=12, punch=18, clarity=35, air=28, width=15),
    "Acoustic":         DialState(bass=18, punch=12, clarity=22, air=25, width=18),
    "Metal":            DialState(bass=30, punch=50, clarity=20, air=15, width=8),
    # Aggressive variants (from penalty_simulator notebook)
    "Aggressive Bass":  DialState(bass=65, punch=20, clarity=12, air=10, width=8),
    "Aggressive Punch": DialState(bass=28, punch=75, clarity=18, air=15, width=10),
}

def list_presets() -> List[str]:
    return list(PRESETS.keys())

def get_preset(name: str) -> DialState:
    return PRESETS[name]

# ---------- 2) Metric-driven Recommendations ----------

@dataclass
class Recommendation:
    name: str                       # a label for this suggestion
    dials: DialState                # suggested dial positions (0..100)
    priority: int                   # lower = earlier to try
    rationale: List[str]            # bullet points (human-readable “why”)
    notes: Optional[str] = None     # extra context

def _clip01(x: float, lo=0.0, hi=100.0) -> float:
    return float(max(lo, min(hi, x)))

def _scale(val: float, in_lo: float, in_hi: float, out_lo: float, out_hi: float) -> float:
    # linear map with clamping
    if in_hi == in_lo:
        return out_lo
    t = (val - in_lo) / (in_hi - in_lo)
    t = max(0.0, min(1.0, t))
    return out_lo + t * (out_hi - out_lo)

def recommend_from_analysis(rep) -> List[Recommendation]:
    """
    Takes an AnalysisReport (from analyze_wav) OR a dict with the same fields we need.
    Returns a sorted list of Recommendation objects.
    """
    # Extract metrics in a defensive way
    def _get(dct, k, default=0.0):
        return float(dct.get(k, default))

    # normalize input
    if hasattr(rep, "bass_energy_pct"):
        # It's an AnalysisReport
        basic = rep.basic
        stereo = rep.stereo
        lufs = rep.lufs_integrated
        tp   = rep.true_peak_dbfs
        bass = rep.bass_energy_pct
        airp = rep.air_energy_pct
        flat = rep.spectral_flatness
    else:
        # Expect a dict-like structure
        basic = rep.get("basic", rep)
        stereo = rep.get("stereo", rep)
        lufs = _get(rep, "lufs_integrated", -20.0)
        tp   = _get(rep, "true_peak_dbfs", -1.5)
        bass = _get(rep, "bass_energy_pct", 40.0)
        airp = _get(rep, "air_energy_pct", 0.2)
        flat = _get(rep, "spectral_flatness", 0.05)

    peak_dbfs = _get(basic, "peak_dbfs", -3.0)
    rms_dbfs  = _get(basic, "rms_dbfs", -20.0)
    crest_db  = _get(basic, "crest_db", 12.0)
    phase_corr = _get(stereo, "phase_correlation", 0.5)
    width      = _get(stereo, "stereo_width", 0.4)

    recs: List[Recommendation] = []

    # --- Heuristics (tunable) ---
    # Tonal balance
    very_bassy  = bass > 65.0
    super_bassy = bass > 80.0
    very_dark   = airp < 0.02
    dark        = airp < 0.1
    brightish   = airp > 0.7

    # Loudness/dynamics context
    very_dynamic = crest_db > 16
    squashed     = crest_db < 8

    # Stereo context
    too_narrow   = width < 0.35
    very_wide    = phase_corr < 0.2

    #################################################################
    # 1) Fix dark + bass-heavy → more Air, some Clarity, moderate Punch
    #################################################################
    if very_bassy and (dark or very_dark):
        air_amt = 25 if very_dark else 18
        clar_amt = 18 if super_bassy else 12
        punch_amt = 30 if very_dynamic else 22
        dials = DialState(
            bass= max(10, _scale(bass, 60, 90, 14, 28)),  # keep some bass but don’t add more
            punch= punch_amt,
            clarity= clar_amt,
            air= air_amt,
            width= 8 if too_narrow else 6
        )
        recs.append(Recommendation(
            name="De-mud & Open Top",
            dials=dials,
            priority=1,
            rationale=[
                f"bass_energy %{bass:.1f} is high → reduce mud via clarity + keep lows controlled",
                f"air_energy %{airp:.3f} is low → add air shelf for brightness",
                f"crest {crest_db:.1f} dB → moderate punch to emphasize transients",
            ],
            notes="Start here if the mix feels boomy/dull."
        ))

    #################################################################
    # 2) If very dynamic + quiet → add Punch, little Bass, gentle Air
    #################################################################
    if very_dynamic and lufs < -18:
        dials = DialState(
            bass= 16,
            punch= 38,
            clarity= 10,
            air= 14,
            width= 8 if too_narrow else 6
        )
        recs.append(Recommendation(
            name="Punch Up the Transients",
            dials=dials,
            priority=2,
            rationale=[
                f"crest {crest_db:.1f} dB (dynamic) and LUFS {lufs:.1f} (quiet) → add transient definition",
                "small air lift to help intelligibility",
            ]
        ))

    #################################################################
    # 3) If midrange congested (not dark, not bright, but high flatness) → Clarity focus
    #################################################################
    if 0.03 < flat < 0.12 and not dark and not brightish:
        dials = DialState(
            bass= 10,
            punch= 18,
            clarity= 26,
            air= 10,
            width= 8
        )
        recs.append(Recommendation(
            name="Clear Midrange",
            dials=dials,
            priority=3,
            rationale=[
                f"spectral_flatness {flat:.3f} suggests dense content → de-mud around 180–230 Hz",
                "moderate punch for definition without aggression",
            ]
        ))

    #################################################################
    # 4) If image is narrow → widen (safely)
    #################################################################
    if too_narrow:
        dials = DialState(
            bass= 14,
            punch= 14,
            clarity= 10,
            air= 12,
            width= _scale(width, 0.2, 0.4, 12, 22)
        )
        recs.append(Recommendation(
            name="Widen Image (Safe)",
            dials=dials,
            priority=4,
            rationale=[
                f"stereo_width {width:.2f} is narrow → add width",
                "tone moves kept gentle to avoid destabilizing center"
            ],
            notes="If mono compatibility is critical, keep width ≤ 15."
        ))

    #################################################################
    # 5) Default safe polish if no strong issues
    #################################################################
    if not recs:
        recs.append(Recommendation(
            name="Balanced Gentle",
            dials=PRESETS["Balanced Gentle"],
            priority=9,
            rationale=["No strong issues detected → start with light, broad polish."]
        ))

    # Stable ordering
    recs.sort(key=lambda r: r.priority)
    return recs

# ---------- 3) Render/Orchestrator Plans (without executing) ----------

def build_premaster_plan_from_recs(
    recs: List[Recommendation],
    limit: int = 3,
    prefix: str = "PM"
) -> List[Tuple[str, DialState]]:
    """
    Convert top-N recommendations to (name, DialState) tuples for RenderEngine.commit_variants().
    This is kept for backward compatibility, but use build_all_variants_plan() for comprehensive variants.
    """
    planned: List[Tuple[str, DialState]] = []
    for i, r in enumerate(recs[:limit], start=1):
        tag = f"{prefix}{i}_{r.name.replace(' ', '')}"
        planned.append((tag, r.dials))
    return planned

def build_all_variants_plan(
    recs: List[Recommendation],
    prefix: str = "PM",
    top_recommendations: int = 3
) -> List[Tuple[str, DialState]]:
    """
    Generate ALL possible variants including all presets and all recommendations.
    Top recommendations get "_recommended" suffix.
    Returns (name, DialState) tuples for RenderEngine.commit_variants().
    """
    planned: List[Tuple[str, DialState]] = []
    
    # 1. Add all preset variants first
    for preset_name, preset_dials in PRESETS.items():
        # Clean up name for filename
        clean_name = preset_name.replace(' ', '').replace('&', 'And')
        tag = f"{prefix}_{clean_name}"
        planned.append((tag, preset_dials))
    
    # 2. Add all recommendation variants
    for i, r in enumerate(recs, start=1):
        clean_name = r.name.replace(' ', '').replace('&', 'And')
        # Mark top recommendations with "_recommended" suffix
        if i <= top_recommendations:
            tag = f"{prefix}_{clean_name}_recommended"
        else:
            tag = f"{prefix}_{clean_name}"
        planned.append((tag, r.dials))
    
    return planned

def recommend_mastering_styles_from_metrics(rep) -> List[Tuple[str, float, str]]:
    """
    Suggest LocalMasterProvider styles from analysis:
    Returns list of tuples: (style, strength 0..1, why)
    """
    lufs = getattr(rep, "lufs_integrated", -20.0)
    crest = rep.basic["crest_db"] if hasattr(rep, "basic") else rep.get("crest_db", 12.0)
    airp  = getattr(rep, "air_energy_pct", 0.2)
    bass  = getattr(rep, "bass_energy_pct", 40.0)

    out: List[Tuple[str,float,str]] = []
    # If dark → try "bright"
    if airp < 0.05:
        out.append(("bright", 0.6, f"Low air ({airp:.3f}) → add sheen"))
    # If bass-heavy → try "neutral" vs "warm" (depending on taste)
    if bass > 70:
        out.append(("neutral", 0.5, f"High bass energy ({bass:.1f}%) → keep low-end in check"))
    else:
        out.append(("warm", 0.5, f"Moderate bass ({bass:.1f}%) → touch of weight"))
    # If very dynamic & quiet → try "loud" moderately
    if crest > 16 and lufs < -18:
        out.append(("loud", 0.55, f"Dynamic ({crest:.1f} dB) & quiet ({lufs:.1f} LUFS) → more forward"))

    # Always keep a neutral baseline
    if not any(s == "neutral" for s,_,_ in out):
        out.insert(0, ("neutral", 0.5, "Baseline reference"))

    # Deduplicate, preserve order
    seen=set(); filtered=[]
    for s in out:
        if s[0] in seen: continue
        seen.add(s[0]); filtered.append(s)
    return filtered

# ---------- 4) Human-readable summary helpers ----------

def recommendation_summary(recs: List[Recommendation]) -> str:
    lines = []
    for r in recs:
        lines.append(f"- {r.name} (priority {r.priority}): "
                     f"B{r.dials.bass:.0f} P{r.dials.punch:.0f} C{r.dials.clarity:.0f} A{r.dials.air:.0f} W{r.dials.width:.0f}")
        for why in r.rationale:
            lines.append(f"    • {why}")
        if r.notes:
            lines.append(f"    ↪ {r.notes}")
    return "\n".join(lines)

print("Presets & Recommendations layer loaded:")
print("- PRESETS dict, list_presets(), get_preset(name)")
print("- recommend_from_analysis(rep) → [Recommendation]")
print("- build_premaster_plan_from_recs(recs, limit) [legacy]")
print("- build_all_variants_plan(recs, prefix, top_recommendations) [comprehensive]")
print("- recommend_mastering_styles_from_metrics(rep)")
print("- recommendation_summary(recs)")
