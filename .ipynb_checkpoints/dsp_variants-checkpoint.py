import numpy as np
from scipy.io import wavfile
from scipy import signal

# --- Shared helpers ---
def _to_float32(x):
    if x.dtype == np.int16:  return x.astype(np.float32) / 32768.0
    if x.dtype == np.int32:  return x.astype(np.float32) / 2147483648.0
    if x.dtype == np.uint8:  return (x.astype(np.float32) - 128.0) / 128.0
    return x.astype(np.float32)

def _ensure_stereo(x):
    if x.ndim == 1: return np.stack([x, x], axis=-1)
    if x.shape[1] == 1: return np.repeat(x, 2, axis=1)
    return x

def _sanitize(x):
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(x, -4.0, 4.0)

def peak_normalize(x, target_db=-1.0, eps=1e-9):
    peak = np.max(np.abs(x))
    if not np.isfinite(peak) or peak < eps:
        return np.zeros_like(x)
    return x * (10**(target_db/20.0) / peak)

# --- Robust Bass Boost (musical low shelf + headroom) ---
def apply_boost_bass_robust(in_wav, out_wav,
                            shelf_hz=100.0, shelf_db=3.0, Q=0.7,
                            headroom_db=-1.0):
    sr, data = wavfile.read(in_wav)
    x = _ensure_stereo(_to_float32(data))
    x = _sanitize(x)

    # RBJ low shelf
    A = 10**(shelf_db/40.0)
    w0 = 2*np.pi*shelf_hz/sr
    alpha = np.sin(w0)/(2*Q)
    cosw0 = np.cos(w0)

    b0 =    A*((A+1) - (A-1)*cosw0 + 2*np.sqrt(A)*alpha)
    b1 =  2*A*((A-1) - (A+1)*cosw0)
    b2 =    A*((A+1) - (A-1)*cosw0 - 2*np.sqrt(A)*alpha)
    a0 =       (A+1) + (A-1)*cosw0 + 2*np.sqrt(A)*alpha
    a1 =  -2*((A-1) + (A+1)*cosw0)
    a2 =       (A+1) + (A-1)*cosw0 - 2*np.sqrt(A)*alpha

    b = np.array([b0, b1, b2]) / a0
    a = np.array([1.0, a1/a0, a2/a0])

    sos = signal.tf2sos(b, a)
    y = signal.sosfilt(sos, x, axis=0)

    y = _sanitize(y)
    y = peak_normalize(y, target_db=headroom_db)

    wavfile.write(out_wav, sr, y.astype(np.float32))
    return y

# --- Robust Kick/Bass Tightening (sidechain duck) ---
def apply_tighten_kick_bass_robust(in_wav, out_wav,
                                   kick_lo=40.0, kick_hi=110.0,
                                   low_cutoff=120.0,
                                   duck_depth_db=4.0,
                                   attack_ms=4.0, release_ms=90.0,
                                   mud_hz=180.0, mud_db=-1.2,
                                   hpf_hz=20.0, headroom_db=-1.0):
    sr, data = wavfile.read(in_wav)
    x = _ensure_stereo(_to_float32(data))
    x = _sanitize(x)

    # HPF cleanup
    sos_hp = signal.butter(2, hpf_hz/(sr*0.5), btype="highpass", output="sos")
    y = signal.sosfilt(sos_hp, x, axis=0)

    # Split bands
    sos_lp = signal.butter(4, low_cutoff/(sr*0.5), btype="lowpass", output="sos")
    sos_kb = signal.butter(4, [kick_lo/(sr*0.5), kick_hi/(sr*0.5)], btype="bandpass", output="sos")
    low = signal.sosfilt(sos_lp, y, axis=0)
    kick = signal.sosfilt(sos_kb, y, axis=0)
    nonkick_low = low - kick

    # Envelope & ducking
    env = np.abs(np.mean(kick, axis=1))
    env = signal.sosfilt(signal.butter(2, 10.0/(sr*0.5), btype="lowpass", output="sos"), env)
    # Normalize env
    env /= np.percentile(env, 95) if env.size else 1.0
    env = np.clip(env, 0.0, 1.0)

    # Smooth
    a_a = np.exp(-1.0/((attack_ms/1000.0)*sr))
    a_r = np.exp(-1.0/((release_ms/1000.0)*sr))
    out_env, prev = np.zeros_like(env), 0.0
    for i, e in enumerate(env):
        prev = a_a*prev + (1-a_a)*e if e > prev else a_r*prev + (1-a_r)*e
        out_env[i] = prev

    floor_gain = 10**(-duck_depth_db/20.0)
    gain_curve = floor_gain + (1.0 - floor_gain) * (1.0 - out_env)
    ducked_nonkick = nonkick_low * gain_curve[:, None]

    # Recombine
    lows_tight = ducked_nonkick + kick
    high = y - low
    y_out = lows_tight + high

    # Anti-mud EQ (optional peaking cut)
    if mud_db != 0.0:
        y_out = signal.sosfilt(signal.iirpeak(mud_hz/(sr*0.5), Q=1.0, gain=mud_db), y_out, axis=0)

    # Final cleanup
    y_out = _sanitize(y_out)
    y_out = peak_normalize(y_out, target_db=headroom_db)

    wavfile.write(out_wav, sr, y_out.astype(np.float32))
    return y_out
