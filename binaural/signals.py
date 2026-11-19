"""Source material: impulse, broadband signal and short speech.

All signals are generated or decoded once into in-memory mono arrays. The
real-time callback only indexes these arrays -- it never touches a file,
which keeps the audio thread free of unbounded I/O.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

SIGNAL_SPEECH_TEXT = "virtual acoustic exhibit"


def impulse_signal(sample_rate: int = 48000,
                   duration_s: float = 1.0) -> np.ndarray:
    """Unit impulse at the first sample, padded to ``duration_s``."""
    x = np.zeros(int(round(duration_s * sample_rate)))
    x[0] = 1.0
    return x


def broadband_signal(sample_rate: int = 48000,
                     duration_s: float = 3.0,
                     seed: int = 20261119) -> np.ndarray:
    """Repeated logarithmic chirps: deterministic, broadband, periodic.

    A chirp train (rather than white noise) gives the ear clear frequency
    sweeps for judging timbre and interaural cues while remaining seamless
    when looped.
    """
    n = int(round(duration_s * sample_rate))
    t = np.arange(n) / sample_rate
    period = 0.5
    phase = 2.0 * np.pi * ((t % period) / period)
    # exponential chirp 100 Hz -> 12 kHz within each period
    f0, f1 = 100.0, min(12000.0, sample_rate * 0.45)
    k = np.log(f1 / f0) / period
    instantaneous = f0 * (np.exp(k * (t % period)) - 1.0) / k
    phase = 2.0 * np.pi * instantaneous
    env = 0.6 * (0.5 + 0.5 * np.cos(2.0 * np.pi * t / period - np.pi))
    x = env * np.sin(phase)
    rng = np.random.default_rng(seed)
    x += 0.02 * rng.standard_normal(n)
    # guarantee exact loop continuity
    x = x - np.linspace(x[0], x[-1], n)
    peak = np.max(np.abs(x))
    return x / peak * 0.8


def _formant_speech(sample_rate: int, text: str,
                    duration_s: float = 2.0) -> np.ndarray:
    """Deterministic voiced-formant fallback resembling short speech.

    Used only when no TTS engine is installed. It synthesizes a sequence of
    syllable-like formant bursts (F1/F2 trajectories with a voiced source),
    which is enough for localization tests even though it carries no words.
    """
    n_syllables = max(3, len(text.split()))
    n = int(round(duration_s * sample_rate))
    t = np.arange(n) / sample_rate
    rng = np.random.default_rng(7)
    f1_centers = rng.uniform(300, 700, n_syllables)
    f2_centers = rng.uniform(900, 2200, n_syllables)
    syllable_len = n // n_syllables
    out = np.zeros(n)
    for i in range(n_syllables):
        s, e = i * syllable_len, min(n, (i + 1) * syllable_len)
        tt = np.arange(e - s) / sample_rate
        f0 = 110.0 + 30.0 * np.sin(2 * np.pi * 3.0 * tt)
        phase = 2 * np.pi * np.cumsum(f0) / sample_rate
        voice = np.sin(phase) + 0.35 * np.sin(2 * phase)
        env = np.sin(np.pi * np.linspace(0, 1, e - s)) ** 2
        seg = voice * env
        for f_center in (f1_centers[i], f2_centers[i]):
            # crude resonator: narrow band-pass via modulated cosine
            seg = seg + 0.5 * env * np.sin(
                2 * np.pi * f_center * tt + phase)
        out[s:e] += seg
    peak = np.max(np.abs(out))
    return out / peak * 0.8 if peak > 0 else out


def speech_signal(sample_rate: int = 48000,
                  text: str = SIGNAL_SPEECH_TEXT) -> np.ndarray:
    """Short spoken phrase via espeak-ng when available, else formant synth."""
    espeak = shutil.which("espeak-ng") or shutil.which("espeak")
    if espeak:
        try:
            with tempfile.TemporaryDirectory() as tmp:
                wav_path = Path(tmp) / "speech.wav"
                subprocess.run(
                    [espeak, "-s", "140", "-w", str(wav_path), text],
                    check=True, capture_output=True, timeout=20)
                data, fs = _read_wav_mono(wav_path)
            if fs != sample_rate:
                data = _resample_linear(data, fs, sample_rate)
            return _normalize(data)
        except (subprocess.SubprocessError, OSError, ValueError):
            pass  # fall through to the built-in synthesizer
    return _formant_speech(sample_rate, text)


def load_user_wav(path: str | Path,
                  target_sample_rate: int = 48000) -> np.ndarray:
    """Load and downmix any user WAV to mono at the renderer sample rate."""
    data, fs = _read_wav_mono(path)
    if fs != target_sample_rate:
        data = _resample_linear(data, fs, target_sample_rate)
    return _normalize(data)


# ----------------------------------------------------------------------
def _read_wav_mono(path: str | Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    data, fs = sf.read(str(path), always_2d=False)
    if data.ndim == 2:
        data = data.mean(axis=1)
    if np.issubdtype(data.dtype, np.integer):
        max_val = float(np.iinfo(data.dtype).max)
        data = data.astype(np.float64) / max_val
    return data.astype(np.float64), int(fs)


def _resample_linear(x: np.ndarray, fs_in: int, fs_out: int) -> np.ndarray:
    if fs_in == fs_out:
        return x
    duration = x.size / fs_in
    n_out = int(round(duration * fs_out))
    t_in = np.arange(x.size) / fs_in
    t_out = np.arange(n_out) / fs_out
    return np.interp(t_out, t_in, x)


def _normalize(x: np.ndarray, peak_target: float = 0.8) -> np.ndarray:
    peak = np.max(np.abs(x))
    return x / peak * peak_target if peak > 0 else x


def build_default_library(sample_rate: int = 48000) -> dict[str, np.ndarray]:
    """The three required exhibit signals, fully loaded into memory."""
    return {
        "impulse": impulse_signal(sample_rate),
        "broadband": broadband_signal(sample_rate),
        "speech": speech_signal(sample_rate),
    }
