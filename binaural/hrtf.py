"""Head-related transfer functions: database, interpolation and SOFA import.

Two backends are supported:

* :class:`ProceduralHRTF` -- a deterministic generic-mannequin model built
  from acoustic components (a signed sinusoidal interaural time difference,
  head shadow, pinna notch, torso reflection). It needs no external data and
  is fully reproducible, but it is *not* an individual human measurement.
* :class:`SOFAHRTF` -- measured HRIRs loaded from a SOFA file (SimpleFreeFieldHRIR
  convention), used whenever the exhibit provides a database file.

Interpolation between grid directions is performed on the complex spectra
(bilinear interpolation in azimuth/elevation), never by crossfading channel
gains. Each rendered direction therefore gets its own continuously varying
pair of binaural impulse responses.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.signal import lfilter

from .coordinates import (
    EAR_DISTANCE,
    SPEED_OF_SOUND,
    direction_to_unit_vector,
    normalize_angle,
)

DEFAULT_SAMPLE_RATE = 48000
DEFAULT_HRIR_LENGTH = 128  # samples (~2.7 ms at 48 kHz)


@dataclass
class HRTFQuery:
    """One interpolated HRIR pair and where it came from."""

    azimuth_deg: float
    elevation_deg: float
    hrir: np.ndarray  # shape (2, N): row 0 = left, row 1 = right
    corner_indices: tuple  # (az_idx0, az_idx1, el_idx0, el_idx1)
    weights: tuple  # (w_az, w_el)
    source: str  # "procedural" or sofa file name


class HRTFBase:
    """Common grid storage and complex-spectrum interpolation."""

    def __init__(self, azimuths: np.ndarray, elevations: np.ndarray,
                 hrirs: np.ndarray, sample_rate: int, source_name: str,
                 metadata: Optional[dict] = None):
        self.azimuths = np.asarray(azimuths, dtype=float)
        self.elevations = np.asarray(elevations, dtype=float)
        self.hrirs = np.asarray(hrirs, dtype=np.float64)
        # shape: (n_az, n_el, 2, N)
        if self.hrirs.shape[:3] != (len(self.azimuths),
                                    len(self.elevations), 2):
            raise ValueError("hrirs must have shape (n_az, n_el, 2, N)")
        self.sample_rate = int(sample_rate)
        self.source_name = source_name
        self.metadata = metadata or {}
        self.hrir_length = self.hrirs.shape[-1]
        self.nfft = int(2 ** np.ceil(np.log2(2 * self.hrir_length)))
        self._spectra = np.fft.rfft(self.hrirs, n=self.nfft, axis=-1)

    # ------------------------------------------------------------------
    def query(self, azimuth_deg: float, elevation_deg: float) -> HRTFQuery:
        """Return an interpolated (left, right) HRIR for one direction."""
        az = normalize_angle(float(azimuth_deg))
        el = float(np.clip(elevation_deg,
                           self.elevations[0], self.elevations[-1]))

        az_idx, w_az = _fractional_index(self.azimuths, az, circular=True)
        el_idx, w_el = _fractional_index(self.elevations, el, circular=False)
        a0, a1 = az_idx
        e0, e1 = el_idx

        s00 = self._spectra[a0, e0]
        s10 = self._spectra[a1, e0]
        s01 = self._spectra[a0, e1]
        s11 = self._spectra[a1, e1]
        s0 = s00 * (1.0 - w_az) + s10 * w_az
        s1 = s01 * (1.0 - w_az) + s11 * w_az
        spectrum = s0 * (1.0 - w_el) + s1 * w_el

        hrir = np.fft.irfft(spectrum, n=self.nfft, axis=-1)
        hrir = hrir[:, : self.hrir_length].copy()
        return HRTFQuery(
            azimuth_deg=az,
            elevation_deg=el,
            hrir=hrir,
            corner_indices=(a0, a1, e0, e1),
            weights=(w_az, w_el),
            source=self.source_name,
        )

    # ------------------------------------------------------------------
    def grid_record(self) -> dict:
        """Serializable description of the database, exported for review."""
        return {
            "source": self.source_name,
            "sample_rate": self.sample_rate,
            "hrir_length_samples": self.hrir_length,
            "azimuths_deg": self.azimuths.tolist(),
            "elevations_deg": self.elevations.tolist(),
            "metadata": self.metadata,
        }


def _fractional_index(grid: np.ndarray, value: float,
                      circular: bool) -> tuple[tuple[int, int], float]:
    """Locate value on a monotonic grid; return ((i0, i1), weight toward i1)."""
    n = len(grid)
    if circular:
        step = abs(grid[1] - grid[0])
        base = grid[0]
        pos = (value - base) / step
        i0 = int(np.floor(pos)) % n
        frac = pos - np.floor(pos)
        i1 = (i0 + 1) % n
        return (i0, i1), float(frac)

    if value <= grid[0]:
        return (0, 0), 0.0
    if value >= grid[-1]:
        return (n - 1, n - 1), 0.0
    i1 = int(np.searchsorted(grid, value))
    i0 = i1 - 1
    w = (value - grid[i0]) / (grid[i1] - grid[i0])
    return (i0, i1), float(w)


# ======================================================================
# Procedural generic-mannequin HRTF
# ======================================================================
class ProceduralHRTF(HRTFBase):
    """Analytic HRIR model used when no measured SOFA database is available.

    Physical ingredients (per ear):

    * a signed sinusoidal interaural time difference law in azimuth,
      ``tau = side * (pi/2) * (a/c) * sin(azimuth)``, realized with
      fractional-sample delays so the field is smooth across the grid;
    * head shadow as a one-pole low-pass whose cutoff drops on the
      contralateral side (frequency-dependent interaural level difference);
    * a pinna reflection whose elevation-dependent delay moves a comb notch
      across the spectrum (the main elevation cue);
    * a damped torso/shoulder reflection (~1.2 ms).

    The two ears share one global gain normalization so interaural level
    differences are preserved.
    """

    def __init__(self, sample_rate: int = DEFAULT_SAMPLE_RATE,
                 hrir_length: int = DEFAULT_HRIR_LENGTH,
                 azimuth_step_deg: float = 5.0,
                 elevation_step_deg: float = 10.0,
                 head_radius_m: float = EAR_DISTANCE):
        # Half-open periodic azimuth grid: -180 .. 180-step. The point
        # +180 is reached by circular interpolation wrapping to -180, which
        # represents the same physical direction, so the seam is continuous.
        azimuths = np.arange(-180.0, 180.0, azimuth_step_deg)
        elevations = np.arange(-90.0, 90.0 + elevation_step_deg / 2,
                               elevation_step_deg)
        n = hrir_length
        hrirs = np.zeros((len(azimuths), len(elevations), 2, n))

        for ia, az in enumerate(azimuths):
            for ie, el in enumerate(elevations):
                unit = direction_to_unit_vector(az, el)
                pair = np.stack([
                    _procedural_ear(unit, +1, sample_rate, n, head_radius_m),
                    _procedural_ear(unit, -1, sample_rate, n, head_radius_m),
                ])
                hrirs[ia, ie] = pair

        peak = np.max(np.abs(hrirs))
        if peak > 0:
            hrirs *= 0.99 / peak

        super().__init__(
            azimuths, elevations, hrirs, sample_rate,
            source_name="procedural-generic-v1",
            metadata={
                "model": ("signed sinusoidal ITD + head shadow + pinna notch "
                          "+ front/back shelf + torso"),
                "head_radius_m": head_radius_m,
                "speed_of_sound_mps": SPEED_OF_SOUND,
                "azimuth_step_deg": azimuth_step_deg,
                "elevation_step_deg": elevation_step_deg,
                "note": "generic mannequin, not an individual measurement",
            },
        )


def _one_pole_lowpass_impulse(delay_samples: float, n: int,
                              cutoff_hz: float, fs: int) -> np.ndarray:
    """One-pole low-pass impulse with fractional-sample onset delay.

    Sub-sample delays are essential for smooth azimuth interpolation: an
    integer-rounded ITD would jump by a whole sample between nearby grid
    directions and create discontinuities larger than the interpolation can
    hide.
    """
    pole = np.exp(-2.0 * np.pi * cutoff_hz / fs)
    d = max(0.0, float(delay_samples))
    base = int(np.floor(d))
    frac = d - base
    half_taps = 4
    onset = np.zeros(n)
    if base >= n:
        return onset
    if frac == 0.0:
        onset[base] = 1.0
    else:
        # k=0 centers the kernel on the fractional position; negative taps
        # pull in pre-cursor samples, positive taps the post-cursor ones.
        k = np.arange(-half_taps, half_taps + 1)
        kernel = np.sinc(k - frac) * np.hanning(2 * half_taps + 1)
        for j, w in zip(k, kernel):
            src = base + j
            if 0 <= src < n:
                onset[src] += w
    return lfilter([1.0], [1.0, -pole], onset)


def _procedural_ear(unit: np.ndarray, side: int, fs: int, n: int,
                    head_radius: float) -> np.ndarray:
    """Build one ear's HRIR. side=+1 left, side=-1 right."""
    ear_axis = np.array([0.0, float(side), 0.0])
    cos_theta = float(np.clip(np.dot(unit, ear_axis), -1.0, 1.0))
    frontness = float(np.clip(unit[0], -1.0, 1.0))  # +1 front, -1 behind

    # --- interaural time difference, fractional samples ---------------
    # Signed sinusoidal ITD law in source azimuth (unit[1] = sin(azimuth)):
    # zero on the median plane, opposite signs for the two ears, and the
    # same sign at the front and back of a given side. The peak magnitude
    # K*a/c with K ~= pi/2 sits in the measured human range, while the law
    # is differentiable over the whole circle and carries sub-sample
    # resolution, so neighboring grid directions never jump a whole sample.
    itd_constant = ((np.pi / 2.0) * head_radius / SPEED_OF_SOUND * fs)
    delay = -float(side) * itd_constant * float(unit[1])

    # --- head shadow: stronger low-pass toward the contralateral ear -
    # Smooth half-cosine shadow factor: zero on the ipsilateral side and
    # differentiable at the 90-degree crossing, so neighboring azimuth grid
    # points never see a slope kink.
    shadow = 0.5 * (1.0 - np.cos(np.pi * np.clip(-cos_theta, 0.0, 1.0)))
    cutoff = 5000.0 - 4200.0 * shadow
    h = _one_pole_lowpass_impulse(delay, n, cutoff, fs)
    # A one-pole ring-down carries more truncated-window energy as the cutoff
    # drops, which would otherwise make the *shadowed* ear louder. Normalize
    # the pulse energy so head shadow only reshapes the spectrum, then apply
    # the intended diffraction loss explicitly to create the ILD.
    pole = np.exp(-2.0 * np.pi * cutoff / fs)
    unit_pulse_energy = (1.0 - pole) / (1.0 + pole)
    h *= np.sqrt(unit_pulse_energy / max(np.sum(h ** 2), 1e-20))
    # extra contralateral broadband attenuation (head diffraction loss)
    h *= 1.0 - 0.45 * shadow

    # --- pinna: elevation-controlled reflection notch ----------------
    el = np.arcsin(np.clip(unit[2], -1.0, 1.0))
    pinna_delay_s = 0.00032 - 0.00022 * np.sin(el)  # 0.10 .. 0.32 ms
    d_pinna = pinna_delay_s * fs
    pinna_gain = 0.40 * (1.0 - 0.3 * shadow)
    pinna = _one_pole_lowpass_impulse(delay + d_pinna, n, 8000.0, fs)
    h = h - pinna_gain * pinna

    # --- front/back pinna coloration ---------------------------------
    # A short (~70 us) reflection is strong for frontal sources and weak
    # behind, shifting a comb notch around 6-8 kHz: the classic generic
    # front/back spectral cue that prevents front and back being identical.
    d_fb = 0.00007 * fs
    fb_gain = 0.35 * (0.5 + 0.5 * frontness) * (1.0 - 0.4 * shadow)
    fb_reflection = _one_pole_lowpass_impulse(
        delay + d_fb, n, 9000.0, fs)
    h = h + fb_gain * fb_reflection
    # rear sources lose more high-frequency energy (pinna/back-of-head)
    rear_loss = 0.75 + 0.25 * frontness  # 0.5 behind .. 1.0 front
    h = _shelf(h, rear_loss, cutoff_hz=3500.0, fs=fs)

    # --- torso / shoulder reflection (~1.2 ms, damped) ---------------
    torso_gain = 0.15 * max(0.0, np.cos(el)) * (0.6 + 0.4 * frontness)
    d_torso = 0.0012 * fs
    torso = _one_pole_lowpass_impulse(delay + d_torso, n, 2000.0, fs)
    h = h + torso_gain * torso

    # --- broadband diffraction loss on the complete ear response ------
    # Pinna/torso reflections would otherwise refill the contralateral
    # energy removed by head shadow. A single smooth factor attenuates the
    # whole shadowed-ear HRIR, producing a physical ~8 dB ILD at 90 degrees
    # while leaving the spectral coloration cues above untouched.
    h *= 1.0 - 0.6 * shadow
    return h


def _shelf(h: np.ndarray, high_factor: float, cutoff_hz: float,
           fs: int) -> np.ndarray:
    """First-order high-frequency gain shelf; high_factor=1 is identity."""
    if abs(high_factor - 1.0) < 1e-12:
        return h
    pole = np.exp(-2.0 * np.pi * cutoff_hz / fs)
    low = lfilter([1.0 - pole], [1.0, -pole], h)
    high = h - low
    return low + high_factor * high


# ======================================================================
# SOFA (measured) HRTF
# ======================================================================
class SOFAHRTF(HRTFBase):
    """Measured HRIRs from a SOFA ``SimpleFreeFieldHRIR`` file.

    SOFA stores positions as spherical (azimuth, elevation, radius) in
    degrees with the same angular convention used here. Positions are
    re-gridded onto a regular azimuth/elevation table by nearest-neighbor
    assignment (measured databases are typically already on such a grid).
    """

    def __init__(self, sofa_path: str,
                 azimuth_step_deg: float = 5.0,
                 elevation_step_deg: float = 5.0):
        try:
            import h5py  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "loading SOFA files requires the optional 'h5py' package"
            ) from exc
        import h5py

        with h5py.File(sofa_path, "r") as f:
            convention = _read_str(f.attrs.get("Conventions", b""))
            if convention and "HRIR" not in convention:
                raise ValueError(f"unsupported SOFA convention: {convention}")
            fs = int(np.array(f["Data.SamplingRate"]).reshape(-1)[0])
            ir = np.asarray(f["Data.IR"][()])  # (M, R, N): R=2 (L, R)
            pos = np.asarray(f["SourcePosition"][()])  # (M, 3) deg,deg,m
            delay = np.asarray(f["Data.Delay"][()]).reshape(-1)

        if ir.ndim != 3 or ir.shape[1] != 2:
            raise ValueError("expected Data.IR of shape (measurements, 2, N)")
        if int(round(delay[0])) != 0:
            # Non-zero SOFA delays would need compensation; reject explicitly.
            raise ValueError("SOFA files with non-zero Data.Delay are not "
                             "supported by this exhibit build")

        az_grid = np.arange(-180.0, 180.0, azimuth_step_deg)
        el_grid = np.arange(-90.0, 90.0 + elevation_step_deg / 2,
                            elevation_step_deg)
        n = ir.shape[-1]
        table = np.zeros((len(az_grid), len(el_grid), 2, n))
        counts = np.zeros((len(az_grid), len(el_grid)), dtype=int)

        for measurement, (az, el, _r) in enumerate(pos):
            ia = int(round(normalize_angle(az) / azimuth_step_deg)) % len(az_grid)
            ie = int(round((el + 90.0) / elevation_step_deg))
            ie = int(np.clip(ie, 0, len(el_grid) - 1))
            table[ia, ie] += ir[measurement]
            counts[ia, ie] += 1

        missing = counts == 0
        if missing.any():
            raise ValueError(
                "SOFA grid has empty cells; choose steps matching the file "
                f"({missing.sum()} missing at steps "
                f"{azimuth_step_deg}/{elevation_step_deg})"
            )
        table /= np.maximum(counts[..., None, None], 1)

        super().__init__(
            az_grid, el_grid, table, fs,
            source_name=str(sofa_path),
            metadata={"convention": convention or "SimpleFreeFieldHRIR",
                      "measurements": int(ir.shape[0])},
        )


def _read_str(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


def load_hrtf(sofa_path: Optional[str] = None,
              sample_rate: int = DEFAULT_SAMPLE_RATE,
              **kwargs) -> HRTFBase:
    """Factory: measured SOFA when a path is given, procedural otherwise."""
    if sofa_path:
        return SOFAHRTF(sofa_path, **kwargs)
    return ProceduralHRTF(sample_rate=sample_rate, **kwargs)
